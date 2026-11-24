"""Contracts for the upres_cove review scene and the compare tool's --demo mode."""

import argparse
import dataclasses
import importlib.util
import os

import pytest

from tools.demos import dam_break_cove as cove
from tools.demos import upres_cove as uc

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIB = 1024 ** 3


def _compare_tool():
    path = os.path.join(REPO, "tools", "profiling", "compare_guided_upres.py")
    spec = importlib.util.spec_from_file_location("compare_guided_upres", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CMP = _compare_tool()


def _coarse_settings():
    from tools.profiling import profile_gpu_solver as P

    ns = argparse.Namespace(
        device="cpu", voxel_size=uc.VOXEL, max_particles=uc.MAX_PARTICLES,
        min_substeps=None, max_substeps=None, max_whitewater=0, no_surface=False,
        whitewater=False, scene_preset="dam_break", no_whitewater=True,
        viscosity=False, surface_tension=False)
    settings = P._build_settings(ns)
    settings.leaf_capacity = uc.LEAF_CAPACITY
    settings.narrow_band_enabled = False
    return settings


def test_scene_is_the_dam_break_cove_at_double_resolution():
    assert uc.DOMAIN_LO == cove.DOMAIN_LO and uc.DOMAIN_HI == cove.DOMAIN_HI
    assert uc.RESERVOIR_LO == cove.RESERVOIR_LO and uc.RESERVOIR_HI == cove.RESERVOIR_HI
    assert uc.VOXEL == pytest.approx(cove.VOXEL / 2)
    assert uc.fine_voxel() == pytest.approx(cove.VOXEL / 4)


def test_particle_count_is_far_above_every_other_demo():
    emitted = uc.expected_particles()
    # The largest shipped reserve is 900k (paddle_stir, deep_plunge); dam_break
    # emits ~256k from the same reservoir at 3 cm.
    assert emitted > 2 * 900_000
    assert emitted == pytest.approx(8 * uc.expected_particles(cove.VOXEL))
    assert emitted * 1.15 < uc.MAX_PARTICLES  # Initial Fill fits the reserve


def test_coarse_leaves_hold_the_reservoir_with_room_for_the_surge():
    cells = [round((h - lo) / uc.VOXEL) for lo, h in zip(uc.RESERVOIR_LO, uc.RESERVOIR_HI,
                                                        strict=True)]
    # One ring of padding leaves around the column (topology expand).
    leaves = 1
    for n in cells:
        leaves *= -(-n // 8) + 2
    assert 3 * leaves < uc.LEAF_CAPACITY


def test_coarse_budget_covers_default_scene_admission():
    # Blender's capacity planner counted the authored source and all colliders.
    assert uc.LEAF_CAPACITY >= 6205


def test_fine_budgets_match_what_guided_upres_allocates():
    from fluidSim_warp_blender51.sim.guided_upres import fine_budget_scales

    coarse = _coarse_settings()
    full = uc.fine_settings(coarse)
    p_scale, _ = fine_budget_scales(uc.UPRES_FACTOR)
    guided_leaves = int(coarse.leaf_capacity * uc.leaf_budget_scale())
    assert full.leaf_capacity == guided_leaves
    assert abs(guided_leaves - uc.FINE_LEAF_CAPACITY) <= 1
    assert full.voxel_size == pytest.approx(coarse.voxel_size / 2)
    assert full.max_particles == coarse.max_particles * 8
    assert full.whitewater_enabled is False
    assert int(coarse.max_particles * p_scale) == 4 * uc.MAX_PARTICLES


def test_each_pass_fits_a_16_gb_card():
    """Coarse plus one fine solve at a time, by the add-on's own estimate."""
    from fluidSim_warp_blender51.sim.guided_upres import fine_budget_scales
    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes

    coarse = _coarse_settings()
    p_scale, _ = fine_budget_scales(uc.UPRES_FACTOR)
    guided = dataclasses.replace(
        coarse, voxel_size=coarse.voxel_size / uc.UPRES_FACTOR,
        max_particles=int(coarse.max_particles * p_scale),
        leaf_capacity=int(coarse.leaf_capacity * uc.leaf_budget_scale()),
        narrow_band_enabled=True, narrow_band_voxels=uc.SHELL_VOXELS,
        whitewater_enabled=False)
    full = uc.fine_settings(coarse)
    base = estimate_solver_bytes(coarse)
    for fine in (guided, full):
        assert base + estimate_solver_bytes(fine) < 14 * GIB


def test_builder_registers_the_opt_in_scene_outside_the_gallery():
    with open(os.path.join(REPO, "tools", "demos", "build_demo_scenes.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    assert '"upres_cove": build_upres_cove' in src
    assert "def author_upres_cove" in src
    assert "upres_cove" not in src.split("DEFAULT_SCENES = ")[1].split("\n")[0]
    with open(os.path.join(REPO, "demos", "MANIFEST.md"), encoding="utf-8") as fh:
        assert "upres_cove" not in fh.read()


def test_preset_mode_keeps_its_historical_defaults():
    args = CMP.parse([])
    assert (args.demo, args.voxel_size, args.factor, args.shell,
            args.core_shells, args.frames) == (None, 0.06, 2, 4.0, 2.0, [20, 40])


def test_demo_mode_takes_the_demo_defaults_unless_overridden():
    args = CMP.resolve_demo_args(CMP.parse(["--demo", "upres_cove"]),
                                 CMP.demo_defaults("upres_cove"))
    assert args.voxel_size is None  # the scene's own resolution
    assert (args.factor, args.shell, args.core_shells) == (
        uc.UPRES_FACTOR, uc.SHELL_VOXELS, uc.CORE_SHELLS)
    assert args.frames == list(uc.REVIEW_FRAMES)
    args = CMP.resolve_demo_args(
        CMP.parse(["--demo", "upres_cove", "--factor", "3", "--frames", "10",
                   "--core-shells", "0"]), CMP.demo_defaults("upres_cove"))
    assert (args.factor, args.core_shells, args.frames) == (3, 0.0, [10])


def test_demos_without_upres_defaults_use_the_generic_ones():
    assert CMP.demo_defaults("no_such_demo") is None
    assert CMP.demo_defaults("tsunami_parthenon") is None  # geometry module only
    args = CMP.resolve_demo_args(CMP.parse(["--demo", "dam_break"]), None)
    assert (args.factor, args.shell, args.core_shells, args.frames) == (2, 4.0, 2.0, [20, 40])


def test_summary_reports_the_speedup_over_full_fine_after_warm_up():
    records = [
        {"coarse_ms": 900.0, "guided_fine_ms": 900.0, "guided_total_ms": 1800.0,
         "full_fine_ms": 900.0},
        {"coarse_ms": 20.0, "guided_fine_ms": 30.0, "guided_total_ms": 50.0,
         "full_fine_ms": 100.0},
        {"coarse_ms": 20.0, "guided_fine_ms": 50.0, "guided_total_ms": 70.0,
         "full_fine_ms": 140.0},
    ]
    out = CMP.summarize(records)
    assert out["mean_guided_total_ms"] == 60.0
    assert out["mean_full_fine_ms"] == 120.0
    assert out["speedup_vs_full_fine"] == 2.0
    no_full = [dict(r, full_fine_ms=None) for r in records]
    assert "speedup_vs_full_fine" not in CMP.summarize(no_full)


class _Anim:
    def __init__(self, action=None, drivers=()):
        self.action, self.drivers = action, list(drivers)


class _Obj:
    def __init__(self, name, role, anim=None, parent=None):
        self.name, self.animation_data, self.parent = name, anim, parent
        self.fluidsim = type("FS", (), {"role": role})()


def test_demo_mode_names_animated_inputs_it_cannot_replay():
    rig = _Obj("Rig", "NONE", anim=_Anim(action="spin"))
    objects = [
        _Obj("Paddle", "COLLIDER", anim=_Anim(action="turn")),
        _Obj("Blade", "COLLIDER", parent=rig),
        _Obj("Driven Source", "SOURCE", anim=_Anim(drivers=["d"])),
        _Obj("Rock", "COLLIDER", anim=_Anim()),
        _Obj("Camera", "NONE", anim=_Anim(action="fly")),
        rig,
    ]
    assert CMP.animated_inputs(objects) == ["Paddle", "Blade", "Driven Source"]


def test_demo_mode_rejects_nla_and_animated_mesh_data():
    stacked = _Obj("NLA source", "SOURCE", anim=_Anim())
    stacked.animation_data.nla_tracks = ["track"]
    deforming = _Obj("Animated mesh", "COLLIDER")
    deforming.data = _Obj("Mesh data", "NONE", anim=_Anim(action="deform"))
    disabled = _Obj("Disabled input", "SOURCE", anim=_Anim(action="move"))
    disabled.fluidsim.enabled = False
    assert CMP.animated_inputs([stacked, deforming, disabled]) == [
        "NLA source", "Animated mesh",
    ]


def test_demo_mode_identifies_enabled_automatic_seeds_from_scene_inputs():
    seed = _Obj("Water seed", "SOURCE")
    seed.fluidsim.enabled = seed.fluidsim.auto_inject_seed = True
    disabled = _Obj("Disabled seed", "SOURCE")
    disabled.fluidsim.enabled = False
    disabled.fluidsim.auto_inject_seed = True
    initial_fill = _Obj("Initial fill", "SOURCE")
    initial_fill.fluidsim.enabled = True
    assert CMP.automatic_seed_objects([seed, disabled, initial_fill]) == ["Water seed"]
