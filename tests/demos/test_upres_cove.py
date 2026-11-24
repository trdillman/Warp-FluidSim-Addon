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


def test_scene_is_the_dam_break_cove_with_finer_voxels():
    assert uc.DOMAIN_LO == cove.DOMAIN_LO and uc.DOMAIN_HI == cove.DOMAIN_HI
    assert uc.RESERVOIR_LO == cove.RESERVOIR_LO and uc.RESERVOIR_HI == cove.RESERVOIR_HI
    assert uc.VOXEL < cove.VOXEL
    assert uc.fine_voxel() == pytest.approx(uc.VOXEL / 2)


def test_coarse_initial_fill_keeps_the_showcase_particle_count():
    emitted = uc.expected_particles()
    # Dam Break emits ~256k from the same reservoir at 3 cm.
    assert emitted > 3 * uc.expected_particles(cove.VOXEL)
    assert emitted == pytest.approx(864_000)
    assert emitted * 1.15 < uc.MAX_PARTICLES  # Initial Fill fits the reserve


def test_coarse_leaves_hold_the_reservoir_with_room_for_the_surge():
    cells = [round((h - lo) / uc.VOXEL) for lo, h in zip(uc.RESERVOIR_LO, uc.RESERVOIR_HI,
                                                        strict=True)]
    # One ring of padding leaves around the column (topology expand).
    leaves = 1
    for n in cells:
        leaves *= -(-n // 8) + 2
    assert 3 * leaves < uc.LEAF_CAPACITY


@pytest.mark.parametrize("fine", [False, True])
def test_default_leaf_budgets_cover_the_entire_domain(fine):
    from fluidSim_warp_blender51.sim.quality_contract import (
        AuthoredQualitySettings,
        resolve_quality,
    )

    coarse = _coarse_settings()
    longest = max(hi - lo for lo, hi in zip(uc.DOMAIN_LO, uc.DOMAIN_HI, strict=True))
    coarse.voxel_size = longest / round(longest / uc.VOXEL)
    settings = uc.fine_settings(coarse) if fine else coarse
    quality = resolve_quality(AuthoredQualitySettings(voxel_size=settings.voxel_size),
                              (uc.DOMAIN_LO, uc.DOMAIN_HI))
    assert quality.total_domain_leaf_cells <= settings.leaf_capacity


@pytest.mark.parametrize("demo_factory", [None, uc])
def test_full_fine_reference_disables_narrow_band(demo_factory):
    coarse = dataclasses.replace(_coarse_settings(), narrow_band_enabled=True)
    full = CMP._full_fine_settings(coarse, uc.UPRES_FACTOR, demo_factory)
    assert not full.narrow_band_enabled
    assert coarse.narrow_band_enabled


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
        _Obj("Moving Drain", "SINK", anim=_Anim(action="slide")),
        _Obj("Swept Field", "FIELD", parent=rig),
        rig,
    ]
    assert CMP.animated_inputs(objects) == [
        "Paddle", "Blade", "Driven Source", "Moving Drain", "Swept Field"]


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


def test_review_scene_is_never_saved_into_the_demo_gate_folder():
    """check_demo_blends.py fails on any demos/*.blend outside its inventory."""
    with open(os.path.join(REPO, "tools", "demos", "build_demo_scenes.py"),
              encoding="utf-8") as fh:
        src = fh.read()
    body = src.split("def build_upres_cove")[1].split("\ndef ")[0]
    assert "save_scene(" not in body
    for call in ("save_playable(", "write_meta(", "render_preview("):
        tail = body.split(call, 1)[1].split(")\n", 1)[0]
        assert "directory=REVIEW_DEMOS" in tail, call
    assert 'os.path.join(REPO, "artifacts", "review_demos")' in src


def test_compare_tool_finds_built_review_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(CMP, "REPO", str(tmp_path))
    assert CMP.demo_blend_path("upres_cove") is None
    review = tmp_path / "artifacts" / "review_demos"
    review.mkdir(parents=True)
    (review / "upres_cove.blend").write_bytes(b"")
    assert CMP.demo_blend_path("upres_cove") == str(review / "upres_cove.blend")
    shipped = tmp_path / "demos"
    shipped.mkdir()
    (shipped / "dam_break.blend").write_bytes(b"")
    assert CMP.demo_blend_path("dam_break") == str(shipped / "dam_break.blend")
