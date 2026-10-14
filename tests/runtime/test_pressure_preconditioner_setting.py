"""The ``pressure_preconditioner`` domain setting, end to end (RNA to PressureSolver)."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from fluidSim_warp_blender51.sim import pressure
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
from settings_doubles import full_settings, split_settings
from test_unregister_resilience import addon_modules  # noqa: F401 - fixture: add-on modules under a stub bpy

PKG = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"


def _rna_call():
    tree = ast.parse((PKG / "properties.py").read_text(encoding="utf-8"))
    domain = next(n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == "FLUIDSIM_PG_domain")
    for item in domain.body:
        if (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                and item.target.id == "pressure_preconditioner"):
            return item.annotation
    raise AssertionError("FLUIDSIM_PG_domain declares no pressure_preconditioner")


def test_rna_enum_has_exactly_the_two_kinds_and_defaults_to_multigrid():
    call = _rna_call()
    keywords = {kw.arg: kw.value for kw in call.keywords}
    items = ast.literal_eval(keywords["items"])
    assert [item[0] for item in items] == [
        pressure.PRECONDITIONER_JACOBI, pressure.PRECONDITIONER_MULTIGRID]
    assert ast.literal_eval(keywords["default"]) == pressure.PRECONDITIONER_MULTIGRID
    assert "update" in keywords  # live: wired to the same request as the other pressure fields
    assert ast.literal_eval(keywords["description"]).strip()


def test_solver_default_equals_rna_default():
    assert SolverSettings().pressure_preconditioner == pressure.PRECONDITIONER_MULTIGRID


def test_build_settings_projects_the_preconditioner(monkeypatch):
    from fluidSim_warp_blender51.blender import solver_settings

    preferences = types.ModuleType("fluidSim_warp_blender51.preferences")
    preferences.configured_device = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.preferences", preferences)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = tuple
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    scene = SimpleNamespace(
        fluidsim=split_settings(full_settings(
            domain_object=None, quality="PREVIEW", use_scene_gravity=True,
            pressure_preconditioner="MULTIGRID")),
        render=SimpleNamespace(fps=24, fps_base=1.0),
        use_gravity=True, gravity=[0, 0, -9.81],
    )
    assert solver_settings.build_settings(scene).pressure_preconditioner == "MULTIGRID"


def _fake_solver(preconditioner):
    seen = []

    class _Pressure:
        preconditioner = pressure.PRECONDITIONER_JACOBI

        def project_frame(self, *_args, **_kwargs):
            seen.append(self.preconditioner)
            return {"iterations": 1, "residual": 0.0, "project_velocity": True}

    fake = SimpleNamespace(
        settings=SolverSettings(pressure_preconditioner=preconditioner),
        pressure_solver=_Pressure(),
        vel=None, voxel_type=SimpleNamespace(data=None), collision_vel=None,
        voxel_counts=SimpleNamespace(data=None),
        sourcing=SimpleNamespace(apply_pressure_sources=lambda dt: None),
        stats=SimpleNamespace(),
        _cfl_speed_limit=lambda dt: 1.0,
        _native_solver_message=lambda: "",
    )
    return fake, seen


def test_solver_hands_its_setting_to_the_pressure_solver():
    fake, seen = _fake_solver("MULTIGRID")
    FluidSolver._pressure(fake, 0.01)
    assert seen == ["MULTIGRID"]
    assert fake.pressure_solver.preconditioner == "MULTIGRID"


def test_live_change_reaches_the_pressure_solver_before_the_next_solve():
    fake, seen = _fake_solver("JACOBI")
    FluidSolver._pressure(fake, 0.01)
    fake.settings.pressure_preconditioner = "MULTIGRID"  # what apply_live_settings does
    FluidSolver._pressure(fake, 0.01)
    fake.settings.pressure_preconditioner = "JACOBI"
    FluidSolver._pressure(fake, 0.01)
    assert seen == ["JACOBI", "MULTIGRID", "JACOBI"]


def test_pending_settings_classifies_it_as_live(monkeypatch):
    from test_pending_settings_registry import _modules  # same stub-bpy loader

    _properties, pending, _bridge = _modules(monkeypatch)
    assert pending.SETTING_TRANSITION_REGISTRY["pressure_preconditioner"] is pending.SettingTier.LIVE
    assert "pressure_preconditioner" in pending._HOT_SETTING_FIELDS
    assert "pressure_preconditioner" in pending._LIVE_SETTING_FIELDS


def test_pressure_panel_shows_it_only_for_pcg():
    source = (PKG / "blender" / "ui_scene.py").read_text(encoding="utf-8")
    block = source[source.index("class FLUIDSIM_PT_sim_pressure"):]
    block = block[:block.index("\nclass ", 10)]
    guarded = block[block.index('if fs.pressure_solver == "PCG"'):]
    assert 'layout.prop(fs, "pressure_preconditioner")' in guarded


def _operator_context(*, active, domain):
    scene = SimpleNamespace(fluidsim=SimpleNamespace(domain_object=domain))
    area = SimpleNamespace(type="PROPERTIES", spaces=SimpleNamespace(active=SimpleNamespace(context="")))
    screen = SimpleNamespace(areas=[area])
    return SimpleNamespace(scene=scene, active_object=active, screen=screen), area


def test_open_full_settings_picks_physics_for_an_active_domain_else_scene(addon_modules):  # noqa: F811
    panels = addon_modules.panels
    domain, other = object(), object()
    for active, domain_object, expected in (
        (domain, domain, "PHYSICS"),
        (other, domain, "SCENE"),
        (None, domain, "SCENE"),
        (domain, None, "SCENE"),
    ):
        context, area = _operator_context(active=active, domain=domain_object)
        reports = []
        op = SimpleNamespace(report=lambda level, text: reports.append((level, text)))
        result = panels.FLUIDSIM_OT_open_scene_properties.execute(op, context)
        assert result == {"FINISHED"}
        assert area.spaces.active.context == expected
        assert expected.capitalize() in reports[0][1]
