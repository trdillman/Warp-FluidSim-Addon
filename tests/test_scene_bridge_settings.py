"""Regression tests for Blender scene-to-solver settings mapping."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeSolverSettings:
    def __init__(self):
        self.pressure_multigrid = 0

    def __setattr__(self, name, value):
        if name == "multigrid":
            raise AssertionError("build_settings must not assign settings.multigrid")
        super().__setattr__(name, value)


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(addons={}),
        evaluated_depsgraph_get=lambda: None,
    )
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(
            frame_change_pre=[],
            playback_pre=[],
            playback_post=[],
            persistent=lambda fn: fn,
        )
    )

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    fake_solver = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    fake_solver.SolverSettings = _FakeSolverSettings

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    return importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")


def _scene(multigrid):
    fs = SimpleNamespace(
        domain_object=None,
        voxel_size=0.1,
        leaf_capacity=64,
        topology_expand=1,
        open_boundaries=False,
        max_particles=1000,
        max_whitewater=0,
        time_scale=1.0,
        cfl=3.0,
        min_substeps=1,
        max_substeps=4,
        use_scene_gravity=False,
        gravity=(0.0, 0.0, -9.81),
        sample_type="FLIP",
        flip_ratio=0.95,
        trace_mode="RK3",
        max_speed=0.0,
        limit_behavior="MAINTAIN",
        limit_percent=0.1,
        particles_per_voxel=8.0,
        count_stiffness=0.0,
        relax_enabled=True,
        relax_strength=0.5,
        relax_mode="LOCAL_VOXEL",
        restitution=0.0,
        friction=0.0,
        pressure_solver="PCG",
        pressure_iterations=80,
        pressure_tolerance=1.0e-5,
        multigrid=multigrid,
        pressure_threshold=0.0,
        viscosity_enabled=False,
        viscosity_solver="PCG",
        viscosity_constant=0.0,
        viscosity_from_particles=False,
        viscosity_iterations=20,
        viscosity_tolerance=1.0e-5,
        surface_enabled=True,
        particle_radius_scale=1.2,
        surface_filter_type="BOX",
        surface_filter_iterations=1,
        surface_filter_strength=0.7,
        surface_dilate_iterations=0,
        surface_dilate_strength=1.0,
        surface_erode_iterations=0,
        surface_erode_strength=1.0,
        surface_smooth_type="BOX",
        surface_smooth_iterations=1,
        surface_smooth_strength=0.7,
        whitewater_enabled=False,
        ww_churn_decay=1.0,
        ww_churn_gain=1.0,
        ww_churn_threshold=0.5,
        ww_emission_rate=0.0,
        ww_min_speed=0.0,
        ww_life_min=1.0,
        ww_life_max=2.0,
        ww_bubble_drag=0.0,
        ww_spray_drag=0.0,
    )
    return SimpleNamespace(
        fluidsim=fs,
        render=SimpleNamespace(fps=24, fps_base=1),
        use_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    )


def test_build_settings_maps_scene_multigrid_to_pressure_multigrid(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.sim.enums import MultiGrid

    settings = scene_bridge.build_settings(_scene("V_CYCLE"))

    assert settings.pressure_multigrid == MultiGrid.V_CYCLE
    assert not hasattr(settings, "multigrid")


def test_build_settings_maps_every_declared_scene_field(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene("OFF")

    settings = scene_bridge.build_settings(scene)

    direct_fields = (
        "voxel_size", "leaf_capacity", "topology_expand", "open_boundaries",
        "max_particles", "max_whitewater", "time_scale", "cfl",
        "min_substeps", "max_substeps", "flip_ratio", "max_speed",
        "limit_percent", "particles_per_voxel", "count_stiffness",
        "relax_enabled", "relax_strength",
        "pressure_iterations", "pressure_tolerance", "pressure_threshold",
        "viscosity_enabled", "viscosity_constant", "viscosity_from_particles",
        "viscosity_iterations", "viscosity_tolerance", "surface_enabled",
        "particle_radius_scale", "surface_filter_iterations",
        "surface_filter_strength", "surface_dilate_iterations",
        "surface_dilate_strength", "surface_erode_iterations",
        "surface_erode_strength", "surface_smooth_iterations",
        "surface_smooth_strength", "whitewater_enabled", "ww_churn_decay",
        "ww_churn_gain", "ww_churn_threshold", "ww_emission_rate",
        "ww_min_speed", "ww_life_min", "ww_life_max", "ww_bubble_drag",
        "ww_spray_drag",
    )
    for name in direct_fields:
        assert getattr(settings, name) == getattr(scene.fluidsim, name), name

    enum_fields = {
        "sample_type": "sample_type",
        "trace_mode": "trace_mode",
        "limit_behavior": "limit_behavior",
        "relax_mode": "relax_mode",
        "pressure_solver": "pressure_solver",
        "viscosity_solver": "viscosity_solver",
        "surface_filter_type": "surface_filter_type",
        "surface_smooth_type": "surface_smooth_type",
    }
    for target, source in enum_fields.items():
        assert getattr(settings, target).name == getattr(scene.fluidsim, source)


def test_realtime_surface_bridge_uses_public_solver_rebuild(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)

    source = inspect.getsource(scene_bridge._apply_realtime_surface_settings)

    assert "rebuild_surface()" in source
    assert "_surface_mesh_cache" not in source
    assert "_surface_mesh_key" not in source
    assert "._surface()" not in source
