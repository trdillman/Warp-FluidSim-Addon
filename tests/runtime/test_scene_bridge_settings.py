"""Regression tests for Blender scene-to-solver settings mapping."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fluidSim_warp_blender51.sim.solver import SolverSettings
from settings_doubles import setting, split_settings


def _build_settings(scene):
    from fluidSim_warp_blender51.blender.solver_settings import build_settings

    return build_settings(scene)


def test_multigrid_feature_is_not_exposed_or_scene_configurable():
    root = Path(__file__).parents[2] / "fluidSim_warp_blender51"

    assert '"multigrid"' not in (root / "panels.py").read_text(encoding="utf-8")
    assert "multigrid:" not in (root / "properties.py").read_text(encoding="utf-8")
    assert "pressure_multigrid" not in (
        root / "blender" / "scene_bridge.py").read_text(encoding="utf-8")


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
    fake_solver.SolverSettings = SolverSettings
    fake_preferences = types.ModuleType("fluidSim_warp_blender51.preferences")
    fake_preferences.configured_device = lambda: "cuda:0"

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver)
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.preferences", fake_preferences)
    for module in (
        "fluidSim_warp_blender51.blender.scene_bridge",
        "fluidSim_warp_blender51.blender.solver_settings",
    ):
        sys.modules.pop(module, None)
    return importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")


def _scene():
    fs = SimpleNamespace(
        domain_object=None,
        voxel_size=0.1,
        leaf_capacity=64,
        topology_expand=1,
        open_boundaries=False,
        domain_sdf_walls=True,
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
        directional_force=(1.0, 2.0, 3.0),
        omnidirectional_force_center=(4.0, 5.0, 6.0),
        omnidirectional_force_strength=-7.0,
        omnidirectional_force_radius=8.0,
        force_diffusion=0.25,
        dissipation_enabled=True,
        dissipation_attribute="viscosity",
        dissipation_rate=0.75,
        dissipation_target=0.5,
        attraction_enabled=True,
        attraction_strength=9.0,
        attraction_max_distance=10.0,
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
        pressure_threshold=0.0,
        viscosity_enabled=False,
        viscosity_solver="PCG",
        viscosity_constant=0.0,
        viscosity_from_particles=False,
        viscosity_iterations=20,
        viscosity_tolerance=1.0e-5,
        surface_enabled=True,
        live_surface_publish_cadence=4,
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
        fluidsim=split_settings(fs),
        render=SimpleNamespace(fps=24, fps_base=1),
        use_gravity=True,
        gravity=(0.0, 0.0, -9.81),
    )


def test_build_settings_maps_every_declared_scene_field(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()

    settings = _build_settings(scene)

    direct_fields = (
        "voxel_size", "leaf_capacity", "topology_expand", "open_boundaries",
        "domain_sdf_walls",
        "max_particles", "max_whitewater", "time_scale", "cfl",
        "min_substeps", "max_substeps", "flip_ratio", "max_speed",
        "directional_force", "omnidirectional_force_center",
        "omnidirectional_force_strength", "omnidirectional_force_radius",
        "force_diffusion", "dissipation_enabled", "dissipation_attribute",
        "dissipation_rate", "dissipation_target", "attraction_enabled",
        "attraction_strength", "attraction_max_distance",
        "particles_per_voxel", "count_stiffness",
        "relax_enabled", "relax_strength",
        "pressure_iterations", "pressure_tolerance", "pressure_threshold",
        "viscosity_enabled", "viscosity_constant", "viscosity_from_particles",
        "viscosity_iterations", "viscosity_tolerance", "surface_enabled",
        "live_surface_publish_cadence",
        "particle_radius_scale", "surface_filter_iterations",
        "surface_filter_strength", "surface_dilate_iterations",
        "surface_dilate_strength", "surface_erode_iterations",
        "surface_erode_strength", "surface_smooth_iterations",
        "surface_smooth_strength", "whitewater_enabled",
        "ww_churn_threshold", "ww_emission_rate",
        "ww_min_speed", "ww_life_min", "ww_life_max", "ww_bubble_drag",
        "ww_spray_drag",
    )
    for name in direct_fields:
        assert getattr(settings, name) == setting(scene.fluidsim, name), name

    for target, source in {
        "speed_limit_percent": "limit_percent",
        "churn_decay": "ww_churn_decay",
        "churn_gain": "ww_churn_gain",
    }.items():
        assert getattr(settings, target) == setting(scene.fluidsim, source)

    enum_fields = {
        "sample_type": "sample_type",
        "trace_mode": "trace_mode",
        "speed_limit_behavior": "limit_behavior",
        "relax_mode": "relax_mode",
        "pressure_solver": "pressure_solver",
        "viscosity_solver": "viscosity_solver",
        "surface_filter_type": "surface_filter_type",
        "surface_smooth_type": "surface_smooth_type",
    }
    for target, source in enum_fields.items():
        assert getattr(settings, target).name == setting(scene.fluidsim, source)


def test_build_settings_detaches_blender_vector_proxies(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()

    class InvalidatableVector:
        def __init__(self, values):
            self.values = values
            self.valid = True

        def __iter__(self):
            if not self.valid:
                raise ReferenceError("RNA vector owner was invalidated")
            return iter(self.values)

    directional = InvalidatableVector((1.0, 2.0, 3.0))
    center = InvalidatableVector((4.0, 5.0, 6.0))
    scene.fluidsim.domain_object.fluidsim_domain.directional_force = directional
    scene.fluidsim.domain_object.fluidsim_domain.omnidirectional_force_center = center

    settings = _build_settings(scene)
    directional.valid = False
    center.valid = False

    assert settings.directional_force == (1.0, 2.0, 3.0)
    assert settings.omnidirectional_force_center == (4.0, 5.0, 6.0)
    assert isinstance(settings.directional_force, tuple)
    assert isinstance(settings.omnidirectional_force_center, tuple)


def test_closed_domain_projects_required_collision_walls(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.fluidsim.domain_object.fluidsim_domain.domain_sdf_walls = False

    settings = _build_settings(scene)

    assert settings.open_boundaries is False
    assert settings.domain_sdf_walls is True
    # Projection must not mutate legacy scene data during read-only preflight.
    assert scene.fluidsim.domain_object.fluidsim_domain.domain_sdf_walls is False


def test_build_settings_maps_ui_aliases_without_shadow_attributes(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.fluidsim.domain_object.fluidsim_domain.limit_percent = 0.37
    scene.fluidsim.domain_object.fluidsim_domain.limit_behavior = "ACCELERATION"
    scene.fluidsim.domain_object.fluidsim_domain.ww_churn_decay = 2.75
    scene.fluidsim.domain_object.fluidsim_domain.ww_churn_gain = 3.25

    settings = _build_settings(scene)

    assert settings.speed_limit_percent == 0.37
    assert settings.speed_limit_behavior.name == "ACCELERATION"
    assert settings.churn_decay == 2.75
    assert settings.churn_gain == 3.25
    assert not {"limit_percent", "limit_behavior", "ww_churn_decay", "ww_churn_gain"} & set(vars(settings))


def test_scene_setting_translation_rejects_undeclared_targets(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import solver_settings

    settings = SolverSettings(voxel_size=9.0)
    monkeypatch.setattr(
        solver_settings, "DIRECT_SETTING_FIELDS",
        solver_settings.DIRECT_SETTING_FIELDS + ("unknown_setting",),
    )
    monkeypatch.setattr(
        solver_settings, "_SETTING_REGISTRY_VALIDATED", False)

    with pytest.raises(ValueError, match="unknown_setting"):
        solver_settings.apply_scene_settings(_scene().fluidsim, settings)
    assert settings.voxel_size == 9.0


def test_realtime_surface_bridge_uses_public_solver_rebuild(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)

    from fluidSim_warp_blender51.blender import pending_settings

    source = inspect.getsource(pending_settings._apply_realtime_surface_settings)

    assert "rebuild_surface()" in source
    assert "_surface_mesh_cache" not in source
    assert "_surface_mesh_key" not in source
    assert "._surface()" not in source


def test_explicit_output_refresh_bypasses_live_surface_cadence(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)

    from fluidSim_warp_blender51.blender import pending_settings

    source = inspect.getsource(pending_settings.request_scene_update)

    assert source.count("force_surface=True") == 2


def test_build_settings_reads_domain_fields_from_migrated_object(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    # Bind through the freshly-imported submodule identity (not the possibly
    # stale ``blender`` package attribute), so the ``domain_bounds`` patch below
    # lands on the same module object whose ``build_settings`` we invoke.
    import fluidSim_warp_blender51.blender.solver_settings as solver_settings

    # Isolate the settings-projection path from domain geometry.
    monkeypatch.setattr(
        solver_settings, "domain_bounds",
        lambda scene: ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))

    domain_group = SimpleNamespace(
        voxel_size=0.02, max_particles=9_000_000,
        use_scene_gravity=False, gravity=(0.0, 0.0, -3.0))
    domain_obj = SimpleNamespace(fluidsim_domain=domain_group)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            schema_version=1, domain_object=domain_obj,
            # Stale mirror values the solver must NOT read post-migration.
            voxel_size=0.5, max_particles=1,
            use_scene_gravity=True, gravity=(0.0, 0.0, 0.0)),
        render=SimpleNamespace(fps=24, fps_base=1),
        use_gravity=True, gravity=(0.0, 0.0, -9.81))

    settings = solver_settings.build_settings(scene)

    # Domain solver config comes from obj.fluidsim_domain, not scene.fluidsim.
    assert settings.voxel_size == 0.02
    assert settings.max_particles == 9_000_000
    # apply_gravity reads use_scene_gravity/gravity from the domain group too:
    # the group opts out of scene gravity, so -3.0 (its own), not -9.81 (scene).
    assert settings.gravity == (0.0, 0.0, -3.0)
