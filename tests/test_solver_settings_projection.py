"""Focused contracts for owned Blender-to-solver value projection."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim.solver import SolverSettings


def test_selected_projection_owns_vectors_and_maps_alias_and_enum():
    from fluidSim_warp_blender51.blender import solver_settings

    fs = SimpleNamespace(
        directional_force=[1, 2, 3],
        limit_percent=0.25,
        limit_behavior="ACCELERATION",
    )
    settings = SolverSettings()

    solver_settings.apply_scene_settings(
        fs, settings,
        {"directional_force", "limit_percent", "limit_behavior"})
    fs.directional_force[0] = 99

    assert settings.directional_force == (1.0, 2.0, 3.0)
    assert settings.speed_limit_percent == 0.25
    assert settings.speed_limit_behavior.name == "ACCELERATION"


def test_registry_validation_fails_closed_for_unknown_target(monkeypatch):
    from fluidSim_warp_blender51.blender import solver_settings

    monkeypatch.setattr(
        solver_settings, "DIRECT_SETTING_FIELDS",
        solver_settings.DIRECT_SETTING_FIELDS + ("not_a_solver_setting",))
    monkeypatch.setattr(solver_settings, "_SETTING_REGISTRY_VALIDATED", False)

    with pytest.raises(ValueError, match="not_a_solver_setting"):
        solver_settings.validate_setting_registry()


def test_build_settings_returns_only_owned_domain_gravity_and_device(monkeypatch):
    from fluidSim_warp_blender51.blender import solver_settings

    preferences = types.ModuleType("fluidSim_warp_blender51.preferences")
    preferences.configured_device = lambda: "cuda:7"
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.preferences", preferences)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            domain_object=None,
            use_scene_gravity=False,
            gravity=[1, 2, 3],
            directional_force=[4, 5, 6],
        ),
        render=SimpleNamespace(fps=30, fps_base=1.25),
        use_gravity=True,
        gravity=[0, 0, -9.81],
    )

    settings = solver_settings.build_settings(scene)
    scene.fluidsim.gravity[0] = 99
    scene.fluidsim.directional_force[0] = 99

    assert settings.device == "cuda:7"
    assert settings.domain_min == (-2.0, -2.0, 0.0)
    assert settings.domain_max == (2.0, 2.0, 4.0)
    assert settings.frame_rate == 24.0
    assert settings.gravity == (1.0, 2.0, 3.0)
    assert settings.directional_force == (4.0, 5.0, 6.0)
