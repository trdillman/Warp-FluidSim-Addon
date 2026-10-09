"""Initialize refuses a blocked narrow-band setup, even when set by script."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import simulation_admission as admission
from fluidSim_warp_blender51.sim import memory_safety
from fluidSim_warp_blender51.sim import narrow_band


def _owner():
    session = SimpleNamespace(solver=None)
    return admission.SimulationAdmission(
        session,
        scene_identity=lambda _scene: 1,
        applied_state_capture=lambda *_a, **_k: None,
        reset_settings_apply=lambda *_a: None,
    )


def _settings(**overrides):
    values = dict(
        device="cuda:0", narrow_band_enabled=True, narrow_band_voxels=3.0,
        species_enabled=False, viscosity_enabled=False,
        viscosity_from_particles=False, export_attributes=False,
        max_particles=1000)
    values.update(overrides)
    return SimpleNamespace(**values)


def _scene(*objects):
    return SimpleNamespace(objects=list(objects), frame_start=1)


def _reached_planning(monkeypatch):
    """Make a refused preflight distinguishable from one that moved on."""
    reached = []
    monkeypatch.setattr(
        memory_safety, "vram_snapshot",
        lambda _device: reached.append("vram") or SimpleNamespace())
    return reached


def test_blocked_setup_is_refused_before_any_planning(monkeypatch):
    reached = _reached_planning(monkeypatch)
    monkeypatch.setattr(
        admission, "build_settings",
        lambda _scene: _settings(species_enabled=True, export_attributes=True))

    with pytest.raises(RuntimeError) as raised:
        _owner()._preflight(_scene())

    message = str(raised.value)
    assert narrow_band.REASON_COLOR_MIXING in message
    assert narrow_band.REASON_EXPORT_ATTRIBUTES in message
    assert reached == []


def test_scalar_field_source_object_refuses(monkeypatch):
    reached = _reached_planning(monkeypatch)
    monkeypatch.setattr(
        admission, "build_settings", lambda _scene: _settings())
    field = SimpleNamespace(
        fluidsim=SimpleNamespace(role="FIELD", field_operation="SCALAR"))

    with pytest.raises(RuntimeError, match="field source"):
        _owner()._preflight(_scene(field))

    assert reached == []


def test_blockers_are_ignored_when_the_mode_is_off(monkeypatch):
    reached = _reached_planning(monkeypatch)
    monkeypatch.setattr(
        admission, "build_settings",
        lambda _scene: _settings(
            narrow_band_enabled=False, species_enabled=True))
    owner = _owner()
    monkeypatch.setattr(
        owner, "_live_solver_bytes", lambda: 0)

    def stop(*_args, **_kwargs):
        raise LookupError("planning reached")

    monkeypatch.setattr(owner, "_plan_scene", stop)

    with pytest.raises(LookupError, match="planning reached"):
        owner._preflight(_scene())

    assert reached == ["vram"]


def test_an_allowed_narrow_band_setup_proceeds_to_planning(monkeypatch):
    reached = _reached_planning(monkeypatch)
    monkeypatch.setattr(
        admission, "build_settings", lambda _scene: _settings())
    owner = _owner()
    monkeypatch.setattr(owner, "_live_solver_bytes", lambda: 0)

    def stop(*_args, **_kwargs):
        raise LookupError("planning reached")

    monkeypatch.setattr(owner, "_plan_scene", stop)

    with pytest.raises(LookupError, match="planning reached"):
        owner._preflight(_scene())

    assert reached == ["vram"]
