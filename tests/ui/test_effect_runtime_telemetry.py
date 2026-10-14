"""Effect runtime telemetry and errors reach the effect panel.

``runtime_error``, ``calls``, ``elements`` and ``submission_ms`` were declared
on each effect and shown by the workspace, but nothing ever wrote them: the
workspace always said "Last Runtime Error: None" and RUNTIME_ERROR could not
happen. The Solver Session now publishes each step's per-frame stats, and a
failed launch's error, into the effect's properties.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_operator_panel_behavior import _install_bpy_stub, _reload_module


def _effects(monkeypatch):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module("fluidSim_warp_blender51.blender.effects")
    refreshed = []
    monkeypatch.setattr(effects, "refresh_stack_state", lambda scene: refreshed.append(scene))
    return effects, refreshed


def _pg(instance_id):
    return SimpleNamespace(instance_id=instance_id, display_name=instance_id,
                           runtime_error="", calls=0, elements=0, submission_ms=0.0)


def _scene(*pgs):
    return SimpleNamespace(fluidsim_effects=SimpleNamespace(items=list(pgs)))


def _record(instance_id, calls, elements, seconds):
    return {"instanceId": instance_id, "calls": calls, "elements": elements,
            "submissionSeconds": seconds}


def test_step_stats_are_published_per_effect_for_the_frame(monkeypatch):
    effects, _refreshed = _effects(monkeypatch)
    vortex, cooling = _pg("vortex"), _pg("cooling")
    scene = _scene(vortex, cooling)

    effects.publish_runtime_telemetry(scene, [
        _record("vortex", 4, 1000, 0.002), _record("vortex", 4, 500, 0.001),
        _record("cooling", 1, 64, 0.0005)])

    assert (vortex.calls, vortex.elements) == (8, 1500)
    assert vortex.submission_ms == pytest.approx(3.0)
    assert (cooling.calls, cooling.elements) == (1, 64)
    assert cooling.submission_ms == pytest.approx(0.5)


def test_unchanged_stats_are_not_rewritten(monkeypatch):
    # An RNA write tags the scene for re-evaluation even when the value is equal.
    effects, _refreshed = _effects(monkeypatch)
    writes = []

    class _Watched(SimpleNamespace):
        def __setattr__(self, name, value):
            writes.append(name)
            super().__setattr__(name, value)

    pg = _Watched(instance_id="vortex", display_name="vortex", runtime_error="",
                  calls=8, elements=1500, submission_ms=3.0)
    writes.clear()
    effects.publish_runtime_telemetry(_scene(pg), [_record("vortex", 8, 1500, 0.003)])
    assert writes == []


def test_a_failed_launch_marks_its_effect_runtime_error(monkeypatch):
    effects, refreshed = _effects(monkeypatch)
    from fluidSim_warp_blender51.sim.effects.runtime import (
        EffectExecutionError, EffectRuntimeError)

    vortex, other = _pg("vortex"), _pg("other")
    scene = _scene(vortex, other)
    error = EffectExecutionError(EffectRuntimeError(
        effect_id="fluidsim.vortex_force", instance_id="vortex", kernel_id="k",
        stage="BEFORE_EMISSION", frame=12, substep_index=2, message="index out of range"))

    effects.publish_runtime_error(scene, error)

    assert "index out of range" in vortex.runtime_error
    assert "frame 12" in vortex.runtime_error
    assert other.runtime_error == ""
    assert refreshed == [scene]  # status recomputed: RUNTIME_ERROR


def test_the_session_publishes_stats_and_errors_around_a_step(monkeypatch):
    _install_bpy_stub(monkeypatch)
    effects = _reload_module("fluidSim_warp_blender51.blender.effects")
    scene_bridge = _reload_module("fluidSim_warp_blender51.blender.scene_bridge")
    import inspect

    source = inspect.getsource(scene_bridge.Session.step)
    assert "publish_runtime_telemetry" in source
    assert "publish_runtime_error" in source
    assert hasattr(effects, "publish_runtime_telemetry")
