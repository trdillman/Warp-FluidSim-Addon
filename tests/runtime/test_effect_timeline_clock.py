"""Custom Effects run on the timeline frame in preview and bake alike.

Regression for review item B25 (decision D6):
(a) preview started the solver counter at 0 relative to the timeline start,
    bake at the absolute start - 1, so a frame-window effect fired on
    different timeline frames (worst in scenes starting at 1001);
(b) INITIALIZATION-cadence effects re-ran after a checkpoint resume, because
    the restored state was not matched by restored cadence history.
"""
from __future__ import annotations

from types import SimpleNamespace

from fluidSim_warp_blender51.sim.effects.enums import EffectCadence
from fluidSim_warp_blender51.sim.effects.runtime import EffectRuntimeManager


def test_session_step_hands_the_timeline_frame_to_the_solver(monkeypatch):
    import test_scene_bridge_cache_playback as harness

    scene_bridge = harness._install_scene_bridge_stubs(monkeypatch)
    scene = harness._scene(frame=1004, playback=False)
    sess = scene_bridge.session()
    seen = []
    sess.solver = SimpleNamespace(
        frame=3, step_frame=lambda: seen.append(sess.solver.effect_frame) or {})
    sess.scene_identity = id(scene)
    try:
        sess.step(scene, write_cache=False, timeline_frame=1004, publish_outputs=False)
    except Exception:  # noqa: BLE001 - only the clock hand-off matters here
        pass
    assert seen == [1004]


def _entry(key, frame_start, cadence=EffectCadence.INITIALIZATION):
    def active(frame, start=frame_start):
        return frame >= start

    kernel = SimpleNamespace(cadence=cadence, frame_start=frame_start,
                             active_on_frame=active)
    return SimpleNamespace(
        key=key,
        instance=SimpleNamespace(frame_start=0, active_on_frame=lambda _f: True),
        compiled_kernel=SimpleNamespace(kernel_definition=kernel))


def test_resume_marks_initialization_effects_that_already_ran():
    manager = EffectRuntimeManager.__new__(EffectRuntimeManager)
    manager._initialized = set()
    early = _entry("early", 1001)
    late = _entry("late", 1050)
    every = _entry("every", 1001, cadence=EffectCadence.ONCE_PER_FRAME)
    manager.by_stage = {"stage": (early, late, every)}

    manager.mark_initialized_through(1020)

    assert manager._initialized == {"early"}
    assert manager._cadence_allows(early, 1021) is False
    assert manager._cadence_allows(late, 1050) is True
