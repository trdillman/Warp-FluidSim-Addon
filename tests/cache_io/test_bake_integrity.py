"""Production Bake backend integrity (review items B20-B24).

- B20: a failure after initialize() replaced the live session restored only
  the Live flag, not the session.
- B21 (decision D4): resuming with a later end frame must extend the cache.
- B22: a bake started from Live did not restore Live, because the restore ran
  while the bake still held the scene and the Live guard refused it.
- B23/B24: admission used whatever frame the playhead was on, so the initial
  state depended on where Bake was clicked, and a resume primed collider
  velocity history at the playhead instead of the last committed frame.
"""
from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51 import blender as blender_package
from fluidSim_warp_blender51.blender import production_bake
from fluidSim_warp_blender51.blender.production_bake import (
    BakeController, BakeStatus, ProductionBakeBackend,
)


class _Scene:
    def __init__(self, frame=37, live=True):
        self.frame_current = frame
        self.frame_start = 1
        self.fluidsim = SimpleNamespace(live=live)

    def frame_set(self, frame):
        self.frame_current = int(frame)

    def as_pointer(self):
        return id(self)


class _Session:
    def __init__(self, scene, fail_sync=False):
        self.scene = scene
        self.fail_sync = fail_sync
        self.initialized_at = []
        self.released = []

    def owns_scene(self, _scene):
        return False

    def initialize(self, scene, depsgraph=None):
        self.initialized_at.append(scene.frame_current)

    def sync_scene(self, _scene, _depsgraph=None):
        if self.fail_sync:
            raise RuntimeError("sync failed after the session was replaced")

    def release(self, scene):
        self.released.append(scene)


@pytest.fixture
def backend(monkeypatch, tmp_path):
    @contextlib.contextmanager
    def suppress():
        yield

    fake_playback = types.ModuleType("fluidSim_warp_blender51.blender.playback")
    fake_playback.suppress_frame_handlers = suppress
    fake_effects = types.ModuleType("fluidSim_warp_blender51.blender.effects")
    fake_effects.effect_identity = lambda _scene: ("effects", ())
    for name, module in (("playback", fake_playback), ("effects", fake_effects)):
        monkeypatch.setitem(sys.modules, f"fluidSim_warp_blender51.blender.{name}", module)
        monkeypatch.setattr(blender_package, name, module, raising=False)
    monkeypatch.setattr(blender_package, "scene_bridge",
                        SimpleNamespace(session=lambda: None), raising=False)

    instance = ProductionBakeBackend()
    instance.cache_path = lambda _scene: tmp_path
    instance._signatures_at = lambda _scene, _frame: {
        "scene_key": "key", "settings_signature": "settings",
        "source_signature": "sources"}
    instance._bpy = lambda: SimpleNamespace(
        context=SimpleNamespace(evaluated_depsgraph_get=lambda: "depsgraph"))
    return instance


def _snapshot(instance, session):
    def snapshot(scene):
        instance.sess = session
        instance._original_frame = int(scene.frame_current)
        instance._original_live = bool(scene.fluidsim.live)
        instance._original_session_active = False
        instance._snapshot_open = True

    instance._snapshot_live_state = snapshot


def test_failure_after_the_session_is_replaced_restores_the_session(backend):
    scene = _Scene()
    session = _Session(scene, fail_sync=True)
    _snapshot(backend, session)

    with pytest.raises(RuntimeError):
        backend.initialize(scene, 5, 9)
    backend.fail(scene, RuntimeError("sync failed"))

    assert session.released == [scene]          # not left holding a bake solver
    assert scene.fluidsim.live is True          # Live put back
    assert scene.frame_current == 37            # playhead put back


def test_admission_runs_at_the_bake_start_not_the_playhead(backend):
    scene = _Scene(frame=37)
    session = _Session(scene, fail_sync=True)
    _snapshot(backend, session)

    with pytest.raises(RuntimeError):
        backend.initialize(scene, 5, 9)

    assert session.initialized_at == [5]


def test_restoring_live_is_not_refused_by_the_bakes_own_guard(backend, monkeypatch):
    scene = _Scene(live=False)

    class _Guarded:
        """Mirrors playback.set_live_enabled: Live cannot turn on mid-bake."""
        _live = False

        @property
        def live(self):
            return self._live

        @live.setter
        def live(self, value):
            if value and production_bake.bake_is_active(scene):
                value = False
            self._live = bool(value)

    scene.fluidsim = _Guarded()
    controller = SimpleNamespace(status=BakeStatus.RUNNING)
    monkeypatch.setitem(production_bake._ACTIVE_BAKES,
                        production_bake.scene_identity(scene), (scene, controller))
    session = _Session(scene)
    backend.sess = session
    backend._original_frame = 37
    backend._original_live = True
    backend._original_session_active = False

    backend._restore_original_state(scene)

    assert scene.fluidsim.live is True
    assert production_bake.bake_is_active(scene)  # the bake still holds it


def test_resume_with_a_later_end_frame_extends_a_complete_cache():
    class _Backend:
        def __init__(self):
            self.calls = []

        def resume_decision(self, _scene, **kwargs):
            from fluidSim_warp_blender51.io.cache_schema import ResumeDecision

            self.calls.append(kwargs)
            if kwargs.get("end", 0) > 10:
                return ResumeDecision(True, 11, "RESUME_READY", "resume at 11")
            return ResumeDecision(False, None, "ALREADY_COMPLETE", "complete through 10")

        def load_manifest(self, _scene):
            return SimpleNamespace(frame_start=1, frame_end=10)

    backend = _Backend()
    controller = BakeController(backend=backend)
    controller.resume(_Scene(), end=20)

    assert controller.next_frame == 11
    assert controller.frame_end == 20
    assert backend.calls[-1] == {"end": 20}
