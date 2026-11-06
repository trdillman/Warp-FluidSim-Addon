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


class _PreviewSession:
    """The pre-bake preview session; the Live callback admits like playback does."""

    def __init__(self):
        self.owner = None
        self.initialized = []
        self.released = []
        self.solver = object()

    def owns_scene(self, scene):
        return scene is self.owner

    def initialize(self, scene, depsgraph=None):
        self.initialized.append(scene)
        self.owner = scene

    def sync_scene(self, _scene, _depsgraph=None):
        pass

    def release(self, scene):
        self.released.append(scene)
        self.owner = None


def _held_edit_world(backend, monkeypatch, *, current_settings, current_reinit="before"):
    from fluidSim_warp_blender51.blender import cache_identity

    suppressed = []

    @contextlib.contextmanager
    def suppress_mode_transitions():
        suppressed.append(True)
        try:
            yield
        finally:
            suppressed.pop()

    sys.modules["fluidSim_warp_blender51.blender.playback"].suppress_mode_transitions = (
        suppress_mode_transitions)
    session = _PreviewSession()

    class _Settings:
        """Mirrors playback.set_live_enabled: turning Live on admits a free session."""
        _live = False

        @property
        def live(self):
            return self._live

        @live.setter
        def live(self, value):
            self._live = bool(value)
            if value and not suppressed and not session.owns_scene(scene):
                session.initialize(scene)

    scene = _Scene(live=False)
    scene.fluidsim = _Settings()
    restored = []
    monkeypatch.setattr(production_bake, "restore_solver",
                        lambda solver, payload: restored.append(payload))
    monkeypatch.setattr(cache_identity, "settings_signature",
                        lambda _scene: current_settings)
    monkeypatch.setattr(backend, "_reinit_signature_now", lambda _scene: current_reinit)
    backend._original_reinit_signature = "before"
    session.owner = scene  # the bake's own session owns the scene until restore
    backend.sess = session
    backend._original_frame = 37
    backend._original_live = True
    backend._original_session_active = True
    backend._original_session_payload = "pre-bake checkpoint"
    backend._original_session_signatures = {
        "scene_key": "key", "settings_signature": "before",
        "source_signature": "sources"}
    return scene, session, restored


def test_a_held_settings_edit_leaves_the_preview_released(backend, monkeypatch):
    """Review finding (e): the restore never compared the captured signature, so
    a held Reinitialize-tier edit restored the pre-bake checkpoint into a solver
    admitted from the edited settings (or failed a COMPLETE bake)."""
    scene, session, restored = _held_edit_world(
        backend, monkeypatch, current_settings="edited", current_reinit="edited")
    session.initialized.clear()

    backend.finish(scene)

    assert restored == []                 # no checkpoint into an incompatible grid
    assert session.initialized == []      # not re-admitted, Live included
    assert session.released == [scene]
    assert scene.fluidsim.live is True
    assert scene.frame_current == 37
    assert "Initialize" in backend.restore_notice


def test_unchanged_settings_still_restore_the_preview(backend, monkeypatch):
    scene, session, restored = _held_edit_world(
        backend, monkeypatch, current_settings="before")

    backend.finish(scene)

    assert restored == ["pre-bake checkpoint"]
    assert session.initialized == [scene]
    assert backend.restore_notice == ""


def test_a_held_live_tier_edit_still_restores_the_preview(backend, monkeypatch):
    # Only a Reinitialize-tier edit changes the grid the checkpoint fits; a
    # live edit (viscosity, say) restores the preview and shows as pending.
    scene, session, restored = _held_edit_world(
        backend, monkeypatch, current_settings="edited", current_reinit="before")

    backend.finish(scene)

    assert restored == ["pre-bake checkpoint"]
    assert session.initialized == [scene]
    assert backend.restore_notice == ""


def test_a_complete_bake_stays_complete_and_says_initialize_is_needed():
    notice = "Settings changed during the bake: Initialize to restart the preview"
    backend = SimpleNamespace(finish=lambda _scene: None, restore_notice=notice)
    controller = BakeController(backend=backend)
    controller.status = BakeStatus.RUNNING
    controller.frame_start, controller.frame_end, controller.next_frame = 1, 2, 3

    controller.tick(_Scene())

    assert controller.status is BakeStatus.COMPLETE
    assert notice in controller.message


def test_preview_restoration_primes_original_frame_before_admission(backend, monkeypatch):
    """A preview at5 must not capture inputs/history from bake-end20."""
    scene = _Scene(frame=20, live=False)
    session = _Session(scene)
    session.solver = object()
    backend.sess = session
    backend._original_frame = 5
    backend._original_live = False
    backend._original_session_active = True
    backend._original_session_payload = object()
    backend._held_settings_changed = lambda _scene: False
    monkeypatch.setattr(production_bake, 'restore_solver', lambda *_args: None)
    backend._restore_original_state(scene)
    assert session.initialized_at == [5]
    assert scene.frame_current == 5


def _resume_with_frozen_points(backend, monkeypatch, tmp_path, *, foreign=False):
    import numpy as np
    from fluidSim_warp_blender51.blender import scene_inputs
    from fluidSim_warp_blender51.io import checkpoint

    key = ('source', 'POINTS', 'dependency')
    original = scene_inputs.SceneInputs()
    original._native_plan_input(key, True, lambda: np.array([[1, 0, 0]], np.float32))
    state = original.export_frozen_state()
    payload = checkpoint.attach_frozen_inputs(
        checkpoint.CheckpointPayload(10, .5, 'settings', 'sources'), state.metadata, state.arrays)
    path = tmp_path / 'checkpoint.npz'
    digest = checkpoint.write_checkpoint_atomic(path, payload)
    entry = SimpleNamespace(checkpoint_file='checkpoint.npz', checkpoint_sha256=digest,
                            checkpoint_size=path.stat().st_size)
    manifest = SimpleNamespace(last_committed_frame=10, frame_start=1, frame_end=20,
        effect_signature='', settings_signature='settings', source_signature='sources', frames={10: entry})
    monkeypatch.setattr(production_bake, 'load_manifest', lambda _path: manifest)
    monkeypatch.setattr(scene_inputs, 'frozen_input_keys', lambda _scene:
                        {('foreign', 'POINTS', 'dependency')} if foreign else {key})
    backend.resume_decision = lambda *_a, **_k: SimpleNamespace(allowed=True, resume_frame=11)
    backend._effect_identity_at = lambda *_a: ('', ())
    restored = []
    monkeypatch.setattr(production_bake, 'restore_solver', lambda solver, saved: restored.append(saved.frame))

    class Session(_Session):
        def __init__(self, scene):
            super().__init__(scene)
            self.solver = SimpleNamespace(frame=10)
            self.scene_inputs = None
            self.dynamic_history_frame = None

        def initialize(self, scene, depsgraph=None, frozen_input_state=None):
            super().initialize(scene, depsgraph)
            self.scene_inputs = scene_inputs.SceneInputs()
            if frozen_input_state is not None:
                self.scene_inputs.restore_frozen_state(frozen_input_state)
            self.points = self.scene_inputs._native_plan_input(key, True,
                lambda: np.array([[scene.frame_current, 0, 0]], np.float32))
            self.dynamic_history_frame = scene.frame_current

    scene = _Scene(frame=37)
    session = Session(scene)
    _snapshot(backend, session)
    return scene, session, restored, path


def test_resume_restores_frozen_points_but_primes_dynamic_history_at_checkpoint(backend, monkeypatch, tmp_path):
    scene, session, restored, _path = _resume_with_frozen_points(backend, monkeypatch, tmp_path)
    assert backend.initialize(scene, 1, 20, resume=True) == 11
    assert session.initialized_at == [10]
    assert session.points.tolist() == [[1, 0, 0]]
    assert session.dynamic_history_frame == 10
    assert restored == [10]


def test_foreign_frozen_object_id_refuses_before_replacing_preview_or_cache(backend, monkeypatch, tmp_path):
    scene, session, restored, path = _resume_with_frozen_points(
        backend, monkeypatch, tmp_path, foreign=True)
    before = path.read_bytes()
    snapshot = []
    backend._snapshot_live_state = lambda _scene: snapshot.append(True)
    with pytest.raises(production_bake.ResumeRefused, match='FROZEN_INPUT_STATE_INVALID'):
        backend.initialize(scene, 1, 20, resume=True)
    assert session.initialized_at == []
    assert snapshot == []
    assert restored == []
    assert path.read_bytes() == before
    assert scene.frame_current == 37
    assert scene.fluidsim.live


@pytest.mark.parametrize('finish', ['finish', 'cancel', 'fail'])
def test_preview_frozen_state_and_original_frame_return_on_every_terminal_path(
        backend, monkeypatch, tmp_path, finish):
    scene, session, restored, _path = _resume_with_frozen_points(backend, monkeypatch, tmp_path)
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs
    import numpy as np

    key = ('source', 'POINTS', 'dependency')
    inputs = SceneInputs()
    inputs._native_plan_input(key, True, lambda: np.array([[1, 0, 0]], np.float32))
    backend.sess = session
    backend._original_frame = 5
    backend._original_live = False
    backend._original_session_active = True
    backend._original_session_payload = SimpleNamespace(frame=5)
    backend._original_frozen_state = inputs.export_frozen_state()
    backend._held_settings_changed = lambda _scene: False
    backend._initialized = True
    scene.frame_set(20)
    getattr(backend, finish)(scene, *([RuntimeError('injected failure')] if finish == 'fail' else []))
    assert session.initialized_at == [5]
    assert session.points.tolist() == [[1, 0, 0]]
    assert session.dynamic_history_frame == 5
    assert restored == [5]
    assert scene.frame_current == 5


def test_failed_preview_input_restore_releases_bake_solver_and_keeps_live_suppressed(
        backend, monkeypatch, tmp_path):
    scene, session, _restored, _path = _resume_with_frozen_points(
        backend, monkeypatch, tmp_path, foreign=True)
    from fluidSim_warp_blender51.blender.scene_inputs import FrozenInputState

    backend.sess = session
    backend._original_frame = 5
    backend._original_live = True
    backend._original_session_active = True
    backend._original_session_payload = SimpleNamespace(frame=5)
    backend._original_frozen_state = FrozenInputState({'schema_version': 1, 'records': []}, {})
    backend._held_settings_changed = lambda _scene: False
    monkeypatch.setattr(blender_package.playback, 'suppress_mode_transitions',
                        contextlib.nullcontext, raising=False)
    with pytest.raises(RuntimeError, match='could not restore pre-bake live state'):
        backend._restore_original_state(scene)
    assert session.released == [scene]
    assert scene.frame_current == 5
    assert session.initialized_at == []
