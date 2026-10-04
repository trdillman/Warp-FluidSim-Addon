"""Regression coverage for artist-facing production bake safety."""

from __future__ import annotations

import importlib
from pathlib import Path
import shutil
import sys
import types
from types import SimpleNamespace

import pytest


def _cache_bytes(cache_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(cache_dir).as_posix(): path.read_bytes()
        for path in cache_dir.rglob("*")
        if path.is_file()
    }


def test_restart_preserves_existing_cache_when_admission_fails_then_allows_retry(
    monkeypatch, tmp_path
):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache_dir = tmp_path / "cache"
    (cache_dir / "frames").mkdir(parents=True)
    (cache_dir / "checkpoints").mkdir()
    (cache_dir / "logs").mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"authoritative-manifest")
    (cache_dir / "frames" / "frame_000001.npz").write_bytes(b"frame-one")
    (cache_dir / "checkpoints" / "checkpoint_000001.npz").write_bytes(
        b"checkpoint-one"
    )
    (cache_dir / "logs" / "bake.jsonl").write_bytes(b"old-log\n")
    before = _cache_bytes(cache_dir)

    class _FaultThenSuccessBackend:
        attempts = 0

        def __init__(self, **_kwargs):
            self.base_dir = cache_dir

        def cache_path(self, _scene):
            return cache_dir

        def initialize(self, _scene, _start, _end, *, resume=False, restart=False):
            assert resume is False
            assert restart is True
            type(self).attempts += 1
            for name in ("fluidsim_cache.json", "frames", "checkpoints", "logs"):
                path = cache_dir / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "fluidsim_cache.json").write_bytes(b"replacement-manifest")
            if type(self).attempts == 1:
                raise RuntimeError("admission rejected")

        def fail(self, _scene, _error):
            return None

        def step_and_commit(self, _scene, frame):
            frames = cache_dir / "frames"
            frames.mkdir(exist_ok=True)
            (frames / f"frame_{frame:06d}.npz").write_bytes(b"replacement-frame")

        def finish(self, _scene):
            return None

    monkeypatch.setattr(
        production_bake, "ProductionBakeBackend", _FaultThenSuccessBackend
    )
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            bake_start=1,
            bake_end=1,
            checkpoint_interval=1,
            cache_compress=False,
        )
    )

    with pytest.raises(RuntimeError, match="admission rejected"):
        cache_bake.run_synchronous(scene, restart=True)

    assert _cache_bytes(cache_dir) == before
    assert production_bake.cache_path_is_active(cache_dir) is False

    result = cache_bake.run_synchronous(scene, restart=True)

    assert result.status is production_bake.BakeStatus.COMPLETE
    assert (cache_dir / "fluidsim_cache.json").read_bytes() == b"replacement-manifest"
    assert (cache_dir / "frames" / "frame_000001.npz").read_bytes() == (
        b"replacement-frame"
    )
    assert production_bake.cache_path_is_active(cache_dir) is False
    assert not tuple(cache_dir.glob(".fluidsim-restart-*"))


def test_restart_restores_exact_baseline_when_failed_writer_creates_new_managed_dirs(
    monkeypatch, tmp_path
):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")
    (cache_dir / "artist-note.txt").write_bytes(b"keep-me")
    before = _cache_bytes(cache_dir)

    class _FailedWriter:
        def __init__(self, **_kwargs):
            self.base_dir = cache_dir

        def cache_path(self, _scene):
            return cache_dir

        def initialize(self, _scene, _start, _end, *, resume=False, restart=False):
            assert not resume and restart
            (cache_dir / "fluidsim_cache.json").write_bytes(b"new-manifest")
            for name in ("frames", "checkpoints", "logs"):
                directory = cache_dir / name
                directory.mkdir()
                (directory / "partial.bin").write_bytes(name.encode("ascii"))
            raise RuntimeError("admission rejected")

        def fail(self, _scene, _error):
            return None

    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _FailedWriter)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            bake_start=1,
            bake_end=1,
            checkpoint_interval=1,
            cache_compress=False,
        )
    )

    with pytest.raises(RuntimeError, match="admission rejected"):
        cache_bake.run_synchronous(scene, restart=True)

    assert _cache_bytes(cache_dir) == before
    assert production_bake.cache_path_is_active(cache_dir) is False


@pytest.mark.parametrize("root_kind", ["absent", "artist_only"])
def test_failed_restart_cleans_new_managed_entries_from_empty_baseline(
    monkeypatch, tmp_path, root_kind
):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache_dir = tmp_path / root_kind
    if root_kind == "artist_only":
        cache_dir.mkdir()
        (cache_dir / "artist-note.txt").write_bytes(b"preserve artist file")
    root_existed = cache_dir.exists()
    before = _cache_bytes(cache_dir) if root_existed else {}

    class _FailedWriter:
        def __init__(self, **_kwargs):
            self.base_dir = cache_dir

        def cache_path(self, _scene):
            return cache_dir

        def initialize(self, _scene, _start, _end, *, resume=False, restart=False):
            assert not resume and restart
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "fluidsim_cache.json").write_bytes(b"partial-manifest")
            frames = cache_dir / "frames"
            frames.mkdir()
            (frames / "partial.bin").write_bytes(b"partial-frame")
            raise RuntimeError("admission rejected")

        def fail(self, _scene, _error):
            return None

    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _FailedWriter)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            bake_start=1,
            bake_end=1,
            checkpoint_interval=1,
            cache_compress=False,
        )
    )

    with pytest.raises(RuntimeError, match="admission rejected"):
        cache_bake.run_synchronous(scene, restart=True)

    if root_existed:
        assert cache_dir.is_dir()
        assert _cache_bytes(cache_dir) == before
    else:
        assert cache_dir.exists() is False
    assert production_bake.cache_path_is_active(cache_dir) is False


def test_partial_restart_staging_failure_preserves_unstaged_originals(
    monkeypatch, tmp_path
):
    from fluidSim_warp_blender51.blender import cache_bake

    cache_dir = tmp_path / "cache"
    (cache_dir / "frames").mkdir(parents=True)
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")
    (cache_dir / "frames" / "frame.bin").write_bytes(b"old-frame")
    before = _cache_bytes(cache_dir)
    real_replace = Path.replace

    def fail_frames(path, target):
        if path == cache_dir / "frames":
            raise OSError("staging blocked")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_frames)

    with pytest.raises(OSError, match="staging blocked"):
        cache_bake._protect_restart_cache(cache_dir)

    assert _cache_bytes(cache_dir) == before
    assert not tuple(cache_dir.glob(".fluidsim-restart-*"))


def test_restart_never_stages_or_clears_user_folders_outside_a_cache(tmp_path):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    project = tmp_path / "project"
    (project / "frames").mkdir(parents=True)
    (project / "frames" / "render_0001.png").write_bytes(b"artist render")
    (project / "logs").mkdir()
    (project / "logs" / "notes.txt").write_bytes(b"artist notes")
    before = _cache_bytes(project)

    with pytest.raises(ValueError, match="not a FluidSim cache"):
        cache_bake._protect_restart_cache(project)
    with pytest.raises(ValueError, match="not a FluidSim cache"):
        production_bake.ProductionBakeBackend()._clear_for_restart(project)

    assert _cache_bytes(project) == before
    assert not tuple(project.glob(".fluidsim-restart-*"))


def test_restore_failure_still_releases_synchronous_cache_reservation(
    monkeypatch, tmp_path
):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")

    class _FailedAdmission:
        def __init__(self, **_kwargs):
            self.base_dir = cache_dir

        def cache_path(self, _scene):
            return cache_dir

        def initialize(self, *_args, **_kwargs):
            raise RuntimeError("admission rejected")

        def fail(self, _scene, _error):
            return None

    def fail_restore(_self):
        raise OSError("restore blocked")

    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _FailedAdmission)
    monkeypatch.setattr(cache_bake._RestartCacheProtection, "restore", fail_restore)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            bake_start=1,
            bake_end=1,
            checkpoint_interval=1,
            cache_compress=False,
        )
    )

    with pytest.raises(OSError, match="restore blocked"):
        cache_bake.run_synchronous(scene, restart=True)

    assert production_bake.cache_path_is_active(cache_dir) is False


def test_old_backup_cleanup_failure_warns_but_successful_bake_finishes(
    monkeypatch, tmp_path
):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")

    class _SuccessfulBackend:
        active_sessions = 0
        finish_calls = 0

        def __init__(self, **_kwargs):
            self.base_dir = cache_dir

        def cache_path(self, _scene):
            return cache_dir

        def initialize(self, _scene, _start, _end, *, resume=False, restart=False):
            assert not resume and restart
            type(self).active_sessions += 1
            (cache_dir / "fluidsim_cache.json").write_bytes(b"new-manifest")

        def step_and_commit(self, _scene, _frame):
            return None

        def finish(self, _scene):
            type(self).active_sessions -= 1
            type(self).finish_calls += 1

        def fail(self, _scene, _error):
            type(self).active_sessions -= 1

    real_rmtree = cache_bake.shutil.rmtree

    def fail_old_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".fluidsim-restart-"):
            raise OSError("backup deletion blocked")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _SuccessfulBackend)
    monkeypatch.setattr(cache_bake.shutil, "rmtree", fail_old_backup_cleanup)
    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            bake_start=1,
            bake_end=1,
            checkpoint_interval=1,
            cache_compress=False,
        )
    )

    with pytest.warns(RuntimeWarning, match="backup cleanup deferred"):
        result = cache_bake.run_synchronous(scene, restart=True)

    assert result.status is production_bake.BakeStatus.COMPLETE
    assert _SuccessfulBackend.finish_calls == 1
    assert _SuccessfulBackend.active_sessions == 0
    assert production_bake.cache_path_is_active(cache_dir) is False
    assert (cache_dir / "fluidsim_cache.json").read_bytes() == b"new-manifest"
    assert len(tuple(cache_dir.glob(".fluidsim-restart-*"))) == 1


def _install_bpy_stub(monkeypatch):
    fake_bpy = types.ModuleType("bpy")

    class _Operator:
        def report(self, *_args, **_kwargs):
            return None

        @classmethod
        def poll_message_set(cls, _message):
            return None

    def _prop(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(Operator=_Operator)
    fake_bpy.props = SimpleNamespace(
        BoolProperty=lambda **kwargs: _prop(**kwargs),
        EnumProperty=lambda **kwargs: _prop(**kwargs),
        IntProperty=lambda **kwargs: _prop(**kwargs),
        StringProperty=lambda **kwargs: _prop(**kwargs),
    )
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.data = SimpleNamespace(scenes=[])
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)
    for name in (
        "fluidSim_warp_blender51.operator_domains.cache",
        "fluidSim_warp_blender51.operator_domains.common",
    ):
        sys.modules.pop(name, None)
    return fake_bpy, importlib.import_module(
        "fluidSim_warp_blender51.operator_domains.cache"
    )


class _Scene:
    def __init__(self, token: int, cache_path: Path):
        self._token = token
        self.removed = False
        self._fluidsim = SimpleNamespace(
            cache_path=str(cache_path),
            checkpoint_interval=1,
            cache_compress=False,
            bake_start=1,
            bake_end=2,
            bake_status="IDLE",
            bake_progress=0.0,
            bake_message="",
            status="unchanged",
        )

    def as_pointer(self):
        return 0 if self.removed else self._token

    @property
    def fluidsim(self):
        if self.removed:
            raise ReferenceError("StructRNA of type Scene has been removed")
        return self._fluidsim


class _ModalBackend:
    def __init__(self, **_kwargs):
        self.base_dir = Path()

    def cache_path(self, scene):
        self.base_dir = Path(scene.fluidsim.cache_path)
        return self.base_dir


class _ModalController:
    instances = []

    def __init__(self, *, backend, progress_callback=None):
        from fluidSim_warp_blender51.blender.production_bake import BakeStatus

        self.backend = backend
        self.progress_callback = progress_callback
        self.status = BakeStatus.IDLE
        self.progress = 0.0
        self.message = ""
        self.error = ""
        self._reserved_cache_path = None
        self.tick_scenes = []
        self.cancel_requests = 0
        type(self).instances.append(self)

    def start(self, _scene, _start, _end, *, restart=False):
        from fluidSim_warp_blender51.blender.production_bake import BakeStatus

        self.restart = bool(restart)
        self.status = BakeStatus.INITIALIZING
        self.message = "Initializing production bake"

    def reserve_cache_path(self, path):
        self._reserved_cache_path = Path(path).resolve()

    def request_cancel(self):
        from fluidSim_warp_blender51.blender.production_bake import BakeStatus

        self.cancel_requests += 1
        self.status = BakeStatus.CANCELLING
        self.message = "Cancellation requested"

    def tick(self, scene):
        from fluidSim_warp_blender51.blender.production_bake import BakeStatus

        self.tick_scenes.append(scene)
        if self.status is BakeStatus.INITIALIZING:
            self.status = BakeStatus.RUNNING
            self.progress = 0.5
            self.message = "Ready to bake"
        elif self.status is BakeStatus.CANCELLING:
            self.status = BakeStatus.CANCELLED
            self.message = "Production bake cancelled"


def _modal_context(scene):
    removed_timers = []
    manager = SimpleNamespace(
        event_timer_add=lambda *_args, **_kwargs: object(),
        event_timer_remove=lambda timer: removed_timers.append(timer),
        modal_handler_add=lambda _operator: None,
    )
    return (
        SimpleNamespace(scene=scene, window_manager=manager, window=object()),
        removed_timers,
    )


def test_modal_bake_remains_bound_to_initiating_scene_after_context_switch(
    monkeypatch, tmp_path
):
    fake_bpy, cache_operators = _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    _ModalController.instances.clear()
    monkeypatch.setattr(production_bake, "BakeController", _ModalController)
    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _ModalBackend)
    scene_a = _Scene(101, tmp_path / "a")
    scene_b = _Scene(202, tmp_path / "b")
    fake_bpy.data.scenes[:] = [scene_a, scene_b]
    context, removed_timers = _modal_context(scene_a)
    operator = cache_operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = False

    assert operator.execute(context) == {"RUNNING_MODAL"}
    context.scene = scene_b
    assert operator.modal(context, SimpleNamespace(type="TIMER", value="NOTHING")) == {
        "RUNNING_MODAL"
    }
    assert cache_operators.FLUIDSIM_OT_bake_cancel().execute(context) == {"FINISHED"}
    result = operator.modal(
        context, SimpleNamespace(type="TIMER", value="NOTHING")
    )

    controller = _ModalController.instances[-1]
    assert result == {"CANCELLED"}
    assert controller.tick_scenes == [scene_a, scene_a]
    assert scene_a.fluidsim.bake_status == production_bake.BakeStatus.CANCELLED.value
    assert scene_b.fluidsim.bake_status == "IDLE"
    assert scene_b.fluidsim.status == "unchanged"
    assert production_bake.cache_path_is_active(tmp_path / "a") is False
    assert len(removed_timers) == 1


def test_modal_bake_cancels_and_unregisters_when_initiating_scene_is_removed(
    monkeypatch, tmp_path
):
    fake_bpy, cache_operators = _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    _ModalController.instances.clear()
    monkeypatch.setattr(production_bake, "BakeController", _ModalController)
    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _ModalBackend)
    scene = _Scene(303, tmp_path / "removed")
    fake_bpy.data.scenes[:] = [scene]
    context, removed_timers = _modal_context(scene)
    operator = cache_operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = False

    assert operator.execute(context) == {"RUNNING_MODAL"}
    controller = _ModalController.instances[-1]
    fake_bpy.data.scenes.clear()
    scene.removed = True
    registration_proxy = SimpleNamespace(as_pointer=lambda: 303)
    try:
        result = operator.modal(
            context, SimpleNamespace(type="TIMER", value="NOTHING")
        )
        assert result == {"CANCELLED"}
        assert controller.cancel_requests == 1
        assert controller.tick_scenes == []
        assert production_bake.cache_path_is_active(tmp_path / "removed") is False
        assert len(removed_timers) == 1
    finally:
        production_bake.clear_active_bake(registration_proxy, controller)


@pytest.mark.parametrize("failure_point", ["timer", "modal_handler"])
def test_modal_start_acquisition_failure_restores_cache_and_releases_ownership(
    monkeypatch, tmp_path, failure_point
):
    fake_bpy, cache_operators = _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    _ModalController.instances.clear()
    monkeypatch.setattr(production_bake, "BakeController", _ModalController)
    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _ModalBackend)
    cache_dir = tmp_path / failure_point
    cache_dir.mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")
    before = _cache_bytes(cache_dir)
    scene = _Scene(404, cache_dir)
    fake_bpy.data.scenes[:] = [scene]
    removed_timers = []

    def event_timer_add(*_args, **_kwargs):
        if failure_point == "timer":
            raise RuntimeError("timer acquisition failed")
        return object()

    def modal_handler_add(_operator):
        if failure_point == "modal_handler":
            raise RuntimeError("modal registration failed")

    manager = SimpleNamespace(
        event_timer_add=event_timer_add,
        event_timer_remove=lambda timer: removed_timers.append(timer),
        modal_handler_add=modal_handler_add,
    )
    context = SimpleNamespace(scene=scene, window_manager=manager, window=object())
    operator = cache_operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = True
    operator.expected_cache_path = ""

    assert operator.execute(context) == {"CANCELLED"}
    assert _cache_bytes(cache_dir) == before
    assert production_bake.cache_path_is_active(cache_dir) is False
    assert len(removed_timers) == (1 if failure_point == "modal_handler" else 0)


def test_removed_scene_restore_failure_still_removes_timer_and_reservation(
    monkeypatch, tmp_path
):
    fake_bpy, cache_operators = _install_bpy_stub(monkeypatch)
    from fluidSim_warp_blender51.blender import production_bake

    _ModalController.instances.clear()
    monkeypatch.setattr(production_bake, "BakeController", _ModalController)
    monkeypatch.setattr(production_bake, "ProductionBakeBackend", _ModalBackend)
    cache_dir = tmp_path / "removed-restore-failure"
    cache_dir.mkdir()
    (cache_dir / "fluidsim_cache.json").write_bytes(b"old-manifest")
    scene = _Scene(505, cache_dir)
    fake_bpy.data.scenes[:] = [scene]
    context, removed_timers = _modal_context(scene)
    operator = cache_operators.FLUIDSIM_OT_bake_start()
    operator.restart_existing = True
    operator.expected_cache_path = ""

    assert operator.execute(context) == {"RUNNING_MODAL"}
    controller = _ModalController.instances[-1]
    scene.removed = True
    fake_bpy.data.scenes.clear()
    from fluidSim_warp_blender51.blender import cache_bake

    monkeypatch.setattr(
        cache_bake._RestartCacheProtection,
        "restore",
        lambda _self: (_ for _ in ()).throw(OSError("restore blocked")),
    )

    assert operator.modal(
        context, SimpleNamespace(type="TIMER", value="NOTHING")
    ) == {"CANCELLED"}
    assert production_bake.cache_path_is_active(cache_dir) is False
    assert len(removed_timers) == 1
    assert controller.cancel_requests == 1
