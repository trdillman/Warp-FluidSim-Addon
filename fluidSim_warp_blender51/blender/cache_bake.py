"""Production Cache Bake lifecycle owner.

The transaction/controller primitives live in ``production_bake``; this module
owns the lifecycle around them (``BakeRun``) and the synchronous sequential
workflow built on it, which remains the public compatibility entry point used
by scripts and tests.
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from .cache_manager import RESTART_MANAGED_ENTRIES, require_restartable_cache_directory
from .scene_access import scene_identity


def _remove_managed_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


class _RestartCacheProtection:
    """Keep the previous managed cache recoverable until admission succeeds."""

    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path).resolve()
        self.backup_path: Path | None = None
        self._root_existed = False
        self._baseline_names: set[str] = set()
        self._staged_names: list[str] = []
        self._staging_complete = False

    @property
    def active(self) -> bool:
        return self.backup_path is not None

    def stage(self) -> None:
        require_restartable_cache_directory(self.cache_path)
        self._root_existed = self.cache_path.exists()
        existing = [
            name
            for name in RESTART_MANAGED_ENTRIES
            if (self.cache_path / name).exists()
            or (self.cache_path / name).is_symlink()
        ]
        self._baseline_names = set(existing)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        backup_path = self.cache_path / f".fluidsim-restart-{uuid4().hex}"
        try:
            backup_path.mkdir()
        except Exception:
            if not self._root_existed:
                try:
                    self.cache_path.rmdir()
                except OSError:
                    pass
            raise
        self.backup_path = backup_path
        try:
            for name in existing:
                (self.cache_path / name).replace(backup_path / name)
                self._staged_names.append(name)
            self._staging_complete = True
        except Exception:
            self.restore()
            raise

    def commit(self) -> None:
        """Discard the old cache only after replacement admission succeeds."""
        backup_path = self.backup_path
        if backup_path is None:
            return
        # Admission made the replacement authoritative. Disable rollback before
        # deleting the backup so a partial cleanup failure can never make a
        # caller remove the valid replacement in an attempted restore.
        self.backup_path = None
        self._root_existed = False
        self._baseline_names.clear()
        self._staged_names.clear()
        self._staging_complete = False
        try:
            shutil.rmtree(backup_path)
        except OSError as exc:
            warnings.warn(
                "Production bake replacement admitted; previous cache backup "
                f"cleanup deferred at {backup_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    def restore(self) -> None:
        """Remove partial replacement data and restore every staged entry."""
        backup_path = self.backup_path
        if backup_path is None:
            return
        names = (
            RESTART_MANAGED_ENTRIES
            if self._staging_complete
            else tuple(self._staged_names)
        )
        for name in names:
            if name in self._baseline_names and name not in self._staged_names:
                continue
            destination = self.cache_path / name
            _remove_managed_entry(destination)
            if name not in self._staged_names:
                continue
            source = backup_path / name
            source.replace(destination)
            self._staged_names.remove(name)
        if backup_path.exists():
            shutil.rmtree(backup_path)
        self.backup_path = None
        root_existed = self._root_existed
        self._root_existed = False
        self._baseline_names.clear()
        self._staging_complete = False
        if not root_existed:
            try:
                self.cache_path.rmdir()
            except OSError:
                pass


def _protect_restart_cache(cache_path: Path | str) -> _RestartCacheProtection:
    protection = _RestartCacheProtection(cache_path)
    protection.stage()
    return protection


class BakeRun:
    """One Production Cache Bake, from lease to release.

    Construction takes everything a bake owns -- the controller, the reserved
    cache path, scene ownership and, for a restart, the staged previous cache --
    and gives all of it back if any step fails. ``tick`` advances the bake and
    settles the staged cache (keep the replacement once it is admitted, return
    the old cache if the bake ends without one). ``close`` is the single exit:
    it returns an un-admitted restart's cache and releases ownership, and is
    safe to call again. Drive it from a loop (synchronous bake, command line)
    or from a timer (the modal operator).

    ``backend`` defaults to a ``ProductionBakeBackend`` built from the scene's
    checkpoint and compression settings; ``start``/``end`` default to the scene's
    authored bake range.
    """

    def __init__(
        self,
        scene,
        *,
        start: int | None = None,
        end: int | None = None,
        resume: bool = False,
        restart: bool = False,
        progress_callback: Callable[[dict], None] | None = None,
        backend=None,
    ):
        from . import production_bake

        fs = scene.fluidsim
        if backend is None:
            backend = production_bake.ProductionBakeBackend(
                checkpoint_interval=int(getattr(fs, "checkpoint_interval", 1)),
                compressed=bool(getattr(fs, "cache_compress", False)),
            )
        self.scene = scene
        self.scene_key = scene_identity(scene)
        self.controller = production_bake.BakeController(
            backend=backend, progress_callback=progress_callback)
        self._restart_cache: _RestartCacheProtection | None = None
        end = int(fs.bake_end if end is None else end)
        try:
            if resume:
                self.controller.resume(scene, end=end)
            else:
                start = int(fs.bake_start if start is None else start)
                self.controller.start(scene, start, end, restart=restart)
            cache_path = backend.cache_path(scene)
            self.controller.reserve_cache_path(cache_path)
            production_bake.register_active_bake(scene, self.controller)
            if restart:
                self._restart_cache = _protect_restart_cache(cache_path)
        except BaseException:
            production_bake.release_active_bake(self.controller)
            raise

    @property
    def finished(self) -> bool:
        """Whether the bake has reached a terminal state (or never started)."""
        from .production_bake import BakeStatus

        return self.controller.status not in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }

    def request_cancel(self) -> None:
        self.controller.request_cancel()

    def tick(self) -> float | None:
        """Advance one transition, then settle the staged cache.

        Returns the controller's suggested timer delay. A failure to settle the
        cache propagates; ``close`` still releases ownership.
        """
        from .production_bake import BakeStatus

        delay = self.controller.tick(self.scene)
        protection = self._restart_cache
        if protection is not None and protection.active:
            status = self.controller.status
            if status is BakeStatus.RUNNING:
                protection.commit()
            elif status in (BakeStatus.CANCELLED, BakeStatus.FAILED):
                protection.restore()
        return delay

    def abandon(self, reason: str) -> None:
        """End a bake whose scene is gone: nothing can be ticked against it.

        Cancels the controller, force-releases the backend's solver session and
        returns an un-admitted restart's old cache. The run ends CANCELLED with
        ``reason``, or FAILED if either cleanup step failed. Call ``close``
        afterwards to release ownership.
        """
        from .production_bake import BakeStatus

        controller = self.controller
        controller.request_cancel()
        errors = []
        session = getattr(getattr(controller, "backend", None), "sess", None)
        force_release = getattr(session, "force_release", None)
        if callable(force_release):
            try:
                force_release(preserve_error=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"session cleanup failed: {exc}")
        try:
            protection = self._restart_cache
            if protection is not None and protection.active:
                protection.restore()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cache restore failed: {exc}")
        if errors:
            detail = "; ".join(errors)
            controller.status = BakeStatus.FAILED
            controller.error = detail
            controller.message = f"Production bake cleanup failed: {detail}"
        else:
            controller.status = BakeStatus.CANCELLED
            controller.message = reason

    def close(self) -> None:
        """Return an un-admitted restart's old cache and release ownership."""
        from .production_bake import release_active_bake

        try:
            protection = self._restart_cache
            if protection is not None and protection.active:
                protection.restore()
        finally:
            release_active_bake(self.controller)


def run_synchronous(
    scene,
    sess=None,
    start: int | None = None,
    end: int | None = None,
    *,
    resume: bool = False,
    restart: bool = False,
    progress_callback: Callable[[dict], None] | None = None,
):
    """Run the authoritative sequential cache transaction to a terminal state."""
    from .production_bake import BakeStatus

    run = BakeRun(
        scene, start=start, end=end, resume=resume, restart=restart,
        progress_callback=progress_callback)
    try:
        while not run.finished:
            run.tick()
        controller = run.controller
        if controller.status is BakeStatus.FAILED:
            raise RuntimeError(controller.error or controller.message)
        return controller.result()
    finally:
        run.close()


def bake(scene, sess, start: int, end: int) -> str:
    """Synchronous compatibility entry point used by scripts and tests."""
    result = run_synchronous(scene, sess, start, end, restart=False)
    return result.cache_path
