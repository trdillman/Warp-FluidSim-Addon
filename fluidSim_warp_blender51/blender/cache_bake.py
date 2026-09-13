"""Sequential Production Cache Bake owner.

The transaction/controller primitives live in ``production_bake``; this module
owns the synchronous sequential workflow and remains the public compatibility
entry point used by scripts and tests.
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

_RESTART_MANAGED_ENTRIES = (
    "fluidsim_cache.json",
    "frames",
    "checkpoints",
    "logs",
)


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
        self._root_existed = self.cache_path.exists()
        existing = [
            name
            for name in _RESTART_MANAGED_ENTRIES
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
            _RESTART_MANAGED_ENTRIES
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


def _settle_restart_cache(
    protection: _RestartCacheProtection | None,
    status,
    *, running_status,
    terminal_statuses,
) -> None:
    if protection is None or not protection.active:
        return
    if status is running_status:
        protection.commit()
    elif status in terminal_statuses:
        protection.restore()


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
    from .production_bake import (
        BakeController,
        BakeStatus,
        ProductionBakeBackend,
        clear_active_bake,
        register_active_bake,
    )

    fs = scene.fluidsim
    start = int(fs.bake_start if start is None else start)
    end = int(fs.bake_end if end is None else end)
    backend = ProductionBakeBackend(
        checkpoint_interval=int(getattr(fs, "checkpoint_interval", 1)),
        compressed=bool(getattr(fs, "cache_compress", False)),
    )
    controller = BakeController(
        backend=backend, progress_callback=progress_callback)
    restart_cache = None
    try:
        if resume:
            controller.resume(scene, end=end)
        else:
            controller.start(scene, start, end, restart=restart)
        controller.reserve_cache_path(backend.cache_path(scene))
        register_active_bake(scene, controller)
        if restart:
            restart_cache = _protect_restart_cache(backend.cache_path(scene))
        while controller.status in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }:
            controller.tick(scene)
            _settle_restart_cache(
                restart_cache,
                controller.status,
                running_status=BakeStatus.RUNNING,
                terminal_statuses={
                    BakeStatus.CANCELLED,
                    BakeStatus.FAILED,
                },
            )
        if controller.status is BakeStatus.FAILED:
            raise RuntimeError(controller.error or controller.message)
        return controller.result()
    finally:
        try:
            if restart_cache is not None and restart_cache.active:
                restart_cache.restore()
        finally:
            clear_active_bake(scene, controller)


def bake(scene, sess, start: int, end: int) -> str:
    """Synchronous compatibility entry point used by scripts and tests."""
    result = run_synchronous(scene, sess, start, end, restart=False)
    return result.cache_path
