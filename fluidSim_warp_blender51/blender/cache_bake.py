"""Sequential Production Cache Bake owner.

The transaction/controller primitives live in ``production_bake``; this module
owns the synchronous sequential workflow and remains the public compatibility
entry point used by scripts and tests.
"""

from __future__ import annotations

from collections.abc import Callable


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
    try:
        if resume:
            controller.resume(scene, end=end)
        else:
            controller.start(scene, start, end, restart=restart)
        controller.reserve_cache_path(backend.cache_path(scene))
        register_active_bake(scene, controller)
        while controller.status in {
            BakeStatus.INITIALIZING,
            BakeStatus.RUNNING,
            BakeStatus.CANCELLING,
        }:
            controller.tick(scene)
        if controller.status is BakeStatus.FAILED:
            raise RuntimeError(controller.error or controller.message)
        return controller.result()
    finally:
        clear_active_bake(scene, controller)


def bake(scene, sess, start: int, end: int) -> str:
    """Synchronous compatibility entry point used by scripts and tests."""
    result = run_synchronous(scene, sess, start, end, restart=False)
    return result.cache_path
