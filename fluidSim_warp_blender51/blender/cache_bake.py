"""Compatibility adapter for the cache-v2 Production Bake owner."""

from __future__ import annotations


def bake(scene, sess, start: int, end: int) -> str:
    """Synchronous compatibility entry point used by scripts and tests."""
    from .production_bake import run_synchronous

    # The legacy "Export Cache Sequence" action is a fresh export, not Resume.
    # The dedicated production-bake UI owns explicit resume/restart choices.
    result = run_synchronous(scene, sess, start, end, restart=True)
    return result.cache_path
