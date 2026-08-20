"""Compatibility adapter for the cache-v2 Production Bake owner."""

from __future__ import annotations


def bake(scene, sess, start: int, end: int) -> str:
    """Synchronous compatibility entry point used by scripts and tests."""
    from .production_bake import run_synchronous

    result = run_synchronous(scene, sess, start, end, restart=False)
    return result.cache_path
