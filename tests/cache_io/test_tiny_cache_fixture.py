"""The tiny cache fixture is valid cache v3 and small."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from tiny_cache import build_tiny_cache  # noqa: E402

from fluidSim_warp_blender51.io import cache, cache_schema  # noqa: E402


def test_tiny_cache_verifies_and_reads(tmp_path):
    base = build_tiny_cache(tmp_path / "c", frame_count=4)
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
    assert cache_schema.verified_contiguous_frames(base, manifest) == (1, 2, 3, 4)
    counts = {len(cache.read_frame(str(base), f)["surface_verts"]) for f in (1, 2, 3, 4)}
    assert len(counts) > 1, "surface vertex count should vary per frame"
    assert sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) < 64 * 1024
