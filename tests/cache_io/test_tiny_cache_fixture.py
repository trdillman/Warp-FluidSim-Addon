"""The tiny cache fixture is valid cache v3 and small."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from tiny_cache import build_tiny_cache

from fluidSim_warp_blender51.blender.ui_cache_state import production_bake_action
from fluidSim_warp_blender51.io import cache, cache_schema


def test_tiny_cache_verifies_and_reads(tmp_path):
    base = build_tiny_cache(tmp_path / "c", frame_count=4)
    manifest = cache_schema.load_manifest(cache_schema.manifest_path(base))
    assert cache_schema.verified_contiguous_frames(base, manifest) == (1, 2, 3, 4)
    counts = {len(cache.read_frame(str(base), f)["surface_verts"]) for f in (1, 2, 3, 4)}
    assert len(counts) > 1, "surface vertex count should vary per frame"
    assert sum(p.stat().st_size for p in base.rglob("*") if p.is_file()) < 64 * 1024


@pytest.mark.parametrize("solver_version", ["3.0.0", "3.1.0", "3.2.0"])
def test_solver_version_does_not_block_cache_playback(tmp_path, solver_version):
    base = build_tiny_cache(tmp_path / solver_version, frame_count=1)
    manifest_file = cache_schema.manifest_path(base)
    manifest = cache_schema.load_manifest(manifest_file)
    manifest.solver_version = solver_version
    cache_schema.write_manifest_atomic(manifest_file, manifest)

    before = manifest_file.read_bytes()
    assert cache.read_frame(str(base), 1) is not None
    assert production_bake_action(base).action == "RESTART"
    assert manifest_file.read_bytes() == before
