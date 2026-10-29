"""Committed metadata is coherent evidence, never an artifact checksum waiver."""
from dataclasses import replace
import os

import pytest

from fluidSim_warp_blender51.io import cache_schema
from fluidSim_warp_blender51.blender import cache_manager, rigid_publication
from tests.cache_io.fixtures.tiny_cache import build_tiny_cache


def test_observation_selects_counts_and_rigid_records_without_mutable_aliases(tmp_path):
    build_tiny_cache(tmp_path)
    path = cache_schema.manifest_path(tmp_path)
    manifest = cache_schema.load_manifest(path)
    record = cache_schema.RigidBodyRecord(
        object_uid="body", pos=(1., 2., 3.), quat=(1., 0., 0., 0.),
        lin_vel=(0., 0., 0.), ang_vel=(0., 0., 0.), rest_com=(0., 0., 0.))
    manifest.frames[2] = replace(manifest.frames[2], rigid_bodies=(record,))
    manifest.format_minor = 1
    cache_schema.write_manifest_atomic(path, manifest)
    observed = cache_schema.observe_committed_sequence(tmp_path)
    assert observed.frames == (1, 2, 3)
    entries = observed.entries((1, 2))
    assert observed.point_counts((1, 2)) == (
        max(e.particle_count for e in entries), max(e.whitewater_count for e in entries))
    assert rigid_publication.cache_records(tmp_path, (2,), sequence=observed) == {2: (record,)}
    observed.manifest.frames.clear()
    assert len(observed.entries()) == 3
    with pytest.raises(ValueError, match="not in the observed"):
        observed.entries((4,))


def test_same_size_same_timestamp_manifest_edit_invalidates_observation(tmp_path):
    build_tiny_cache(tmp_path)
    observed = cache_schema.observe_committed_sequence(tmp_path)
    path = cache_schema.manifest_path(tmp_path)
    stamp = path.stat()
    raw = path.read_bytes()
    changed = raw.replace(b'tiny-cache-sources', b'tiny-cache-SOURCES')
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError):
        observed.revalidate()


def test_later_corruption_keeps_inspection_prefix_but_refuses_complete_export(tmp_path):
    build_tiny_cache(tmp_path)
    observed = cache_schema.observe_committed_sequence(tmp_path)
    entry = observed.entries((3,))[0]
    path = tmp_path / entry.frame_file
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    inspection = cache_manager.inspect_cache(tmp_path)
    assert inspection.last_complete_frame == 2
    assert not inspection.healthy
    with pytest.raises(cache_schema.SequenceIntegrityError, match="checksum mismatch"):
        observed.revalidate()
    with pytest.raises(cache_schema.SequenceIntegrityError):
        cache_schema.observe_committed_sequence(tmp_path)


def test_manifest_replacement_during_verification_is_refused(tmp_path, monkeypatch):
    build_tiny_cache(tmp_path)
    original = cache_schema.verified_contiguous_frames
    path = cache_schema.manifest_path(tmp_path)

    def replace_after_verification(base, manifest):
        frames = original(base, manifest)
        path.write_bytes(path.read_bytes() + b" ")
        return frames

    monkeypatch.setattr(cache_schema, "verified_contiguous_frames", replace_after_verification)
    with pytest.raises(ValueError, match="changed during integrity"):
        cache_schema.observe_committed_sequence(tmp_path)
