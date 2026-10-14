"""Manifest commit and read work without changing a byte (review P07).

Before: every commit re-validated every entry (several times) and deep-copied
the manifest twice, so a bake's commits were O(N^2); every displayed frame and
every Bake-panel redraw re-parsed and re-validated the whole manifest. Now
entries are validated once, serialization converts each entry once, and
unchanged manifest bytes are parsed once. The written bytes, the returned
manifests and every refusal must stay exactly what they were.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fixtures.tiny_cache import build_tiny_cache

from fluidSim_warp_blender51.io import cache, cache_schema
from fluidSim_warp_blender51.io.cache_schema import (
    CacheManifestV3, FrameEntry, load_manifest, manifest_path, write_manifest_atomic,
)


def _manifest(frames: int, *, narrow_band: bool = True) -> CacheManifestV3:
    manifest = CacheManifestV3.new(
        solver_version="3.1.0", scene_key="scene", settings_signature="settings",
        source_signature="sources", frame_start=10, frame_end=10 + frames + 5,
        source_schema_version=4, transfer_mode="APIC",
        grid_spec={"resolution_divisions": 96, "voxel_size": 0.0416666,
                   "domain_size": [4.0, 2.5, 3.25], "leaf_dim": 8},
        narrow_band={"enabled": narrow_band, "band_voxels": 3.0 if narrow_band else 0.0},
        platform={"os": "Windows-11", "gpu_name": "RTX", "driver": "12.9",
                  "warp": "1.17.0"},
        timestamp="2026-10-12T00:00:00Z",
        effect_signature="e" * 64,
        effect_inventory=({"instance_id": "a", "effect_id": "x.y",
                           "parameters": {"gain": 0.5, "axes": [1, 2]}},),
    )
    for index in range(frames):
        frame = 10 + index
        manifest = manifest.with_committed_frame(_entry(frame), timestamp=(
            f"2026-10-12T00:{index // 60:02d}:{index % 60:02d}Z"))
    return manifest


def _entry(frame: int, **overrides) -> FrameEntry:
    values = dict(
        frame=frame, frame_file=f"frames/frame_{frame:06d}.npz",
        frame_sha256=f"{frame:064x}", frame_size=1000 + frame,
        checkpoint_file=f"checkpoints/checkpoint_{frame:06d}.npz",
        checkpoint_sha256=f"{frame * 7:064x}", checkpoint_size=5000 + frame,
        completed_at="2026-10-12T01:02:03Z", solver_frame=frame,
        simulation_time=frame / 24.0 + 1e-13, settings_signature="settings",
        source_signature="sources", source_frame_stamp=f"{frame * 3:064x}",
        whitewater_count=frame % 5, particle_count=100 * frame)
    values.update(overrides)
    return FrameEntry(**values)


def _reference_bytes(manifest: CacheManifestV3) -> bytes:
    """The pre-P07 writer: validate, asdict twice, stream through json.dump."""
    manifest.validate()
    manifest.validate()
    payload = asdict(manifest)
    payload["effect_inventory"] = [dict(item) for item in manifest.effect_inventory]
    payload["frames"] = {
        str(frame): asdict(manifest.frames[frame]) for frame in sorted(manifest.frames)}
    handle = io.StringIO(newline="\n")
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    return handle.getvalue().encode("utf-8")


@pytest.mark.parametrize("frames", [0, 1, 37])
@pytest.mark.parametrize("narrow_band", [False, True])
def test_written_manifest_bytes_are_unchanged(tmp_path, frames, narrow_band):
    manifest = _manifest(frames, narrow_band=narrow_band)
    path = manifest_path(tmp_path)
    write_manifest_atomic(path, manifest)
    assert path.read_bytes() == _reference_bytes(manifest)


def test_to_dict_matches_asdict():
    manifest = _manifest(5)
    expected = asdict(manifest)
    expected["effect_inventory"] = [dict(item) for item in manifest.effect_inventory]
    expected["frames"] = {str(f): asdict(manifest.frames[f]) for f in sorted(manifest.frames)}
    assert manifest.to_dict() == expected


def test_a_commit_does_not_revalidate_committed_entries(monkeypatch):
    manifest = _manifest(200)
    calls = []
    original = cache_schema._require_timestamp
    monkeypatch.setattr(cache_schema, "_require_timestamp",
                        lambda value, name: calls.append(name) or original(value, name))
    candidate = manifest.with_committed_frame(_entry(210))
    candidate.to_dict()
    # Two manifest timestamps per validation, one for the new entry: not 200.
    assert calls.count("completed_at") == 1
    assert len(calls) <= 8


def test_validation_still_refuses_what_it_refused():
    manifest = _manifest(3)
    entry = manifest.frames[10]
    with pytest.raises(ValueError, match="does not match frame"):
        entry.validate(key=11)  # memoized entry: the key is still checked
    bad = _entry(13, completed_at="not a time")
    for _ in range(2):  # a failure is never remembered as a pass
        with pytest.raises(ValueError, match="ISO-8601"):
            manifest.with_committed_frame(bad)
    drift = replace(manifest, frames={**manifest.frames,
                                      13: _entry(13, settings_signature="other")})
    with pytest.raises(ValueError, match="settings signature drift"):
        drift.validate()


def test_load_manifest_returns_independent_equal_copies(tmp_path):
    manifest = _manifest(4)
    path = manifest_path(tmp_path)
    write_manifest_atomic(path, manifest)
    first = load_manifest(path)
    assert first == manifest
    first.frames.clear()
    first.grid_spec["voxel_size"] = 9.0
    first.effect_inventory[0]["parameters"]["axes"].append(3)
    second = load_manifest(path)
    assert second == manifest
    assert second is not first


def test_load_manifest_sees_a_rewrite_with_the_same_size_and_mtime(tmp_path):
    path = manifest_path(tmp_path)
    write_manifest_atomic(path, _manifest(3))
    before = load_manifest(path)
    stat = path.stat()
    changed = replace(_manifest(3), scene_key="SCENE")  # same length
    write_manifest_atomic(path, changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    after = load_manifest(path)
    assert before.scene_key == "scene" and after.scene_key == "SCENE"


def test_a_corrupted_manifest_is_refused_every_time(tmp_path):
    path = manifest_path(tmp_path)
    write_manifest_atomic(path, _manifest(2))
    load_manifest(path)
    text = path.read_text(encoding="utf-8").replace('"leaf_dim": 8', '"leaf_dim": 4')
    path.write_text(text, encoding="utf-8")
    for _ in range(2):
        with pytest.raises(ValueError, match="leaf_dim"):
            load_manifest(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid cache manifest"):
        load_manifest(path)
    assert cache.detect_cache_format(tmp_path) == cache.CACHE_FORMAT_VERSION
    path.write_text(json.dumps({"format_version": 2}), encoding="utf-8")
    assert cache.detect_cache_format(tmp_path) == 2


def test_playback_sees_each_newly_committed_frame(tmp_path):
    base = build_tiny_cache(tmp_path / "cache", frame_count=2)
    assert cache.available_frames(str(base)) == [1, 2]
    assert cache.read_frame(str(base), 3) is None
    first = cache.read_frame(str(base), 1)
    # Extend the cache by one frame: same writer, same manifest file.
    other = build_tiny_cache(tmp_path / "other", frame_count=3)
    for name in ("frames/frame_000003.npz", "checkpoints/checkpoint_000003.npz"):
        target = base / name
        target.write_bytes((other / name).read_bytes())
    manifest = load_manifest(manifest_path(base))
    manifest = replace(manifest, frame_end=3).with_committed_frame(
        load_manifest(manifest_path(other)).frames[3])
    write_manifest_atomic(manifest_path(base), manifest)
    assert cache.available_frames(str(base)) == [1, 2, 3]
    assert cache.read_sequence_metadata(str(base))["frame_count"] == 3
    third = cache.read_frame(str(base), 3)
    assert int(third["frame"]) == 3
    again = cache.read_frame(str(base), 1)
    assert set(again) == set(first)
    for key in first:
        assert again[key].tobytes() == first[key].tobytes()


def test_playback_still_hashes_the_frame_on_every_read(tmp_path):
    base = build_tiny_cache(tmp_path / "cache", frame_count=2)
    cache.read_frame(str(base), 2)
    frame = Path(base) / "frames" / "frame_000002.npz"
    data = bytearray(frame.read_bytes())
    data[-30] ^= 0xFF
    frame.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        cache.read_frame(str(base), 2)
