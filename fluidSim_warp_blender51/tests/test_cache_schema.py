from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fluidsim_testpkg.io.cache_schema import (
    CacheManifestV2,
    FrameEntry,
    load_manifest,
    sha256_file,
    verify_frame_entry,
    write_manifest_atomic,
)


def _manifest() -> CacheManifestV2:
    return CacheManifestV2.new(
        solver_version="1.0.248",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        frame_start=1,
        frame_end=10,
        checkpoint_interval=1,
        timestamp="2026-08-12T10:00:00Z",
    )


def _entry(frame: int = 1, **changes) -> FrameEntry:
    values = dict(
        frame=frame,
        frame_file=f"frames/frame_{frame:06d}.npz",
        frame_sha256="a" * 64,
        frame_size=100,
        checkpoint_file=f"checkpoints/checkpoint_{frame:06d}.npz",
        checkpoint_sha256="b" * 64,
        checkpoint_size=200,
        completed_at="2026-08-12T10:00:01Z",
        solver_frame=frame,
        simulation_time=frame / 24.0,
        settings_signature="settings",
        source_signature="sources",
    )
    values.update(changes)
    return FrameEntry(**values)


def test_manifest_atomic_round_trip(tmp_path: Path):
    path = tmp_path / "fluidsim_cache.json"
    manifest = _manifest().with_committed_frame(
        _entry(), timestamp="2026-08-12T10:00:01Z"
    )
    write_manifest_atomic(path, manifest)
    assert load_manifest(path) == manifest
    assert not path.with_name(path.name + ".tmp").exists()


def test_unknown_format_is_rejected(tmp_path: Path):
    path = tmp_path / "fluidsim_cache.json"
    payload = _manifest().to_dict()
    payload["format_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported cache format"):
        load_manifest(path)


def test_unknown_manifest_field_is_rejected(tmp_path: Path):
    path = tmp_path / "fluidsim_cache.json"
    payload = _manifest().to_dict()
    payload["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_manifest(path)


def test_frame_commit_must_be_contiguous():
    manifest = _manifest()
    with pytest.raises(ValueError, match="next frame is 1"):
        manifest.with_committed_frame(_entry(2))


def test_manifest_rejects_gap_even_when_constructed_directly():
    manifest = _manifest()
    manifest.frames = {1: _entry(1), 3: _entry(3)}
    with pytest.raises(ValueError, match="contiguous"):
        manifest.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_sha256", "A" * 64, "SHA-256"),
        ("frame_file", "../frame.npz", "safe relative"),
        ("checkpoint_file", "/tmp/checkpoint.npz", "safe relative"),
        ("frame_size", 0, ">= 1"),
        ("simulation_time", float("nan"), "finite"),
        ("solver_frame", 2, "does not match"),
    ],
)
def test_frame_entry_strict_validation(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(_entry(), **{field: value}).validate()


def test_frame_signature_must_match_manifest():
    manifest = _manifest()
    with pytest.raises(ValueError, match="settings signature drift"):
        manifest.with_committed_frame(_entry(settings_signature="different"))


def test_verify_frame_entry_checks_size_and_hash(tmp_path: Path):
    frames = tmp_path / "frames"
    checkpoints = tmp_path / "checkpoints"
    frames.mkdir()
    checkpoints.mkdir()
    frame_path = frames / "frame_000001.npz"
    checkpoint_path = checkpoints / "checkpoint_000001.npz"
    frame_path.write_bytes(b"frame")
    checkpoint_path.write_bytes(b"checkpoint")
    entry = _entry(
        frame_sha256=sha256_file(frame_path),
        frame_size=frame_path.stat().st_size,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_size=checkpoint_path.stat().st_size,
    )
    verify_frame_entry(tmp_path, entry)
    frame_path.write_bytes(b"Frame")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_frame_entry(tmp_path, entry)


def test_load_missing_manifest_returns_none(tmp_path: Path):
    assert load_manifest(tmp_path / "missing.json") is None
