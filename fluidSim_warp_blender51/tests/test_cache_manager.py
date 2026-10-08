from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fluidsim_testpkg.blender.cache_manager import (
    free_cache,
    inspect_cache,
    move_cache,
    remove_orphans,
    rename_cache,
    validate_leaf_name,
)
from fluidsim_testpkg.io.cache import CACHE_FORMAT_VERSION
from fluidsim_testpkg.io.cache_schema import (
    LEGACY_2X_CACHE_MESSAGE,
    LEGACY_2X_FORMAT_VERSION,
    CacheManifestV3,
    FrameEntry,
    manifest_path,
    write_manifest_atomic,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_2x_cache(base: Path) -> None:
    """A minimal FluidSim 2.x (format 2) cache directory with one frame."""
    frame = base / "frames" / "frame_000001.npz"
    checkpoint = base / "checkpoints" / "checkpoint_000001.npz"
    frame.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    frame.write_bytes(b"frame-2x")
    checkpoint.write_bytes(b"checkpoint-2x")
    manifest = {
        "format_version": 2,
        "checkpoint_format_version": 1,
        "solver_version": "2.1.0",
        "scene_key": "scene",
        "settings_signature": "settings",
        "source_signature": "sources",
        "frame_start": 1,
        "frame_end": 5,
        "checkpoint_interval": 1,
        "created_at": "2026-09-01T10:00:00Z",
        "updated_at": "2026-09-01T10:00:01Z",
        "effect_signature": "",
        "effect_inventory": [],
        "frames": {"1": {
            "frame": 1,
            "frame_file": "frames/frame_000001.npz",
            "frame_sha256": _digest(b"frame-2x"),
            "frame_size": 8,
            "checkpoint_file": "checkpoints/checkpoint_000001.npz",
            "checkpoint_sha256": _digest(b"checkpoint-2x"),
            "checkpoint_size": 13,
            "completed_at": "2026-09-01T10:00:01Z",
            "solver_frame": 1,
            "simulation_time": 0.04,
            "settings_signature": "settings",
            "source_signature": "sources",
        }},
    }
    (base / "fluidsim_cache.json").write_text(json.dumps(manifest), encoding="utf-8")


def _tree(base: Path) -> dict[str, bytes]:
    return {
        item.relative_to(base).as_posix(): item.read_bytes()
        for item in sorted(base.rglob("*")) if item.is_file()
    }


def _make_cache(base: Path, frames: int = 2) -> CacheManifestV3:
    manifest = CacheManifestV3.new(
        solver_version="3.0.0",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        frame_start=1,
        frame_end=max(frames, 2),
        source_schema_version=1,
        transfer_mode="FLIP",
        grid_spec={
            "resolution_divisions": 64, "voxel_size": 0.0625,
            "domain_size": [4.0, 4.0, 4.0], "leaf_dim": 8},
        narrow_band={"enabled": False, "band_voxels": 0.0},
        platform={
            "os": "Windows-11", "gpu_name": "NVIDIA RTX", "driver": "560.94",
            "warp": "1.17.0"},
        timestamp="2026-08-11T13:00:00Z",
        effect_signature="e" * 64,
        effect_inventory=({"instance_id": "effect-1", "effect_id": "test.vortex"},),
    )
    for frame in range(1, frames + 1):
        frame_payload = f"frame-{frame}".encode()
        checkpoint_payload = f"checkpoint-{frame}".encode()
        frame_path = base / "frames" / f"frame_{frame:06d}.npz"
        checkpoint_path = base / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(frame_payload)
        checkpoint_path.write_bytes(checkpoint_payload)
        manifest = manifest.with_committed_frame(
            FrameEntry(
                frame=frame,
                frame_file=frame_path.relative_to(base).as_posix(),
                frame_sha256=_digest(frame_payload),
                frame_size=len(frame_payload),
                checkpoint_file=checkpoint_path.relative_to(base).as_posix(),
                checkpoint_sha256=_digest(checkpoint_payload),
                checkpoint_size=len(checkpoint_payload),
                completed_at=f"2026-08-11T13:00:0{frame}Z",
                solver_frame=frame,
                simulation_time=frame / 24.0,
                settings_signature="settings",
                source_signature="sources",
                source_frame_stamp="c" * 64,
                whitewater_count=0,
                particle_count=1,
            ),
            timestamp=f"2026-08-11T13:00:0{frame}Z",
        )
    write_manifest_atomic(manifest_path(base), manifest)
    return manifest


def test_inspection_is_manifest_authoritative_and_reports_orphans(tmp_path):
    _make_cache(tmp_path)
    orphan = tmp_path / "frames" / "frame_000003.npz"
    orphan.write_bytes(b"uncommitted")

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == CACHE_FORMAT_VERSION == 3
    assert inspection.message == "Cache v3: 2 verified committed frame(s)"
    assert inspection.committed_frames == (1, 2)
    assert inspection.last_resumable_frame == 2
    assert inspection.healthy is True
    assert inspection.effect_signature == "e" * 64
    assert inspection.effect_inventory[0]["instance_id"] == "effect-1"
    assert inspection.orphan_files == ("frames/frame_000003.npz",)
    assert remove_orphans(tmp_path) == ("frames/frame_000003.npz",)
    assert not orphan.exists()


def test_integrity_failure_fails_closed(tmp_path):
    _make_cache(tmp_path)
    (tmp_path / "frames" / "frame_000002.npz").write_bytes(b"corrupt")

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == CACHE_FORMAT_VERSION
    assert inspection.healthy is False
    assert inspection.compatible is False
    assert inspection.message == "Cache v3 failed integrity verification"
    assert inspection.last_complete_frame == 1
    assert "frames/frame_000002.npz" in inspection.corrupt_files


def test_a_corrupt_current_manifest_is_invalid_not_a_2x_cache(tmp_path):
    (tmp_path / "fluidsim_cache.json").write_text("{not json", encoding="utf-8")

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == CACHE_FORMAT_VERSION
    assert inspection.compatible is False
    assert inspection.healthy is False
    assert inspection.corrupt_files == ("fluidsim_cache.json",)
    assert "invalid" in inspection.message
    assert inspection.message != LEGACY_2X_CACHE_MESSAGE
    assert "cache-v2" not in inspection.message.lower()


def test_fluidsim_2x_cache_is_refused_with_the_rebake_message(tmp_path):
    _write_2x_cache(tmp_path)
    before = _tree(tmp_path)

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == LEGACY_2X_FORMAT_VERSION == 2
    assert inspection.compatible is False
    assert inspection.message == LEGACY_2X_CACHE_MESSAGE
    assert inspection.committed_frames == ()
    assert inspection.last_complete_frame is None
    assert inspection.last_resumable_frame is None
    assert inspection.orphan_files == ()
    assert remove_orphans(tmp_path) == ()
    assert _tree(tmp_path) == before, "inspection must never touch a 2.x cache"


def test_fluidsim_2x_cache_can_be_freed(tmp_path):
    cache = tmp_path / "cache"
    _write_2x_cache(cache)

    assert free_cache(cache) is None

    assert not cache.exists()


def test_fluidsim_2x_cache_moves_byte_for_byte_without_raising(tmp_path):
    source = tmp_path / "source"
    _write_2x_cache(source)
    before = _tree(source)

    destination = move_cache(source, tmp_path / "destination")

    assert not source.exists()
    assert _tree(destination) == before
    moved = inspect_cache(destination)
    assert moved.format_version == LEGACY_2X_FORMAT_VERSION
    assert moved.message == LEGACY_2X_CACHE_MESSAGE
    assert rename_cache(destination, "renamed").name == "renamed"


def test_fluidsim_1_0_cache_is_refused_but_can_be_freed(tmp_path):
    """3.0 reads cache format 3 only; a 1.0 cache must say to re-bake."""
    (tmp_path / "fluidsim_sequence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fluidsim_00001.npz").write_bytes(b"legacy")

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == 1
    assert inspection.compatible is False
    assert inspection.committed_frames == ()
    assert inspection.last_complete_frame is None
    assert inspection.last_resumable_frame is None
    assert "FluidSim 1.0" in inspection.message and "bake again" in inspection.message
    assert free_cache(tmp_path) is None
    assert not tmp_path.exists()


def test_move_rename_and_free_preserve_verified_cache(tmp_path):
    source = tmp_path / "source"
    _make_cache(source)
    destination = move_cache(source, tmp_path / "destination")
    assert not source.exists()
    assert inspect_cache(destination).healthy

    renamed = rename_cache(destination, "renamed")
    assert renamed.name == "renamed"
    assert inspect_cache(renamed).committed_frames == (1, 2)

    assert free_cache(renamed) is None
    assert not renamed.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", r"a\\b"])
def test_rename_rejects_non_leaf_names(name):
    with pytest.raises(ValueError):
        validate_leaf_name(name)


def test_cache_mutations_reject_an_arbitrary_directory(tmp_path):
    arbitrary = tmp_path / "project-files"
    arbitrary.mkdir()
    sentinel = arbitrary / "important.blend"
    sentinel.write_bytes(b"user data")

    with pytest.raises(ValueError, match="not a recognized FluidSim cache"):
        free_cache(arbitrary)
    with pytest.raises(ValueError, match="not a recognized FluidSim cache"):
        move_cache(arbitrary, tmp_path / "moved")
    with pytest.raises(ValueError, match="not a recognized FluidSim cache"):
        rename_cache(arbitrary, "renamed")

    assert sentinel.read_bytes() == b"user data"
    assert arbitrary.is_dir()


def test_interrupted_first_transaction_is_recognized_as_managed_cache(tmp_path):
    from fluidsim_testpkg.blender.cache_manager import is_managed_cache_directory

    partial = tmp_path / "partial"
    frame = partial / "frames" / "frame_000001.npz.abc.tmp"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"uncommitted")

    assert is_managed_cache_directory(partial) is True


RESTART_ENTRIES = ("fluidsim_cache.json", "frames", "checkpoints", "logs")


def test_restart_refuses_to_replace_user_folders_outside_a_cache(tmp_path):
    from fluidsim_testpkg.blender.cache_manager import require_restartable_cache_directory

    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    (project / "logs" / "notes.txt").write_bytes(b"user data")
    (project / "frames").mkdir()

    with pytest.raises(ValueError, match="not a FluidSim cache") as refused:
        require_restartable_cache_directory(project, RESTART_ENTRIES)

    assert "replace logs in" in str(refused.value)
    assert (project / "logs" / "notes.txt").read_bytes() == b"user data"


def test_restart_allows_new_empty_or_recognized_cache_directories(tmp_path):
    from fluidsim_testpkg.blender.cache_manager import require_restartable_cache_directory

    missing = tmp_path / "missing"
    empty = tmp_path / "empty"
    (empty / "frames").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "logs").mkdir(parents=True)
    (cache / "logs" / "bake.jsonl").write_bytes(b"old-log")
    (cache / "fluidsim_cache.json").write_bytes(b"manifest")

    for path in (missing, empty, cache):
        assert require_restartable_cache_directory(path, RESTART_ENTRIES) == path.resolve()
