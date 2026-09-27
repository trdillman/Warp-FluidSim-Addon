from __future__ import annotations

import hashlib
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
from fluidsim_testpkg.io.cache_schema import (
    CacheManifestV2,
    FrameEntry,
    manifest_path,
    write_manifest_atomic,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _make_cache(base: Path, frames: int = 2) -> CacheManifestV2:
    manifest = CacheManifestV2.new(
        solver_version="1.0.248",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        frame_start=1,
        frame_end=max(frames, 2),
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

    assert inspection.format_version == 2
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

    assert inspection.healthy is False
    assert inspection.compatible is False
    assert inspection.last_complete_frame == 1
    assert "frames/frame_000002.npz" in inspection.corrupt_files


def test_legacy_cache_is_playback_only(tmp_path):
    (tmp_path / "fluidsim_sequence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fluidsim_000001.npz").write_bytes(b"legacy")

    inspection = inspect_cache(tmp_path)

    assert inspection.format_version == 1
    assert inspection.last_complete_frame == 1
    assert inspection.last_resumable_frame is None
    assert "playback only" in inspection.message


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
