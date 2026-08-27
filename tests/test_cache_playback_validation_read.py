"""Focused cache-v2 playback-read validation regressions."""

from pathlib import Path

import numpy as np
import pytest

from fluidSim_warp_blender51.io.cache import read_frame_for_playback
from fluidSim_warp_blender51.io.cache_schema import (
    CacheManifestV2,
    FrameEntry,
    sha256_file,
    write_manifest_atomic,
)


def _payload(frame):
    empty_vec3 = np.zeros((0, 3), dtype=np.float32)
    empty_scalar = np.zeros(0, dtype=np.float32)
    return {
        "format_version": np.array(2, dtype=np.int32),
        "frame": np.array(frame, dtype=np.int32),
        "solver_frame": np.array(frame, dtype=np.int32),
        "simulation_time": np.array(float(frame), dtype=np.float64),
        "settings_signature": np.array("settings-signature"),
        "source_signature": np.array("source-signature"),
        "scene_key": np.array("scene-key"),
        "effect_signature": np.array(""),
        "effect_inventory_json": np.array("[]"),
        "positions": empty_vec3,
        "velocities": empty_vec3,
        "particle_age": empty_scalar,
        "particle_life": empty_scalar,
        "particle_temperature": empty_scalar,
        "particle_viscosity": empty_scalar,
        "particle_density": empty_scalar,
        "particle_time_offset": empty_scalar,
        "whitewater": empty_vec3,
        "whitewater_velocities": empty_vec3,
        "whitewater_age": empty_scalar,
        "whitewater_life": empty_scalar,
    }


def _write_two_frame_cache(base: Path):
    manifest = CacheManifestV2.new(
        solver_version="test-solver",
        scene_key="scene-key",
        settings_signature="settings-signature",
        source_signature="source-signature",
        frame_start=1,
        frame_end=2,
    )
    for frame in (1, 2):
        frame_path = base / "frames" / f"frame_{frame:06d}.npz"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(frame_path, **_payload(frame))
        checkpoint_path = base / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"checkpoint-{frame}".encode("ascii"))
        manifest = manifest.with_committed_frame(FrameEntry(
            frame=frame,
            frame_file=frame_path.relative_to(base).as_posix(),
            frame_sha256=sha256_file(frame_path),
            frame_size=frame_path.stat().st_size,
            checkpoint_file=checkpoint_path.relative_to(base).as_posix(),
            checkpoint_sha256=sha256_file(checkpoint_path),
            checkpoint_size=checkpoint_path.stat().st_size,
            completed_at="2026-08-26T00:00:00+00:00",
            solver_frame=frame,
            simulation_time=float(frame),
            settings_signature="settings-signature",
            source_signature="source-signature",
        ))
    write_manifest_atomic(base / "fluidsim_cache.json", manifest)
    return manifest


def test_playback_read_carries_identity_and_validates_each_frame_sha(tmp_path):
    manifest = _write_two_frame_cache(tmp_path)

    first = read_frame_for_playback(str(tmp_path), 1)
    assert first is not None
    assert int(first.frame_data["frame"]) == 1
    assert first.metadata["scene_key"] == manifest.scene_key
    assert first.metadata_identity[0] == 2

    second_path = tmp_path / "frames" / "frame_000002.npz"
    second_path.write_bytes(second_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match=r"frame 2 size mismatch"):
        read_frame_for_playback(str(tmp_path), 2)
