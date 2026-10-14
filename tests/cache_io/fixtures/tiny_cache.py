"""Build a tiny, valid cache-v3 sequence from numpy data, with no solver and no Warp.

Used by the solver-free playback test and by the Alembic export acceptance script.
The surface is a UV sphere whose tessellation changes every frame (so the vertex
count varies, like a real liquid surface) and whose centre drifts along +x.
Frames go through the real cache writer and manifest transaction; the only
stand-in is each checkpoint, a small placeholder file the manifest hashes
(nothing here resumes a bake).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fluidSim_warp_blender51.io import cache, cache_schema

SETTINGS_SIGNATURE = "tiny-cache-settings"
SOURCE_SIGNATURE = "tiny-cache-sources"


def sphere_mesh(frame: int) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere of radius 0.5 centred at x = 0.1 * frame; vertex count varies per frame."""
    segments = 4 + frame
    rings = 3 + frame % 3
    verts = [(0.0, 0.0, 0.5)]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        for seg in range(segments):
            theta = 2.0 * np.pi * seg / segments
            verts.append((0.5 * np.sin(phi) * np.cos(theta),
                          0.5 * np.sin(phi) * np.sin(theta),
                          0.5 * np.cos(phi)))
    verts.append((0.0, 0.0, -0.5))
    verts = np.asarray(verts, dtype=np.float32)
    verts[:, 0] += 0.1 * frame
    south = len(verts) - 1
    tris = []
    for seg in range(segments):
        nxt = (seg + 1) % segments
        tris.append((0, 1 + seg, 1 + nxt))
        for ring in range(rings - 2):
            a = 1 + ring * segments
            b = a + segments
            tris.append((a + seg, b + seg, b + nxt))
            tris.append((a + seg, b + nxt, a + nxt))
        last = 1 + (rings - 2) * segments
        tris.append((south, last + nxt, last + seg))
    return verts, np.asarray(tris, dtype=np.int32)


def particle_line(frame: int, count: int) -> np.ndarray:
    """``count`` particles on a line along +y, starting at x = 0.1 * frame."""
    particles = np.zeros((count, 3), dtype=np.float32)
    particles[:, 0] = 0.1 * frame
    particles[:, 1] = 1.0e-3 * np.arange(count, dtype=np.float32)
    return particles


def particle_speed(frame: int, count: int) -> np.ndarray:
    """The published ``speed`` attribute: 0.5 * frame on every particle."""
    return np.full(count, 0.5 * frame, dtype=np.float32)


def _payload(frame: int, settings_signature: str, particle_count=None):
    verts, tris = sphere_mesh(frame)
    if particle_count is None:
        particles = np.asarray([[0.1 * frame, 0.0, 0.0], [0.1 * frame, 0.1, 0.0]],
                               dtype=np.float32)
        output_attributes, particle_inventory = {}, ()
    else:
        particles = particle_line(frame, particle_count)
        output_attributes = {
            "particles": {"speed": particle_speed(frame, particle_count)}}
        particle_inventory = ("speed",)
    n = len(particles)
    zeros = np.zeros(n, dtype=np.float32)
    return SimpleNamespace(
        frame=frame,
        settings_signature=settings_signature,
        positions=particles,
        velocities=np.zeros((n, 3), dtype=np.float32),
        whitewater=np.zeros((0, 3), dtype=np.float32),
        surface_verts=verts,
        surface_tris=tris,
        voxel_size=0.1,
        surface_voxel_size=0.1,
        surface_resolution_multiplier=1.0,
        surface_resolution_mode=0,
        surface_update_cadence=1,
        surface_smooth_iterations=0,
        particle_attributes={
            "age": zeros, "life": zeros + 1.0, "temperature": zeros,
            "viscosity": zeros, "density": zeros + 1.0, "time_offset": zeros,
        },
        whitewater_attributes={
            "velocity": np.zeros((0, 3), dtype=np.float32),
            "age": np.zeros(0, dtype=np.float32),
            "life": np.zeros(0, dtype=np.float32),
            "type": np.zeros(0, dtype=np.int32),
        },
        output_attributes=output_attributes,
        attribute_inventory={"surface": (), "particles": particle_inventory,
                             "whitewater": ("type",)},
        whitewater_required=0,
        whitewater_available=0,
        whitewater_admitted=0,
        whitewater_dropped=0,
        whitewater_policy="NONE",
        whitewater_status="OK",
    )


def build_tiny_cache(base_dir, *, frame_start=1, frame_count=3,
                     settings_signature=SETTINGS_SIGNATURE,
                     particle_counts=None) -> Path:
    """Write ``frame_count`` committed frames under ``base_dir`` and return it.

    ``particle_counts`` (one per frame) swaps the two fixed particles for
    :func:`particle_line` points carrying a cached ``speed`` attribute.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    frame_end = frame_start + frame_count - 1
    manifest = cache_schema.CacheManifestV3.new(
        solver_version="3.0.0",
        scene_key="tiny-cache-scene",
        settings_signature=settings_signature,
        source_signature=SOURCE_SIGNATURE,
        frame_start=frame_start,
        frame_end=frame_end,
        source_schema_version=1,
        transfer_mode="FLIP",
        grid_spec={"resolution_divisions": 16, "voxel_size": 0.1,
                   "domain_size": [1.6, 1.6, 1.6], "leaf_dim": 8},
        narrow_band={"enabled": False, "band_voxels": 0.0},
        platform={"os": "test", "gpu_name": "none", "driver": "0", "warp": "none"},
    )
    for frame in range(frame_start, frame_end + 1):
        stamp = hashlib.sha256(f"tiny-frame-{frame}".encode()).hexdigest()
        payload = _payload(
            frame, settings_signature,
            None if particle_counts is None
            else particle_counts[frame - frame_start])
        frame_path, frame_sha, frame_size = cache.write_frame_v3_atomic(
            base, payload, frame,
            scene_key_value="tiny-cache-scene",
            settings_signature=settings_signature,
            source_signature=SOURCE_SIGNATURE,
            source_frame_stamp=stamp,
            simulation_time=frame / 24.0,
        )
        checkpoint = base / "checkpoints" / f"checkpoint_{frame:06d}.npz"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"placeholder checkpoint %d" % frame)
        entry = cache_schema.FrameEntry(
            frame=frame,
            frame_file=frame_path.relative_to(base).as_posix(),
            frame_sha256=frame_sha,
            frame_size=frame_size,
            checkpoint_file=checkpoint.relative_to(base).as_posix(),
            checkpoint_sha256=cache_schema.sha256_file(checkpoint),
            checkpoint_size=checkpoint.stat().st_size,
            completed_at="2026-10-09T00:00:00Z",
            solver_frame=frame,
            simulation_time=frame / 24.0,
            settings_signature=settings_signature,
            source_signature=SOURCE_SIGNATURE,
            source_frame_stamp=stamp,
            whitewater_count=0,
            particle_count=len(payload.positions),
        )
        manifest = manifest.with_committed_frame(entry)
        cache_schema.write_manifest_atomic(cache_schema.manifest_path(base), manifest)
    return base
