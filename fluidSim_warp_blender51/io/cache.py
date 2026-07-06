"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import zipfile

import numpy as np

CACHE_FORMAT_VERSION = 1
CACHE_SEQUENCE_METADATA = "fluidsim_sequence.json"


def _frame_path(base_dir: str, frame: int) -> str:
    return os.path.join(base_dir, f"fluidsim_{frame:05d}.npz")


def _metadata_path(base_dir: str) -> str:
    return os.path.join(base_dir, CACHE_SEQUENCE_METADATA)


def _optional_particle_attr(solver, name: str):
    try:
        return solver.particles.attr(name).numpy()[:solver.particles.host_count]
    except AttributeError:
        return np.zeros(0, dtype=np.float32)


def _optional_whitewater_attr(solver, name: str):
    try:
        return getattr(solver.whitewater, name).numpy()[:solver.whitewater.host_count]
    except AttributeError:
        return np.zeros(0, dtype=np.float32)


def resolve_frame_range(
        scene,
        *,
        use_custom: bool,
        frame_start: int,
        frame_end: int) -> tuple[int, int]:
    """Resolve a cache operation range from scene defaults or custom override."""
    if use_custom:
        start = int(frame_start)
        end = int(frame_end)
    else:
        start = int(scene.frame_start)
        end = int(scene.frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    return start, end


def write_sequence_metadata(
        base_dir: str,
        frame_start: int,
        frame_end: int,
        *,
        source: str = "simulation") -> str:
    """Write sequence-level cache metadata next to per-frame .npz files."""
    os.makedirs(base_dir, exist_ok=True)
    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "source": str(source),
    }
    path = _metadata_path(base_dir)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{CACHE_SEQUENCE_METADATA}.",
        suffix=".tmp",
        dir=base_dir,
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def read_sequence_metadata(base_dir: str) -> dict | None:
    path = _metadata_path(base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cache metadata format_version "
            f"{data.get('format_version')} in {path}; "
            f"expected {CACHE_FORMAT_VERSION}."
        )
    return data


def write_frame(scene, sess) -> str:
    import bpy

    fs = scene.fluidsim
    base = bpy.path.abspath(fs.cache_path)
    os.makedirs(base, exist_ok=True)
    solver = sess.solver
    verts, tris = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    if solver.settings.surface_enabled:
        verts, tris = solver.surface_mesh()
    path = _frame_path(base, solver.frame)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(path)}.",
        suffix=".tmp.npz",
        dir=base,
    )
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            format_version=np.array(CACHE_FORMAT_VERSION, dtype=np.int32),
            positions=solver.particle_positions(),
            velocities=solver.particle_velocities(),
            whitewater=solver.whitewater_positions(),
            surface_verts=verts,
            surface_tris=tris,
            frame=solver.frame,
            voxel_size=np.array(
                getattr(solver.settings, "voxel_size", 0.05), dtype=np.float32),
            particle_age=_optional_particle_attr(solver, "age"),
            particle_life=_optional_particle_attr(solver, "life"),
            particle_temperature=_optional_particle_attr(solver, "temperature"),
            particle_viscosity=_optional_particle_attr(solver, "viscosity"),
            particle_density=_optional_particle_attr(solver, "density"),
            whitewater_velocities=_optional_whitewater_attr(solver, "velocity"),
            whitewater_age=_optional_whitewater_attr(solver, "age"),
            whitewater_life=_optional_whitewater_attr(solver, "life"),
        )
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def read_frame(base_dir: str, frame: int) -> dict | None:
    path = _frame_path(base_dir, frame)
    if not os.path.exists(path):
        return None
    try:
        loaded = np.load(path, allow_pickle=False)
    except (EOFError, OSError, ValueError, pickle.UnpicklingError, zipfile.BadZipFile):
        return None

    with loaded as data:
        if "format_version" not in data.files:
            raise ValueError(
                f"Cache file {path} is missing format_version; regenerate the cache."
            )
        version = int(np.asarray(data["format_version"]).item())
        if version != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported cache format_version {version} in {path}; "
                f"expected {CACHE_FORMAT_VERSION}. Regenerate the cache."
            )
        try:
            return {k: data[k] for k in data.files}
        except (ValueError, pickle.UnpicklingError):
            return None


def available_frames(base_dir: str) -> list:
    if not os.path.isdir(base_dir):
        return []
    frames = []
    for name in os.listdir(base_dir):
        if name.startswith("fluidsim_") and name.endswith(".npz"):
            try:
                frames.append(int(name[9:-4]))
            except ValueError:
                pass
    return sorted(frames)


def validate_contiguous_frames(
        base_dir: str,
        frame_start: int,
        frame_end: int) -> list[int]:
    """Return cache frames missing from the inclusive requested range."""
    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    present = set(available_frames(base_dir))
    return [frame for frame in range(start, end + 1) if frame not in present]
