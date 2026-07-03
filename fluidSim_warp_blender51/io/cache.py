"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import zipfile

import numpy as np

CACHE_FORMAT_VERSION = 1


def _frame_path(base_dir: str, frame: int) -> str:
    return os.path.join(base_dir, f"fluidsim_{frame:05d}.npz")


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
