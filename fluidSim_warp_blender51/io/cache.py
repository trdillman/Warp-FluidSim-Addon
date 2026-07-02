"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.
"""

from __future__ import annotations

import os

import numpy as np


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
    np.savez_compressed(
        path,
        positions=solver.particle_positions(),
        velocities=solver.particle_velocities(),
        whitewater=solver.whitewater_positions(),
        surface_verts=verts,
        surface_tris=tris,
        frame=solver.frame,
    )
    return path


def read_frame(base_dir: str, frame: int) -> dict | None:
    path = _frame_path(base_dir, frame)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


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
