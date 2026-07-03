"""Optional export tooling: surface OBJ and particle PLY snapshots."""

from __future__ import annotations

import os

import numpy as np


def _require_existing_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Output parent directory does not exist: {parent}")


def export_surface_obj(solver, path: str) -> str:
    verts, tris = solver.surface_mesh()
    _require_existing_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# FluidSim Warp surface export\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return path


def export_points_ply(positions: np.ndarray, path: str,
                      velocities: np.ndarray | None = None) -> str:
    n = len(positions)
    has_vel = velocities is not None and len(velocities) == n
    _require_existing_parent(path)
    with open(path, "wb") as fh:
        header = ["ply", "format binary_little_endian 1.0",
                  f"element vertex {n}",
                  "property float x", "property float y", "property float z"]
        if has_vel:
            header += ["property float vx", "property float vy",
                       "property float vz"]
        header += ["end_header"]
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        if has_vel:
            data = np.hstack([positions.astype(np.float32),
                              velocities.astype(np.float32)])
        else:
            data = positions.astype(np.float32)
        fh.write(np.ascontiguousarray(data).tobytes())
    return path


def export_frame(solver, base_path: str) -> list:
    """Export current frame surface + particles + whitewater next to base."""
    _require_existing_parent(base_path)
    paths = []
    if solver.settings.surface_enabled:
        paths.append(export_surface_obj(solver, base_path + "_surface.obj"))
    pos = solver.particle_positions()
    if len(pos):
        paths.append(export_points_ply(
            pos, base_path + "_particles.ply",
            velocities=solver.particle_velocities()))
    ww = solver.whitewater_positions()
    if len(ww):
        paths.append(export_points_ply(ww, base_path + "_whitewater.ply"))
    return paths
