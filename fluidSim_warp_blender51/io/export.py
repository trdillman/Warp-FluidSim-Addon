"""Optional export tooling: surface OBJ and particle PLY snapshots."""

from __future__ import annotations

import os

import numpy as np


def _host_frame(solver):
    from ..sim.frame_output import HostFrame
    return solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)


def _require_existing_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Output parent directory does not exist: {parent}")


def export_surface_obj(solver, path: str, *, host=None) -> str:
    _require_existing_parent(path)
    verts, tris = (host or _host_frame(solver)).surface_mesh
    metadata = solver.surface.metadata()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# FluidSim Warp surface export\n")
        fh.write(f"# surface_voxel_size {metadata['voxel_size']:.8g}\n")
        fh.write(
            f"# surface_resolution_mode {int(solver.settings.surface_resolution_mode)}\n"
        )
        fh.write(
            "# surface_resolution_multiplier "
            f"{float(solver.settings.surface_resolution_multiplier):.8g}\n"
        )
        fh.write(
            f"# surface_smooth_iterations {int(solver.settings.surface_smooth_iterations)}\n"
        )
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    return path


def export_points_ply(positions: np.ndarray, path: str,
                      velocities: np.ndarray | None = None) -> str:
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    n = len(positions)
    has_vel = velocities is not None
    if has_vel:
        velocities = np.asarray(velocities)
        if velocities.ndim != 2 or velocities.shape != positions.shape:
            raise ValueError("velocities must match positions shape (N, 3)")
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
    host = _host_frame(solver)
    paths = []
    if solver.settings.surface_enabled:
        paths.append(export_surface_obj(
            solver, base_path + "_surface.obj", host=host))
    pos = host.positions
    if len(pos):
        paths.append(export_points_ply(
            pos, base_path + "_particles.ply",
            velocities=host.velocities))
    ww = host.whitewater
    if len(ww):
        paths.append(export_points_ply(ww, base_path + "_whitewater.ply"))
    return paths
