"""Shared fixtures for the 4.0 two-way rigid coupling tests (lane R)."""

from __future__ import annotations

import numpy as np

H = 1.0 / 48.0


def box_mesh_arrays(center, half_extents):
    """Closed, outward-wound box: (8, 3) float32 points, (36,) int32 indices."""
    cx, cy, cz = center
    hx, hy, hz = half_extents
    points = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    tris = np.array([
        0, 2, 1, 0, 3, 2,  # -z
        4, 5, 6, 4, 6, 7,  # +z
        0, 1, 5, 0, 5, 4,  # -y
        3, 7, 6, 3, 6, 2,  # +y
        0, 4, 7, 0, 7, 3,  # -x
        1, 2, 6, 1, 6, 5,  # +x
    ], dtype=np.int32)
    return points, tris


def box_collider(wp, device, center, half_extents, **fields):
    from fluidSim_warp_blender51.sim.solver import ColliderDesc

    points, tris = box_mesh_arrays(center, half_extents)
    mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                   indices=wp.array(tris, dtype=wp.int32, device=device))
    return ColliderDesc(
        mesh=mesh, center=tuple(float(v) for v in center),
        aabb_min=tuple(float(v) for v in points.min(axis=0)),
        aabb_max=tuple(float(v) for v in points.max(axis=0)),
        **fields)


def lattice(lo, hi, spacing, exclude=None):
    """Particles on a regular lattice in [lo, hi), minus an optional AABB."""
    axes = [np.arange(lo[i] + 0.5 * spacing, hi[i], spacing) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = np.stack([g.ravel() for g in grid], axis=1)
    if exclude is not None:
        emin, emax = (np.asarray(v) for v in exclude)
        inside = np.all((points > emin) & (points < emax), axis=1)
        points = points[~inside]
    return np.ascontiguousarray(points, dtype=np.float32)
