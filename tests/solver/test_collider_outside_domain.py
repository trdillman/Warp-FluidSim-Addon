"""A collider with no domain overlap has no influence instead of aborting.

Regression for review item B09 (owner decision D7, 2026-10-12): the topology
rebuild marked every collider AABB unconditionally, so a collider entirely
outside the domain (or animating out of it) raised ValueError from
``mark_world_box`` and playback released the session. The capacity planner
already treated such bounds as zero demand.
"""
from __future__ import annotations

import numpy as np
import pytest


def _tetra_collider(wp, device, lo, size):
    from fluidSim_warp_blender51.sim.solver import ColliderDesc

    base = np.array(lo, dtype=np.float32)
    pts = np.array([base, base + (size, 0, 0), base + (0, size, 0),
                    base + (0, 0, size)], dtype=np.float32)
    tris = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)
    mesh = wp.Mesh(points=wp.array(pts, dtype=wp.vec3, device=device),
                   indices=wp.array(tris, dtype=wp.int32, device=device))
    return ColliderDesc(mesh=mesh, aabb_min=tuple(pts.min(axis=0)),
                        aabb_max=tuple(pts.max(axis=0)))


def test_collider_outside_the_domain_is_ignored(cuda_device):
    wp = pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(-0.4, -0.4, 0.0), domain_max=(0.4, 0.4, 0.8),
        voxel_size=0.05, leaf_capacity=64, max_particles=4096,
        max_whitewater=1024, device=cuda_device)
    solver = FluidSolver(settings)
    try:
        rng = np.random.default_rng(0)
        solver.inject_particles(rng.uniform(
            (-0.2, -0.2, 0.1), (0.2, 0.2, 0.3), (500, 3)).astype(np.float32))
        inside = _tetra_collider(wp, cuda_device, (-0.3, -0.3, 0.05), 0.1)
        outside = _tetra_collider(wp, cuda_device, (1.0, 1.0, 1.0), 0.2)
        solver.set_colliders([inside, outside])

        solver.step_frame()

        assert solver.frame == 1
        assert solver.particles.refresh_host_count() == 500
    finally:
        solver.release()
