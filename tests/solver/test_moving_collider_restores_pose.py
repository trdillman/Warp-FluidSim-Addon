"""A moving collider must hand the scene's cached mesh back at its authored pose.

Regression for review item B11: ``_update_moving_colliders`` writes
extrapolated sub-frame points into the collider mesh, which is the scene's
cached evaluated mesh. When the collider then stopped, the scene reused that
unchanged cached mesh with zero velocity, so collision geometry stayed at the
extrapolated pose, ahead of the real object (probe: 0.75 units).
"""
from __future__ import annotations

import numpy as np
import pytest


def test_moving_collider_mesh_is_restored_after_each_frame(cuda_device):
    wp = pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.solver import (
        ColliderDesc, FluidSolver, SolverSettings,
    )

    settings = SolverSettings(
        domain_min=(-0.4, -0.4, 0.0), domain_max=(0.4, 0.4, 0.8),
        voxel_size=0.05, leaf_capacity=64, max_particles=4096,
        max_whitewater=1024, min_substeps=3, max_substeps=3,
        device=cuda_device)
    solver = FluidSolver(settings)
    try:
        rng = np.random.default_rng(3)
        solver.inject_particles(rng.uniform(
            (-0.2, -0.2, 0.1), (0.2, 0.2, 0.3), (500, 3)).astype(np.float32))
        authored = np.array([[-0.1, -0.1, 0.05], [0.1, -0.1, 0.05],
                             [-0.1, 0.1, 0.05], [-0.1, -0.1, 0.25]],
                            dtype=np.float32)
        tris = np.array([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3], dtype=np.int32)
        mesh = wp.Mesh(points=wp.array(authored, dtype=wp.vec3, device=cuda_device),
                       indices=wp.array(tris, dtype=wp.int32, device=cuda_device))
        bounds = dict(aabb_min=tuple(authored.min(axis=0)),
                      aabb_max=tuple(authored.max(axis=0)))

        solver.set_colliders([ColliderDesc(
            mesh=mesh, linear_velocity=(3.0, 0.0, 0.0), **bounds)])
        solver.step_frame()
        np.testing.assert_allclose(mesh.points.numpy(), authored, atol=1e-6)

        # The collider stops: the scene hands the same cached mesh back.
        solver.set_colliders([ColliderDesc(mesh=mesh, **bounds)])
        solver.step_frame()
        np.testing.assert_allclose(mesh.points.numpy(), authored, atol=1e-6)
    finally:
        solver.release()
