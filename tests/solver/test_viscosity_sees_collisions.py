"""The viscosity solve must see mesh-collider voxels as collision.

Regression for review item B08: the grid clear zeroes ``voxel_type`` every
substep and ``VOXEL_COLLISION`` was re-derived from ``collision_sdf`` only in
``_collision_faces``, after the viscosity solve. Viscosity's collision guards
therefore never fired for mesh colliders or domain walls.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_viscosity_solve_receives_mesh_collider_voxels(cuda_device):
    wp = pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.constants import VOXEL_COLLISION
    from fluidSim_warp_blender51.sim.solver import (
        ColliderDesc, FluidSolver, SolverSettings,
    )

    settings = SolverSettings(
        domain_min=(-0.4, -0.4, 0.0), domain_max=(0.4, 0.4, 0.8),
        voxel_size=0.05, leaf_capacity=64, max_particles=4096,
        max_whitewater=1024, max_substeps=1, viscosity_enabled=True,
        viscosity_constant=0.05, device=cuda_device)
    solver = FluidSolver(settings)
    try:
        rng = np.random.default_rng(2)
        solver.inject_particles(rng.uniform(
            (-0.2, -0.2, 0.1), (0.2, 0.2, 0.4), (2000, 3)).astype(np.float32))
        # A closed box collider straddling the liquid.
        lo, hi = np.array((-0.1, -0.1, 0.05)), np.array((0.1, 0.1, 0.25))
        corners = np.array([[x, y, z] for z in (lo[2], hi[2])
                            for y in (lo[1], hi[1]) for x in (lo[0], hi[0])],
                           dtype=np.float32)
        faces = np.array([0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
                          2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5],
                         dtype=np.int32)
        mesh = wp.Mesh(points=wp.array(corners, dtype=wp.vec3, device=cuda_device),
                       indices=wp.array(faces, dtype=wp.int32, device=cuda_device))
        solver.set_colliders([ColliderDesc(mesh=mesh, aabb_min=tuple(lo),
                                           aabb_max=tuple(hi))])
        seen = []
        original = solver.viscosity_solver.solve

        def solve(vel, field, voxel_type, *args, **kwargs):
            wp.synchronize_device(cuda_device)
            seen.append(voxel_type.numpy()[:solver._nvox].copy())
            return original(vel, field, voxel_type, *args, **kwargs)

        solver.viscosity_solver.solve = solve
        solver.step_frame()

        assert seen, "viscosity solve did not run"
        assert int(np.count_nonzero(seen[0] & VOXEL_COLLISION)) > 0
    finally:
        solver.release()
