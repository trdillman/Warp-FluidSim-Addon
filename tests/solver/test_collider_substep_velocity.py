"""B38: collider surface velocity at a sub-frame pose uses the moved centre.

``_update_moving_colliders`` re-poses a translating and rotating collider to
``centre + R(t) (x - centre) + v t``. The collision velocity written by
``k_rasterize_collider_material`` is ``v + w x (p - c)``; with ``c`` left at
the frame-start centre the rotating part gains a spurious ``w x (v t)``,
about 0.18 m/s at 6 rad/s, 1 m/s and t = 0.03 s.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import constants as C  # noqa: E402
from fluidSim_warp_blender51.sim.solver import (  # noqa: E402
    ColliderDesc,
    FluidSolver,
    SolverSettings,
)

H = 0.05
CENTRE = np.array([0.8, 0.8, 0.8])
HALF = 0.1
LINEAR = np.array([1.0, 0.0, 0.0])
ANGULAR = np.array([0.0, 0.0, 6.0])
ELAPSED = 0.03


def _box_mesh(device):
    corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)],
                       np.float32) * HALF + CENTRE.astype(np.float32)
    faces = np.array([
        [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],   # -z, +z
        [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],   # -y, +y
        [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],   # -x, +x
    ], np.int32)
    return wp.Mesh(points=wp.array(corners, dtype=wp.vec3, device=device),
                   indices=wp.array(faces.ravel(), dtype=wp.int32, device=device))


def test_rotating_translating_collider_velocity_uses_the_moved_centre(cuda_device):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.6, 1.6, 1.6),
        voxel_size=H, leaf_capacity=64, max_particles=64,
        # no domain-wall SDF, so every finite SDF voxel is the collider's
        domain_sdf_walls=False,
        whitewater_enabled=False, surface_enabled=False, device=cuda_device)
    solver = FluidSolver(settings)
    try:
        mesh = _box_mesh(cuda_device)
        solver.set_colliders([ColliderDesc(
            mesh=mesh, linear_velocity=tuple(LINEAR), angular_velocity=tuple(ANGULAR),
            center=tuple(CENTRE), aabb_min=tuple(CENTRE - HALF),
            aabb_max=tuple(CENTRE + HALF))])
        solver._rebuild_topology()
        solver._clear_grid_fields()
        solver._update_moving_colliders(ELAPSED)
        solver._apply_collisions()
        solver._restore_collider_poses()

        nl = int(solver.grid.num_leaves.numpy()[0])
        n = nl * C.LEAF_VOLUME
        sdf = solver.collision_sdf.data.numpy()[:n]
        vel = solver.collision_vel.numpy()[:n]
        lc = solver.grid.leaf_coords.numpy()[:nl]
        flat = np.arange(n)
        slot, local = flat // C.LEAF_VOLUME, flat % C.LEAF_VOLUME
        ijk = np.stack([lc[slot, 0] * 8 + (local & 7),
                        lc[slot, 1] * 8 + ((local >> 3) & 7),
                        lc[slot, 2] * 8 + ((local >> 6) & 7)], 1)
        p = (ijk + 0.5) * H
        hit = sdf < 1.0e5
        assert hit.sum() > 100
        moved_centre = CENTRE + LINEAR * ELAPSED
        expected = LINEAR + np.cross(ANGULAR, p[hit] - moved_centre)
        error = np.abs(vel[hit] - expected).max()
        assert error < 1.0e-4, f"collision velocity off by {error:.4f} m/s"
    finally:
        solver.release()
