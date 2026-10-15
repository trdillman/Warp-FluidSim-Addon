"""4.0 lane R: the pressure face gather sees every voxel a body owns.

The gather walks a voxel box around each body's bounding sphere instead of
the whole grid (the crown_splash per-frame budget). The box must still cover
every collision voxel the body owns, including when it is clamped against
the domain walls: the gathered solid volume equals the host count of those
voxels.
"""

from __future__ import annotations

import numpy as np
import pytest

from _rigid_harness import H, box_collider, lattice

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (0.5, 0.5, 0.5)


@pytest.mark.parametrize("center", [
    (0.31, 0.24, 0.20),   # free in the tank
    (0.06, 0.45, 0.05),   # against three domain walls
])
def test_face_gather_volume_matches_owned_collision_voxels(cuda_device, center):
    import warp as wp
    from fluidSim_warp_blender51.sim.constants import VOXEL_COLLISION
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    points = lattice(DOMAIN_MIN, (0.5, 0.5, 0.25), 0.5 * H,
                     exclude=((center[0] - 0.08, center[1] - 0.08, center[2] - 0.08),
                              (center[0] + 0.08, center[1] + 0.08, center[2] + 0.08)))
    solver = FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=1024, max_particles=len(points) + 8192,
        surface_enabled=False, whitewater_enabled=False, device=cuda_device))
    try:
        solver.inject_particles(points)
        body = box_collider(
            wp, cuda_device, center, (0.07, 0.05, 0.04), dynamic=True,
            object_uid="block", density=600.0, drag=1.0)
        solver.set_colliders([body])
        solver.step_frame()
        rank = solver.rigid.attached[0][0]
        nv = int(solver._nvox)
        owner = solver._collision_owner.numpy()[:nv]
        vtype = solver.voxel_type.data.numpy()[:nv]
        owned = int(np.count_nonzero((owner == rank) & ((vtype & VOXEL_COLLISION) != 0)))
        row = solver.rigid.gather(solver, 1.0 / 240.0)[0]
    finally:
        solver.release()
    assert owned > 0
    h = float(np.float32(H))  # the kernel sees the float32 voxel size
    assert row[12] == pytest.approx(owned * h ** 3, rel=1.0e-9)
