"""P18: force-field proxies skip faces with no weight before evaluating fields.

``k_forces_proxy_fields`` used to evaluate every field for all three faces of
every active voxel and only then check the face weight. The result for valid
faces must be unchanged bit for bit; this pins it against the old body.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import kernels_fluid as KF  # noqa: E402
from fluidSim_warp_blender51.sim.fields import SparseGrid, voxel_coord_of_flat  # noqa: E402
from fluidSim_warp_blender51.sim.solver import ForceFieldDesc  # noqa: E402

H = 0.05


@wp.kernel
def _k_reference(
    fields: wp.array(dtype=KF.ForceFieldProxy), num_fields: int,
    leaf_coords: wp.array(dtype=wp.vec3i), num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3, voxel_size: float,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    time: float, dt: float,
):
    # The pre-P18 body: evaluate every face, then gate on the weight.
    tid = wp.tid()
    if tid >= num_leaves[0] * 512:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    velocity = wp.vec3(u[tid], v[tid], w[tid])
    for axis in range(3):
        point = origin + wp.vec3(
            (float(ijk[0]) + wp.where(axis == 0, 0.0, 0.5)) * voxel_size,
            (float(ijk[1]) + wp.where(axis == 1, 0.0, 0.5)) * voxel_size,
            (float(ijk[2]) + wp.where(axis == 2, 0.0, 0.5)) * voxel_size,
        )
        acceleration = wp.vec3(0.0)
        for field_index in range(num_fields):
            acceleration += KF.force_field_proxy_acceleration(
                fields[field_index], point, velocity, time)
        if axis == 0 and wu[tid] > 0.0:
            u[tid] = u[tid] + acceleration[0] * dt
        elif axis == 1 and wv[tid] > 0.0:
            v[tid] = v[tid] + acceleration[1] * dt
        elif axis == 2 and ww[tid] > 0.0:
            w[tid] = w[tid] + acceleration[2] * dt


def test_weight_gated_proxy_forces_match_the_ungated_body(cuda_device):
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8), voxel_size=H,
                      leaf_capacity=8, device=cuda_device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.799, 0.799, 0.799))
    grid.finalize_topology()
    n = grid.voxel_capacity
    rng = np.random.default_rng(7)
    vel = rng.normal(0.0, 1.0, size=(3, n)).astype(np.float32)
    weights = rng.uniform(0.0, 2.0, size=(3, n)).astype(np.float32)
    weights[rng.uniform(size=(3, n)) < 0.6] = 0.0
    weights[:, :50] = 0.0  # whole voxels of air
    fields = wp.array([
        ForceFieldDesc(force_type=11, strength=3.0, size=0.3, noise=1.0,
                       center=(0.4, 0.4, 0.4), seed=3).to_struct(),
        ForceFieldDesc(force_type=3, shape=0, strength=2.0, center=(0.4, 0.4, 0.4),
                       direction=(0.0, 0.0, 1.0), max_distance=1.0).to_struct(),
        ForceFieldDesc(force_type=12, linear_drag=0.4, quadratic_drag=0.1).to_struct(),
    ], dtype=KF.ForceFieldProxy, device=cuda_device)
    results = []
    for kernel in (_k_reference, KF.k_forces_proxy_fields):
        u, v, w = (wp.array(vel[i], dtype=float, device=cuda_device) for i in range(3))
        wu, wv, ww = (wp.array(weights[i], dtype=float, device=cuda_device)
                      for i in range(3))
        wp.launch(kernel, dim=n, device=cuda_device,
                  inputs=[fields, 3, grid.leaf_coords, grid.num_leaves, grid.origin, H,
                          u, v, w, wu, wv, ww, 0.37, 1.0 / 96.0])
        results.append(np.stack([u.numpy(), v.numpy(), w.numpy()]))
    reference, gated = results
    assert not np.array_equal(reference, vel), "fields must change valid faces"
    assert np.array_equal(gated, reference)
