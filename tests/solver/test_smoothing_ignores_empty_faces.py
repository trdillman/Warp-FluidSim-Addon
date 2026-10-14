"""Explicit velocity smoothing must not average in empty (zero-weight) faces.

Regression for review items B07 (the JACOBI viscosity smoother,
``viscosity.k_viscosity_smooth``) and B31 (the Diffusion force,
``kernels_fluid.k_forces_diffusion``). Both averaged every allocated
neighbour face, including air faces that P2G leaves at zero velocity with
zero weight, so a uniformly moving liquid lost momentum at its free surface.
A uniform velocity field has nothing to smooth: every liquid face must keep
its velocity.
"""
from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

DEVICE = "cpu"
VELOCITY = -5.0


def _uniform_block():
    from fluidSim_warp_blender51.sim.fields import SparseGrid

    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
                      voxel_size=0.1, leaf_capacity=1, device=DEVICE)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    n = grid.voxel_capacity
    local = np.arange(n)
    ijk = np.stack([local & 7, (local >> 3) & 7, (local >> 6) & 7], axis=1)
    liquid = (ijk >= 2).all(axis=1) & (ijk < 6).all(axis=1)
    velocity = np.where(liquid, VELOCITY, 0.0).astype(np.float32)
    weight = np.where(liquid, 1.0, 0.0).astype(np.float32)
    arrays = [wp.array(velocity, dtype=float, device=DEVICE) for _ in range(3)]
    weights = [wp.array(weight, dtype=float, device=DEVICE) for _ in range(3)]
    outputs = [wp.zeros(n, dtype=float, device=DEVICE) for _ in range(3)]
    return grid, n, liquid, arrays, weights, outputs


def test_viscosity_smoother_keeps_a_uniformly_moving_block():
    from fluidSim_warp_blender51.sim.constants import VOXEL_FLUID
    from fluidSim_warp_blender51.sim.viscosity import k_viscosity_smooth

    grid, n, liquid, (u, v, w), (wu, wv, ww), (uo, vo, wo) = _uniform_block()
    voxel_type = wp.array(np.where(liquid, VOXEL_FLUID, 0).astype(np.int32),
                          device=DEVICE)
    nu = wp.full(n, 0.05, dtype=float, device=DEVICE)
    wp.launch(k_viscosity_smooth, dim=n, device=DEVICE,
              inputs=[u, v, w, wu, wv, ww, nu, voxel_type, grid.leaf_table,
                      grid.leaf_coords, grid.voxels_dim, grid.leaves_dim,
                      grid.num_leaves, 0.1, 1.0 / 96.0],
              outputs=[uo, vo, wo])
    for out in (uo, vo, wo):
        np.testing.assert_allclose(out.numpy()[liquid], VELOCITY, rtol=1e-6)


def test_diffusion_force_keeps_a_uniformly_moving_block():
    from fluidSim_warp_blender51.sim.kernels_fluid import k_forces_diffusion

    grid, n, liquid, (u, v, w), (wu, wv, ww), (uo, vo, wo) = _uniform_block()
    wp.launch(k_forces_diffusion, dim=n, device=DEVICE,
              inputs=[u, v, w, wu, wv, ww, grid.leaf_table, grid.leaf_coords,
                      grid.voxels_dim, grid.leaves_dim, grid.num_leaves, 0.5],
              outputs=[uo, vo, wo])
    for out in (uo, vo, wo):
        np.testing.assert_allclose(out.numpy()[liquid], VELOCITY, rtol=1e-6)


def test_smoothing_still_diffuses_a_velocity_difference():
    from fluidSim_warp_blender51.sim.kernels_fluid import k_forces_diffusion

    grid, n, liquid, (u, v, w), (wu, wv, ww), (uo, vo, wo) = _uniform_block()
    local = np.arange(n)
    values = u.numpy()
    spike = int(np.nonzero(liquid & ((local & 7) == 3) & (((local >> 3) & 7) == 3)
                           & ((local >> 6) == 3))[0][0])
    values[spike] = 0.0
    u.assign(values)
    wp.launch(k_forces_diffusion, dim=n, device=DEVICE,
              inputs=[u, v, w, wu, wv, ww, grid.leaf_table, grid.leaf_coords,
                      grid.voxels_dim, grid.leaves_dim, grid.num_leaves, 0.5],
              outputs=[uo, vo, wo])
    assert uo.numpy()[spike] == pytest.approx(0.5 * VELOCITY)
