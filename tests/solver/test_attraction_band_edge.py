"""B33: collider attraction at the edge of the rasterized collider band.

``k_reset_collision_grid`` fills the collider SDF with a 1e6 far value and
each collider writes finite distances only inside its query band. Attraction
differentiates neighbouring voxels; at the band edge one neighbour is the far
value, which used to produce a ~1e6/(2h) gradient and a velocity spike of
order 1e5 m/s with open boundaries (no domain-wall SDF underneath).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import kernels_fluid as KF  # noqa: E402
from fluidSim_warp_blender51.sim.fields import SparseGrid  # noqa: E402

H = 0.05
FAR = 1.0e6  # solver._clear_grid_fields far value


def _voxel_centres(grid):
    nl = int(grid.num_leaves.numpy()[0])
    lc = grid.leaf_coords.numpy()[:nl]
    flat = np.arange(nl * 512)
    slot, local = flat // 512, flat % 512
    ijk = np.stack([lc[slot, 0] * 8 + (local & 7),
                    lc[slot, 1] * 8 + ((local >> 3) & 7),
                    lc[slot, 2] * 8 + ((local >> 6) & 7)], 1)
    return (ijk + 0.5) * H


def _attract(device, sdf_of_centres):
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8), voxel_size=H,
                      leaf_capacity=8, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.799, 0.799, 0.799))
    grid.finalize_topology()
    pc = _voxel_centres(grid)
    n = pc.shape[0]
    cap = grid.voxel_capacity
    sdf = np.zeros(cap, np.float32)
    sdf[:n] = sdf_of_centres(pc)
    a_sdf = wp.array(sdf, dtype=float, device=device)
    u = wp.zeros(cap, dtype=float, device=device)
    v = wp.zeros(cap, dtype=float, device=device)
    w = wp.zeros(cap, dtype=float, device=device)
    ones = wp.array(np.ones(cap, np.float32), dtype=float, device=device)
    dt = 0.01
    wp.launch(KF.k_attract_sdf_gradient_direction, dim=cap, device=device,
              inputs=[a_sdf, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                      grid.leaves_dim, grid.num_leaves, H, 1.0, 1.0, dt,
                      u, v, w, ones, ones, ones])
    vel = np.stack([u.numpy()[:n], v.numpy()[:n], w.numpy()[:n]], 1)
    return pc, vel, dt


def _sphere_band(pc, band):
    d = np.linalg.norm(pc - 0.4, axis=1) - 0.05
    out = np.full(pc.shape[0], FAR, np.float32)
    inside = np.abs(d) <= band
    out[inside] = d[inside]
    return out, d


def test_attraction_ignores_far_field_neighbours_at_the_band_edge(cuda_device):
    band = 0.3
    pc, vel, dt = _attract(cuda_device, lambda p: _sphere_band(p, band)[0])
    _, d = _sphere_band(pc, band)
    speed = np.linalg.norm(vel, axis=1)
    # a unit-gradient SDF gives |dv| ~= strength * dt per axis component sum;
    # sqrt(3) * dt bounds any central or one-sided difference of it
    assert speed.max() <= np.sqrt(3.0) * dt * 1.01, (
        f"band-edge spike {speed.max():.4g} m/s at d={d[speed.argmax()]:.3f}")
    # inside the band the pull is unchanged: toward the sphere, magnitude dt
    interior = (d > 2 * H) & (d < band - 2 * H)
    assert np.allclose(speed[interior], dt, rtol=0.05)


def test_attraction_with_walls_is_unchanged_inside_the_band(cuda_device):
    """Closed-wall composition has no far values; output must not change."""
    band = 0.3

    def walled(pc):
        sdf, _ = _sphere_band(pc, band)
        outside = np.max(np.maximum(-pc, pc - 0.8), axis=1)
        return np.minimum(sdf, -outside).astype(np.float32)

    pc, vel, dt = _attract(cuda_device, walled)
    # reference: central difference over the full grid, a missing neighbour
    # (outside the domain) replaced by the centre value
    sdf = walled(pc)
    ijk = np.rint(pc / H - 0.5).astype(int)
    lookup = {tuple(c): i for i, c in enumerate(ijk)}
    expected = np.zeros_like(vel)
    for i, c in enumerate(ijk):
        if sdf[i] > 1.0:
            continue
        for axis in range(3):
            e = np.zeros(3, int)
            e[axis] = 1
            hi = lookup.get(tuple(c + e))
            lo = lookup.get(tuple(c - e))
            s_hi = sdf[hi] if hi is not None else sdf[i]
            s_lo = sdf[lo] if lo is not None else sdf[i]
            expected[i, axis] = -(s_hi - s_lo) / (2.0 * H) * dt
    np.testing.assert_allclose(vel, expected, rtol=1.0e-5, atol=1.0e-9)
