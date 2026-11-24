"""Guided shell upres: the guide kernels on a known coarse liquid.

Run on CPU with ``FLUIDSIM_TEST_DEVICE=cpu``; on a GPU host they use CUDA.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import guided_upres as GU
from fluidSim_warp_blender51.sim.fields import (
    ScalarField,
    SparseGrid,
    voxel_coord_of_flat,
)

SIZE = (0.64, 0.64, 0.64)
COARSE_H = 0.04
FINE_H = 0.02
CENTRE = (0.32, 0.32, 0.32)
RADIUS = 0.16
VEL = (0.5, -0.25, 1.0)


@wp.kernel
def _k_sphere(sdf: wp.array(dtype=float), leaf_coords: wp.array(dtype=wp.vec3i),
              origin: wp.vec3, h: float, c: wp.vec3, r: float):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                             float(ijk[2]) + 0.5)
    sdf[tid] = wp.length(x - c) - r


@wp.kernel
def _k_centres(leaf_coords: wp.array(dtype=wp.vec3i), origin: wp.vec3,
               h: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    out[tid] = origin + h * wp.vec3(float(ijk[0]) + 0.5, float(ijk[1]) + 0.5,
                                    float(ijk[2]) + 0.5)


def _grid(device, h, lo=(0.08, 0.08, 0.08), hi=(0.56, 0.56, 0.56)):
    g = SparseGrid(origin=(0.0, 0.0, 0.0), size=SIZE, voxel_size=h,
                   leaf_capacity=512, device=device)
    g.begin_topology()
    g.mark_world_box(lo, hi)
    g.finalize_topology()
    return g


def _coarse_frame(device, time, velocity=VEL, radius=RADIUS):
    g = _grid(device, COARSE_H)
    sdf = ScalarField(g, name="coarse_sdf")
    wp.launch(_k_sphere, dim=g.voxel_capacity, device=device,
              inputs=[sdf.data, g.leaf_coords, g.origin, COARSE_H,
                      wp.vec3(*CENTRE), radius])
    n = g.voxel_capacity

    def full(value):
        return wp.full(n, float(value), dtype=float, device=device)

    vel = SimpleNamespace(u=full(velocity[0]), v=full(velocity[1]),
                          w=full(velocity[2]), wu=full(1.0), wv=full(1.0),
                          ww=full(1.0))
    return GU.GuideFrame(g, sdf, vel, time)


def _centres(grid, device):
    n = int(grid.active_voxel_count)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(_k_centres, dim=n, device=device,
              inputs=[grid.leaf_coords, grid.origin, FINE_H], outputs=[out])
    return out.numpy()


def test_phi_takes_the_coarse_liquid_eroded_by_the_shell(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=3)
    frame = _coarse_frame(dev, 0.0)
    guide.set_frames(frame, None)
    fine = _grid(dev, FINE_H)
    limit = 8.0 * FINE_H
    phi = wp.full(fine.voxel_capacity, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)

    n = int(fine.active_voxel_count)
    x = _centres(fine, dev)
    p = phi.numpy()[:n]
    d = np.linalg.norm(x - np.asarray(CENTRE), axis=1) - RADIUS
    shell = 3 * FINE_H
    # Deep coarse liquid is liquid in the fine solve...
    deep = d < -shell - COARSE_H
    assert deep.sum() > 100 and np.all(p[deep] < 0.0)
    # ...and the shell itself is left to the fine particles.
    band = d > -shell + COARSE_H
    assert np.all(p[band] >= 0.0)


def test_deep_fine_liquid_cannot_drift_outside_the_coarse_surface(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = _grid(dev, FINE_H)
    limit = 8.0 * FINE_H
    # Fine thinks everything is deep liquid.
    phi = wp.full(fine.voxel_capacity, -limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    n = int(fine.active_voxel_count)
    x = _centres(fine, dev)
    p = phi.numpy()[:n]
    d = np.linalg.norm(x - np.asarray(CENTRE), axis=1) - RADIUS
    far_out = d > 3 * 2 * FINE_H + COARSE_H
    assert far_out.sum() > 100
    assert np.all(p[far_out] > -2 * FINE_H)


def test_interior_velocity_follows_the_guide_and_blends_in_time(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    a = _coarse_frame(dev, 0.0, velocity=(0.0, 0.0, 0.0))
    b = _coarse_frame(dev, 1.0, velocity=(1.0, -2.0, 4.0))
    guide.set_frames(a, b)
    guide.set_time(0.25)
    assert guide.alpha == pytest.approx(0.25)
    fine = _grid(dev, FINE_H)
    n = fine.voxel_capacity
    limit = 8.0 * FINE_H
    phi = wp.full(n, limit, dtype=float, device=dev)
    guide.apply_phi(fine, phi, fine.active_voxel_count, limit, dev)
    nb = [wp.full(n, 9.0, dtype=float, device=dev) for _ in range(3)]
    guide.apply_velocity(fine, phi, *nb, fine.active_voxel_count, dev)
    m = int(fine.active_voxel_count)
    p = phi.numpy()[:m]
    deep = p < -2 * FINE_H
    shallow = p >= -2 * FINE_H
    assert deep.sum() > 50
    for arr, expected in zip(nb, (0.25, -0.5, 1.0)):
        values = arr.numpy()[:m]
        np.testing.assert_allclose(values[deep], expected, atol=1e-5)
        # Shell and air keep their own velocity.
        assert np.all(values[shallow] == 9.0)


def test_mark_leaves_covers_the_coarse_liquid(cuda_device):
    dev = cuda_device
    guide = GU.ShellGuide(shell_voxels=2)
    guide.set_frames(_coarse_frame(dev, 0.0), None)
    fine = SparseGrid(origin=(0.0, 0.0, 0.0), size=SIZE, voxel_size=FINE_H,
                      leaf_capacity=512, device=dev)
    fine.begin_topology()
    guide.mark_leaves(fine, dev)
    fine.finalize_topology()
    x = _centres(fine, dev)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    assert np.all(lo <= np.asarray(CENTRE) - RADIUS)
    assert np.all(hi >= np.asarray(CENTRE) + RADIUS)
