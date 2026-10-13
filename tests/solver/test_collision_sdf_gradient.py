"""REVIEW-017: collider normals at the edge of the sparse topology."""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.fields import (  # noqa: E402
    SparseGrid,
    voxel_center_world,
    voxel_coord_of_flat,
)
from fluidSim_warp_blender51.sim.kernels_fluid import collision_sdf_gradient  # noqa: E402

VOXEL = 0.05
PLANE_Z = 0.2


@wp.kernel
def _k_plane_sdf(sdf: wp.array(dtype=float),
                 leaf_coords: wp.array(dtype=wp.vec3i),
                 origin: wp.vec3, voxel_size: float, plane_z: float):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    sdf[tid] = voxel_center_world(ijk, origin, voxel_size)[2] - plane_z


@wp.kernel
def _k_gradient(points: wp.array(dtype=wp.vec3), sdf: wp.array(dtype=float),
                leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
                leaves_dim: wp.vec3i, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = collision_sdf_gradient(
        points[tid], sdf, leaf_table, voxels_dim, leaves_dim)


def _gradients(device, points):
    # 16^3 voxel domain with only the first 8^3 leaf allocated, so x > 8.5
    # is inside the domain but has no voxels under it.
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
                      voxel_size=VOXEL, leaf_capacity=8, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == 1
    sdf = wp.zeros(grid.voxel_capacity, dtype=float, device=device)
    wp.launch(_k_plane_sdf, dim=grid.voxel_capacity, device=device,
              inputs=[sdf, grid.leaf_coords, grid.origin, VOXEL, PLANE_Z])
    pts = wp.array(points, dtype=wp.vec3, device=device)
    out = wp.zeros(len(points), dtype=wp.vec3, device=device)
    wp.launch(_k_gradient, dim=len(points), device=device,
              inputs=[pts, sdf, grid.leaf_table, grid.voxels_dim,
                      grid.leaves_dim, out])
    return out.numpy()


def test_gradient_ignores_samples_outside_the_sparse_topology(cuda_device):
    """A +x sample past the last leaf used to read the 1e9 fallback.

    The collider is the plane z = 0.2, so the true normal is +z everywhere.
    At x = 8.2 voxels the centre and -x samples still see allocated voxels
    but the +x sample (8.7) sees none.
    """
    interior, edge, far = _gradients(
        cuda_device, [(4.0, 4.0, 3.0), (8.2, 4.0, 3.0), (12.0, 4.0, 3.0)])

    # one-voxel central difference of a unit-gradient SDF in world units
    assert interior == pytest.approx((0.0, 0.0, VOXEL), abs=1.0e-6)
    # edge: the x component must come from the valid side only (plane has
    # no x dependence), never from the fallback
    assert edge == pytest.approx((0.0, 0.0, VOXEL), abs=1.0e-6)
    # no voxels anywhere near: no direction rather than a fallback diagonal
    assert far == pytest.approx((0.0, 0.0, 0.0))


def test_whitewater_collision_uses_the_shared_gradient():
    """The whitewater kernel had its own forward differences off the fallback."""
    from fluidSim_warp_blender51.sim import whitewater

    source = inspect.getsource(whitewater.k_advect_whitewater.func)
    assert "collision_sdf_gradient(" in source
    assert "1.0e6) - csdf" not in source
