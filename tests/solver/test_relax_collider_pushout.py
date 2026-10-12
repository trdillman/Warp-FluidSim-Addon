"""B32: particle relaxation projects embedded particles out of colliders.

``k_particles_relax`` pushes a particle that ends up inside a collider back
out along the collider SDF gradient. Its central differences read one voxel
either side; at the domain boundary the outer sample has no voxel under it
and returns the 1e9 fallback, which used to swamp the normal and push the
particle *into* a floor collider. Sibling of REVIEW-017
(``collision_sdf_gradient``).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import kernels_fluid as KF
from fluidSim_warp_blender51.sim.fields import (
    SparseGrid,
    voxel_center_world,
    voxel_coord_of_flat,
)

VOXEL = 0.05


@wp.kernel
def _k_floor_sdf(sdf: wp.array(dtype=float), leaf_coords: wp.array(dtype=wp.vec3i),
                 origin: wp.vec3, voxel_size: float, floor_z: float):
    tid = wp.tid()
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    sdf[tid] = voxel_center_world(ijk, origin, voxel_size)[2] - floor_z


def _relax_one(device, start, floor_z):
    """Relax a lone particle against the floor collider ``z < floor_z``."""
    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4),
                      voxel_size=VOXEL, leaf_capacity=8, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    grid.finalize_topology()
    nv = grid.voxel_capacity
    sdf = wp.zeros(nv, dtype=float, device=device)
    wp.launch(_k_floor_sdf, dim=nv, device=device,
              inputs=[sdf, grid.leaf_coords, grid.origin, VOXEL, floor_z])
    pos = wp.array([start], dtype=wp.vec3, device=device)
    count = wp.array([1], dtype=wp.int32, device=device)
    counts = wp.zeros(nv, dtype=wp.int32, device=device)
    pvox = wp.zeros(1, dtype=wp.int32, device=device)
    arrival = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(KF.k_particle_to_voxel_count_and_update_index, dim=1, device=device,
              inputs=[pos, count, grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                      grid.origin, VOXEL], outputs=[counts, pvox, arrival])
    starts = wp.zeros(nv, dtype=wp.int32, device=device)
    KF.voxel_count_to_starts(counts, starts)
    unordered = wp.zeros(1, dtype=wp.int32, device=device)
    indices = wp.zeros(1, dtype=wp.int32, device=device)
    KF.scatter_particles_to_bins(pvox, arrival, count, 1, counts, starts, unordered, indices)
    out = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.launch(KF.k_particles_relax, dim=1, device=device,
              inputs=[pos, count, pvox, counts, starts, indices,
                      grid.leaf_table, grid.leaf_coords, grid.voxels_dim, grid.leaves_dim,
                      grid.origin, VOXEL, 0.5 * VOXEL, 0.5, 0,
                      sdf, wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.4, 0.4, 0.4)],
              outputs=[out])
    return out.numpy()[0]


def test_relax_pushes_floor_particle_up_at_the_domain_minimum(cuda_device):
    """Floor z < 0.03 straddles the domain's lower face; particle at z=0.02.

    The -z one-voxel sample lies below the domain (no voxel), so the old code
    differenced against 1e9, got a -z normal and moved the particle to
    z ~= 0.015 -- deeper inside the floor.

    The depth itself is the renormalised trilinear sample, which below the
    first voxel centre (z = 0.025) is that voxel's value, -0.005; the push is
    1.01 times that along +z.
    """
    floor_z = 0.03
    after = _relax_one(cuda_device, (0.2, 0.2, 0.02), floor_z)
    assert after[0] == pytest.approx(0.2, abs=1.0e-6)
    assert after[1] == pytest.approx(0.2, abs=1.0e-6)
    assert after[2] > 0.02, f"particle pushed deeper into the floor: z={after[2]}"
    assert after[2] == pytest.approx(0.02 + 1.01 * 0.005, abs=1.0e-6)


def test_relax_interior_pushout_is_unchanged(cuda_device):
    """Away from the domain edge the projection is the plain central difference."""
    floor_z = 0.23
    after = _relax_one(cuda_device, (0.2, 0.2, 0.22), floor_z)
    assert after[0] == pytest.approx(0.2, abs=1.0e-6)
    assert after[1] == pytest.approx(0.2, abs=1.0e-6)
    # phi = -0.01 along a unit +z normal, pushed 1.01 * 0.01
    assert after[2] == pytest.approx(0.22 + 0.0101, abs=1.0e-6)
