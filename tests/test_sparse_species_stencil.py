"""Flat sparse-leaf stencil and conditional halo admission contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from fluidSim_warp_blender51.sim.capacity_planner import (
    CapacityVerdict,
    DemandQuality,
    plan_initialization,
)
from fluidSim_warp_blender51.sim.fields import (
    LEAF_VOLUME,
    neighbor_flat_index,
    positive_face_neighbor,
)


@dataclass(frozen=True)
class Footprint:
    stable_id: str
    particle_demand: int
    particle_quality: DemandQuality
    leaf_coords: tuple[tuple[int, int, int], ...]


@wp.kernel
def _query_neighbors(
    flats: wp.array(dtype=wp.int32),
    axes: wp.array(dtype=wp.int32),
    directions: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    neighbors: wp.array(dtype=wp.int32),
    positive_neighbors: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    neighbors[tid] = neighbor_flat_index(
        flats[tid], axes[tid], directions[tid], leaf_table, leaf_coords,
        voxels_dim, leaves_dim)
    positive_neighbors[tid] = positive_face_neighbor(
        flats[tid], axes[tid], leaf_table, leaf_coords,
        voxels_dim, leaves_dim)


def _flat(slot: int, x: int, y: int, z: int) -> int:
    return slot * LEAF_VOLUME + x + 8 * (y + 8 * z)


def _cpu_neighbor(flat, axis, direction, table, coords, voxel_dims, leaf_dims):
    slot, local = divmod(flat, LEAF_VOLUME)
    x = local & 7
    y = (local >> 3) & 7
    z = (local >> 6) & 7
    voxel = [coords[slot][0] * 8 + x,
             coords[slot][1] * 8 + y,
             coords[slot][2] * 8 + z]
    voxel[axis] += direction
    if any(value < 0 or value >= voxel_dims[i]
           for i, value in enumerate(voxel)):
        return -1
    leaf = tuple(value >> 3 for value in voxel)
    table_index = leaf[0] + leaf_dims[0] * (
        leaf[1] + leaf_dims[1] * leaf[2])
    neighbor_slot = table[table_index]
    if neighbor_slot < 0:
        return -1
    local = (voxel[0] & 7) + 8 * ((voxel[1] & 7) + 8 * (voxel[2] & 7))
    return neighbor_slot * LEAF_VOLUME + local


def test_warp_neighbors_match_cpu_oracle_across_interior_faces_edges_and_corners():
    leaf_dims = (3, 3, 3)
    coords = [(x, y, z) for z in range(3) for y in range(3) for x in range(3)]
    table = np.arange(27, dtype=np.int32)
    cases = []
    center_slot = 1 + 3 * (1 + 3 * 1)
    for local in ((3, 4, 5), (7, 2, 2), (2, 7, 2), (2, 2, 7),
                  (7, 7, 2), (7, 7, 7), (0, 0, 0)):
        for axis in range(3):
            for direction in (-1, 1):
                cases.append((_flat(center_slot, *local), axis, direction))

    flats = np.asarray([item[0] for item in cases], dtype=np.int32)
    axes = np.asarray([item[1] for item in cases], dtype=np.int32)
    directions = np.asarray([item[2] for item in cases], dtype=np.int32)
    neighbors = wp.zeros(len(cases), dtype=wp.int32, device="cpu")
    positive = wp.zeros(len(cases), dtype=wp.int32, device="cpu")
    wp.launch(
        _query_neighbors, dim=len(cases),
        inputs=[
            wp.array(flats, dtype=wp.int32, device="cpu"),
            wp.array(axes, dtype=wp.int32, device="cpu"),
            wp.array(directions, dtype=wp.int32, device="cpu"),
            wp.array(table, dtype=wp.int32, device="cpu"),
            wp.array(np.asarray(coords, dtype=np.int32), dtype=wp.vec3i, device="cpu"),
            wp.vec3i(24, 24, 24), wp.vec3i(*leaf_dims),
        ],
        outputs=[neighbors, positive], device="cpu",
    )

    expected = np.asarray([
        _cpu_neighbor(flat, axis, direction, table, coords, (24, 24, 24), leaf_dims)
        for flat, axis, direction in cases
    ], dtype=np.int32)
    expected_positive = np.asarray([
        _cpu_neighbor(flat, axis, 1, table, coords, (24, 24, 24), leaf_dims)
        for flat, axis, _direction in cases
    ], dtype=np.int32)
    np.testing.assert_array_equal(neighbors.numpy(), expected)
    np.testing.assert_array_equal(positive.numpy(), expected_positive)


def test_missing_neighbor_is_impermeable_and_positive_face_has_one_owner():
    # Slots are deliberately not leaf-table order, proving lookup uses the table.
    coords = np.asarray(((1, 0, 0), (0, 0, 0)), dtype=np.int32)
    table = np.asarray((1, 0, -1), dtype=np.int32)
    cases = [(_flat(1, 7, 3, 3), 0, 1), (_flat(0, 0, 3, 3), 0, -1),
             (_flat(0, 7, 3, 3), 0, 1)]
    flats = np.asarray([item[0] for item in cases], dtype=np.int32)
    axes = np.asarray([item[1] for item in cases], dtype=np.int32)
    directions = np.asarray([item[2] for item in cases], dtype=np.int32)
    neighbors = wp.zeros(3, dtype=wp.int32, device="cpu")
    positive = wp.zeros(3, dtype=wp.int32, device="cpu")
    wp.launch(
        _query_neighbors, dim=3,
        inputs=[wp.array(flats, dtype=wp.int32, device="cpu"),
                wp.array(axes, dtype=wp.int32, device="cpu"),
                wp.array(directions, dtype=wp.int32, device="cpu"),
                wp.array(table, dtype=wp.int32, device="cpu"),
                wp.array(coords, dtype=wp.vec3i, device="cpu"),
                wp.vec3i(24, 8, 8), wp.vec3i(3, 1, 1)],
        outputs=[neighbors, positive], device="cpu",
    )
    result = neighbors.numpy()
    assert result[0] == _flat(0, 0, 3, 3)
    assert result[1] == _flat(1, 7, 3, 3)
    assert result[2] == -1
    # Only the lower voxel owns the admitted internal face; the reverse lookup
    # exists but is never returned as positive ownership.
    assert positive.numpy()[0] == _flat(0, 0, 3, 3)
    assert positive.numpy()[1] != _flat(1, 7, 3, 3)


def _settings(**overrides):
    values = {
        "domain_min": (0.0, 0.0, 0.0),
        "domain_max": (24.0, 8.0, 8.0),
        "voxel_size": 1.0,
        "max_particles": 100,
        "leaf_capacity": 3,
        "particles_per_voxel": 8.0,
        "topology_expand": 0,
        "max_whitewater": 0,
        "surface_enabled": False,
        "species_enabled": False,
    }
    values.update(overrides)
    return values


def _resources():
    return {"gpu_total_bytes": 16 * 1024**3, "gpu_free_bytes": 14 * 1024**3}


def test_species_admission_requires_one_leaf_halo_and_disabled_topology_is_unchanged():
    source = (Footprint("source", 1, DemandQuality.EXACT, ((1, 0, 0),)),)
    disabled = plan_initialization(_settings(leaf_capacity=1), source, (), (), _resources())
    enabled = plan_initialization(
        _settings(leaf_capacity=1, species_enabled=True), source, (), (), _resources())

    assert disabled.verdict is CapacityVerdict.ADMITTED
    assert disabled.required_leaf_coords == ((1, 0, 0),)
    assert enabled.verdict is CapacityVerdict.BLOCKED
    assert enabled.required_leaf_coords == ((0, 0, 0), (1, 0, 0), (2, 0, 0))
    assert any(item.code == "LEAF_CAPACITY" and item.required == 3
               for item in enabled.blockers)
