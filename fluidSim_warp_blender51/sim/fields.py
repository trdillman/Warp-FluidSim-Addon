"""Sparse tiled leaf-grid field storage for the FluidSim Warp port.

The original OpenCL solver stores fields in 8x8x8 leaf tiles addressed through
a leaf table, with leaf-local voxel addressing and global start-vector
reconstruction. This module reproduces that model with Warp arrays:

- ``leaf_table``: dense leaf-pointer table, leaf coord -> slot or -1
- ``leaf_coords``: slot -> leaf coordinate (start-vector reconstruction)
- field data:      flat arrays of ``capacity * 512`` entries, slot-major

No bpy imports allowed in this module.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from .fastlaunch import launch as fast_launch

wp.set_module_options({"enable_backward": False})

from .constants import (
    LEAF_DIM,
    LEAF_MASK,
    LEAF_SHIFT,
    LEAF_VOLUME,
    VOXEL_COLLISION,
)
from .quality_contract import AuthoredQualitySettings, resolve_quality

# Warp address helpers (shared by every kernel module)


@wp.func
def leaf_linear_index(lc: wp.vec3i, leaves_dim: wp.vec3i) -> wp.int32:
    """Linear index of a leaf coordinate in the dense leaf table (x fastest)."""
    return lc[0] + leaves_dim[0] * (lc[1] + leaves_dim[1] * lc[2])


@wp.func
def leaf_coord_of_voxel(ijk: wp.vec3i) -> wp.vec3i:
    return wp.vec3i(ijk[0] >> LEAF_SHIFT, ijk[1] >> LEAF_SHIFT, ijk[2] >> LEAF_SHIFT)


@wp.func
def leaf_local_of_voxel(ijk: wp.vec3i) -> wp.int32:
    """Leaf-local voxel offset (x fastest) in [0, 512)."""
    return (
        (ijk[0] & LEAF_MASK)
        + ((ijk[1] & LEAF_MASK) << LEAF_SHIFT)
        + ((ijk[2] & LEAF_MASK) << (2 * LEAF_SHIFT))
    )


@wp.func
def voxel_in_domain(ijk: wp.vec3i, voxels_dim: wp.vec3i) -> wp.bool:
    return (
        ijk[0] >= 0
        and ijk[1] >= 0
        and ijk[2] >= 0
        and ijk[0] < voxels_dim[0]
        and ijk[1] < voxels_dim[1]
        and ijk[2] < voxels_dim[2]
    )


@wp.func
def voxel_flat_index(
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
) -> wp.int32:
    """Global voxel coord -> flat field index, or -1 if unallocated/out of domain."""
    if not voxel_in_domain(ijk, voxels_dim):
        return wp.int32(-1)
    lc = leaf_coord_of_voxel(ijk)
    slot = leaf_table[leaf_linear_index(lc, leaves_dim)]
    if slot < 0:
        return wp.int32(-1)
    return slot * LEAF_VOLUME + leaf_local_of_voxel(ijk)


@wp.func
def voxel_coord_of_flat(
    leaf_coords: wp.array(dtype=wp.vec3i),
    flat: wp.int32,
) -> wp.vec3i:
    """Flat field index -> global voxel coord (start-vector reconstruction)."""
    slot = flat // LEAF_VOLUME
    local = flat - slot * LEAF_VOLUME
    lc = leaf_coords[slot]
    li = local & LEAF_MASK
    lj = (local >> LEAF_SHIFT) & LEAF_MASK
    lk = (local >> (2 * LEAF_SHIFT)) & LEAF_MASK
    return wp.vec3i(
        (lc[0] << LEAF_SHIFT) + li,
        (lc[1] << LEAF_SHIFT) + lj,
        (lc[2] << LEAF_SHIFT) + lk,
    )


@wp.func
def voxel_center_world(ijk: wp.vec3i, origin: wp.vec3, voxel_size: float) -> wp.vec3:
    return origin + wp.vec3(
        (float(ijk[0]) + 0.5) * voxel_size,
        (float(ijk[1]) + 0.5) * voxel_size,
        (float(ijk[2]) + 0.5) * voxel_size,
    )


@wp.func
def world_to_voxel(p: wp.vec3, origin: wp.vec3, voxel_size: float) -> wp.vec3:
    """World position -> continuous voxel-space coordinate."""
    inv = 1.0 / voxel_size
    return wp.vec3((p[0] - origin[0]) * inv, (p[1] - origin[1]) * inv, (p[2] - origin[2]) * inv)


@wp.func
def voxel_floor(p: wp.vec3) -> wp.vec3i:
    return wp.vec3i(int(wp.floor(p[0])), int(wp.floor(p[1])), int(wp.floor(p[2])))


# Scalar field sampling.


@wp.func
def field_read_f(
    data: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    default: float,
) -> float:
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return default
    return data[flat]


@wp.func
def field_read_i(
    data: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    default: wp.int32,
) -> wp.int32:
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return default
    return data[flat]


@wp.func
def sample_scalar_trilinear(
    data: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    vp: wp.vec3,
    default: float,
) -> float:
    """Trilinear sample of a voxel-centred scalar field at voxel-space pos."""
    q = vp - wp.vec3(0.5, 0.5, 0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    acc = float(0.0)
    wsum = float(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                w = wx * wy * wz
                if flat >= 0:
                    acc += w * data[flat]
                    wsum += w
    if wsum > 1.0e-8:
        return acc / wsum
    return default


@wp.func
def sample_vec3_trilinear(
    data: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    vp: wp.vec3,
    default: wp.vec3,
) -> wp.vec3:
    """Trilinear sample of a voxel-centred vector field."""
    q = vp - wp.vec3(0.5, 0.5, 0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    acc = wp.vec3(0.0)
    wsum = float(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                weight = wx * wy * wz
                if flat >= 0:
                    acc += weight * data[flat]
                    wsum += weight
    if wsum > 1.0e-8:
        return acc / wsum
    return default


# Staggered MAC velocity sampling.
# Component c face samples live at voxel lower faces:
#   u stored at (i-1/2, j, k) -> voxel-space sample point offset (0, .5, .5)


@wp.func
def sample_mac_component(
    data: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    vp: wp.vec3,
    axis: wp.int32,
) -> float:
    """Sample one staggered velocity component; unallocated faces are skipped
    and weights renormalized. ``weight`` marks valid faces (>0)."""
    off = wp.vec3(0.5, 0.5, 0.5)
    if axis == 0:
        off = wp.vec3(0.0, 0.5, 0.5)
    elif axis == 1:
        off = wp.vec3(0.5, 0.0, 0.5)
    else:
        off = wp.vec3(0.5, 0.5, 0.0)
    q = vp - off
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    acc = float(0.0)
    wsum = float(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                w = wx * wy * wz
                if flat >= 0 and weight[flat] > 0.0:
                    acc += w * data[flat]
                    wsum += w
    if wsum > 1.0e-8:
        return acc / wsum
    return 0.0


@wp.func
def sample_mac_velocity(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    vp: wp.vec3,
) -> wp.vec3:
    return wp.vec3(
        sample_mac_component(u, wu, leaf_table, voxels_dim, leaves_dim, vp, 0),
        sample_mac_component(v, wv, leaf_table, voxels_dim, leaves_dim, vp, 1),
        sample_mac_component(w, ww, leaf_table, voxels_dim, leaves_dim, vp, 2),
    )


@wp.func
def is_collision_voxel(vt: wp.int32) -> wp.bool:
    return (vt & VOXEL_COLLISION) != 0


# Topology kernels


@wp.kernel
def k_clear_leaf_topology(
    leaf_table: wp.array(dtype=wp.int32),
    leaf_flags: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    leaf_table[tid] = -1
    leaf_flags[tid] = 0


@wp.kernel
def k_mark_leaves_from_points(
    points: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    leaf_flags: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(points[tid], origin, voxel_size)
    ijk = voxel_floor(vp)
    if not voxel_in_domain(ijk, voxels_dim):
        return
    lc = leaf_coord_of_voxel(ijk)
    leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1


@wp.kernel
def k_mark_leaves_box(
    lo: wp.vec3i,
    hi: wp.vec3i,
    leaves_dim: wp.vec3i,
    leaf_flags: wp.array(dtype=wp.int32),
):
    """Mark all leaves overlapping a voxel-space AABB [lo, hi] (inclusive)."""
    tid = wp.tid()
    llo = wp.vec3i(lo[0] >> LEAF_SHIFT, lo[1] >> LEAF_SHIFT, lo[2] >> LEAF_SHIFT)
    lhi = wp.vec3i(hi[0] >> LEAF_SHIFT, hi[1] >> LEAF_SHIFT, hi[2] >> LEAF_SHIFT)
    nx = lhi[0] - llo[0] + 1
    ny = lhi[1] - llo[1] + 1
    nz = lhi[2] - llo[2] + 1
    if tid >= nx * ny * nz:
        return
    x = tid % nx
    y = (tid // nx) % ny
    z = tid // (nx * ny)
    lc = wp.vec3i(llo[0] + x, llo[1] + y, llo[2] + z)
    if lc[0] < 0 or lc[1] < 0 or lc[2] < 0:
        return
    if lc[0] >= leaves_dim[0] or lc[1] >= leaves_dim[1] or lc[2] >= leaves_dim[2]:
        return
    leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1


@wp.kernel
def k_expand_leaf_flags(
    leaf_flags_in: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    leaf_flags_out: wp.array(dtype=wp.int32),
):
    """coreGridExpand analogue at leaf granularity: dilate flags by one leaf."""
    tid = wp.tid()
    nx = leaves_dim[0]
    ny = leaves_dim[1]
    x = tid % nx
    y = (tid // nx) % ny
    z = tid // (nx * ny)
    flagged = wp.int32(0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                xx = x + dx
                yy = y + dy
                zz = z + dz
                if xx < 0 or yy < 0 or zz < 0:
                    continue
                if xx >= nx or yy >= ny or zz >= leaves_dim[2]:
                    continue
                if leaf_flags_in[xx + nx * (yy + ny * zz)] != 0:
                    flagged = 1
    if flagged != 0:
        leaf_flags_out[tid] = 1


@wp.kernel
def k_mark_unallocated_leaves(
    leaf_flags: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    pending: wp.array(dtype=wp.int32),
):
    """Predicate feeding the allocation scan: flagged but not yet allocated."""
    tid = wp.tid()
    if leaf_flags[tid] != 0 and leaf_table[tid] < 0:
        pending[tid] = 1
    else:
        pending[tid] = 0


@wp.kernel
def k_allocate_leaves(
    pending: wp.array(dtype=wp.int32),
    pending_rank: wp.array(dtype=wp.int32),
    base: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    capacity: wp.int32,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    overflow: wp.array(dtype=wp.int32),
):
    """Assign leaf slots in linear leaf-coordinate order.

    Slot order is the exclusive prefix rank of the allocation predicate, not
    the arrival order of an atomic cursor. Every flat voxel index in the solver
    is ``leaf_slot * LEAF_VOLUME + offset``, so an order-dependent slot map
    permutes the whole grid index space between otherwise identical runs and
    makes every downstream accumulation, scatter and reduction irreproducible.
    ``num_leaves`` is published with ``atomic_max``, which is order-invariant,
    and is still raised past ``capacity`` so the overflow readback keeps
    reporting the full required count.
    """
    tid = wp.tid()
    if pending[tid] == 0:
        return
    slot = base[0] + pending_rank[tid]
    wp.atomic_max(num_leaves, 0, slot + 1)
    if slot >= capacity:
        overflow[0] = 1
        return
    leaf_table[tid] = slot
    nx = leaves_dim[0]
    ny = leaves_dim[1]
    leaf_coords[slot] = wp.vec3i(tid % nx, (tid // nx) % ny, tid // (nx * ny))


# Python-side containers


class TopologyCapacityError(RuntimeError):
    """Recoverable sparse-topology refusal that leaves the live grid intact."""

    def __init__(self, required: int, capacity: int):
        self.required = int(required)
        self.capacity = int(capacity)
        super().__init__(
            f"FluidSim: leaf capacity exceeded ({self.required} > {self.capacity}); "
            "increase the domain leaf allocation.")


def _world_box_voxels(grid, lo_world, hi_world):
    """Inclusive voxel index range of a world box, before any clamping."""
    h = grid.voxel_size
    lo = [int(np.floor((lo_world[i] - grid.origin[i]) / h)) for i in range(3)]
    hi = [int(np.floor((hi_world[i] - grid.origin[i]) / h)) for i in range(3)]
    return lo, hi


class SparseGrid:
    """Sparse leaf-tile grid over a world-space domain box."""

    def __init__(self, origin, size, voxel_size, leaf_capacity, device):
        self.device = device
        bounds = (
            tuple(float(value) for value in origin),
            tuple(float(origin[i]) + float(size[i]) for i in range(3)),
        )
        self.quality = resolve_quality(
            AuthoredQualitySettings(voxel_size=float(voxel_size)), bounds)
        self.voxel_size = self.quality.voxel_size
        nx, ny, nz = self.quality.padded_voxel_dimensions
        self.voxels_dim = wp.vec3i(nx, ny, nz)
        self.leaves_dim = wp.vec3i(*self.quality.leaf_dimensions)
        self.num_leaf_cells = (
            self.leaves_dim[0] * self.leaves_dim[1] * self.leaves_dim[2]
        )
        self.origin = wp.vec3(float(origin[0]), float(origin[1]), float(origin[2]))
        self.capacity = int(min(leaf_capacity, self.num_leaf_cells))

        # Live lookup/count buffers are never cleared while candidate demand is
        # still unadmitted. Candidate table/coords are swapped into place only
        # after the overflow readback proves the rebuild fits.
        self.leaf_table = wp.zeros(self.num_leaf_cells, dtype=wp.int32, device=device)
        self.leaf_coords = wp.zeros(self.capacity, dtype=wp.vec3i, device=device)
        self.num_leaves = wp.zeros(1, dtype=wp.int32, device=device)
        self.overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self._candidate_leaf_table = wp.zeros(
            self.num_leaf_cells, dtype=wp.int32, device=device)
        self._candidate_leaf_coords = wp.zeros(
            self.capacity, dtype=wp.vec3i, device=device)
        self._candidate_num_leaves = wp.zeros(1, dtype=wp.int32, device=device)
        self._candidate_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        # Flags are candidate-only scratch and do not participate in live field
        # lookup, so one pair can be reused across transactions.
        self.leaf_flags = wp.zeros(self.num_leaf_cells, dtype=wp.int32, device=device)
        self.leaf_flags_tmp = wp.zeros(self.num_leaf_cells, dtype=wp.int32, device=device)
        # Allocation-order scratch: the pending predicate, its exclusive prefix
        # rank, and the slot base captured before the scan publishes new slots.
        self._leaf_pending = wp.zeros(self.num_leaf_cells, dtype=wp.int32, device=device)
        self._leaf_pending_rank = wp.zeros(
            self.num_leaf_cells, dtype=wp.int32, device=device)
        self._leaf_slot_base = wp.zeros(1, dtype=wp.int32, device=device)
        self._candidate_pending = False
        self.active_leaf_count = 0  # host mirror, refreshed by refresh_topology_counts

    # -- properties ---------------------------------------------------------
    @property
    def voxel_capacity(self) -> int:
        return self.capacity * LEAF_VOLUME

    @property
    def active_voxel_count(self) -> int:
        return self.active_leaf_count * LEAF_VOLUME

    def total_voxels(self) -> int:
        return int(self.voxels_dim[0]) * int(self.voxels_dim[1]) * int(self.voxels_dim[2])

    # -- topology lifecycle ---------------------------------------------------
    def begin_topology(self):
        self._candidate_pending = False
        fast_launch(
            k_clear_leaf_topology,
            dim=self.num_leaf_cells,
            inputs=[self._candidate_leaf_table, self.leaf_flags],
            device=self.device,
        )
        self._candidate_num_leaves.zero_()
        self._candidate_overflow.zero_()

    def mark_points(self, points: wp.array, count: wp.array, max_points: int):
        fast_launch(
            k_mark_leaves_from_points,
            dim=max_points,
            inputs=[
                points, count, self.origin, self.voxel_size,
                self.voxels_dim, self.leaves_dim, self.leaf_flags,
            ],
            device=self.device,
        )

    def overlaps_world_box(self, lo_world, hi_world):
        """Whether a world box touches at least one domain voxel."""
        lo, hi = _world_box_voxels(self, lo_world, hi_world)
        return not any(hi[i] < 0 or lo[i] > int(self.voxels_dim[i]) - 1
                       for i in range(3))

    def mark_world_box(self, lo_world, hi_world):
        lo, hi = _world_box_voxels(self, lo_world, hi_world)
        # A box with no domain overlap must be rejected, not clamped: clamping
        # glues it to the domain boundary corner and marks leaves there, so an
        # out-of-domain source appears to work from the wrong place. A box that
        # only partly overlaps is still clamped to the overlapping part.
        if any(hi[i] < 0 or lo[i] > int(self.voxels_dim[i]) - 1 for i in range(3)):
            raise ValueError(
                f"world box {tuple(lo_world)} .. {tuple(hi_world)} lies entirely "
                f"outside the simulation domain")
        lo = [max(0, min(int(self.voxels_dim[i]) - 1, lo[i])) for i in range(3)]
        hi = [max(0, min(int(self.voxels_dim[i]) - 1, hi[i])) for i in range(3)]
        n_leaves = 1
        for i in range(3):
            n_leaves *= (hi[i] >> LEAF_SHIFT) - (lo[i] >> LEAF_SHIFT) + 1
        fast_launch(
            k_mark_leaves_box,
            dim=n_leaves,
            inputs=[
                wp.vec3i(*lo), wp.vec3i(*hi), self.leaves_dim, self.leaf_flags,
            ],
            device=self.device,
        )

    def expand(self, iterations=1):
        for _ in range(iterations):
            wp.copy(self.leaf_flags_tmp, self.leaf_flags)
            fast_launch(
                k_expand_leaf_flags,
                dim=self.num_leaf_cells,
                inputs=[self.leaf_flags_tmp, self.leaves_dim, self.leaf_flags],
                device=self.device,
            )

    def finalize_topology(self, sync: bool = True) -> int:
        # Slot assignment is a prefix rank over linear leaf coordinates, so the
        # admitted topology is identical for identical flags. The base is the
        # already-admitted count, captured before the allocation kernel raises
        # it, which keeps the incremental (non-cleared candidate table) path
        # behaving exactly as the atomic cursor did.
        fast_launch(
            k_mark_unallocated_leaves,
            dim=self.num_leaf_cells,
            inputs=[self.leaf_flags, self._candidate_leaf_table],
            outputs=[self._leaf_pending],
            device=self.device,
        )
        wp.utils.array_scan(
            self._leaf_pending, self._leaf_pending_rank, inclusive=False)
        wp.copy(self._leaf_slot_base, self._candidate_num_leaves)
        fast_launch(
            k_allocate_leaves,
            dim=self.num_leaf_cells,
            inputs=[
                self._leaf_pending, self._leaf_pending_rank,
                self._leaf_slot_base, self.leaves_dim, self.capacity,
                self._candidate_leaf_table, self._candidate_leaf_coords,
                self._candidate_num_leaves, self._candidate_overflow,
            ],
            device=self.device,
        )
        self._candidate_pending = True
        if not sync:
            return self.active_leaf_count
        return self.refresh_topology_counts()

    def refresh_topology_counts(self) -> int:
        if not self._candidate_pending:
            return self.active_leaf_count
        count = int(self._candidate_num_leaves.numpy()[0])
        if int(self._candidate_overflow.numpy()[0]) != 0:
            self._candidate_pending = False
            raise TopologyCapacityError(count, self.capacity)
        # Keep live Warp array identities stable. Solver subsystems and queued
        # kernels bind these arrays by object identity; swapping the Python
        # handles after admission made later phases observe alternating stale
        # topology buffers and injected catastrophic vertical energy. Copy the
        # admitted candidate into the stable live storage instead.
        wp.copy(self.leaf_table, self._candidate_leaf_table)
        wp.copy(self.leaf_coords, self._candidate_leaf_coords)
        wp.copy(self.num_leaves, self._candidate_num_leaves)
        wp.copy(self.overflow, self._candidate_overflow)
        self._candidate_pending = False
        self.active_leaf_count = count
        return self.active_leaf_count

    def restore_topology(self, leaf_coords) -> int:
        """Restore an audited slot-ordered topology without partial mutation."""
        coords = np.asarray(leaf_coords)
        if coords.ndim != 2 or coords.shape[1:] != (3,):
            raise ValueError("grid_leaf_coords must have shape (N, 3)")
        if coords.dtype.kind not in "iu":
            raise ValueError("grid_leaf_coords must contain integers")
        coords = np.ascontiguousarray(coords, dtype=np.int32)
        count = int(len(coords))
        if count > self.capacity:
            raise TopologyCapacityError(count, self.capacity)
        dimensions = tuple(int(self.leaves_dim[index]) for index in range(3))
        if count:
            if np.any(coords < 0):
                raise ValueError("grid_leaf_coords contains a negative coordinate")
            for axis, dimension in enumerate(dimensions):
                if np.any(coords[:, axis] >= dimension):
                    raise ValueError(
                        f"grid_leaf_coords exceeds leaf dimension on axis {axis}")
            if len({tuple(int(value) for value in row) for row in coords}) != count:
                raise ValueError("grid_leaf_coords contains duplicate leaves")

        table = np.full(self.num_leaf_cells, -1, dtype=np.int32)
        for slot, (x, y, z) in enumerate(coords):
            linear = int(x) + dimensions[0] * (
                int(y) + dimensions[1] * int(z))
            table[linear] = slot
        padded_coords = np.zeros((self.capacity, 3), dtype=np.int32)
        if count:
            padded_coords[:count] = coords

        # All validation and host construction completed before the first live
        # device write. Keep array identities stable just like normal topology
        # admission does.
        table_dev = wp.array(table, dtype=wp.int32, device=self.device)
        coords_dev = wp.array(padded_coords, dtype=wp.vec3i, device=self.device)
        count_dev = wp.array([count], dtype=wp.int32, device=self.device)
        wp.copy(self.leaf_table, table_dev)
        wp.copy(self.leaf_coords, coords_dev)
        wp.copy(self.num_leaves, count_dev)
        self.overflow.zero_()
        wp.copy(self._candidate_leaf_table, table_dev)
        wp.copy(self._candidate_leaf_coords, coords_dev)
        wp.copy(self._candidate_num_leaves, count_dev)
        self._candidate_overflow.zero_()
        self.leaf_flags.zero_()
        self.leaf_flags_tmp.zero_()
        self._candidate_pending = False
        self.active_leaf_count = count
        return count


class ScalarField:
    """Scalar field stored over sparse leaf slots (float32/float16/int32)."""

    def __init__(self, grid: SparseGrid, dtype=float, name=""):
        self.grid = grid
        self.dtype = dtype
        self.name = name
        self.data = wp.zeros(grid.voxel_capacity, dtype=dtype, device=grid.device)

    def zero(self):
        self.data.zero_()


class MacVelocityField:
    """Staggered MAC velocity: per-voxel lower-face samples + validity weights."""

    def __init__(self, grid: SparseGrid):
        self.grid = grid
        n = grid.voxel_capacity
        dev = grid.device
        self.u = wp.zeros(n, dtype=float, device=dev)
        self.v = wp.zeros(n, dtype=float, device=dev)
        self.w = wp.zeros(n, dtype=float, device=dev)
        self.wu = wp.zeros(n, dtype=float, device=dev)
        self.wv = wp.zeros(n, dtype=float, device=dev)
        self.ww = wp.zeros(n, dtype=float, device=dev)

    def zero(self):
        for a in (self.u, self.v, self.w, self.wu, self.wv, self.ww):
            a.zero_()

    def arrays(self):
        return (self.u, self.v, self.w, self.wu, self.wv, self.ww)
