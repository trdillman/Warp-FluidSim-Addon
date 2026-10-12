"""GPU-resident particle storage for liquid and whitewater particles.

All particle state stays device-resident during playback. Compaction and
binning are performed on the GPU (scan-based), matching the original
particle indexing/sorting kernel family. No bpy imports allowed here.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from .fastlaunch import launch as fast_launch

from .checkpoint_state import host_prefix
from .constants import LEAF_VOLUME


PHYSICAL_ATTR_NAMES = ("age", "life", "temperature", "viscosity", "density")
OPTIONAL_VEC4_ATTR_NAMES = ("color",)
OPTIONAL_INT_ATTR_NAMES = ("source_id", "uid")


@wp.kernel
def k_clamp_count(
    count: wp.array(dtype=wp.int32),
    capacity: int,
):
    if count[0] < 0:
        count[0] = 0
    elif count[0] > capacity:
        count[0] = capacity


@wp.kernel
def k_set_count(
    count: wp.array(dtype=wp.int32),
    value: int,
):
    count[0] = value


@wp.kernel
def k_reset_alive(
    count: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    alive[tid] = wp.where(tid < count[0], 1, 0)


@wp.kernel
def k_compact_alive_scatter(
    alive: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),  # exclusive scan of alive
    count_in: wp.array(dtype=wp.int32),
    src_pos: wp.array(dtype=wp.vec3),
    src_vel: wp.array(dtype=wp.vec3),
    src_f0: wp.array(dtype=float),  # age
    src_f1: wp.array(dtype=float),  # life
    src_f2: wp.array(dtype=float),  # temperature
    src_f3: wp.array(dtype=float),  # viscosity
    src_f4: wp.array(dtype=float),  # density/extra
    src_time_offset: wp.array(dtype=float),
    dst_pos: wp.array(dtype=wp.vec3),
    dst_vel: wp.array(dtype=wp.vec3),
    dst_f0: wp.array(dtype=float),
    dst_f1: wp.array(dtype=float),
    dst_f2: wp.array(dtype=float),
    dst_f3: wp.array(dtype=float),
    dst_f4: wp.array(dtype=float),
    dst_time_offset: wp.array(dtype=float),
    count_out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    n = count_in[0]
    if tid == 0 and n == 0:
        count_out[0] = 0
    if tid >= n:
        return
    if tid == n - 1:
        count_out[0] = offsets[tid] + alive[tid]
    if alive[tid] == 0:
        return
    j = offsets[tid]
    dst_pos[j] = src_pos[tid]
    dst_vel[j] = src_vel[tid]
    dst_f0[j] = src_f0[tid]
    dst_f1[j] = src_f1[tid]
    dst_f2[j] = src_f2[tid]
    dst_f3[j] = src_f3[tid]
    dst_f4[j] = src_f4[tid]
    dst_time_offset[j] = src_time_offset[tid]


@wp.kernel
def k_compact_vec4_scatter(
    alive: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    count_in: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.vec4),
    target: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid < count_in[0] and alive[tid] != 0:
        target[offsets[tid]] = source[tid]


@wp.kernel
def k_compact_int_scatter(
    alive: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    count_in: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.int32),
    target: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < count_in[0] and alive[tid] != 0:
        target[offsets[tid]] = source[tid]


@wp.kernel
def k_compact_mat33_scatter(
    alive: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    count_in: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.mat33),
    target: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    if tid < count_in[0] and alive[tid] != 0:
        target[offsets[tid]] = source[tid]


@wp.kernel
def k_zero_mat33_tail(
    count: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.mat33),
):
    """Zero every slot at or beyond the live count.

    Emission claims fresh slots past the count without writing the affine
    channel, so the channel must already be zero there ("emission writes zero
    affine").  Compaction leaves stale matrices in that tail; this restores it.
    """
    tid = wp.tid()
    if tid >= count[0]:
        values[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def k_last_offset_count(
    alive: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    count_in: wp.array(dtype=wp.int32),
    count_out: wp.array(dtype=wp.int32),
):
    n = count_in[0]
    if n == 0:
        count_out[0] = 0
    else:
        count_out[0] = offsets[n - 1] + alive[n - 1]


class ParticleSystem:
    """Double-buffered liquid particles with an explicit optional inventory.

    Physical float channels are always allocated because solver kernels own
    them.  Product channels (color/source identity/UID) are allocated only when
    selected before initialization, preserving the preview memory contract.
    """

    ATTR_NAMES = PHYSICAL_ATTR_NAMES
    OPTIONAL_VEC4_ATTR_NAMES = OPTIONAL_VEC4_ATTR_NAMES
    OPTIONAL_INT_ATTR_NAMES = OPTIONAL_INT_ATTR_NAMES

    def __init__(self, capacity: int, device, optional_channels=()):
        self.capacity = int(capacity)
        self.device = device
        selected = frozenset(str(name) for name in optional_channels)
        unknown = selected - set(OPTIONAL_VEC4_ATTR_NAMES) - set(OPTIONAL_INT_ATTR_NAMES)
        if unknown:
            raise ValueError(f"unsupported optional particle channels: {sorted(unknown)}")
        self.optional_channels = tuple(
            name for name in (*OPTIONAL_VEC4_ATTR_NAMES, *OPTIONAL_INT_ATTR_NAMES)
            if name in selected)
        n = self.capacity
        self.position = wp.zeros(n, dtype=wp.vec3, device=device)
        self.velocity = wp.zeros(n, dtype=wp.vec3, device=device)
        self.attrs = {
            name: wp.zeros(n, dtype=float, device=device) for name in self.ATTR_NAMES
        }
        self.vec4_attrs = {
            name: wp.zeros(n, dtype=wp.vec4, device=device)
            for name in OPTIONAL_VEC4_ATTR_NAMES if name in selected
        }
        self.int_attrs = {
            name: wp.zeros(n, dtype=wp.int32, device=device)
            for name in OPTIONAL_INT_ATTR_NAMES if name in selected
        }
        # Kernels receive stable typed placeholders when a channel is disabled.
        self._dummy_vec4 = wp.zeros(1, dtype=wp.vec4, device=device)
        self._dummy_int = wp.zeros(1, dtype=wp.int32, device=device)
        # APIC affine matrix (world units, 1/s), double buffered for compaction.
        # Allocated lazily, zeroed, only while the APIC transfer is active
        # (``ensure_affine``).  Invariant: every slot at or beyond ``count`` is
        # zero, so emission needs no explicit write.
        self.affine = None
        self._affine2 = None
        self._dummy_affine = wp.zeros(1, dtype=wp.mat33, device=device)
        # Per-particle time residual channel. Always zero since the removal of
        # ST-FLIP; kept because it is a public cache/checkpoint/effects channel.
        self.time_offset = wp.zeros(n, dtype=float, device=device)
        self._position2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._velocity2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._attrs2 = {
            name: wp.zeros(n, dtype=float, device=device) for name in self.ATTR_NAMES
        }
        self._vec4_attrs2 = {
            name: wp.zeros(n, dtype=wp.vec4, device=device)
            for name in self.vec4_attrs
        }
        self._int_attrs2 = {
            name: wp.zeros(n, dtype=wp.int32, device=device)
            for name in self.int_attrs
        }
        self._time_offset2 = wp.zeros(n, dtype=float, device=device)
        self.alive = wp.zeros(n, dtype=wp.int32, device=device)
        self.scan_tmp = wp.zeros(n, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self._count2 = wp.zeros(1, dtype=wp.int32, device=device)
        self.voxel_index = wp.zeros(n, dtype=wp.int32, device=device)
        self.host_count = 0
        self.next_uid = 1

    @property
    def affine_allocated(self) -> bool:
        return self.affine is not None

    def ensure_affine(self):
        """Allocate the zeroed affine channel if needed and return it."""
        if self.affine is None:
            self.affine = wp.zeros(
                self.capacity, dtype=wp.mat33, device=self.device)
            self._affine2 = wp.zeros(
                self.capacity, dtype=wp.mat33, device=self.device)
        return self.affine

    def release_affine(self) -> None:
        """Free the affine channel (non-APIC transfer: nothing maintains it)."""
        self.affine = None
        self._affine2 = None

    def affine_for_kernels(self):
        """The real channel, or a length-1 placeholder when not allocated."""
        return self.affine if self.affine is not None else self._dummy_affine

    def has_optional_channel(self, name: str) -> bool:
        return name in self.vec4_attrs or name in self.int_attrs

    def optional_array(self, name: str):
        if name in self.vec4_attrs:
            return self.vec4_attrs[name]
        if name in self.int_attrs:
            return self.int_attrs[name]
        if name in OPTIONAL_VEC4_ATTR_NAMES:
            return self._dummy_vec4
        if name in OPTIONAL_INT_ATTR_NAMES:
            return self._dummy_int
        raise KeyError(name)

    def attr(self, name: str) -> wp.array:
        if name == "time_offset":
            return self.time_offset
        if name in self.attrs:
            return self.attrs[name]
        if name in self.vec4_attrs:
            return self.vec4_attrs[name]
        if name in self.int_attrs:
            return self.int_attrs[name]
        raise KeyError(name)

    def clamp_count(self):
        fast_launch(k_clamp_count, dim=1, inputs=[self.count, self.capacity],
                  device=self.device)

    def refresh_host_count(self) -> int:
        self.clamp_count()
        self.host_count = int(self.count.numpy()[0])
        return self.host_count

    def clear(self):
        self.count.zero_()
        self.time_offset.zero_()
        for value in (*self.vec4_attrs.values(), *self.int_attrs.values()):
            value.zero_()
        if self.affine is not None:
            self.affine.zero_()
            self._affine2.zero_()
        self.host_count = 0
        self.next_uid = 1

    def resized_copy(self, capacity: int) -> "ParticleSystem":
        """Allocate a larger pool and copy the complete live prefix."""
        capacity = int(capacity)
        if capacity <= self.capacity:
            raise ValueError("resized particle capacity must be larger")
        count = self.refresh_host_count()
        grown = ParticleSystem(capacity, self.device, self.optional_channels)
        if count:
            wp.copy(grown.position, self.position, count=count)
            wp.copy(grown.velocity, self.velocity, count=count)
            for name in self.ATTR_NAMES:
                wp.copy(grown.attrs[name], self.attrs[name], count=count)
            for name in self.vec4_attrs:
                wp.copy(grown.vec4_attrs[name], self.vec4_attrs[name], count=count)
            for name in self.int_attrs:
                wp.copy(grown.int_attrs[name], self.int_attrs[name], count=count)
            wp.copy(grown.time_offset, self.time_offset, count=count)
            if self.affine is not None:
                wp.copy(grown.ensure_affine(), self.affine, count=count)
            wp.copy(grown.alive, self.alive, count=count)
            wp.copy(grown.voxel_index, self.voxel_index, count=count)
        if self.affine is not None:
            grown.ensure_affine()
        wp.copy(grown.count, self.count, count=1)
        grown.host_count = count
        grown.next_uid = int(self.next_uid)
        return grown

    def checkpoint_arrays(self, count: int) -> dict[str, np.ndarray]:
        """Return enabled product channels as owned checkpoint arrays."""
        result = {}
        for name, value in self.vec4_attrs.items():
            result[f"particle_attr_{name}"] = np.ascontiguousarray(
                host_prefix(value, count), dtype=np.float32)
        for name, value in self.int_attrs.items():
            result[f"particle_attr_{name}"] = np.ascontiguousarray(
                host_prefix(value, count), dtype=np.int32)
        # Empty (0, 9) means "no affine state" (non-APIC writer); it is only
        # persisted while the channel is allocated, i.e. under APIC.
        if self.affine is not None and count:
            result["particle_affine"] = np.ascontiguousarray(
                host_prefix(self.affine, count).reshape(count, 9), dtype=np.float32)
        else:
            result["particle_affine"] = np.zeros((0, 9), dtype=np.float32)
        return result

    def restore_checkpoint_prefix(self, arrays, count: int) -> None:
        """Validate and restore the complete persistent liquid prefix."""
        count = int(count)
        if count < 0 or count > self.capacity:
            raise ValueError(
                f"particle checkpoint count {count} exceeds capacity {self.capacity}")
        required = {
            "particle_position": ((count, 3), np.float32),
            "particle_velocity": ((count, 3), np.float32),
            "particle_alive": ((count,), np.int32),
            "particle_time_offset": ((count,), np.float32),
        }
        for name in self.ATTR_NAMES:
            required[f"particle_attr_{name}"] = ((count,), np.float32)
        for name in self.vec4_attrs:
            required[f"particle_attr_{name}"] = ((count, 4), np.float32)
        for name in self.int_attrs:
            required[f"particle_attr_{name}"] = ((count,), np.int32)
        normalized = {}
        for name, (shape, dtype) in required.items():
            if name not in arrays:
                raise ValueError(f"particle checkpoint is missing {name}")
            value = np.asarray(arrays[name])
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
            value = np.ascontiguousarray(value, dtype=dtype)
            if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
            normalized[name] = value
        affine_rows = None
        saved_affine = arrays.get("particle_affine") if hasattr(
            arrays, "get") else None
        if saved_affine is not None and np.asarray(saved_affine).shape[0] != 0:
            saved_affine = np.ascontiguousarray(saved_affine, dtype=np.float32)
            if saved_affine.shape != (count, 9):
                raise ValueError(
                    f"particle_affine has shape {saved_affine.shape}; "
                    f"expected {(count, 9)}")
            if not np.all(np.isfinite(saved_affine)):
                raise ValueError("particle_affine contains non-finite values")
            affine_rows = saved_affine
        alive = normalized["particle_alive"]
        if np.any((alive != 0) & (alive != 1)):
            raise ValueError("particle_alive must contain only 0 or 1")
        if "uid" in self.int_attrs:
            uid = normalized["particle_attr_uid"]
            if np.any(uid <= 0) or len(np.unique(uid)) != count:
                raise ValueError("particle_attr_uid must contain unique positive IDs")

        for value in (
                self.position, self.velocity, self.time_offset, self.alive,
                self.voxel_index, self.scan_tmp, self._position2,
                self._velocity2, self._time_offset2):
            value.zero_()
        for value in (
                *self.attrs.values(), *self._attrs2.values(),
                *self.vec4_attrs.values(), *self._vec4_attrs2.values(),
                *self.int_attrs.values(), *self._int_attrs2.values()):
            value.zero_()
        if self.affine is not None:
            self.affine.zero_()
            self._affine2.zero_()
        if count:
            if self.affine is not None and affine_rows is not None:
                # A checkpoint from another transfer mode carries no affine
                # (or it is dropped by a non-APIC solver): zero is valid.
                wp.copy(self.affine, wp.array(
                    affine_rows.reshape(count, 3, 3), dtype=wp.mat33,
                    device=self.device), count=count)
            wp.copy(self.position, wp.array(
                normalized["particle_position"], dtype=wp.vec3, device=self.device),
                count=count)
            wp.copy(self.velocity, wp.array(
                normalized["particle_velocity"], dtype=wp.vec3, device=self.device),
                count=count)
            wp.copy(self.alive, wp.array(
                alive, dtype=wp.int32, device=self.device), count=count)
            wp.copy(self.time_offset, wp.array(
                normalized["particle_time_offset"], dtype=float, device=self.device),
                count=count)
            for name in self.ATTR_NAMES:
                wp.copy(self.attrs[name], wp.array(
                    normalized[f"particle_attr_{name}"], dtype=float,
                    device=self.device), count=count)
            for name in self.vec4_attrs:
                wp.copy(self.vec4_attrs[name], wp.array(
                    normalized[f"particle_attr_{name}"], dtype=wp.vec4,
                    device=self.device), count=count)
            for name in self.int_attrs:
                wp.copy(self.int_attrs[name], wp.array(
                    normalized[f"particle_attr_{name}"], dtype=wp.int32,
                    device=self.device), count=count)
        wp.copy(self.count, wp.array([count], dtype=wp.int32, device=self.device),
                count=1)
        self._count2.zero_()
        self.host_count = count

    def compact(self, count_hint: int | None = None):
        """Remove dead particles while keeping every enabled channel aligned."""
        n = self.capacity if count_hint is None else max(
            0, min(int(count_hint), self.capacity))
        if n <= 0:
            self.count.zero_()
            self.time_offset.zero_()
            self.host_count = 0
            return
        self.clamp_count()
        alive = self.alive if count_hint is None else self.alive[:n]
        scan_tmp = self.scan_tmp if count_hint is None else self.scan_tmp[:n]
        count_in = self.count
        count_out = self._count2
        if count_hint is not None:
            fast_launch(k_set_count, dim=1, inputs=[self._count2, n],
                      device=self.device)
            count_in = self._count2
            count_out = self.count
        wp.utils.array_scan(alive, scan_tmp, inclusive=False)
        fast_launch(k_compact_alive_scatter, dim=n, inputs=[
            self.alive, self.scan_tmp, count_in,
            self.position, self.velocity,
            self.attrs["age"], self.attrs["life"], self.attrs["temperature"],
            self.attrs["viscosity"], self.attrs["density"], self.time_offset,
            self._position2, self._velocity2,
            self._attrs2["age"], self._attrs2["life"],
            self._attrs2["temperature"], self._attrs2["viscosity"],
            self._attrs2["density"], self._time_offset2, count_out,
        ], device=self.device)
        for name in self.vec4_attrs:
            fast_launch(k_compact_vec4_scatter, dim=n, inputs=[
                self.alive, self.scan_tmp, count_in,
                self.vec4_attrs[name], self._vec4_attrs2[name],
            ], device=self.device)
        for name in self.int_attrs:
            fast_launch(k_compact_int_scatter, dim=n, inputs=[
                self.alive, self.scan_tmp, count_in,
                self.int_attrs[name], self._int_attrs2[name],
            ], device=self.device)
        if self.affine is not None:
            fast_launch(k_compact_mat33_scatter, dim=n, inputs=[
                self.alive, self.scan_tmp, count_in,
                self.affine, self._affine2,
            ], device=self.device)
        self.position, self._position2 = self._position2, self.position
        self.velocity, self._velocity2 = self._velocity2, self.velocity
        self.time_offset, self._time_offset2 = self._time_offset2, self.time_offset
        for name in self.ATTR_NAMES:
            self.attrs[name], self._attrs2[name] = self._attrs2[name], self.attrs[name]
        for name in self.vec4_attrs:
            self.vec4_attrs[name], self._vec4_attrs2[name] = (
                self._vec4_attrs2[name], self.vec4_attrs[name])
        for name in self.int_attrs:
            self.int_attrs[name], self._int_attrs2[name] = (
                self._int_attrs2[name], self.int_attrs[name])
        if self.affine is not None:
            self.affine, self._affine2 = self._affine2, self.affine
        if count_out is self._count2:
            wp.copy(self.count, self._count2)
        if self.affine is not None:
            fast_launch(k_zero_mat33_tail, dim=n,
                        inputs=[self.count, self.affine], device=self.device)
        fast_launch(k_reset_alive, dim=n, inputs=[self.count, self.alive],
                  device=self.device)


class ParticleBins:
    """Voxel binning (counting sort) of particles over sparse leaf slots.

    Reproduces the original chain:
      fluidSimParticleToVoxelCountAndUpdateVoxelIndex ->
      fluidSimVoxelCountToVoxelParticleArrayStart ->
      fluidSimParticleToVoxelParticleArray (+ optional in-bin sort)

    The scatter places each particle at the arrival slot the count kernel
    reserved and then ranks each bin by particle index, rather than keeping
    the arrival order of an atomic cursor, so within-bin order is always
    ascending particle index.  Both the relaxation sub-sampling and the P2G
    gather read bins positionally, so an arbitrary within-bin order makes
    those stages irreproducible between runs of the same input.
    ``arrival`` and ``unordered`` are the per-particle scratch for that.
    """

    def __init__(self, particle_capacity: int, voxel_capacity: int, device):
        self.device = device
        self.particle_capacity = int(particle_capacity)
        self.voxel_capacity = int(voxel_capacity)
        self.counts = wp.zeros(voxel_capacity, dtype=wp.int32, device=device)
        self.starts = wp.zeros(voxel_capacity, dtype=wp.int32, device=device)
        self.indices = wp.zeros(
            self.particle_capacity, dtype=wp.int32, device=device)
        self.arrival = wp.zeros(
            self.particle_capacity, dtype=wp.int32, device=device)
        self.unordered = wp.zeros(
            self.particle_capacity, dtype=wp.int32, device=device)

    def zero(self):
        self.counts.zero_()
