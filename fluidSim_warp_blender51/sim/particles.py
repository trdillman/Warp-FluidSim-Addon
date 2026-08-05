"""GPU-resident particle storage for liquid and whitewater particles.

All particle state stays device-resident during playback. Compaction and
binning are performed on the GPU (scan-based), matching the original
particle indexing/sorting kernel family. No bpy imports allowed here.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .constants import LEAF_VOLUME


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
    if tid >= n:
        return
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
    if tid == n - 1:
        count_out[0] = j + 1


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
    """Double-buffered liquid particle set with named float attributes.

    Attribute channels (fixed inventory, matching the original solver's
    particle attribute surface): age, life, temperature, viscosity, density.
    """

    ATTR_NAMES = ("age", "life", "temperature", "viscosity", "density")

    def __init__(self, capacity: int, device):
        self.capacity = int(capacity)
        self.device = device
        n = self.capacity
        self.position = wp.zeros(n, dtype=wp.vec3, device=device)
        self.velocity = wp.zeros(n, dtype=wp.vec3, device=device)
        self.attrs = {
            name: wp.zeros(n, dtype=float, device=device) for name in self.ATTR_NAMES
        }
        # ST-FLIP residual: delta_t = t_global - t_particle.  Stored as
        # float32 for broad Warp-device compatibility; the paper notes that
        # half precision is sufficient, but Blender/Warp builds vary in
        # float16 atomic/indexing support.
        self.time_offset = wp.zeros(n, dtype=float, device=device)
        # scratch (compaction targets)
        self._position2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._velocity2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._attrs2 = {
            name: wp.zeros(n, dtype=float, device=device) for name in self.ATTR_NAMES
        }
        self._time_offset2 = wp.zeros(n, dtype=float, device=device)
        self.alive = wp.zeros(n, dtype=wp.int32, device=device)
        self.scan_tmp = wp.zeros(n, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self._count2 = wp.zeros(1, dtype=wp.int32, device=device)
        # per-particle voxel index cache (fluidSimUpdateParticleVoxelIndex)
        self.voxel_index = wp.zeros(n, dtype=wp.int32, device=device)
        self.host_count = 0

    def attr(self, name: str) -> wp.array:
        if name in {"time_offset", "st_time_offset"}:
            return self.time_offset
        return self.attrs[name]

    def clamp_count(self):
        wp.launch(
            k_clamp_count, dim=1, inputs=[self.count, self.capacity],
            device=self.device,
        )

    def refresh_host_count(self) -> int:
        self.clamp_count()
        self.host_count = int(self.count.numpy()[0])
        return self.host_count

    def clear(self):
        self.count.zero_()
        self.time_offset.zero_()
        self.host_count = 0

    def resized_copy(self, capacity: int) -> "ParticleSystem":
        """Allocate a larger pool and copy the live prefix without swapping it.

        Callers own the final atomic swap.  If allocation or copying raises,
        this object remains the authoritative particle pool.
        """
        capacity = int(capacity)
        if capacity <= self.capacity:
            raise ValueError("resized particle capacity must be larger")
        count = self.refresh_host_count()
        grown = ParticleSystem(capacity, self.device)
        if count:
            wp.copy(grown.position, self.position, count=count)
            wp.copy(grown.velocity, self.velocity, count=count)
            for name in self.ATTR_NAMES:
                wp.copy(grown.attrs[name], self.attrs[name], count=count)
            wp.copy(grown.time_offset, self.time_offset, count=count)
            wp.copy(grown.alive, self.alive, count=count)
            wp.copy(grown.voxel_index, self.voxel_index, count=count)
        wp.copy(grown.count, self.count, count=1)
        grown.host_count = count
        return grown

    def compact(self, count_hint: int | None = None):
        """Remove dead particles (alive==0) with a GPU scan + scatter.

        Args:
            count_hint: Optional upper bound for the live particle prefix. When
                provided by the solver, scan and scatter only that active range.
        """
        n = self.capacity if count_hint is None else max(0, min(int(count_hint), self.capacity))
        if n <= 0:
            self.count.zero_()
            self.time_offset.zero_()
            self.host_count = 0
            return
        self.clamp_count()

        # The solver keeps host_count as an upper bound on device count. Use
        # that bound here so compaction does not silently return to capacity
        # scans after active-work launch bounds have trimmed the hot path.
        alive = self.alive if count_hint is None else self.alive[:n]
        scan_tmp = self.scan_tmp if count_hint is None else self.scan_tmp[:n]
        count_in = self.count
        count_out = self._count2
        if count_hint is not None:
            wp.launch(k_set_count, dim=1, inputs=[self._count2, n],
                      device=self.device)
            count_in = self._count2
            count_out = self.count
        wp.utils.array_scan(alive, scan_tmp, inclusive=False)
        wp.launch(
            k_last_offset_count,
            dim=1,
            inputs=[self.alive, self.scan_tmp, count_in, count_out],
            device=self.device,
        )
        wp.launch(
            k_compact_alive_scatter,
            dim=n,
            inputs=[
                self.alive, self.scan_tmp, count_in,
                self.position, self.velocity,
                self.attrs["age"], self.attrs["life"], self.attrs["temperature"],
                self.attrs["viscosity"], self.attrs["density"],
                self.time_offset,
                self._position2, self._velocity2,
                self._attrs2["age"], self._attrs2["life"], self._attrs2["temperature"],
                self._attrs2["viscosity"], self._attrs2["density"],
                self._time_offset2,
                count_out,
            ],
            device=self.device,
        )
        # swap buffers
        self.position, self._position2 = self._position2, self.position
        self.velocity, self._velocity2 = self._velocity2, self.velocity
        self.time_offset, self._time_offset2 = self._time_offset2, self.time_offset
        for k in self.ATTR_NAMES:
            self.attrs[k], self._attrs2[k] = self._attrs2[k], self.attrs[k]
        if count_out is self._count2:
            wp.copy(self.count, self._count2)
        # Re-mark the surviving prefix as alive so the next scan is valid.
        wp.launch(k_reset_alive, dim=n,
                  inputs=[self.count, self.alive], device=self.device)


class ParticleBins:
    """Voxel binning (counting sort) of particles over sparse leaf slots.

    Reproduces the original chain:
      fluidSimParticleToVoxelCountAndUpdateVoxelIndex ->
      fluidSimVoxelCountToVoxelParticleArrayStart ->
      fluidSimParticleToVoxelParticleArray (+ optional in-bin sort)
    """

    def __init__(self, particle_capacity: int, voxel_capacity: int, device):
        self.device = device
        self.particle_capacity = int(particle_capacity)
        self.voxel_capacity = int(voxel_capacity)
        self.counts = wp.zeros(voxel_capacity, dtype=wp.int32, device=device)
        self.starts = wp.zeros(voxel_capacity, dtype=wp.int32, device=device)
        self.cursor = wp.zeros(voxel_capacity, dtype=wp.int32, device=device)
        self.indices = wp.zeros(particle_capacity, dtype=wp.int32, device=device)

    def zero(self):
        self.counts.zero_()
