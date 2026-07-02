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
    dst_pos: wp.array(dtype=wp.vec3),
    dst_vel: wp.array(dtype=wp.vec3),
    dst_f0: wp.array(dtype=float),
    dst_f1: wp.array(dtype=float),
    dst_f2: wp.array(dtype=float),
    dst_f3: wp.array(dtype=float),
    dst_f4: wp.array(dtype=float),
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
        # scratch (compaction targets)
        self._position2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._velocity2 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._attrs2 = {
            name: wp.zeros(n, dtype=float, device=device) for name in self.ATTR_NAMES
        }
        self.alive = wp.zeros(n, dtype=wp.int32, device=device)
        self.scan_tmp = wp.zeros(n, dtype=wp.int32, device=device)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self._count2 = wp.zeros(1, dtype=wp.int32, device=device)
        # per-particle voxel index cache (fluidSimUpdateParticleVoxelIndex)
        self.voxel_index = wp.zeros(n, dtype=wp.int32, device=device)
        self.host_count = 0

    def attr(self, name: str) -> wp.array:
        return self.attrs[name]

    def refresh_host_count(self) -> int:
        self.host_count = int(self.count.numpy()[0])
        return self.host_count

    def clear(self):
        self.count.zero_()
        self.host_count = 0

    def compact(self):
        """Remove dead particles (alive==0) with a GPU scan + scatter."""
        n = self.capacity
        wp.utils.array_scan(self.alive, self.scan_tmp, inclusive=False)
        wp.launch(
            k_last_offset_count,
            dim=1,
            inputs=[self.alive, self.scan_tmp, self.count, self._count2],
            device=self.device,
        )
        wp.launch(
            k_compact_alive_scatter,
            dim=n,
            inputs=[
                self.alive, self.scan_tmp, self.count,
                self.position, self.velocity,
                self.attrs["age"], self.attrs["life"], self.attrs["temperature"],
                self.attrs["viscosity"], self.attrs["density"],
                self._position2, self._velocity2,
                self._attrs2["age"], self._attrs2["life"], self._attrs2["temperature"],
                self._attrs2["viscosity"], self._attrs2["density"],
                self._count2,
            ],
            device=self.device,
        )
        # swap buffers
        self.position, self._position2 = self._position2, self.position
        self.velocity, self._velocity2 = self._velocity2, self.velocity
        for k in self.ATTR_NAMES:
            self.attrs[k], self._attrs2[k] = self._attrs2[k], self.attrs[k]
        wp.copy(self.count, self._count2)
        # Re-mark the surviving prefix as alive so the next scan is valid.
        wp.launch(k_reset_alive, dim=self.capacity,
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
