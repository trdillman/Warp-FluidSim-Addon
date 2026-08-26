"""GPU-resident conservative transport for two miscible clear-dye species."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME
from .fields import positive_face_neighbor, voxel_flat_index, voxel_floor, world_to_voxel
from .registry import register_fn


EPS = 1.0e-12
BEFORE, SOURCES, SINKS, AFTER = 0, 2, 4, 6
MINIMUM, MAX_LOADING, MIN_ALPHA, RECEIPT_SIZE = 8, 9, 10, 11
_FLOAT32_BYTES = 4
_VEC4_BYTES = 16


@dataclass(frozen=True)
class SpeciesTransportAllocationBytes:
    """Pure-integer inventory matching one SpeciesTransport allocation."""

    particle_channels_bytes: int = 0
    particle_delta_bytes: int = 0
    voxel_scratch_bytes: int = 0
    receipt_bytes: int = 0
    host_cache_staging_bytes: int = 0

    @property
    def device_persistent_bytes(self) -> int:
        return int(
            self.particle_channels_bytes
            + self.particle_delta_bytes
            + self.voxel_scratch_bytes
            + self.receipt_bytes
        )

    @property
    def particle_resize_candidate_bytes(self) -> int:
        """Incremental species bytes allocated by a particle-capacity grow."""
        return int(self.particle_channels_bytes + self.particle_delta_bytes)


def species_transport_allocation_bytes(
        particle_capacity, voxel_capacity, enabled
) -> SpeciesTransportAllocationBytes:
    """Return the exact named bytes allocated for live species transport."""
    if not enabled:
        return SpeciesTransportAllocationBytes()
    particle_capacity = max(0, int(particle_capacity))
    voxel_capacity = max(0, int(voxel_capacity))
    return SpeciesTransportAllocationBytes(
        # ParticleSystem owns primary and compaction buffers for both channels.
        particle_channels_bytes=(
            particle_capacity * 2 * (_FLOAT32_BYTES + _VEC4_BYTES)),
        particle_delta_bytes=particle_capacity * _VEC4_BYTES,
        # mix_volume/outgoing plus species_mass/delta. Positive-face validity
        # and raw conductance are recomputed from sparse topology in the flux
        # kernel rather than persisted in per-face scratch arrays.
        voxel_scratch_bytes=voxel_capacity * (
            2 * _FLOAT32_BYTES + 2 * _VEC4_BYTES),
        receipt_bytes=(1 + RECEIPT_SIZE) * _FLOAT32_BYTES,
        # Cache/checkpoint host staging is owned outside SpeciesTransport.
        host_cache_staging_bytes=0,
    )


@dataclass(frozen=True)
class SpeciesPalette:
    ids: tuple[str, str]
    absorption_rgb: tuple[tuple[float, float, float], tuple[float, float, float]]
    reference_path_length: float
    palette_hash: str

    @classmethod
    def create(cls, *, ids, absorption_rgb, reference_path_length):
        ids = tuple(str(value) for value in ids)
        absorption = tuple(tuple(float(v) for v in row) for row in absorption_rgb)
        path = float(reference_path_length)
        if len(ids) != 2 or any(not value for value in ids) or ids[0] == ids[1]:
            raise ValueError("species palette requires two distinct stable IDs")
        if (len(absorption) != 2 or any(len(row) != 3 for row in absorption)
                or any(not math.isfinite(v) or v < 0.0
                       for row in absorption for v in row)):
            raise ValueError("species absorption must be two nonnegative finite RGB triples")
        if not math.isfinite(path) or path <= 0.0:
            raise ValueError("species reference path length must be finite and positive")
        payload = {"ids": ids, "absorption_rgb": absorption,
                   "reference_path_length": path}
        digest = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cls(ids, absorption, path, digest)


@dataclass(frozen=True)
class SpeciesReceipt:
    before: tuple[float, float]
    sources: tuple[float, float]
    sinks: tuple[float, float]
    after: tuple[float, float]
    residual: tuple[float, float]
    min_concentration: float
    max_loading: float
    limiter_alpha: float
    diffusion_substeps: int


@wp.kernel
def k_species_p2g_clear(
    num_leaves: wp.array(dtype=wp.int32),
    mix_volume: wp.array(dtype=float), mass: wp.array(dtype=wp.vec4),
    delta: wp.array(dtype=wp.vec4), outgoing: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    mix_volume[tid] = 0.0
    mass[tid] = wp.vec4(0.0)
    delta[tid] = wp.vec4(0.0)
    outgoing[tid] = 0.0


@wp.kernel
def k_species_p2g_accumulate(
    positions: wp.array(dtype=wp.vec3), volumes: wp.array(dtype=float),
    concentration: wp.array(dtype=wp.vec4), count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, origin: wp.vec3, voxel_size: float,
    mix_volume: wp.array(dtype=float), mass: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    q = world_to_voxel(positions[tid], origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    volume, c = volumes[tid], concentration[tid]
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz))
                if flat >= 0:
                    weight = (wp.where(dx == 1, f[0], 1.0 - f[0])
                              * wp.where(dy == 1, f[1], 1.0 - f[1])
                              * wp.where(dz == 1, f[2], 1.0 - f[2]))
                    amount = volume * weight
                    wp.atomic_add(mix_volume, flat, amount)
                    wp.atomic_add(mass, flat, wp.vec4(
                        amount * c[0], amount * c[1], 0.0, 0.0))


@wp.func
def species_face_is_permeable(
    volume_i: float, volume_j: float,
    collision_i: float, collision_j: float,
    collision_mask_enabled: int,
) -> bool:
    """Return whether one liquid-liquid face can carry species flux."""
    if volume_i <= EPS or volume_j <= EPS:
        return False
    if collision_mask_enabled != 0:
        # Match the solver-wide collision SDF convention: strictly negative
        # samples are inside solid geometry; zero is a valid fluid boundary
        # value and must not suppress species diffusion.
        if collision_i < 0.0 or collision_j < 0.0:
            return False
    return True


@wp.kernel
def k_species_face_conductance_measure(
    volume: wp.array(dtype=float), leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, num_leaves: wp.array(dtype=wp.int32), raw: float,
    collision_sdf: wp.array(dtype=float), collision_mask_enabled: int,
    outgoing: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME or volume[tid] <= EPS:
        return
    for axis in range(3):
        neighbor = positive_face_neighbor(
            tid, axis, leaf_table, leaf_coords, voxels_dim, leaves_dim)
        if neighbor >= 0 and species_face_is_permeable(
                volume[tid], volume[neighbor],
                collision_sdf[tid], collision_sdf[neighbor],
                collision_mask_enabled):
            wp.atomic_add(outgoing, tid, raw)
            wp.atomic_add(outgoing, neighbor, raw)


@wp.kernel
def k_species_node_scale(
    volume: wp.array(dtype=float), outgoing: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32), scale: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if volume[tid] > EPS and outgoing[tid] > EPS:
        scale[tid] = wp.min(1.0, volume[tid] / outgoing[tid])
    else:
        scale[tid] = 0.0


@wp.kernel
def k_species_face_flux_apply(
    volume: wp.array(dtype=float), mass: wp.array(dtype=wp.vec4),
    outgoing: wp.array(dtype=float), leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, num_leaves: wp.array(dtype=wp.int32),
    raw: float, collision_sdf: wp.array(dtype=float),
    collision_mask_enabled: int, delta: wp.array(dtype=wp.vec4),
    global_alpha: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid == 0:
        global_alpha[0] = 1.0
    if tid >= num_leaves[0] * LEAF_VOLUME or volume[tid] <= EPS:
        return
    ci = mass[tid] / volume[tid]
    for axis in range(3):
        neighbor = positive_face_neighbor(
            tid, axis, leaf_table, leaf_coords, voxels_dim, leaves_dim)
        if neighbor < 0:
            continue
        if not species_face_is_permeable(
                volume[tid], volume[neighbor],
                collision_sdf[tid], collision_sdf[neighbor],
                collision_mask_enabled):
            continue
        cj = mass[neighbor] / volume[neighbor]
        scale_i = float(0.0)
        if outgoing[tid] > EPS:
            scale_i = wp.min(1.0, volume[tid] / outgoing[tid])
        scale_j = float(0.0)
        if outgoing[neighbor] > EPS:
            scale_j = wp.min(1.0, volume[neighbor] / outgoing[neighbor])
        conductance = raw * wp.min(scale_i, scale_j)
        change = wp.vec4(conductance * (cj[0] - ci[0]),
                         conductance * (cj[1] - ci[1]), 0.0, 0.0)
        wp.atomic_add(delta, tid, change)
        wp.atomic_add(delta, neighbor, -change)


@wp.func
def species_alpha_candidate(value: wp.vec4, d: wp.vec4) -> float:
    alpha = float(1.0)
    if (wp.isnan(d[0]) or wp.isinf(d[0])
            or wp.isnan(d[1]) or wp.isinf(d[1])):
        alpha = 0.0
    else:
        if d[0] < 0.0:
            alpha = wp.min(alpha, value[0] / -d[0])
        elif d[0] > 0.0:
            alpha = wp.min(alpha, (1.0 - value[0]) / d[0])
        if d[1] < 0.0:
            alpha = wp.min(alpha, value[1] / -d[1])
        elif d[1] > 0.0:
            alpha = wp.min(alpha, (1.0 - value[1]) / d[1])
        total_delta = d[0] + d[1]
        # Ignore sub-ULP cancellation left by tangent projection/interpolation.
        # Treating an O(1e-15) positive sum as real at a fully loaded particle
        # would force the global alpha to zero and freeze all diffusion.
        if total_delta > EPS:
            alpha = wp.min(alpha, (1.0 - value[0] - value[1]) / total_delta)
    return wp.max(0.0, wp.min(1.0, alpha))


@wp.kernel
def k_species_g2p_delta(
    positions: wp.array(dtype=wp.vec3), count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, origin: wp.vec3, voxel_size: float,
    volume: wp.array(dtype=float), grid_delta: wp.array(dtype=wp.vec4),
    concentration: wp.array(dtype=wp.vec4),
    particle_delta: wp.array(dtype=wp.vec4), global_alpha: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    q = world_to_voxel(positions[tid], origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    result = wp.vec4(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz))
                if flat >= 0 and volume[flat] > EPS:
                    weight = (wp.where(dx == 1, f[0], 1.0 - f[0])
                              * wp.where(dy == 1, f[1], 1.0 - f[1])
                              * wp.where(dz == 1, f[2], 1.0 - f[2]))
                    result += weight * grid_delta[flat] / volume[flat]
    # When a fully loaded mixture has equal-and-opposite species changes,
    # float32 scatter/interpolation can leave an O(1e-7) cancellation error.
    # Project only that numerically tangent case.  A single-species Gaussian
    # may also touch loading 1 at its peak, but its non-cancelling delta must
    # diffuse loading normally and must never be converted into species B.
    value = concentration[tid]
    d0, d1 = result[0], result[1]
    simplex_tolerance = float(1.0e-6)
    delta_l1 = wp.abs(d0) + wp.abs(d1)
    cancelling_full_mixture = (
        value[0] + value[1] >= 1.0 - simplex_tolerance
        and d0 * d1 < 0.0
        and (wp.abs(d0 + d1) <= simplex_tolerance
             or wp.abs(d0 + d1)
             <= 1.0e-3 * wp.max(delta_l1, EPS)))
    if cancelling_full_mixture:
        normal = 0.5 * (d0 + d1)
        d0 = d0 - normal
        d1 = d1 - normal
        if ((value[0] >= 1.0 - simplex_tolerance and d0 > 0.0)
                or (value[1] >= 1.0 - simplex_tolerance and d1 > 0.0)
                or (value[0] <= simplex_tolerance and d0 < 0.0)
                or (value[1] <= simplex_tolerance and d1 < 0.0)):
            d0, d1 = 0.0, 0.0
    # Atomic accumulation may also leave an isolated sub-float32 outward
    # delta at a pure simplex vertex (for example +3e-9 at c0 == 1).  Such a
    # direction violates the diffusion maximum principle and carries no
    # resolvable transport, so do not let it freeze the global limiter.
    delta_l1 = wp.abs(d0) + wp.abs(d1)
    if (delta_l1 <= simplex_tolerance
            and ((value[0] >= 1.0 - simplex_tolerance and d0 > 0.0)
                 or (value[1] >= 1.0 - simplex_tolerance and d1 > 0.0)
                 or (value[0] <= simplex_tolerance and d0 < 0.0)
                 or (value[1] <= simplex_tolerance and d1 < 0.0))):
        d0, d1 = 0.0, 0.0
    d = wp.vec4(d0, d1, 0.0, 0.0)
    particle_delta[tid] = d
    candidate = species_alpha_candidate(value, d)
    if candidate < 1.0:
        wp.atomic_min(global_alpha, 0, candidate)


@wp.kernel
def k_species_particle_alpha(
    c: wp.array(dtype=wp.vec4), delta: wp.array(dtype=wp.vec4),
    count: wp.array(dtype=wp.int32), candidates: wp.array(dtype=float),
    global_alpha: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    alpha = species_alpha_candidate(c[tid], delta[tid])
    candidates[tid] = alpha
    if alpha < 1.0:
        wp.atomic_min(global_alpha, 0, alpha)


@wp.kernel
def k_species_delta_apply(
    c: wp.array(dtype=wp.vec4), delta: wp.array(dtype=wp.vec4),
    count: wp.array(dtype=wp.int32), alpha: wp.array(dtype=float),
    receipt: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    value, d, scale = c[tid], delta[tid], alpha[0]
    updated0 = value[0] + scale * d[0]
    updated1 = value[1] + scale * d[1]
    # Full mixtures are projected onto the simplex tangent in G2P. Preserve
    # that invariant exactly here instead of accumulating independent float32
    # additions until viewport validation sees c0 + c1 > 1 + tolerance.
    if (value[0] + value[1] >= 1.0 - 1.0e-6
            and wp.abs(d[0] + d[1]) <= EPS):
        updated1 = 1.0 - updated0
    updated = wp.vec4(updated0, updated1, 0.0, 0.0)
    c[tid] = updated
    if tid == 0:
        wp.atomic_min(receipt, MIN_ALPHA, scale)


@wp.kernel
def k_species_bounds_budget_reduce(
    volume: wp.array(dtype=float), c: wp.array(dtype=wp.vec4),
    count: wp.array(dtype=wp.int32), offset: int,
    receipt: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    value = c[tid]
    wp.atomic_add(receipt, offset, volume[tid] * value[0])
    wp.atomic_add(receipt, offset + 1, volume[tid] * value[1])
    wp.atomic_min(receipt, MINIMUM, wp.min(value[0], value[1]))
    wp.atomic_max(receipt, MAX_LOADING, value[0] + value[1])


@wp.kernel
def k_species_preview_derive(
    c: wp.array(dtype=wp.vec4), count: int, absorption_a: wp.vec3,
    absorption_b: wp.vec3, path: float, preview: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid >= count:
        return
    value = c[tid]
    absorb = path * (value[0] * absorption_a + value[1] * absorption_b)
    preview[tid] = wp.vec4(wp.exp(-absorb[0]), wp.exp(-absorb[1]),
                           wp.exp(-absorb[2]), 1.0)


@wp.kernel
def _k_init_alpha(alpha: wp.array(dtype=float)):
    alpha[0] = 1.0


@wp.kernel
def _k_init_receipt(receipt: wp.array(dtype=float)):
    tid = wp.tid()
    if tid < RECEIPT_SIZE:
        receipt[tid] = 0.0
    if tid == MINIMUM:
        receipt[tid] = 3.402823466e38
    elif tid == MIN_ALPHA:
        receipt[tid] = 1.0


@wp.kernel
def _k_record_amount(receipt: wp.array(dtype=float), offset: int,
                     x: float, y: float):
    receipt[offset] = receipt[offset] + x
    receipt[offset + 1] = receipt[offset + 1] + y


@wp.kernel
def _k_record_amount_device(
    receipt: wp.array(dtype=float), offset: int,
    amount: wp.array(dtype=float),
):
    receipt[offset] = receipt[offset] + amount[0]
    receipt[offset + 1] = receipt[offset + 1] + amount[1]


register_fn(k_species_p2g_clear, "species_p2g_clear")
register_fn(k_species_p2g_accumulate, "species_p2g_accumulate")
register_fn(k_species_face_conductance_measure, "species_face_conductance_measure")
register_fn(k_species_node_scale, "species_node_scale")
register_fn(k_species_face_flux_apply, "species_face_flux_apply")
register_fn(k_species_g2p_delta, "species_g2p_delta")
register_fn(k_species_particle_alpha, "species_particle_alpha")
register_fn(k_species_delta_apply, "species_delta_apply")
register_fn(k_species_bounds_budget_reduce, "species_bounds_budget_reduce")
register_fn(k_species_preview_derive, "species_preview_derive")


class SpeciesTransport:
    def __init__(self, particles, grid, device, *, enabled: bool,
                 diffusivity: float, max_diffusion_substeps: int,
                 palette: SpeciesPalette, collision_sdf=None,
                 collision_mask_enabled: bool = False):
        if not enabled:
            raise ValueError("SpeciesTransport must not be instantiated when disabled")
        if not (particles.has_optional_channel("represented_volume")
                and particles.has_optional_channel("species_concentration")):
            raise ValueError("species transport requires both particle channels")
        self.particles, self.grid, self.device = particles, grid, device
        self.diffusivity = float(diffusivity)
        self.max_diffusion_substeps = int(max_diffusion_substeps)
        self.palette = palette
        if not math.isfinite(self.diffusivity) or self.diffusivity < 0.0:
            raise ValueError("species diffusivity must be finite and nonnegative")
        if self.max_diffusion_substeps < 1:
            raise ValueError("species maximum diffusion substeps must be positive")
        pc, vc = particles.capacity, grid.voxel_capacity
        self.particle_delta = wp.zeros(pc, dtype=wp.vec4, device=device)
        self.mix_volume = wp.zeros(vc, dtype=float, device=device)
        self.species_mass = wp.zeros(vc, dtype=wp.vec4, device=device)
        self.species_delta = wp.zeros(vc, dtype=wp.vec4, device=device)
        self.outgoing_conductance = wp.zeros(vc, dtype=float, device=device)
        # Low-level fixtures may disable collision masking and reuse this
        # existing full-size scratch array as the unused kernel placeholder.
        # Production passes the solver-owned collision_sdf.data by identity.
        self.collision_sdf = (
            collision_sdf if collision_sdf is not None else self.mix_volume)
        if len(self.collision_sdf) < vc:
            raise ValueError("species collision SDF must cover voxel capacity")
        self.collision_mask_enabled = bool(collision_mask_enabled)
        self.global_alpha = wp.zeros(1, dtype=float, device=device)
        self._receipt = wp.zeros(RECEIPT_SIZE, dtype=float, device=device)
        self._bounds_reducer = None
        self.last_bounds_reducer_error = ""
        if wp.get_device(device).is_cuda:
            try:
                from .cuda_reduction import CudaSpeciesBoundsReduction

                self._bounds_reducer = CudaSpeciesBoundsReduction(device)
            except Exception as exc:
                self.last_bounds_reducer_error = f"{type(exc).__name__}: {exc}"
        self._frame_open = False
        self._after_captured = False
        self._last_receipt = None
        self.frame_diffusion_substeps = 0
        self.reset()

    @staticmethod
    def estimate_bytes(particle_capacity, voxel_capacity, enabled):
        return species_transport_allocation_bytes(
            particle_capacity, voxel_capacity, enabled).device_persistent_bytes

    def _particle_dim(self):
        return max(1, min(self.particles.capacity, self.particles.host_count))

    def _voxel_dim(self):
        return max(1, self.grid.active_voxel_count)

    def _init_frame(self):
        wp.launch(_k_init_receipt, dim=RECEIPT_SIZE,
                  inputs=[self._receipt], device=self.device)
        self.frame_diffusion_substeps = 0
        self._frame_open = True
        self._after_captured = False

    def reset(self):
        for value in (self.particle_delta,
                      self.mix_volume, self.species_mass, self.species_delta,
                      self.outgoing_conductance, self.global_alpha):
            value.zero_()
        self._frame_open = False
        self._after_captured = False
        self._last_receipt = None
        self.frame_diffusion_substeps = 0

    def resized_copy(self, particles):
        """Create a growth candidate without duplicating grid-axis resources.

        Capacity growth is an atomic copy/swap transaction.  Until the swap,
        this candidate is never dispatched, so the unchanged grid scratch and
        receipt arrays can be shared by exact Python object identity.  On
        rollback the authoritative transport retains them; on commit the new
        transport retains them after the old wrapper is released.
        """
        if int(particles.capacity) <= int(self.particles.capacity):
            raise ValueError("resized species capacity must be larger")
        if not (particles.has_optional_channel("represented_volume")
                and particles.has_optional_channel("species_concentration")):
            raise ValueError("species transport requires both particle channels")

        particle_delta = wp.zeros(
            int(particles.capacity), dtype=wp.vec4, device=self.device)
        grown = object.__new__(type(self))
        grown.particles = particles
        grown.grid = self.grid
        grown.device = self.device
        grown.diffusivity = self.diffusivity
        grown.max_diffusion_substeps = self.max_diffusion_substeps
        grown.palette = self.palette
        grown.collision_sdf = getattr(self, "collision_sdf", self.mix_volume)
        grown.collision_mask_enabled = bool(getattr(
            self, "collision_mask_enabled", False))
        grown.particle_delta = particle_delta
        for name in (
                "mix_volume", "species_mass", "species_delta",
                "outgoing_conductance", "global_alpha", "_receipt"):
            setattr(grown, name, getattr(self, name))
        grown._bounds_reducer = getattr(self, "_bounds_reducer", None)
        grown.last_bounds_reducer_error = getattr(
            self, "last_bounds_reducer_error", None)
        grown._frame_open = self._frame_open
        grown._after_captured = self._after_captured
        grown._last_receipt = self._last_receipt
        grown.frame_diffusion_substeps = self.frame_diffusion_substeps
        return grown

    def _reduce(self, offset):
        if self._bounds_reducer is not None:
            self._bounds_reducer.launch(
                self.particles.attr("represented_volume"),
                self.particles.attr("species_concentration"),
                self.particles.count, self._receipt,
                launch_bound=self._particle_dim(), after_offset=offset,
                minimum_offset=MINIMUM, max_loading_offset=MAX_LOADING)
        else:
            wp.launch(k_species_bounds_budget_reduce, dim=self._particle_dim(),
                      inputs=[self.particles.attr("represented_volume"),
                              self.particles.attr("species_concentration"),
                              self.particles.count, offset, self._receipt],
                      device=self.device)

    def begin_frame(self):
        """Open one receipt ledger and capture its pre-mutation budget."""
        if not self._frame_open:
            self._init_frame()
            if self._last_receipt is None:
                self._reduce(BEFORE)

    def diffuse(self, dt):
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("species diffusion dt must be finite and nonnegative")
        h = float(self.grid.voxel_size)
        count = (max(1, int(math.ceil(6.0 * self.diffusivity * dt / (h * h))))
                 if self.diffusivity > 0.0 and dt > 0.0 else 0)
        if count > self.max_diffusion_substeps:
            raise ValueError("species diffusion substep demand exceeds admission")
        if count == 0:
            return
        self.begin_frame()
        p, g = self.particles, self.grid
        raw = self.diffusivity * (dt / count) * h
        for _ in range(count):
            wp.launch(k_species_p2g_clear, dim=self._voxel_dim(), inputs=[
                g.num_leaves, self.mix_volume, self.species_mass,
                self.species_delta, self.outgoing_conductance],
                device=self.device)
            wp.launch(k_species_p2g_accumulate, dim=self._particle_dim(), inputs=[
                p.position, p.attr("represented_volume"),
                p.attr("species_concentration"), p.count, g.leaf_table,
                g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
                self.mix_volume, self.species_mass], device=self.device)
            wp.launch(k_species_face_conductance_measure, dim=self._voxel_dim(),
                inputs=[self.mix_volume, g.leaf_table, g.leaf_coords,
                        g.voxels_dim, g.leaves_dim, g.num_leaves, raw,
                        self.collision_sdf, int(self.collision_mask_enabled),
                        self.outgoing_conductance],
                device=self.device)
            wp.launch(k_species_face_flux_apply, dim=self._voxel_dim(), inputs=[
                self.mix_volume, self.species_mass, self.outgoing_conductance,
                g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                g.num_leaves, raw, self.collision_sdf,
                int(self.collision_mask_enabled), self.species_delta,
                self.global_alpha],
                device=self.device)
            wp.launch(k_species_g2p_delta, dim=self._particle_dim(), inputs=[
                p.position, p.count, g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size, self.mix_volume, self.species_delta,
                p.attr("species_concentration"), self.particle_delta,
                self.global_alpha], device=self.device)
            wp.launch(k_species_delta_apply, dim=self._particle_dim(), inputs=[
                p.attr("species_concentration"), self.particle_delta, p.count,
                self.global_alpha, self._receipt], device=self.device)
        self.frame_diffusion_substeps += count

    def record_source(self, amount):
        self.begin_frame()
        wp.launch(_k_record_amount, dim=1,
                  inputs=[self._receipt, SOURCES, float(amount[0]), float(amount[1])],
                  device=self.device)

    def record_source_device(self, amount):
        """Accumulate a two-value device budget without a host synchronization."""
        self.begin_frame()
        wp.launch(_k_record_amount_device, dim=1,
                  inputs=[self._receipt, SOURCES, amount], device=self.device)

    def record_sink(self, amount):
        self.begin_frame()
        wp.launch(_k_record_amount, dim=1,
                  inputs=[self._receipt, SINKS, float(amount[0]), float(amount[1])],
                  device=self.device)

    @property
    def bounds_reduction_backend(self):
        return "native_cuda" if self._bounds_reducer is not None else "warp_atomic"

    def compaction_sink_ledger(self):
        """Open the frame and expose only sink accounting to compaction."""
        self.begin_frame()
        return self._receipt, SINKS

    def capture_compacted_after(self):
        """Reduce the actual current particle state after compaction."""
        self.begin_frame()
        self._reduce(AFTER)
        self._after_captured = True

    def finalize_frame_receipt(self):
        if not self._frame_open and self._last_receipt is not None:
            last = self._last_receipt
            return SpeciesReceipt(
                last.after, (0.0, 0.0), (0.0, 0.0), last.after,
                (0.0, 0.0), last.min_concentration, last.max_loading,
                1.0, 0)
        if not self._frame_open and self.particles.host_count <= 0:
            receipt = SpeciesReceipt(
                (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                (0.0, 0.0), 0.0, 0.0, 1.0, 0)
            self._last_receipt = receipt
            return receipt
        self.begin_frame()
        if not self._after_captured:
            self._reduce(AFTER)
        values = self._receipt.numpy()
        sources, sinks = tuple(values[2:4]), tuple(values[4:6])
        before = (self._last_receipt.after if self._last_receipt is not None
                  else tuple(values[0:2]))
        after = tuple(values[6:8])
        residual = (after[0] - before[0] - sources[0] + sinks[0],
                    after[1] - before[1] - sources[1] + sinks[1])
        minimum = float(values[MINIMUM])
        receipt = SpeciesReceipt(
            before, sources, sinks, after, residual,
            0.0 if minimum > 1.0e30 else minimum,
            float(values[MAX_LOADING]), float(values[MIN_ALPHA]),
            self.frame_diffusion_substeps)
        self._frame_open = False
        self._after_captured = False
        self._last_receipt = receipt
        return receipt

    def derive_preview(self, count):
        count = max(0, min(int(count), self.particles.capacity))
        a = self.palette.absorption_rgb
        wp.launch(k_species_preview_derive, dim=max(1, count), inputs=[
            self.particles.attr("species_concentration"), count,
            wp.vec3(*a[0]), wp.vec3(*a[1]), self.palette.reference_path_length,
            self.particle_delta], device=self.device)
        return self.particle_delta

    def checkpoint_arrays(self, count):
        return {"particle_represented_volume": np.ascontiguousarray(
                    self.particles.attr("represented_volume").numpy()[:count]),
                "particle_species_concentration": np.ascontiguousarray(
                    self.particles.attr("species_concentration").numpy()[:count])}

    def checkpoint_scalars(self):
        return {"species_schema_version": 1,
                "species_palette_hash": self.palette.palette_hash}

    def restore_checkpoint(self, arrays, scalars, count):
        if (scalars.get("species_schema_version") != 1
                or scalars.get("species_palette_hash") != self.palette.palette_hash):
            raise ValueError("species resume state is incompatible; regenerate the cache")
        volume = np.asarray(arrays["particle_represented_volume"], dtype=np.float32)
        c = np.asarray(arrays["particle_species_concentration"], dtype=np.float32)
        if volume.shape != (count,) or c.shape != (count, 4):
            raise ValueError("species resume state is incompatible; regenerate the cache")
        self.particles.validate_species_birth_state(volume, c)
        wp.copy(self.particles.attr("represented_volume"),
                wp.array(volume, dtype=float, device=self.device), count=count)
        wp.copy(self.particles.attr("species_concentration"),
                wp.array(c, dtype=wp.vec4, device=self.device), count=count)
        self.reset()
