"""GPU-resident artistic local mixing for two miscible clear-dye species."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME
from .fields import voxel_flat_index, voxel_floor, world_to_voxel
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
    particle_preview_bytes: int = 0
    voxel_scratch_bytes: int = 0
    receipt_bytes: int = 0
    host_cache_staging_bytes: int = 0

    @property
    def device_persistent_bytes(self) -> int:
        return int(
            self.particle_channels_bytes
            + self.particle_preview_bytes
            + self.voxel_scratch_bytes
            + self.receipt_bytes
        )

    @property
    def particle_resize_candidate_bytes(self) -> int:
        """Incremental species bytes allocated by a particle-capacity grow."""
        return int(self.particle_channels_bytes + self.particle_preview_bytes)


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
        # GPU preview colors are persistent because callers consume them after
        # the deriving launch returns.
        particle_preview_bytes=particle_capacity * _VEC4_BYTES,
        # The artistic local exchange needs only represented volume and species
        # amount on the sparse grid. There is no face or limiter scratch.
        voxel_scratch_bytes=voxel_capacity * (_FLOAT32_BYTES + _VEC4_BYTES),
        receipt_bytes=RECEIPT_SIZE * _FLOAT32_BYTES,
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
    # Legacy cache/report field names retained at this compatibility boundary.
    # Local mixing has no limiter and executes once per admitted solver substep.
    limiter_alpha: float
    diffusion_substeps: int


@dataclass(frozen=True, eq=False)
class ExchangeToken:
    """Identifies one staged exchange owned by one global solver substep."""

    _transport_identity: object
    serial: int
    frame: int
    substep: int


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
def species_local_mixture_at_position(
    position: wp.vec3,
    leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, origin: wp.vec3, voxel_size: float,
    volume: wp.array(dtype=float), mass: wp.array(dtype=wp.vec4),
    fallback: wp.vec4,
) -> wp.vec4:
    q = world_to_voxel(position, origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    represented = float(0.0)
    represented_mass = wp.vec4(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz))
                if flat >= 0 and volume[flat] > EPS:
                    weight = (wp.where(dx == 1, f[0], 1.0 - f[0])
                              * wp.where(dy == 1, f[1], 1.0 - f[1])
                              * wp.where(dz == 1, f[2], 1.0 - f[2]))
                    represented += weight * volume[flat]
                    represented_mass += weight * mass[flat]
    if represented > EPS:
        local = represented_mass / represented
        if (not wp.isnan(local[0]) and not wp.isinf(local[0])
                and not wp.isnan(local[1]) and not wp.isinf(local[1])):
            return wp.vec4(local[0], local[1], 0.0, 0.0)
    return fallback


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
def k_species_surface_color(
    vertices_world: wp.array(dtype=wp.vec3), vertex_count: int,
    leaf_table: wp.array(dtype=wp.int32), voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i, origin: wp.vec3, voxel_size: float,
    volume: wp.array(dtype=float), mass: wp.array(dtype=wp.vec4),
    absorption_a: wp.vec3, absorption_b: wp.vec3, path: float,
    colors: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid >= vertex_count:
        return
    q = world_to_voxel(vertices_world[tid], origin, voxel_size) - wp.vec3(0.5)
    base = voxel_floor(q)
    f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    represented = float(0.0)
    represented_mass = wp.vec4(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                flat = voxel_flat_index(
                    leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz))
                if flat >= 0:
                    weight = (wp.where(dx == 1, f[0], 1.0 - f[0])
                              * wp.where(dy == 1, f[1], 1.0 - f[1])
                              * wp.where(dz == 1, f[2], 1.0 - f[2]))
                    represented += weight * volume[flat]
                    represented_mass += weight * mass[flat]
    concentration = wp.vec4(0.0)
    if represented > EPS:
        concentration = represented_mass / represented
    absorb = path * (
        concentration[0] * absorption_a + concentration[1] * absorption_b)
    colors[tid] = wp.vec4(
        wp.exp(-absorb[0]), wp.exp(-absorb[1]), wp.exp(-absorb[2]), 1.0)


@wp.kernel
def k_species_surface_field_clear(
    num_leaves: wp.array(dtype=wp.int32), volume: wp.array(dtype=float),
    mass: wp.array(dtype=wp.vec4),
):
    tid = wp.tid()
    if tid < num_leaves[0] * LEAF_VOLUME:
        volume[tid] = 0.0
        mass[tid] = wp.vec4(0.0)


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


register_fn(k_species_p2g_accumulate, "species_p2g_accumulate")
register_fn(k_species_bounds_budget_reduce, "species_bounds_budget_reduce")
register_fn(k_species_preview_derive, "species_preview_derive")
register_fn(k_species_surface_color, "species_surface_color")
register_fn(k_species_surface_field_clear, "species_surface_field_clear")


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
            raise ValueError("species local mixing rate must be finite and nonnegative")
        if self.max_diffusion_substeps < 1:
            raise ValueError("legacy species step limit must be positive")
        pc, vc = particles.capacity, grid.voxel_capacity
        self.particle_preview = wp.zeros(pc, dtype=wp.vec4, device=device)
        self.mix_volume = wp.zeros(vc, dtype=float, device=device)
        self.species_mass = wp.zeros(vc, dtype=wp.vec4, device=device)
        # Low-level fixtures may disable collision masking and reuse this
        # existing full-size scratch array as the unused kernel placeholder.
        # Production passes the solver-owned collision_sdf.data by identity.
        self.collision_sdf = (
            collision_sdf if collision_sdf is not None else self.mix_volume)
        if len(self.collision_sdf) < vc:
            raise ValueError("species collision SDF must cover voxel capacity")
        self.collision_mask_enabled = bool(collision_mask_enabled)
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
        self._transport_identity = object()
        self._exchange_serial = 0
        self._pending_exchange = None
        self._exchange_phase = None
        self._pending_exchange_fraction = 0.0
        self._noop_exchange_owner = None
        self._surface_scratch_key = None
        self._surface_volume = None
        self._surface_mass = None
        self._direct_exchange_state = None
        self._direct_exchange_index = 0
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
        self._pending_exchange = None
        self._exchange_phase = None
        self._pending_exchange_fraction = 0.0
        self._noop_exchange_owner = None
        self._surface_scratch_key = None
        self._surface_volume = None
        self._surface_mass = None
        self._direct_exchange_state = None
        self._direct_exchange_index = 0
        self._frame_open = True
        self._after_captured = False

    def reset(self):
        for value in (self.particle_preview, self.mix_volume, self.species_mass):
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

        particle_preview = wp.zeros(
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
        grown.particle_preview = particle_preview
        for name in ("mix_volume", "species_mass", "_receipt"):
            setattr(grown, name, getattr(self, name))
        grown._bounds_reducer = getattr(self, "_bounds_reducer", None)
        grown.last_bounds_reducer_error = getattr(
            self, "last_bounds_reducer_error", None)
        grown._frame_open = self._frame_open
        grown._after_captured = self._after_captured
        grown._last_receipt = self._last_receipt
        grown.frame_diffusion_substeps = self.frame_diffusion_substeps
        grown._transport_identity = object()
        grown._exchange_serial = self._exchange_serial
        grown._pending_exchange = None
        grown._exchange_phase = None
        grown._pending_exchange_fraction = 0.0
        grown._noop_exchange_owner = None
        grown._surface_scratch_key = None
        grown._surface_volume = None
        grown._surface_mass = None
        grown._direct_exchange_state = None
        grown._direct_exchange_index = self._direct_exchange_index
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

    def begin_exchange(self, dt, rate=None, *, frame=-1, substep=-1):
        """Open one artistic local-mixture exchange owned by a solver substep."""
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("species local mixing dt must be finite and nonnegative")
        rate = self.diffusivity if rate is None else float(rate)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("species local mixing rate must be finite and nonnegative")
        owner = (int(frame), int(substep))
        if (self._pending_exchange is not None
                or self._noop_exchange_owner is not None):
            raise RuntimeError("species exchange is already pending")
        self._exchange_serial += 1
        if rate == 0.0 or dt == 0.0:
            self._noop_exchange_owner = (*owner, self._exchange_serial)
            return None
        self.begin_frame()
        token = ExchangeToken(
            self._transport_identity, self._exchange_serial, *owner)
        self._pending_exchange = token
        self._exchange_phase = "deposit_pending"
        self._pending_exchange_fraction = max(
            0.0, min(1.0, -math.expm1(-rate * dt)))
        return token

    def _validate_exchange_token(self, token, frame, substep):
        if not isinstance(token, ExchangeToken):
            raise RuntimeError("species exchange token is foreign")
        if token._transport_identity is not self._transport_identity:
            raise RuntimeError("species exchange token is foreign")
        if token is not self._pending_exchange:
            raise RuntimeError("species exchange token is stale")
        if (token.frame, token.substep) != (int(frame), int(substep)):
            raise RuntimeError("species exchange token owner does not match")

    def complete_exchange_deposit(self, token, *, frame=-1, substep=-1):
        """Record completion of the same-stream fused P2G mixture deposit."""
        self._validate_exchange_token(token, frame, substep)
        if self._exchange_phase != "deposit_pending":
            raise RuntimeError("species exchange deposit is out of order")
        self._exchange_phase = "deposit_complete"

    def prepare_exchange_g2p(self, token, *, frame=-1, substep=-1):
        """Return the fixed blend fraction and admit the fused G2P launch."""
        self._validate_exchange_token(token, frame, substep)
        if self._exchange_phase != "deposit_complete":
            raise RuntimeError("species exchange G2P stage is out of order")
        self._exchange_phase = "g2p_pending"
        return self._pending_exchange_fraction

    def finish_exchange(self, token, *, frame=-1, substep=-1):
        """Close only after the same-stream fused local-mixture G2P is queued."""
        if token is None:
            if self._pending_exchange is not None:
                raise RuntimeError("None cannot finish a pending species exchange")
            owner = self._noop_exchange_owner
            if owner is None or owner[:2] != (int(frame), int(substep)):
                raise RuntimeError("no-op species exchange finish is stale")
            self._noop_exchange_owner = None
            return
        self._validate_exchange_token(token, frame, substep)
        if self._exchange_phase != "g2p_pending":
            raise RuntimeError("species exchange G2P is incomplete")
        self._pending_exchange = None
        self._exchange_phase = None
        self._pending_exchange_fraction = 0.0
        self.frame_diffusion_substeps += 1

    def diffuse(self, dt):
        """Legacy-named driver exercising only fused P2G/G2P mixture work."""
        from types import SimpleNamespace
        from .kernels_fluid import (
            k_particle_set_grid_data_with_species,
            k_staggered_field_to_particle_with_species,
        )

        state = self._direct_exchange_state
        if state is None:
            capacity = self.grid.voxel_capacity
            state = SimpleNamespace(
                u=wp.zeros(capacity, dtype=float, device=self.device),
                v=wp.zeros(capacity, dtype=float, device=self.device),
                w=wp.zeros(capacity, dtype=float, device=self.device),
                wu=wp.zeros(capacity, dtype=float, device=self.device),
                wv=wp.zeros(capacity, dtype=float, device=self.device),
                ww=wp.zeros(capacity, dtype=float, device=self.device),
                old_u=wp.zeros(capacity, dtype=float, device=self.device),
                old_v=wp.zeros(capacity, dtype=float, device=self.device),
                old_w=wp.zeros(capacity, dtype=float, device=self.device),
                sampled_velocity=wp.zeros(
                    self.particles.capacity, dtype=wp.vec3, device=self.device),
            )
            self._direct_exchange_state = state
        owner = self._direct_exchange_index
        self._direct_exchange_index += 1
        token = self.begin_exchange(
            dt, self.diffusivity, frame=-2, substep=owner)
        if token is None:
            self.finish_exchange(None, frame=-2, substep=owner)
            return
        p, g = self.particles, self.grid
        voxel_dim = self._voxel_dim()
        self.mix_volume[:voxel_dim].zero_()
        self.species_mass[:voxel_dim].zero_()
        wp.launch(k_particle_set_grid_data_with_species, dim=self._particle_dim(),
            inputs=[p.position, p.velocity, p.attr("represented_volume"),
                    p.attr("species_concentration"), p.count, g.leaf_table,
                    g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
                    state.u, state.v, state.w, state.wu, state.wv, state.ww,
                    self.mix_volume, self.species_mass], device=self.device)
        self.complete_exchange_deposit(token, frame=-2, substep=owner)
        fraction = self.prepare_exchange_g2p(token, frame=-2, substep=owner)
        wp.launch(k_staggered_field_to_particle_with_species,
            dim=self._particle_dim(), inputs=[
                p.position, p.count, state.u, state.v, state.w,
                state.wu, state.wv, state.ww,
                state.old_u, state.old_v, state.old_w,
                g.leaf_table, g.voxels_dim, g.leaves_dim, g.origin,
                g.voxel_size, 0, 0.0, self.mix_volume, self.species_mass,
                fraction, p.attr("species_concentration"),
                state.sampled_velocity], device=self.device)
        self.finish_exchange(token, frame=-2, substep=owner)

    def _surface_scratch(self, grid):
        key = (id(grid), int(grid.voxel_capacity))
        if self._surface_scratch_key != key:
            self._surface_volume = wp.zeros(
                grid.voxel_capacity, dtype=float, device=self.device)
            self._surface_mass = wp.zeros(
                grid.voxel_capacity, dtype=wp.vec4, device=self.device)
            self._surface_scratch_key = key
        return self._surface_volume, self._surface_mass

    def derive_surface_colors(self, vertices_world, vertex_count, *, grid=None):
        """Derive post-advection Beer-Lambert colors without particle readback."""
        vertex_count = max(0, int(vertex_count))
        colors = wp.empty(vertex_count, dtype=wp.vec4, device=self.device)
        if vertex_count == 0:
            return colors
        if self._pending_exchange is not None:
            raise RuntimeError("surface colors requested during a pending exchange")
        p, g = self.particles, (self.grid if grid is None else grid)
        volume, mass = self._surface_scratch(g)
        voxel_dim = max(1, int(g.active_voxel_count))
        wp.launch(k_species_surface_field_clear, dim=voxel_dim, inputs=[
            g.num_leaves, volume, mass], device=self.device)
        wp.launch(k_species_p2g_accumulate, dim=self._particle_dim(), inputs=[
            p.position, p.attr("represented_volume"),
            p.attr("species_concentration"), p.count, g.leaf_table,
            g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
            volume, mass], device=self.device)
        absorption = self.palette.absorption_rgb
        wp.launch(k_species_surface_color, dim=vertex_count, inputs=[
            vertices_world, vertex_count, g.leaf_table, g.voxels_dim,
            g.leaves_dim, g.origin, g.voxel_size, volume,
            mass, wp.vec3(*absorption[0]),
            wp.vec3(*absorption[1]), self.palette.reference_path_length,
            colors], device=self.device)
        return colors

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
            self.particle_preview], device=self.device)
        return self.particle_preview

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
