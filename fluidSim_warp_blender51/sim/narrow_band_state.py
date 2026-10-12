"""Device state and per-substep driver for narrow-band FLIP (lane J).

``FluidSolver`` creates a :class:`NarrowBand` only when ``narrow_band_enabled``
is on; with the mode off this module is not even imported by a running solve,
so none of its allocations or launches exist.  The pure rules (what the mode
refuses to combine with) live in ``narrow_band.py``; the kernels in
``kernels_narrow_band.py``.

The persistent liquid level set ``phi`` (metres, negative inside) is the only
description of the deep interior: the particles live in a thin band under the
free surface and the rest of the liquid is carried by ``phi`` plus a persisted
MAC velocity field.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from . import constants as C
from .checkpoint_state import host_prefix
from .fastlaunch import launch as fast_launch
from .kernels_narrow_band import (
    k_nb_advect_scalar,
    k_nb_adjust_interior_counts,
    k_nb_advect_velocity,
    k_nb_classify,
    k_nb_eikonal_inward,
    k_nb_fill_interior_velocity,
    k_nb_kill_interior,
    k_nb_mark_leaves_from_phi,
    k_nb_mark_render_leaves,
    k_nb_remap_leaf_field_default,
    k_nb_reseed_commit,
    k_nb_reseed_demand,
    k_nb_reseed_reserve,
    k_nb_set_fluid_from_phi,
    k_nb_store_velocity,
    k_nb_union_band,
    k_nb_union_into_render_sdf,
    k_nb_union_into_solver_sdf,
)
from .surface import k_build_surface_sdf, k_surface_sdf_reset

# Seed mix: frame and substep decorrelate the stream, the constant keeps a
# zero frame from seeding zero.
_SEED_FRAME = 1000003
_SEED_SUBSTEP = 7919
_SEED_BASE = 0x9E3779B1
_SEED_MASK = 0x7FFFFFFF

# Added to interior voxel counts around sourcing; far above any emit_rate.
INTERIOR_COUNT_CREDIT = 1 << 16

# A band cell holding fewer particles than this is reseeded (a sparse cell).
# Reseeding is also capped by the particles the band has killed (``owed``).
RESEED_TRIGGER = 2


class NarrowBand:
    """Persistent level set, interior velocity and the band reseed machinery."""

    def __init__(self, solver):
        self.solver = solver
        s = solver.settings
        g = solver.grid
        dev = solver.device
        vc = g.voxel_capacity
        self.band_voxels = max(1.0, float(s.narrow_band_voxels))
        h = float(g.voxel_size)
        self.voxel_size = h
        self.band = self.band_voxels * h
        # Distances are clamped to this many metres either side of the surface.
        self.limit = (self.band_voxels + 2.0) * h
        # Deeper than this the advected value is trusted over the particles.
        self.pin_depth = max(1.0, self.band_voxels - 1.0) * h
        self.eikonal_iterations = int(math.ceil(self.band_voxels)) + 2

        self.phi = wp.full(vc, self.limit, dtype=float, device=dev)
        self.phi_tmp = wp.full(vc, self.limit, dtype=float, device=dev)
        self.splat = wp.full(vc, self.limit, dtype=float, device=dev)
        self.nb_u = wp.zeros(vc, dtype=float, device=dev)
        self.nb_v = wp.zeros(vc, dtype=float, device=dev)
        self.nb_w = wp.zeros(vc, dtype=float, device=dev)
        self.nb_tmp_u = wp.zeros(vc, dtype=float, device=dev)
        self.nb_tmp_v = wp.zeros(vc, dtype=float, device=dev)
        self.nb_tmp_w = wp.zeros(vc, dtype=float, device=dev)
        self.ones = wp.full(vc, 1.0, dtype=float, device=dev)
        self.cls = wp.zeros(vc, dtype=wp.int32, device=dev)
        self.liquid_count = wp.zeros(1, dtype=wp.int32, device=dev)
        self.demand = wp.zeros(vc, dtype=wp.int32, device=dev)
        self.offsets = wp.zeros(vc, dtype=wp.int32, device=dev)
        self.slot_base = wp.zeros(1, dtype=wp.int32, device=dev)
        self.slot_limit = wp.zeros(1, dtype=wp.int32, device=dev)
        self.overflow_flag = wp.zeros(1, dtype=wp.int32, device=dev)
        # Host-visible state.
        self.reseed_overflow = False
        self.reseed_epoch = 0
        self.liquid_voxels = 0
        self.owed = wp.zeros(1, dtype=wp.int32, device=dev)
        self._substep_index = 0
        self._check_particle_channels()

    def reset(self):
        """Back to the initial state (no liquid known, no overflow)."""
        self.phi.fill_(self.limit)
        self.phi_tmp.fill_(self.limit)
        for array in (self.nb_u, self.nb_v, self.nb_w, self.nb_tmp_u,
                      self.nb_tmp_v, self.nb_tmp_w):
            array.zero_()
        self.overflow_flag.zero_()
        self.reseed_overflow = False
        self.reseed_epoch = 0
        self.liquid_voxels = 0
        self.owed.zero_()

    def restore_seed_state(self, seed_state):
        """Reload the checkpointed reseed epoch (empty means a fresh start)."""
        self.reseed_epoch = int(seed_state[0]) if len(seed_state) else 0
        self.owed.fill_(int(seed_state[1]) if len(seed_state) > 1 else 0)

    # Guards

    def _check_particle_channels(self):
        """Reseeded particles carry no product channel; the blockers forbid them."""
        channels = tuple(self.solver.particles.optional_channels)
        if channels:
            raise RuntimeError(
                "narrow-band FLIP cannot reseed particles that carry "
                f"{channels}; the settings blockers should have refused this")

    # Topology hooks (called from FluidSolver._rebuild_topology)

    def mark_leaves(self):
        """Keep every leaf that still holds liquid, before the rebuild."""
        g = self.solver.grid
        fast_launch(
            k_nb_mark_leaves_from_phi, dim=max(1, self._old_voxels()),
            inputs=[self.phi, g.leaf_coords, g.num_leaves, g.leaves_dim,
                    g.leaf_flags],
            device=self.solver.device)

    def _old_voxels(self):
        return max(1, int(self.solver.grid.active_voxel_count))

    def remap(self, old_leaf_table):
        """Carry phi and the persisted velocity across the rebuild."""
        sv = self.solver
        g = sv.grid
        nv = sv._nvox
        fast_launch(
            k_nb_remap_leaf_field_default, dim=nv,
            inputs=[self.phi, old_leaf_table, g.leaf_coords, g.num_leaves,
                    g.leaves_dim, self.limit],
            outputs=[self.phi_tmp], device=sv.device)
        self.phi, self.phi_tmp = self.phi_tmp, self.phi
        for name in ("u", "v", "w"):
            current = getattr(self, f"nb_{name}")
            spare = getattr(self, f"nb_tmp_{name}")
            fast_launch(
                k_nb_remap_leaf_field_default, dim=nv,
                inputs=[current, old_leaf_table, g.leaf_coords, g.num_leaves,
                        g.leaves_dim, 0.0],
                outputs=[spare], device=sv.device)
            setattr(self, f"nb_{name}", spare)
            setattr(self, f"nb_tmp_{name}", current)

    # Per-substep: before the pressure solve

    def prepare_grid(self, dt):
        """Give liquid the particles no longer witness a velocity and a mask."""
        sv = self.solver
        g = sv.grid
        nv = sv._nvox
        vel = sv.vel
        fast_launch(
            k_nb_advect_velocity, dim=nv,
            inputs=[self.nb_u, self.nb_v, self.nb_w, self.ones,
                    g.leaf_table, g.leaf_coords, g.num_leaves,
                    g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size, dt],
            outputs=[self.nb_tmp_u, self.nb_tmp_v, self.nb_tmp_w],
            device=sv.device)
        self.nb_u, self.nb_tmp_u = self.nb_tmp_u, self.nb_u
        self.nb_v, self.nb_tmp_v = self.nb_tmp_v, self.nb_v
        self.nb_w, self.nb_tmp_w = self.nb_tmp_w, self.nb_w
        fast_launch(
            k_nb_fill_interior_velocity, dim=nv,
            inputs=[self.phi, sv.voxel_type.data,
                    self.nb_u, self.nb_v, self.nb_w,
                    g.leaf_table, g.leaf_coords, g.num_leaves,
                    g.voxels_dim, g.leaves_dim,
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
            device=sv.device)
        fast_launch(
            k_nb_set_fluid_from_phi, dim=nv,
            inputs=[self.phi, g.num_leaves], outputs=[sv.voxel_type.data],
            device=sv.device)

    def credit_interior_counts(self):
        """Make the particle-free interior look full to the source top-up."""
        self._adjust_interior_counts(INTERIOR_COUNT_CREDIT)

    def release_interior_counts(self):
        self._adjust_interior_counts(-INTERIOR_COUNT_CREDIT)

    def _adjust_interior_counts(self, delta):
        sv = self.solver
        fast_launch(
            k_nb_adjust_interior_counts, dim=sv._nvox,
            inputs=[self.phi, sv.grid.num_leaves, self.band, int(delta)],
            outputs=[sv.voxel_counts.data], device=sv.device)

    # Per-substep: after advection

    def update_phi(self, dt):
        """Advect phi, re-derive the surface from the particles, classify."""
        sv = self.solver
        g = sv.grid
        nv = sv._nvox
        dev = sv.device
        vel = sv.vel
        # The projected velocity is the one interior liquid will carry.
        fast_launch(
            k_nb_store_velocity, dim=nv,
            inputs=[vel.u, vel.v, vel.w, g.num_leaves],
            outputs=[self.nb_u, self.nb_v, self.nb_w], device=dev)
        fast_launch(
            k_nb_advect_scalar, dim=nv,
            inputs=[self.phi, vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                    sv.collision_sdf.data, g.leaf_table, g.leaf_coords,
                    g.num_leaves, g.voxels_dim, g.leaves_dim, g.origin,
                    g.voxel_size, dt, self.limit],
            outputs=[self.phi_tmp], device=dev)
        self._splat_particles()
        fast_launch(
            k_nb_union_band, dim=nv,
            inputs=[self.phi_tmp, self.splat, sv.collision_sdf.data,
                    g.num_leaves, self.pin_depth, self.limit],
            outputs=[self.phi], device=dev)
        for _ in range(self.eikonal_iterations):
            fast_launch(
                k_nb_eikonal_inward, dim=nv,
                inputs=[self.phi, sv.collision_sdf.data, g.leaf_table,
                        g.leaf_coords, g.num_leaves, g.voxels_dim,
                        g.leaves_dim, g.voxel_size, self.limit],
                outputs=[self.phi_tmp], device=dev)
            self.phi, self.phi_tmp = self.phi_tmp, self.phi
        self._classify()

    def _splat_particles(self):
        sv = self.solver
        g = sv.grid
        s = sv.settings
        h = float(g.voxel_size)
        radius = 0.5 * float(s.voxel_size) * float(s.particle_radius_scale)
        splat_voxels = max(1, int(s.surface_splat_radius),
                           int(math.ceil(radius / h + 1.0)))
        fast_launch(
            k_surface_sdf_reset, dim=sv._nvox,
            inputs=[self.splat, g.num_leaves, self.limit], device=sv.device)
        fast_launch(
            k_build_surface_sdf, dim=sv._np_dim(),
            inputs=[sv.particles.position, sv.particles.count, g.leaf_table,
                    g.voxels_dim, g.leaves_dim, g.origin, h, radius,
                    splat_voxels, self.limit],
            outputs=[self.splat], device=sv.device)

    def _classify(self):
        sv = self.solver
        g = sv.grid
        self.liquid_count.zero_()
        fast_launch(
            k_nb_classify, dim=sv._nvox,
            inputs=[self.phi, g.num_leaves, self.band],
            outputs=[self.cls, self.liquid_count], device=sv.device)

    # Per-substep: particle lifecycle

    def kill_interior(self):
        sv = self.solver
        g = sv.grid
        p = sv.particles
        fast_launch(
            k_nb_kill_interior, dim=sv._np_dim(),
            inputs=[p.position, p.count, self.phi, g.leaf_table,
                    g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
                    self.band],
            outputs=[p.alive, self.owed], device=sv.device)

    def reseed(self, substep_index):
        """Top the band back up to the authored density, deterministically."""
        sv = self.solver
        g = sv.grid
        nv = sv._nvox
        dev = sv.device
        p = sv.particles
        vel = sv.vel
        target = int(min(C.MAX_PARTICLES_PER_VOXEL,
                         max(1, round(float(sv.settings.particles_per_voxel)))))
        fast_launch(
            k_nb_reseed_demand, dim=nv,
            inputs=[self.cls, self.phi, sv.collision_sdf.data,
                    sv.voxel_counts.data, g.num_leaves,
                    g.voxel_size, RESEED_TRIGGER, max(1, target // 2)],
            outputs=[self.demand], device=dev)
        wp.utils.array_scan(self.demand[:nv], self.offsets[:nv], inclusive=False)
        fast_launch(
            k_nb_reseed_reserve, dim=1,
            inputs=[self.demand, self.offsets, nv, p.capacity, self.owed, p.count],
            outputs=[self.slot_base, self.slot_limit, self.overflow_flag],
            device=dev)
        seed = (sv.frame * _SEED_FRAME + int(substep_index) * _SEED_SUBSTEP
                + _SEED_BASE) & _SEED_MASK
        fast_launch(
            k_nb_reseed_commit, dim=nv,
            inputs=[self.demand, self.offsets, g.leaf_coords, g.num_leaves,
                    g.origin, g.voxel_size, int(seed),
                    self.slot_base, self.slot_limit,
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    p.position, p.velocity, p.attr("age"), p.attr("life"),
                    p.attr("temperature"), p.attr("viscosity"),
                    p.attr("density"), p.attr("time_offset"), p.alive],
            device=dev)
        self.reseed_epoch += 1
        p.refresh_host_count()

    # Frame boundary

    def end_frame(self):
        """Read the per-frame scalars (after the frame's synchronize)."""
        self.liquid_voxels = int(self.liquid_count.numpy()[0])
        if int(self.overflow_flag.numpy()[0]):
            self.reseed_overflow = True
        return self.liquid_voxels

    # Consumers of the liquid surface

    def union_into_solver_sdf(self, sdf):
        """``sdf = min(sdf, phi)`` a full cell below the surface."""
        sv = self.solver
        fast_launch(
            k_nb_union_into_solver_sdf, dim=sv._nvox,
            inputs=[sdf.data, self.phi, sv.grid.num_leaves,
                    float(sv.grid.voxel_size)],
            device=sv.device)

    def mark_render_leaves(self, render_grid):
        """Allocate render leaves under interior liquid (before finalize)."""
        sv = self.solver
        g = sv.grid
        fast_launch(
            k_nb_mark_render_leaves, dim=sv._nvox,
            inputs=[self.phi, g.leaf_coords, g.num_leaves, g.origin,
                    g.voxel_size, float(g.voxel_size), render_grid.origin,
                    render_grid.voxel_size, render_grid.voxels_dim,
                    render_grid.leaves_dim, render_grid.leaf_flags],
            device=sv.device)

    def union_into_render_sdf(self, render_grid, render_sdf):
        sv = self.solver
        g = sv.grid
        fast_launch(
            k_nb_union_into_render_sdf,
            dim=max(1, render_grid.active_voxel_count),
            inputs=[render_sdf.data, render_grid.leaf_coords,
                    render_grid.num_leaves, render_grid.origin,
                    render_grid.voxel_size, self.phi, g.leaf_table,
                    g.voxels_dim, g.leaves_dim, g.origin, g.voxel_size,
                    float(g.voxel_size)],
            device=sv.device)

    # Host accessors (tests, cache producer)

    def active_voxels(self):
        return int(self.solver.grid.refresh_topology_counts()) * C.LEAF_VOLUME

    def phi_numpy(self):
        """Persistent level set over the active leaf slots, float32."""
        return self.phi.numpy()[:self.active_voxels()]

    def leaf_phi(self):
        """``(leaf_coords int32 (N,3), phi float32 (N, 512))`` for the cache."""
        count = int(self.solver.grid.refresh_topology_counts())
        coords = host_prefix(self.solver.grid.leaf_coords, count).astype(np.int32)
        data = host_prefix(self.phi, count * C.LEAF_VOLUME).reshape(
            count, C.LEAF_VOLUME).astype(np.float32)
        return coords, data

    def solver_sdf_at(self, points):
        """The liquid SDF the solver consumers see, sampled at world points.

        ``min(particle splat, phi)`` with phi taken only a full cell below
        the surface, exactly the union applied to the solver SDF.
        """
        sv = self.solver
        g = sv.grid
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        h = float(g.voxel_size)
        origin = np.array([g.origin[0], g.origin[1], g.origin[2]], dtype=np.float64)
        ijk = np.floor((pts - origin) / h).astype(np.int64)
        count = int(g.refresh_topology_counts())
        coords = g.leaf_coords.numpy()[:count].astype(np.int64)
        phi = self.phi.numpy()
        slot_of = {tuple(c): i for i, c in enumerate(coords)}
        out = np.full(len(pts), self.limit, dtype=np.float32)
        for n, c in enumerate(ijk):
            leaf = tuple(int(v) for v in c >> C.LEAF_SHIFT)
            slot = slot_of.get(leaf)
            if slot is None:
                continue
            li, lj, lk = (int(v) & C.LEAF_MASK for v in c)
            local = li + (lj << C.LEAF_SHIFT) + (lk << (2 * C.LEAF_SHIFT))
            out[n] = phi[slot * C.LEAF_VOLUME + local]
        splat = self._splat_host(pts)
        return np.minimum(out, splat).astype(np.float32)

    def _splat_host(self, pts):
        """Nearest particle sphere distance at ``pts`` (host, brute force)."""
        sv = self.solver
        s = sv.settings
        radius = 0.5 * float(s.voxel_size) * float(s.particle_radius_scale)
        pos = sv.particle_positions().astype(np.float64)
        if len(pos) == 0:
            return np.full(len(pts), self.limit, dtype=np.float32)
        out = np.empty(len(pts), dtype=np.float32)
        for n, point in enumerate(pts):
            out[n] = np.sqrt(((pos - point) ** 2).sum(axis=1)).min() - radius
        return out
