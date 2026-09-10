"""Liquid surface SDF construction, smoothing and SDF->collision conversion.

Ports of the fluid_surface_collision kernel group:
  fluidSimBuildSurfaceSdf, fluidSimSmoothSurfaceSdfDialate,
  fluidSimSmoothSurfaceSdfErode, fluidSimSmoothSurfaceSdfBox,
  fluidSimSmoothSurfaceSdfMeanCurvatureFlow, fluidSimSdfToAxisCollision,
  fluidSimExpandAndShiftAxisVelocityInColliderAxis
"""

from __future__ import annotations

import math
import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .quality_contract import AuthoredQualitySettings, resolve_quality
from .surface_tension_math import (
    normal_support_voxels, required_sdf_splat_radius_voxels)
from .fields import (
    leaf_linear_index,
    leaf_coord_of_voxel,
    leaf_local_of_voxel,
    voxel_in_domain,
    voxel_flat_index,
    voxel_coord_of_flat,
    voxel_center_world,
    world_to_voxel,
    field_read_f,
    field_read_i,
)
from .registry import register_fn


# --- particle-SDF splat modes -------------------------------------------------
#
# "ball" skips splat cells that provably cannot change the output; "legacy" is
# the original full-cube splat, kept so before/after pairs can be measured from
# one binary. See k_build_surface_sdf for the identity argument.
SDF_SPLAT_MODE_BALL = "ball"
SDF_SPLAT_MODE_LEGACY = "legacy"
SDF_SPLAT_MODES = (SDF_SPLAT_MODE_BALL, SDF_SPLAT_MODE_LEGACY)

# Passed as the kernel's max_distance in legacy mode. Any value that no real
# distance can reach disables the cull without needing a second kernel or a
# per-cell branch. Representable in float32 (max ~3.4e38).
SDF_SPLAT_NO_CULL = 1.0e30

_default_sdf_splat_mode = SDF_SPLAT_MODE_BALL


def set_default_sdf_splat_mode(mode: str) -> None:
    """Set the splat mode new SurfacePipelines use. Benchmark/diagnostic hook."""

    global _default_sdf_splat_mode
    if mode not in SDF_SPLAT_MODES:
        raise ValueError(
            "unknown sdf splat mode %r; expected one of %r" % (mode, SDF_SPLAT_MODES))
    _default_sdf_splat_mode = mode


def default_sdf_splat_mode() -> str:
    return _default_sdf_splat_mode


# --- surface-tension band construction modes ---------------------------------
#
# Surface tension needs the union-of-spheres distance to stay valid out to
# ``support_voxels`` so the capillary normal/curvature stencil never reads a
# truncated value.  ``required_sdf_splat_radius_voxels`` buys that by widening
# the splat, which is exact but costs particles x R^3.
#
#   "splat"     - the default.  Every particle writes the full band radius.
#                 Exact by construction.
#   "propagate" - seed only the radius the particle support itself justifies,
#                 remember which particle won each seeded voxel, then carry
#                 those closest points outward across the active voxels.  Cost
#                 scales with active voxels x passes instead of particles x R^3.
#                 Each pass recomputes the distance from the receiving voxel's
#                 own centre to the neighbour's closest particle, so a voxel's
#                 value is always an exact Euclidean distance to *some* real
#                 particle -- never an underestimate, and never a smeared
#                 average.  It is an upper bound of the true field, and equal
#                 to it wherever the true nearest particle's point reaches the
#                 voxel.  It is not exact, so it is opt-in until measured.
SDF_BAND_MODE_SPLAT = "splat"
SDF_BAND_MODE_PROPAGATE = "propagate"
SDF_BAND_MODES = (SDF_BAND_MODE_SPLAT, SDF_BAND_MODE_PROPAGATE)

# Propagation seeds an unclamped field so "no closest point yet" is
# distinguishable from "far away but known", then clamps to far_distance at the
# end.  Both values are exactly representable in float32 and are far outside any
# real distance, and the gap between them leaves room for the ``- radius`` term.
SDF_BAND_FAR = 1.0e20
SDF_BAND_INVALID = 1.0e19

# Propagation is the default.  It is not bit-identical to the splat, so
# the justification is a bound rather than an equality: the seed radius
# `band_seed_splat_radius_voxels` derives makes the field *exact*
# everywhere the capillary force and its normal/curvature stencil are
# read, and outside that window every value is still a real
# particle-to-voxel distance, so the field is a strict upper bound that
# never underestimates.  Measured on dam_break: the normal direction
# moves 3.0e-5 degrees and curvature 2e-6 relative, for -34% of
# build_sdf.  `SDF_BAND_MODE_SPLAT` remains available and is what the
# accuracy tests compare against.
_default_sdf_band_mode = SDF_BAND_MODE_PROPAGATE


def set_default_sdf_band_mode(mode: str) -> None:
    """Set the band mode new SurfacePipelines use. Benchmark/diagnostic hook."""

    global _default_sdf_band_mode
    if mode not in SDF_BAND_MODES:
        raise ValueError(
            "unknown sdf band mode %r; expected one of %r" % (mode, SDF_BAND_MODES))
    _default_sdf_band_mode = mode


def default_sdf_band_mode() -> str:
    return _default_sdf_band_mode


_default_sdf_band_seed_radius: int | None = None


def set_default_sdf_band_seed_radius(radius: int | None) -> None:
    """Override the derived seed radius; ``None`` restores the derivation."""
    global _default_sdf_band_seed_radius
    _default_sdf_band_seed_radius = (
        None if radius is None else max(1, int(radius)))


def default_sdf_band_seed_radius() -> int | None:
    return _default_sdf_band_seed_radius


def band_seed_splat_radius_voxels(r_vox: float, band_voxels: float,
                                  base_splat: int, smoothing_iterations: int = 0,
                                  sdf_filter_iterations: int = 0) -> int:
    """Seed radius that makes the propagated band exact where it is read.

    A voxel is seeded by its *true* nearest particle whenever that particle
    lies within Chebyshev radius ``seed`` cells of it, and Chebyshev distance
    never exceeds Euclidean, so every voxel with ``sdf <= seed * h - r`` is
    seeded by the true minimiser and carries the exact value. ``apply()``
    reads normals over ``normal_support_voxels(band_voxels,
    smoothing_iterations)`` -- one cell beyond the force band plus one per
    smoothing pass -- and configured SDF filters (dilate/erode/smooth/filter)
    expand that support further, so the seed has to reach the same support
    ``splat`` is computed from, not just ``band_voxels + 1``. Beyond that
    window the propagated value is a strict upper bound that converges from
    above with each pass.
    """
    # normal_support_voxels() requires a positive band; an inactive band (0.0,
    # used when surface tension is off) has no capillary stencil to protect,
    # so fall back to the one-voxel margin its own max(band_voxels, 0.5)
    # clamp would otherwise apply.
    if band_voxels > 0.0:
        support = (
            normal_support_voxels(band_voxels, smoothing_iterations)
            + sdf_filter_iterations
        )
    else:
        support = 1.0
    return max(int(base_splat), int(math.ceil(r_vox + support)))


def band_propagation_passes(band_splat: int, seed_splat: int) -> int:
    """Passes needed to carry a seeded closest point out to ``band_splat``.

    A seeded voxel sits within Chebyshev radius ``seed_splat`` of its
    particle's cell, and one pass moves a point one cell in the Chebyshev
    metric, so ``band_splat - seed_splat`` passes reach every cell the splat
    would have written -- provided the cells in between are active.  That
    proviso is the reason 3c measures the result instead of trusting it.
    """

    return max(0, int(band_splat) - int(seed_splat))


class SurfacePipeline:
    """Owns liquid-surface fields, cadence, extraction and mesh caching."""

    #: Per-instance splat-mode override; None means follow the module default.
    #: Declared at class scope because tests build pipelines with
    #: ``object.__new__`` and never run ``__init__``.
    sdf_splat_mode: str | None = None

    def splat_mode(self) -> str:
        return getattr(self, "sdf_splat_mode", None) or default_sdf_splat_mode()

    #: Per-instance band-mode override; None means follow the module default.
    sdf_band_mode: str | None = None

    def band_mode(self) -> str:
        return getattr(self, "sdf_band_mode", None) or default_sdf_band_mode()

    # Class scope for the same reason as the mode above: tests build pipelines
    # with object.__new__ and never run __init__.
    sdf_band_seed_radius: int | None = None

    def band_seed_radius(self) -> int | None:
        override = getattr(self, "sdf_band_seed_radius", None)
        return override if override else default_sdf_band_seed_radius()

    def _ensure_band_buffers(self, capacity, device):
        """Lazily size the closest-point scratch to the field capacity."""

        capacity = max(1, int(capacity))
        if getattr(self, "_band_capacity", 0) >= capacity:
            return
        self._band_closest = wp.zeros(capacity, dtype=wp.int32, device=device)
        self._band_points = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._band_points_tmp = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._band_capacity = capacity

    def release_band_buffers(self):
        """Drop the propagation scratch. Safe to call when the mode is off."""

        self._band_closest = None
        self._band_points = None
        self._band_points_tmp = None
        self._band_capacity = 0

    def __init__(self, solver):
        from .fields import ScalarField

        self.solver = solver
        g = solver.grid
        self.sdf = ScalarField(g, name="surface_sdf")
        self.tmp = ScalarField(g, name="surface_tmp")
        self.extractor = SurfaceExtractor(g, solver.device)
        self.grid = g
        self.render_sdf = self.sdf
        self.render_tmp = self.tmp
        self.render_extractor = self.extractor
        self.grid_key = None
        self.nvox = max(1, g.voxel_capacity)
        self.topology_generation = 0
        self.mesh_cache = None
        self.mesh_key = None
        self.built_key = None
        self.solver_sdf_key = None
        self.solver_sdf_error = ""

    def mode(self):
        from .enums import SurfaceResolutionMode

        mode = int(getattr(self.solver.settings, "surface_resolution_mode", int(SurfaceResolutionMode.SIMULATION)))
        valid = {int(SurfaceResolutionMode.SIMULATION), int(SurfaceResolutionMode.MULTIPLIER), int(SurfaceResolutionMode.CUSTOM)}
        return mode if mode in valid else int(SurfaceResolutionMode.SIMULATION)

    @staticmethod
    def voxel_size_for(settings):
        resolved = getattr(settings, "resolved_quality", None)
        if resolved is None:
            resolved = resolve_quality(
                settings, (settings.domain_min, settings.domain_max))
        return resolved.surface_voxel_size

    def voxel_size(self):
        return self.voxel_size_for(self.solver.settings)

    def leaf_capacity(self, h):
        s = self.solver.settings
        surface_quality = resolve_quality(
            AuthoredQualitySettings(voxel_size=float(h)),
            (s.domain_min, s.domain_max))
        total = surface_quality.total_domain_leaf_cells
        if abs(h - float(s.voxel_size)) <= 1.0e-8:
            return min(int(self.solver.grid.capacity), int(total))
        scale = max(1.0, float(getattr(s, "surface_leaf_capacity_scale", 8.0)))
        return int(min(total, max(1, math.ceil(float(s.leaf_capacity) * scale))))

    def resource_key(self):
        s = self.solver.settings
        h = float(self.voxel_size())
        return (round(h, 12), self.leaf_capacity(h), max(8, int(getattr(s, "surface_max_dense_dim", 512))),
                tuple(float(x) for x in s.domain_min), tuple(float(x) for x in s.domain_max))

    def settings_signature(self):
        s = self.solver.settings
        return (int(getattr(s, "surface_resolution_mode", 0)), round(float(getattr(s, "surface_resolution_multiplier", 1.0)), 6),
                round(float(getattr(s, "surface_voxel_size", 0.0)), 9), round(float(getattr(s, "surface_leaf_capacity_scale", 1.0)), 6),
                int(getattr(s, "surface_max_dense_dim", 512)), int(getattr(s, "surface_update_cadence", 1)),
                bool(getattr(s, "surface_build_lowres_sdf", False)), round(float(s.particle_radius_scale), 6), int(s.surface_splat_radius),
                int(s.surface_filter_type), int(s.surface_filter_iterations), round(float(s.surface_filter_strength), 6),
                int(s.surface_dilate_iterations), round(float(s.surface_dilate_strength), 6), int(s.surface_erode_iterations),
                round(float(s.surface_erode_strength), 6), int(s.surface_smooth_type), int(s.surface_smooth_iterations),
                round(float(s.surface_smooth_strength), 6), bool(getattr(s, "spacetime_enabled", False)),
                bool(getattr(s, "spacetime_resync_surface", True)), self.topology_generation)

    def build_key(self):
        solver = self.solver
        return (solver.frame, round(float(solver.sim_time), 12), self.settings_signature())

    def invalidate(self):
        self.mesh_cache = self.mesh_key = None
        host = getattr(self.solver, "_host_frame_cache", None)
        if host is not None:
            host.invalidate_surface()

    def ensure_resources(self, force=False):
        from .fields import SparseGrid, ScalarField

        solver = self.solver
        s = solver.settings
        h = self.voxel_size()
        max_dense = max(8, int(getattr(s, "surface_max_dense_dim", 512)))
        if abs(h - float(solver.grid.voxel_size)) <= 1.0e-8:
            self.grid, self.render_sdf, self.render_tmp, self.render_extractor = solver.grid, self.sdf, self.tmp, self.extractor
            self.render_extractor.max_dense_dim = max_dense
            if force:
                self.render_extractor.reset_buffers()
            self.grid_key = ("simulation", max_dense)
            self.nvox = getattr(solver, "_nvox", max(1, solver.grid.active_voxel_count))
            self.invalidate()
            return
        key = self.resource_key()
        if not force and self.grid_key == key:
            self.render_extractor.max_dense_dim = max_dense
            return
        dmin, dmax = np.asarray(s.domain_min, dtype=np.float64), np.asarray(s.domain_max, dtype=np.float64)
        grid = SparseGrid(origin=dmin, size=np.maximum(dmax - dmin, h), voxel_size=h,
                          leaf_capacity=self.leaf_capacity(h), device=solver.device)
        self.grid = grid
        self.render_sdf = ScalarField(grid, name="render_surface_sdf")
        self.render_tmp = ScalarField(grid, name="render_surface_tmp")
        self.render_extractor = SurfaceExtractor(grid, solver.device, max_dense_dim=max_dense)
        self.grid_key, self.nvox = key, max(1, grid.active_voxel_count)
        self.topology_generation += 1
        self.invalidate()

    @staticmethod
    def filter_kernel(filter_type):
        from .enums import SdfFilterType

        return {int(SdfFilterType.DIALATE): k_smooth_surface_sdf_dilate,
                int(SdfFilterType.ERODE): k_smooth_surface_sdf_erode,
                int(SdfFilterType.BOX): k_smooth_surface_sdf_box,
                int(SdfFilterType.MEAN_CURVATURE_FLOW): k_smooth_surface_sdf_mean_curvature}[int(filter_type)]

    def build_sdf(self, grid, sdf, tmp, nvox, positions, voxel_size):
        solver, s = self.solver, self.solver.settings
        h, nvox = float(voxel_size), max(1, int(nvox))
        radius = 0.5 * float(s.voxel_size) * float(s.particle_radius_scale)
        splat = max(1, int(s.surface_splat_radius), int(math.ceil(radius / h + 1.0)))
        far_distance = 4.0 * h
        band_voxels = 0.0
        smoothing = 0
        sdf_filter_iterations = 0
        if (grid is solver.grid
                and bool(getattr(s, "surface_tension_enabled", False))
                and float(getattr(s, "surface_tension_coefficient", 0.0)) > 0.0):
            band_voxels = float(getattr(s, "surface_tension_band_voxels", 1.5))
            smoothing = int(getattr(
                s, "surface_tension_smoothing_iterations", 1))
            sdf_filter_iterations = sum(max(0, int(value)) for value in (
                s.surface_filter_iterations,
                s.surface_dilate_iterations,
                s.surface_erode_iterations,
                s.surface_smooth_iterations,
            ))
            support_voxels = (
                normal_support_voxels(band_voxels, smoothing)
                + sdf_filter_iterations
            )
            far_distance = max(far_distance, (support_voxels + 1.0) * h)
            splat = max(
                splat,
                required_sdf_splat_radius_voxels(
                    radius / h, band_voxels, smoothing,
                    sdf_filter_iterations),
            )
        else:
            # Mirrors SurfaceTensionSolver.apply()'s own check-and-release: once
            # the capillary band is inactive (tension disabled or coefficient
            # <= 0), the propagation scratch it fed is unused for the rest of
            # the solver session unless tension comes back on, so drop it here
            # rather than holding it for the session's lifetime.
            self.release_band_buffers()
        # Equal to far_distance in "ball" mode: the reset below writes
        # far_distance into every cell, so a candidate at or beyond it can never
        # win an atomic_min. "legacy" passes an unreachable distance so a
        # before/after pair can be measured from one binary.
        max_distance = (
            SDF_SPLAT_NO_CULL
            if self.splat_mode() == SDF_SPLAT_MODE_LEGACY
            else far_distance
        )
        self.last_sdf_splat = {
            "mode": self.splat_mode(),
            "splatRadiusVoxels": int(splat),
            "farDistance": float(far_distance),
            "maxDistance": float(max_distance),
            "voxelSize": float(h),
            "particleRadius": float(radius),
        }
        seed_override = self.band_seed_radius()
        seed_splat = int(seed_override) if seed_override else (
            band_seed_splat_radius_voxels(
                radius / h, band_voxels,
                max(1, int(s.surface_splat_radius),
                    int(math.ceil(radius / h + 1.0))),
                smoothing, sdf_filter_iterations))
        passes = (band_propagation_passes(splat, seed_splat)
                  if self.band_mode() == SDF_BAND_MODE_PROPAGATE else 0)
        self.last_sdf_splat["bandMode"] = self.band_mode()
        self.last_sdf_splat["seedSplatRadiusVoxels"] = int(seed_splat)
        self.last_sdf_splat["propagationPasses"] = int(passes)
        if passes > 0:
            sdf.data, tmp.data = self._build_sdf_by_propagation(
                grid, sdf, tmp, nvox, positions, h, radius,
                seed_splat, passes, far_distance)
        else:
            wp.launch(
                k_surface_sdf_reset, dim=nvox,
                inputs=[sdf.data, grid.num_leaves, far_distance],
                device=solver.device)
            wp.launch(k_build_surface_sdf, dim=solver._np_dim(), inputs=[positions, solver.particles.count, grid.leaf_table,
                      grid.voxels_dim, grid.leaves_dim, grid.origin, h, radius, splat, max_distance], outputs=[sdf.data], device=solver.device)
        def apply(kernel, iterations, strength):
            for _ in range(max(0, int(iterations))):
                wp.launch(kernel, dim=nvox, inputs=[sdf.data, grid.leaf_table, grid.leaf_coords, grid.voxels_dim,
                          grid.leaves_dim, grid.num_leaves, h, float(strength)], outputs=[tmp.data], device=solver.device)
                sdf.data, tmp.data = tmp.data, sdf.data
        apply(self.filter_kernel(int(s.surface_filter_type)), s.surface_filter_iterations, s.surface_filter_strength)
        apply(k_smooth_surface_sdf_dilate, s.surface_dilate_iterations, s.surface_dilate_strength)
        apply(k_smooth_surface_sdf_erode, s.surface_erode_iterations, s.surface_erode_strength)
        apply(self.filter_kernel(int(s.surface_smooth_type)), s.surface_smooth_iterations, s.surface_smooth_strength)
        return sdf

    def _build_sdf_by_propagation(self, grid, sdf, tmp, nvox, positions,
                                  h, radius, seed_splat, passes, far_distance):
        """Seed a small splat, then carry its closest points out to the band.

        Returns the (sdf, tmp) data arrays in their post-ping-pong order, in
        the same convention the smoothing filters below already use.
        """

        solver = self.solver
        dev = solver.device
        self._ensure_band_buffers(sdf.data.shape[0], dev)
        closest, points = self._band_closest, self._band_points
        points_tmp = self._band_points_tmp
        sdf_data, tmp_data = sdf.data, tmp.data

        # Seed unclamped: "no closest point yet" has to stay distinguishable
        # from "known, and far", or propagation would carry the fill value.
        wp.launch(k_surface_sdf_reset, dim=nvox,
                  inputs=[sdf_data, grid.num_leaves, SDF_BAND_FAR], device=dev)
        wp.launch(k_surface_band_index_reset, dim=nvox,
                  inputs=[grid.num_leaves], outputs=[closest], device=dev)
        wp.launch(k_build_surface_sdf, dim=solver._np_dim(),
                  inputs=[positions, solver.particles.count, grid.leaf_table,
                          grid.voxels_dim, grid.leaves_dim, grid.origin, h,
                          radius, seed_splat, SDF_SPLAT_NO_CULL],
                  outputs=[sdf_data], device=dev)
        wp.launch(k_seed_surface_sdf_closest, dim=solver._np_dim(),
                  inputs=[positions, solver.particles.count, grid.leaf_table,
                          grid.voxels_dim, grid.leaves_dim, grid.origin, h,
                          radius, seed_splat, sdf_data],
                  outputs=[closest], device=dev)
        wp.launch(k_gather_surface_band_points, dim=nvox,
                  inputs=[closest, positions, grid.num_leaves],
                  outputs=[points], device=dev)

        for _ in range(int(passes)):
            wp.launch(k_propagate_surface_sdf_band, dim=nvox,
                      inputs=[sdf_data, points, grid.leaf_table,
                              grid.leaf_coords, grid.voxels_dim,
                              grid.leaves_dim, grid.num_leaves, grid.origin,
                              h, radius],
                      outputs=[tmp_data, points_tmp], device=dev)
            sdf_data, tmp_data = tmp_data, sdf_data
            points, points_tmp = points_tmp, points

        self._band_points, self._band_points_tmp = points, points_tmp
        wp.launch(k_clamp_surface_sdf, dim=nvox,
                  inputs=[grid.num_leaves, far_distance], outputs=[sdf_data],
                  device=dev)
        return sdf_data, tmp_data

    def rebuild_topology(self, positions):
        solver, grid = self.solver, self.grid
        if grid is solver.grid:
            self.nvox = solver._nvox
        else:
            grid.begin_topology(); grid.mark_points(positions, solver.particles.count, solver._np_dim())
            grid.expand(max(1, int(getattr(solver.settings, "topology_expand", 1))))
            grid.finalize_topology(sync=True)
            self.nvox = max(1, grid.active_voxel_count)
            self.topology_generation += 1
        return self.nvox

    def solver_sdf_build_key(self):
        solver, s = self.solver, self.solver.settings
        return (
            int(getattr(solver, "frame", 0)),
            round(float(getattr(solver, "sim_time", 0.0)), 12),
            round(float(s.voxel_size), 12),
            round(float(s.particle_radius_scale), 6),
            int(s.surface_splat_radius),
            int(s.surface_filter_type), int(s.surface_filter_iterations),
            round(float(s.surface_filter_strength), 6),
            int(s.surface_dilate_iterations), round(float(s.surface_dilate_strength), 6),
            int(s.surface_erode_iterations), round(float(s.surface_erode_strength), 6),
            int(s.surface_smooth_type), int(s.surface_smooth_iterations),
            round(float(s.surface_smooth_strength), 6),
        )

    def invalidate_solver_sdf(self):
        """Mark the simulation SDF stale after particle positions change."""
        self.solver_sdf_key = None

    def invalidate_particle_state(self):
        """Invalidate every surface product after an out-of-step particle edit."""

        self.built_key = None
        self.solver_sdf_key = None
        self.invalidate()

    def ensure_solver_sdf_current(self):
        """Build the simulation-grid SDF independently of render surfacing.

        Whitewater calls this immediately before emission/advection. Launch
        failures are converted to a dependency result so whitewater can fail
        closed instead of sampling zero-initialized or stale storage.
        """
        solver, s = self.solver, self.solver.settings
        key = self.solver_sdf_build_key()
        if getattr(self, "solver_sdf_key", None) == key:
            return True
        try:
            positions = solver._surface_positions()
            self.build_sdf(
                solver.grid, self.sdf, self.tmp, solver._nvox, positions,
                float(s.voxel_size))
        except Exception as exc:  # noqa: BLE001 - dependency boundary
            self.solver_sdf_key = None
            self.solver_sdf_error = f"Simulation liquid SDF unavailable: {exc}"
            return False
        self.solver_sdf_key = key
        self.solver_sdf_error = ""
        return True

    def build(self, build_render=True, build_lowres=None):
        solver, s = self.solver, self.solver.settings
        if build_lowres is None:
            build_lowres = bool(getattr(s, "surface_build_lowres_sdf", False))
        if build_lowres and not self.ensure_solver_sdf_current():
            if build_render and s.surface_enabled:
                raise RuntimeError(self.solver_sdf_error)
            return False
        if not build_render or not s.surface_enabled:
            return True
        positions = solver._surface_positions()
        self.ensure_resources()
        if self.grid is solver.grid:
            if not build_lowres and not self.ensure_solver_sdf_current():
                raise RuntimeError(self.solver_sdf_error)
            self.nvox = solver._nvox
        else:
            self.build_sdf(self.grid, self.render_sdf, self.render_tmp, self.rebuild_topology(positions), positions, float(self.grid.voxel_size))
        self.built_key = self.build_key()
        self.invalidate()
        return True

    def after_substep(self):
        from .enums import SurfaceUpdateCadence

        s = self.solver.settings
        sdf_ready = True
        with self.solver._phase("surface"):
            # The solver-grid SDF is a whitewater dependency, not a render
            # product. Always refresh it before whitewater, regardless of
            # render mesh enablement, visibility, resolution, or cadence.
            if s.whitewater_enabled:
                sdf_ready = self.ensure_solver_sdf_current()
            cadence = int(getattr(s, "surface_update_cadence", int(SurfaceUpdateCadence.FRAME)))
            if (sdf_ready and s.surface_enabled
                    and cadence == int(SurfaceUpdateCadence.SUBSTEP)):
                self.build(build_render=True, build_lowres=not s.whitewater_enabled)
        return sdf_ready

    def after_frame(self):
        from .enums import SurfaceUpdateCadence

        s = self.solver.settings
        cadence = int(getattr(s, "surface_update_cadence", int(SurfaceUpdateCadence.FRAME)))
        if s.surface_enabled and cadence == int(SurfaceUpdateCadence.FRAME):
            with self.solver._phase("surface"):
                self.build(build_render=True, build_lowres=bool(
                    getattr(s, "surface_build_lowres_sdf", False)))

    def rebuild(self):
        self.built_key = None
        self.invalidate()
        self.build()

    def reset(self):
        self.built_key = None
        self.solver_sdf_key = None
        self.solver_sdf_error = ""
        self.invalidate()
        self.extractor.reset_buffers()
        self.grid_key = None
        self.ensure_resources(force=True)

    def mesh(self):
        if self.solver.settings.surface_enabled and self.built_key != self.build_key():
            self.build(build_render=True, build_lowres=False)
        key = self.build_key()
        if self.mesh_cache is None or self.mesh_key != key:
            self.mesh_cache = self.render_extractor.extract(self.render_sdf)
            self.mesh_key = key
        return self.mesh_cache

    def metadata(self):
        s = self.solver.settings
        leaves = int(getattr(self.grid, "active_leaf_count", 0))
        return {"voxel_size": self.voxel_size(), "resolution_mode": int(s.surface_resolution_mode),
                "resolution_multiplier": float(s.surface_resolution_multiplier),
                "update_cadence": int(s.surface_update_cadence), "smooth_iterations": int(s.surface_smooth_iterations),
                "active_leaves": leaves, "active_voxels": leaves * C.LEAF_VOLUME}


# Surface SDF build (union of particle spheres, splatted into the grid)


@wp.kernel
def k_surface_sdf_reset(
    sdf: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    default_value: float,
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    sdf[tid] = default_value


@wp.kernel
def k_build_surface_sdf(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    particle_radius: float,
    splat_radius_voxels: int,
    max_distance: float,
    sdf: wp.array(dtype=float),
):
    """Union-of-spheres distance splat. sdf holds distance-to-surface
    (negative inside), quantized to int via atomic_min on scaled values.

    We store float distances directly using atomic_min emulation:
    warp supports wp.atomic_min on float arrays, so use it directly.

    ``max_distance`` shrinks the splat from the full ``(2r+1)^3`` cube to the
    ball that can actually change the output, and the shrink is
    output-identical rather than approximate. The field is reset to exactly
    ``far_distance`` before this kernel runs and atomic_min only ever lowers a
    cell, so every cell holds a value <= far_distance at all times. A candidate
    with ``d >= far_distance`` therefore loses every atomic_min it could enter,
    and skipping it changes no bit of the output. Pass SDF_SPLAT_NO_CULL to
    keep the whole cube (used to measure before/after pairs from one binary).

    The skipping is done by narrowing the loop bounds, not by testing each
    cell. That distinction is the whole optimization, and it was established by
    measurement: a per-cell ``if d >= max_distance: continue`` made this kernel
    ~33% SLOWER, because computing ``d`` in order to skip a cell pays exactly
    the ``wp.length`` sqrt the skip was meant to save. The splat is ALU-bound,
    not atomic-bound. Deriving the ranges instead costs two sqrts per ``dk``
    and per ``(dk, dj)`` and visits only cells inside the ball.

    The bounds are a strict superset of the ball. Each level widens its integer
    range by one cell, so an excluded cell satisfies
    ``|e_x| > rx + 1`` and hence ``|e|^2 > rad^2 + 2*rx + 1 > rad^2`` -- it is
    outside the ball with a full voxel of margin, far beyond any float
    rounding in the sqrt. The loop body is byte-for-byte the pre-optimization
    body, so cells that ARE visited cannot disagree with it either.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    p = positions[tid]
    vp = world_to_voxel(p, origin, voxel_size)
    ci = int(wp.floor(vp[0]))
    cj = int(wp.floor(vp[1]))
    ck = int(wp.floor(vp[2]))
    r = splat_radius_voxels

    # Offset of a candidate cell centre from the particle, in voxels, is
    # ``e = (d + 0.5) - f`` per axis, with ``f`` the particle's fractional
    # position inside its own cell.
    fx = vp[0] - float(ci)
    fy = vp[1] - float(cj)
    fz = vp[2] - float(ck)

    # Keep a cell only if ``|e| * voxel_size - particle_radius < max_distance``.
    # Clamped to the cube's own reach: no cell in the cube has ``|e|`` above
    # sqrt(3) * (r + 1) < 2 * (r + 1), so clamping there excludes nothing, and
    # it stops SDF_SPLAT_NO_CULL from overflowing float32 when squared
    # (1e30 / h squared is ~1e63 -> inf -> garbage integer bounds).
    rad = wp.min((max_distance + particle_radius) / voxel_size,
                 2.0 * (float(r) + 1.0))
    rad2 = rad * rad

    dk0 = wp.max(-r, int(wp.ceil(fz - 0.5 - rad)) - 1)
    dk1 = wp.min(r, int(wp.floor(fz - 0.5 + rad)) + 1)
    for dk in range(dk0, dk1 + 1):
        ez = float(dk) + 0.5 - fz
        ry2 = rad2 - ez * ez
        if ry2 <= 0.0:
            continue
        ry = wp.sqrt(ry2)
        dj0 = wp.max(-r, int(wp.ceil(fy - 0.5 - ry)) - 1)
        dj1 = wp.min(r, int(wp.floor(fy - 0.5 + ry)) + 1)
        for dj in range(dj0, dj1 + 1):
            ey = float(dj) + 0.5 - fy
            rx2 = ry2 - ey * ey
            if rx2 <= 0.0:
                continue
            rx = wp.sqrt(rx2)
            di0 = wp.max(-r, int(wp.ceil(fx - 0.5 - rx)) - 1)
            di1 = wp.min(r, int(wp.floor(fx - 0.5 + rx)) + 1)
            for di in range(di0, di1 + 1):
                ijk = wp.vec3i(ci + di, cj + dj, ck + dk)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                if flat < 0:
                    continue
                cw = voxel_center_world(ijk, origin, voxel_size)
                d = wp.length(cw - p) - particle_radius
                wp.atomic_min(sdf, flat, d)


register_fn(k_build_surface_sdf, "fluidSimBuildSurfaceSdf",
            notes="Union-of-spheres particle splat with atomic min distance.")


# --- closest-point band propagation ------------------------------------------
#
# The splat pays particles x R^3 to fill the capillary support band.  These
# kernels pay active_voxels x passes instead: seed the small radius the
# particle support justifies, remember the winning particle per seeded voxel,
# then hand those points outward one cell at a time.  See SDF_BAND_MODES.


@wp.kernel
def k_surface_band_index_reset(
    num_leaves: wp.array(dtype=wp.int32),
    closest: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    closest[tid] = wp.int32(2147483647)


@wp.kernel
def k_seed_surface_sdf_closest(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    particle_radius: float,
    splat_radius_voxels: int,
    sdf: wp.array(dtype=float),
    closest: wp.array(dtype=wp.int32),
):
    """Record which particle produced each seeded voxel's distance.

    Runs the same loop and the same distance expression as
    ``k_build_surface_sdf``, against the field that kernel has already
    finished writing, so only a particle that actually achieved the stored
    minimum can pass the test.  Ties resolve to the lowest particle index,
    which makes the seeding deterministic regardless of atomic ordering.

    Kept as a second 32-bit pass rather than fused into the splat.  Fusing
    needs a 64-bit (distance, index) key, and on sm_89 the contended int64
    ``atomic_min`` that requires costs 2.9-3.6x more than the pass it saves
    (seed radius 4, 200k particles: 1.71 ms two-pass vs 5.05 ms fused).
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    p = positions[tid]
    vp = world_to_voxel(p, origin, voxel_size)
    ci = int(wp.floor(vp[0]))
    cj = int(wp.floor(vp[1]))
    ck = int(wp.floor(vp[2]))
    r = splat_radius_voxels
    for dk in range(-r, r + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                ijk = wp.vec3i(ci + di, cj + dj, ck + dk)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                if flat < 0:
                    continue
                cw = voxel_center_world(ijk, origin, voxel_size)
                d = wp.length(cw - p) - particle_radius
                if d <= sdf[flat]:
                    wp.atomic_min(closest, flat, tid)


@wp.kernel
def k_gather_surface_band_points(
    closest: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    num_leaves: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
):
    """Materialise the winning particle's position per voxel.

    Propagation reads this every pass; dereferencing the index through
    ``positions`` each time would add a second dependent load to the inner
    loop for no benefit.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    idx = closest[tid]
    if idx >= 0 and idx < wp.int32(2147483647):
        points[tid] = positions[idx]
    else:
        points[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_propagate_surface_sdf_band(
    src_sdf: wp.array(dtype=float),
    src_points: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    particle_radius: float,
    dst_sdf: wp.array(dtype=float),
    dst_points: wp.array(dtype=wp.vec3),
):
    """One outward pass of closest-point propagation over the active voxels.

    Every candidate distance is recomputed from *this* voxel's centre to the
    neighbour's closest particle, so the value stays an exact Euclidean
    union-of-spheres distance to a real particle.  It can only ever be an
    upper bound of the true field, and it is exact wherever the true nearest
    particle's point reaches the voxel.  It never interpolates, so it cannot
    manufacture a distance no particle supports.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    cw = voxel_center_world(ijk, origin, voxel_size)
    best = src_sdf[tid]
    best_point = src_points[tid]
    for dk in range(-1, 2):
        for dj in range(-1, 2):
            for di in range(-1, 2):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                n = wp.vec3i(ijk[0] + di, ijk[1] + dj, ijk[2] + dk)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
                if flat < 0:
                    continue
                if src_sdf[flat] >= SDF_BAND_INVALID:
                    continue
                cand = src_points[flat]
                d = wp.length(cw - cand) - particle_radius
                if d < best:
                    best = d
                    best_point = cand
    dst_sdf[tid] = best
    dst_points[tid] = best_point


@wp.kernel
def k_clamp_surface_sdf(
    num_leaves: wp.array(dtype=wp.int32),
    limit: float,
    sdf: wp.array(dtype=float),
):
    """Fold the unclamped propagation field back onto the splat's convention."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    sdf[tid] = wp.min(sdf[tid], limit)


register_fn(k_propagate_surface_sdf_band, "fluidSimPropagateSurfaceSdfBand",
            notes="Closest-point band extension for the capillary SDF support.")


# Surface smoothing filters


@wp.func
def _sdf_read(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    fallback: float,
):
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return fallback
    return src[flat]


@wp.kernel
def k_smooth_surface_sdf_dilate(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Morphological dilate of the liquid region: take min over 6-neighborhood."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    m = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            m = wp.min(m, _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center))
    dst[tid] = center + (m - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_dilate, "fluidSimSmoothSurfaceSdfDialate",
            notes="Min-filter dilation; source name keeps original spelling.")


@wp.kernel
def k_smooth_surface_sdf_erode(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Morphological erode of the liquid region: take max over 6-neighborhood."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    m = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            m = wp.max(m, _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center))
    dst[tid] = center + (m - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_erode, "fluidSimSmoothSurfaceSdfErode",
            notes="Max-filter erosion.")


@wp.kernel
def k_smooth_surface_sdf_box(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Box (mean) filter over the 7-point stencil, blended by strength."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    acc = center
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = wp.vec3i(ijk[0], ijk[1], ijk[2])
            n[axis] = n[axis] + step
            acc += _sdf_read(src, leaf_table, voxels_dim, leaves_dim, n, center)
    mean = acc / 7.0
    dst[tid] = center + (mean - center) * wp.clamp(strength, 0.0, 1.0)


register_fn(k_smooth_surface_sdf_box, "fluidSimSmoothSurfaceSdfBox",
            notes="7-point box smoothing.")


@wp.kernel
def k_smooth_surface_sdf_mean_curvature(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dst: wp.array(dtype=float),
):
    """Mean curvature flow smoothing step: phi += dt * kappa * |grad phi|.

    kappa*|grad| is approximated with the standard second-order stencil.
    Only voxels near the interface (|phi| < 3 voxels) are updated.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = src[tid]
    h = voxel_size
    if wp.abs(c) > 3.0 * h:
        dst[tid] = c
        return
    xp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]), c)
    xm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2]), c)
    yp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]), c)
    ym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2]), c)
    zp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] + 1), c)
    zm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] - 1), c)

    gx = (xp - xm) / (2.0 * h)
    gy = (yp - ym) / (2.0 * h)
    gz = (zp - zm) / (2.0 * h)
    gxx = (xp - 2.0 * c + xm) / (h * h)
    gyy = (yp - 2.0 * c + ym) / (h * h)
    gzz = (zp - 2.0 * c + zm) / (h * h)

    # Mixed derivatives.
    xpyp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1] + 1, ijk[2]), c)
    xpym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1] - 1, ijk[2]), c)
    xmyp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1] + 1, ijk[2]), c)
    xmym = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1] - 1, ijk[2]), c)
    xpzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2] + 1), c)
    xpzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2] - 1), c)
    xmzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2] + 1), c)
    xmzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] - 1, ijk[1], ijk[2] - 1), c)
    ypzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2] + 1), c)
    ypzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2] - 1), c)
    ymzp = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2] + 1), c)
    ymzm = _sdf_read(src, leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] - 1, ijk[2] - 1), c)

    gxy = (xpyp - xpym - xmyp + xmym) / (4.0 * h * h)
    gxz = (xpzp - xpzm - xmzp + xmzm) / (4.0 * h * h)
    gyz = (ypzp - ypzm - ymzp + ymzm) / (4.0 * h * h)

    g2 = gx * gx + gy * gy + gz * gz
    if g2 < 1.0e-12:
        dst[tid] = c
        return

    # kappa |grad phi| = (gxx(gy^2+gz^2) + gyy(gx^2+gz^2) + gzz(gx^2+gy^2)
    #                     - 2(gx gy gxy + gx gz gxz + gy gz gyz)) / g2
    num = (
        gxx * (gy * gy + gz * gz)
        + gyy * (gx * gx + gz * gz)
        + gzz * (gx * gx + gy * gy)
        - 2.0 * (gx * gy * gxy + gx * gz * gxz + gy * gz * gyz)
    )
    kappa_grad = num / g2
    # Stable explicit step: dt <= h^2/6, scaled by user strength.
    dt = wp.clamp(strength, 0.0, 1.0) * (h * h) / 6.0
    dst[tid] = c + dt * kappa_grad


register_fn(k_smooth_surface_sdf_mean_curvature, "fluidSimSmoothSurfaceSdfMeanCurvatureFlow",
            notes="Explicit mean curvature flow with stable step size.")


# SDF -> axis collision classification (per-face solid fractions/bits)


@wp.kernel
def k_sdf_to_axis_collision(
    sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    threshold: float,
    voxel_type: wp.array(dtype=wp.int32),
):
    """Classify collision voxels + per-axis face collision bits from an SDF.

    A voxel is solid when sdf < threshold. The face bits CollisionX/Y/Z mark
    faces whose lower-axis neighbor OR the voxel itself is solid; these are
    consumed by pressure projection and velocity masking.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = sdf[tid]

    vt = voxel_type[tid]
    vt &= ~(C.VOXEL_COLLISION | C.VOXEL_COLLISION_X | C.VOXEL_COLLISION_Y | C.VOXEL_COLLISION_Z)

    solid_c = c < threshold
    if solid_c:
        vt |= C.VOXEL_COLLISION
        vt &= ~(
            C.VOXEL_FLUID | C.VOXEL_SINK |
            C.VOXEL_SOURCE_REGION_TRACKING
        )

    for axis in range(3):
        n = wp.vec3i(ijk[0], ijk[1], ijk[2])
        n[axis] = n[axis] - 1
        flat_n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
        solid_n = True
        if flat_n >= 0:
            solid_n = sdf[flat_n] < threshold
        elif voxel_in_domain(n, voxels_dim):
            solid_n = False
        if solid_c or solid_n:
            if axis == 0:
                vt |= C.VOXEL_COLLISION_X
            elif axis == 1:
                vt |= C.VOXEL_COLLISION_Y
            else:
                vt |= C.VOXEL_COLLISION_Z
    voxel_type[tid] = vt


register_fn(k_sdf_to_axis_collision, "fluidSimSdfToAxisCollision",
            notes="Voxel + per-axis-face collision bits from collision SDF.")


@wp.kernel
def k_expand_shift_axis_velocity_in_collider(
    u_in: wp.array(dtype=float),
    v_in: wp.array(dtype=float),
    w_in: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    collision_vel: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    u_out: wp.array(dtype=float),
    v_out: wp.array(dtype=float),
    w_out: wp.array(dtype=float),
):
    """Replace face velocities on collision faces with the collider velocity
    and extrapolate fluid velocity one ring into the collider so that
    interpolation near walls sees sane values.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    vt = voxel_type[tid]

    cu = u_in[tid]
    cv = v_in[tid]
    cw = w_in[tid]

    if (vt & C.VOXEL_COLLISION_X) != 0:
        cu = collision_vel[tid][0]
    if (vt & C.VOXEL_COLLISION_Y) != 0:
        cv = collision_vel[tid][1]
    if (vt & C.VOXEL_COLLISION_Z) != 0:
        cw = collision_vel[tid][2]

    if (vt & C.VOXEL_COLLISION) != 0:
        # Average neighboring non-collision voxel velocities into the collider
        # cell (simple one-ring extrapolation), keeping the normal component
        # from the collider so separation is preserved.
        acc = wp.vec3(0.0)
        n_count = float(0.0)
        for axis in range(3):
            for step in range(-1, 2, 2):
                n = wp.vec3i(ijk[0], ijk[1], ijk[2])
                n[axis] = n[axis] + step
                flat_n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
                if flat_n < 0:
                    continue
                if (voxel_type[flat_n] & C.VOXEL_COLLISION) != 0:
                    continue
                acc += wp.vec3(u_in[flat_n], v_in[flat_n], w_in[flat_n])
                n_count += 1.0
        if n_count > 0.0:
            avg = acc / n_count
            if (vt & C.VOXEL_COLLISION_X) == 0:
                cu = avg[0]
            if (vt & C.VOXEL_COLLISION_Y) == 0:
                cv = avg[1]
            if (vt & C.VOXEL_COLLISION_Z) == 0:
                cw = avg[2]

    u_out[tid] = cu
    v_out[tid] = cv
    w_out[tid] = cw


register_fn(k_expand_shift_axis_velocity_in_collider,
            "fluidSimExpandAndShiftAxisVelocityInColliderAxis",
            notes="Collider face velocity override + one-ring extrapolation.")


# Marching cubes surface extraction (host orchestrated, GPU compute)


class SurfaceExtractor:
    """Extracts a triangle mesh from the liquid surface SDF using
    warp's built-in marching cubes on a dense sub-grid.

    The dense region is bounded by the axis-aligned bounding box of the
    active leaves (plus a small background margin) instead of the whole
    domain, so extraction cost scales with the liquid extent. Sub-grid
    dimensions are padded to multiples of ``_PAD`` and grow monotonically
    within a session to avoid reallocating the marching cubes buffers
    every frame.
    """

    _PAD = 32

    def __init__(self, grid, device, max_dense_dim=256):
        self.grid = grid
        self.device = device
        self.max_dense_dim = int(max_dense_dim)
        self._mc = None
        self._dense = None
        self._dense_dim = None
        self._bounds = wp.zeros(6, dtype=wp.int32, device=device)

    def _ensure_dense(self, dims):
        nx, ny, nz = dims
        if self._dense is not None and self._dense_dim == (nx, ny, nz):
            return
        self._dense = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self._dense_dim = (nx, ny, nz)
        self._mc = wp.MarchingCubes(
            nx=nx, ny=ny, nz=nz,
            max_verts=nx * ny * nz,
            max_tris=2 * nx * ny * nz,
            device=self.device,
        )

    def reset_buffers(self):
        """Release cached dense extraction buffers after a simulation reset."""
        self._mc = None
        self._dense = None
        self._dense_dim = None

    def _empty(self):
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))

    def _active_voxel_box(self):
        """Voxel-space (lo, hi exclusive) AABB of active leaves, or None."""
        g = self.grid
        n_leaves = int(getattr(g, "active_leaf_count", 0))
        if n_leaves <= 0:
            return None
        wp.launch(
            k_leaf_coord_bounds_init, dim=1,
            inputs=[g.voxels_dim], outputs=[self._bounds],
            device=self.device,
        )
        wp.launch(
            k_leaf_coord_bounds, dim=n_leaves,
            inputs=[g.leaf_coords, g.num_leaves],
            outputs=[self._bounds],
            device=self.device,
        )
        b = self._bounds.numpy()
        if b[3] < b[0]:  # no valid leaf wrote bounds
            return None
        lo = b[:3] * C.LEAF_DIM
        hi = (b[3:] + 1) * C.LEAF_DIM
        return lo, hi

    def extract(self, sdf_field, iso=0.0):
        g = self.grid
        box = self._active_voxel_box()
        if box is None:
            return self._empty()
        lo, hi = box
        vdim = np.array([int(g.voxels_dim[0]), int(g.voxels_dim[1]),
                         int(g.voxels_dim[2])], dtype=np.int64)
        # Margin of background voxels so marching cubes closes the surface.
        lo = np.maximum(lo - 2, 0)
        hi = np.minimum(hi + 2, vdim)
        span = hi - lo
        if np.any(span > self.max_dense_dim):
            raise RuntimeError(
                "FluidSim: surface extraction dense span "
                f"{tuple(int(x) for x in span)} exceeds Surface Max Dense Dim "
                f"{int(self.max_dense_dim)}; increase Surface Max Dense Dim "
                "or lower the surface resolution multiplier."
            )
        # Pad to stable bucket sizes (monotonic growth caps reallocations).
        dims = np.minimum(
            np.minimum(-(-span // self._PAD) * self._PAD, vdim),
            self.max_dense_dim)
        if self._dense_dim is not None:
            dims = np.maximum(dims, np.minimum(
                np.array(self._dense_dim, dtype=np.int64),
                np.minimum(vdim, self.max_dense_dim)))
        dims = np.maximum(dims, np.minimum(np.minimum(8, vdim),
                                           self.max_dense_dim))
        # Anchor the padded box on lo; shift back when it overruns the domain.
        lo = np.maximum(np.minimum(lo, vdim - dims), 0)
        nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
        self._ensure_dense((nx, ny, nz))
        wp.launch(
            k_sparse_to_dense_sdf,
            dim=nx * ny * nz,
            inputs=[
                sdf_field.data, g.leaf_table, g.voxels_dim, g.leaves_dim,
                wp.vec3i(nx, ny, nz),
                wp.vec3i(int(lo[0]), int(lo[1]), int(lo[2])),
                4.0 * g.voxel_size,
            ],
            outputs=[self._dense],
            device=self.device,
        )
        self._mc.surface(self._dense, iso)
        verts = self._mc.verts.numpy()
        tris = self._mc.indices.numpy().reshape(-1, 3)
        # Vertices are in dense sub-box index space; convert to world.
        vs = np.float32(g.voxel_size)
        offset = (np.asarray(g.origin, dtype=np.float32)
                  + (lo.astype(np.float32) + np.float32(0.5)) * vs)
        verts = np.asarray(verts, dtype=np.float32)
        verts *= vs
        verts += offset
        return verts, tris


@wp.kernel
def k_leaf_coord_bounds_init(
    voxels_dim: wp.vec3i,
    bounds: wp.array(dtype=wp.int32),
):
    for a in range(3):
        bounds[a] = voxels_dim[a]  # min slots start above any leaf coord
        bounds[3 + a] = -1  # max slots start below any leaf coord


@wp.kernel
def k_leaf_coord_bounds(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    bounds: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0]:
        return
    lc = leaf_coords[tid]
    for a in range(3):
        wp.atomic_min(bounds, a, lc[a])
        wp.atomic_max(bounds, 3 + a, lc[a])


@wp.kernel
def k_sparse_to_dense_sdf(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    dense_dim: wp.vec3i,
    dense_origin: wp.vec3i,
    background: float,
    dense: wp.array3d(dtype=float),
):
    tid = wp.tid()
    nx = dense_dim[0]
    ny = dense_dim[1]
    nz = dense_dim[2]
    if tid >= nx * ny * nz:
        return
    k = tid // (nx * ny)
    j = (tid - k * nx * ny) // nx
    i = tid - k * nx * ny - j * nx
    ijk = wp.vec3i(i + dense_origin[0], j + dense_origin[1],
                   k + dense_origin[2])
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    val = background
    if flat >= 0:
        val = src[flat]
    dense[i, j, k] = val
