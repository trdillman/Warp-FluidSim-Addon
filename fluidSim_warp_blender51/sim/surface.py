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


class SurfacePipeline:
    """Owns liquid-surface fields, cadence, extraction and mesh caching."""

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
        from .enums import SurfaceResolutionMode

        sim_h = max(float(settings.voxel_size), 1.0e-9)
        mode = int(getattr(settings, "surface_resolution_mode", int(SurfaceResolutionMode.SIMULATION)))
        if mode == int(SurfaceResolutionMode.CUSTOM):
            custom_h = float(getattr(settings, "surface_voxel_size", 0.0))
            if custom_h > 0.0:
                return max(custom_h, sim_h / 16.0)
        if mode == int(SurfaceResolutionMode.MULTIPLIER):
            return sim_h / max(0.25, min(float(getattr(settings, "surface_resolution_multiplier", 1.0)), 8.0))
        return sim_h

    def voxel_size(self):
        return self.voxel_size_for(self.solver.settings)

    def leaf_capacity(self, h):
        s = self.solver.settings
        size = np.maximum(np.asarray(s.domain_max, dtype=np.float64) - np.asarray(s.domain_min, dtype=np.float64), h)
        dims = [max(C.LEAF_DIM, int(np.ceil(v / h))) for v in size]
        dims = [(v + C.LEAF_MASK) & ~C.LEAF_MASK for v in dims]
        total = (dims[0] >> C.LEAF_SHIFT) * (dims[1] >> C.LEAF_SHIFT) * (dims[2] >> C.LEAF_SHIFT)
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
                bool(getattr(s, "surface_build_lowres_sdf", True)), round(float(s.particle_radius_scale), 6), int(s.surface_splat_radius),
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
        wp.launch(k_surface_sdf_reset, dim=nvox, inputs=[sdf.data, grid.num_leaves, 4.0 * h], device=solver.device)
        radius = 0.5 * float(s.voxel_size) * float(s.particle_radius_scale)
        splat = max(1, int(s.surface_splat_radius), int(math.ceil(radius / h + 1.0)))
        wp.launch(k_build_surface_sdf, dim=solver._np_dim(), inputs=[positions, solver.particles.count, grid.leaf_table,
                  grid.voxels_dim, grid.leaves_dim, grid.origin, h, radius, splat], outputs=[sdf.data], device=solver.device)
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
            build_lowres = bool(getattr(s, "surface_build_lowres_sdf", True))
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
            if s.surface_enabled and cadence == int(SurfaceUpdateCadence.SUBSTEP):
                self.build(build_render=True, build_lowres=not s.whitewater_enabled)
        return sdf_ready

    def after_frame(self):
        from .enums import SurfaceUpdateCadence

        s = self.solver.settings
        cadence = int(getattr(s, "surface_update_cadence", int(SurfaceUpdateCadence.FRAME)))
        if s.surface_enabled and cadence == int(SurfaceUpdateCadence.FRAME):
            with self.solver._phase("surface"):
                self.build(build_render=True, build_lowres=bool(
                    getattr(s, "surface_build_lowres_sdf", True)))

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


# ---------------------------------------------------------------------------
# Surface SDF build (union of particle spheres, splatted into the grid)
# ---------------------------------------------------------------------------


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
    sdf: wp.array(dtype=float),
):
    """Union-of-spheres distance splat. sdf holds distance-to-surface
    (negative inside), quantized to int via atomic_min on scaled values.

    We store float distances directly using atomic_min emulation:
    warp supports wp.atomic_min on float arrays, so use it directly.
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
                wp.atomic_min(sdf, flat, d)


register_fn(k_build_surface_sdf, "fluidSimBuildSurfaceSdf",
            notes="Union-of-spheres particle splat with atomic min distance.")


# ---------------------------------------------------------------------------
# Surface smoothing filters
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SDF -> axis collision classification (per-face solid fractions/bits)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Marching cubes surface extraction (host orchestrated, GPU compute)
# ---------------------------------------------------------------------------


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
