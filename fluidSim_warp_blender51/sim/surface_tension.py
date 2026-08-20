"""Continuum-surface-force surface tension for FluidSim.

The implementation uses the simulation-grid liquid signed-distance field (SDF):

* the outward interface normal is ``normalize(grad(phi))``;
* signed curvature is ``-div(normal)`` so a spherical drop receives an inward
  force with the negative-inside SDF convention used by :mod:`sim.surface`;
* a compact cosine delta distributes the interface force over a configurable
  narrow band;
* cell-centred acceleration is averaged onto the staggered MAC faces.

Only two temporary ``vec3`` fields are allocated.  They are reused as the
normal ping-pong buffers and the final acceleration field, so the persistent
scratch cost is 24 bytes per allocated sparse-grid voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .fields import field_read_f, voxel_coord_of_flat, voxel_flat_index
from .registry import register_fn

from .surface_tension_math import (
    capillary_dt,
    compact_cosine_delta,
    normal_support_voxels,
    rayleigh_lamb_frequency,
    required_capillary_substeps,
    required_sdf_splat_radius_voxels,
    required_topology_expand_leaves,
)

__all__ = (
    "DEFAULT_CURVATURE_CLAMP_VOXELS",
    "SurfaceTensionDiagnostics",
    "SurfaceTensionSolver",
    "capillary_dt",
    "compact_cosine_delta",
    "normal_support_voxels",
    "rayleigh_lamb_frequency",
    "required_capillary_substeps",
    "required_sdf_splat_radius_voxels",
    "required_topology_expand_leaves",
)

DEFAULT_CURVATURE_CLAMP_VOXELS = 2.0


@wp.func
def _read_vec3(
    data: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    default: wp.vec3,
) -> wp.vec3:
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return default
    return data[flat]


@wp.kernel
def k_surface_tension_normals(
    sdf: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    normal_band: float,
    normals: wp.array(dtype=wp.vec3),
):
    """Compute outward SDF normals in and around the force band."""

    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    normals[tid] = wp.vec3(0.0, 0.0, 0.0)
    center = sdf[tid]
    if wp.abs(center) > normal_band or collision_sdf[tid] < 0.0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    h = voxel_size
    xp = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]), center)
    xm = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0] - 1, ijk[1], ijk[2]), center)
    yp = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]), center)
    ym = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0], ijk[1] - 1, ijk[2]), center)
    zp = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0], ijk[1], ijk[2] + 1), center)
    zm = field_read_f(sdf, leaf_table, voxels_dim, leaves_dim,
                      wp.vec3i(ijk[0], ijk[1], ijk[2] - 1), center)
    gradient = wp.vec3(xp - xm, yp - ym, zp - zm) / (2.0 * h)
    magnitude = wp.length(gradient)
    if magnitude > 1.0e-6:
        normals[tid] = gradient / magnitude


register_fn(
    k_surface_tension_normals,
    "fluidSimSurfaceTensionNormals",
    notes="SDF gradient normalized inside the capillary narrow band.",
)


@wp.kernel
def k_surface_tension_smooth_normals(
    src: wp.array(dtype=wp.vec3),
    sdf: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    normal_band: float,
    dst: wp.array(dtype=wp.vec3),
):
    """Normalize a 7-point box average of the interface normal field."""

    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    dst[tid] = wp.vec3(0.0, 0.0, 0.0)
    if wp.abs(sdf[tid]) > normal_band or collision_sdf[tid] < 0.0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = src[tid]
    accumulated = center
    count = 1.0
    for axis in range(3):
        for step in range(-1, 2, 2):
            neighbor = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if axis == 0:
                neighbor = wp.vec3i(ijk[0] + step, ijk[1], ijk[2])
            elif axis == 1:
                neighbor = wp.vec3i(ijk[0], ijk[1] + step, ijk[2])
            else:
                neighbor = wp.vec3i(ijk[0], ijk[1], ijk[2] + step)
            value = _read_vec3(
                src, leaf_table, voxels_dim, leaves_dim,
                neighbor, wp.vec3(0.0, 0.0, 0.0))
            if wp.length(value) > 1.0e-6:
                accumulated += value
                count += 1.0
    averaged = accumulated / count
    magnitude = wp.length(averaged)
    if magnitude > 1.0e-6:
        dst[tid] = averaged / magnitude


register_fn(
    k_surface_tension_smooth_normals,
    "fluidSimSurfaceTensionSmoothNormals",
    notes="Optional normalized 7-point smoothing for SDF normals.",
)


@wp.kernel
def k_surface_tension_acceleration(
    normals: wp.array(dtype=wp.vec3),
    sdf: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    half_width: float,
    coefficient: float,
    density: float,
    curvature_limit: float,
    acceleration: wp.array(dtype=wp.vec3),
    active_count: wp.array(dtype=wp.int32),
    max_abs_curvature: wp.array(dtype=float),
    max_acceleration: wp.array(dtype=float),
):
    """Evaluate ``sigma * (-div(n)) * delta(phi) * n / density``."""

    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    acceleration[tid] = wp.vec3(0.0, 0.0, 0.0)
    phi = sdf[tid]
    if wp.abs(phi) >= half_width or collision_sdf[tid] < 0.0:
        return
    center = normals[tid]
    if wp.length(center) <= 1.0e-6:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    nxp = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]), center)
    nxm = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0] - 1, ijk[1], ijk[2]), center)
    nyp = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]), center)
    nym = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0], ijk[1] - 1, ijk[2]), center)
    nzp = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0], ijk[1], ijk[2] + 1), center)
    nzm = _read_vec3(normals, leaf_table, voxels_dim, leaves_dim,
                     wp.vec3i(ijk[0], ijk[1], ijk[2] - 1), center)
    divergence = (
        nxp[0] - nxm[0] + nyp[1] - nym[1] + nzp[2] - nzm[2]
    ) / (2.0 * voxel_size)
    curvature = wp.clamp(-divergence, -curvature_limit, curvature_limit)
    delta = 0.5 * (1.0 + wp.cos(3.141592653589793 * phi / half_width)) / half_width
    value = center * (coefficient * curvature * delta / density)
    magnitude = wp.length(value)
    # Both comparisons are false for NaN; reject it as well as overflow.
    if not (magnitude >= 0.0 and magnitude < 1.0e30):
        return
    acceleration[tid] = value
    wp.atomic_add(active_count, 0, 1)
    wp.atomic_max(max_abs_curvature, 0, wp.abs(curvature))
    wp.atomic_max(max_acceleration, 0, magnitude)


register_fn(
    k_surface_tension_acceleration,
    "fluidSimSurfaceTensionAcceleration",
    notes="Compact cosine CSF acceleration with bounded curvature.",
)


@wp.kernel
def k_surface_tension_accumulate_frame_active_count(
    active_count: wp.array(dtype=wp.int32),
    frame_max_active_count: wp.array(dtype=wp.int32),
):
    """Retain the maximum active interface-voxel count across substeps."""

    if wp.tid() == 0:
        wp.atomic_max(frame_max_active_count, 0, active_count[0])


register_fn(
    k_surface_tension_accumulate_frame_active_count,
    "fluidSimSurfaceTensionAccumulateFrameActiveCount",
    notes="Device-side frame maximum for capillary active voxels.",
)


@wp.kernel
def k_apply_surface_tension_to_mac(
    acceleration: wp.array(dtype=wp.vec3),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    dt: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
):
    """Average cell acceleration to each lower MAC face and integrate it."""

    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    if collision_sdf[tid] < 0.0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    current = acceleration[tid]

    xm_ijk = wp.vec3i(ijk[0] - 1, ijk[1], ijk[2])
    ym_ijk = wp.vec3i(ijk[0], ijk[1] - 1, ijk[2])
    zm_ijk = wp.vec3i(ijk[0], ijk[1], ijk[2] - 1)
    xm = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, xm_ijk)
    ym = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ym_ijk)
    zm = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, zm_ijk)

    if wu[tid] > 0.0:
        value = current[0]
        if xm >= 0 and collision_sdf[xm] >= 0.0:
            value = 0.5 * (value + acceleration[xm][0])
        u[tid] += value * dt
    if wv[tid] > 0.0:
        value = current[1]
        if ym >= 0 and collision_sdf[ym] >= 0.0:
            value = 0.5 * (value + acceleration[ym][1])
        v[tid] += value * dt
    if ww[tid] > 0.0:
        value = current[2]
        if zm >= 0 and collision_sdf[zm] >= 0.0:
            value = 0.5 * (value + acceleration[zm][2])
        w[tid] += value * dt


register_fn(
    k_apply_surface_tension_to_mac,
    "fluidSimApplySurfaceTensionToMac",
    notes="Cell-centred capillary acceleration averaged onto weighted MAC faces.",
)


@dataclass(frozen=True)
class SurfaceTensionDiagnostics:
    active_voxels: int = 0
    max_abs_curvature: float = 0.0
    max_acceleration: float = 0.0
    temporary_bytes: int = 0
    submission_seconds: float = 0.0
    status: str = "DISABLED"
    message: str = ""


class SurfaceTensionSolver:
    """Lazy resources and launch sequencing for the CSF implementation."""

    def __init__(self, solver):
        self.solver = solver
        self._normal_a = None
        self._normal_b = None
        self._active_count = wp.zeros(1, dtype=wp.int32, device=solver.device)
        self._frame_max_active_count = wp.zeros(
            1, dtype=wp.int32, device=solver.device)
        self._max_abs_curvature = wp.zeros(1, dtype=float, device=solver.device)
        self._max_acceleration = wp.zeros(1, dtype=float, device=solver.device)
        self._last_status = "DISABLED"
        self._last_message = ""
        self._submission_seconds = 0.0
        self._accounted_bytes = 0

    @property
    def temporary_bytes(self) -> int:
        return int(self._accounted_bytes)

    def _enabled(self) -> bool:
        settings = self.solver.settings
        return bool(getattr(settings, "surface_tension_enabled", False)) and float(
            getattr(settings, "surface_tension_coefficient", 0.0)
        ) > 0.0

    def _set_accounted_bytes(self, value: int) -> None:
        value = max(0, int(value))
        delta = value - int(self._accounted_bytes)
        profiler = getattr(self.solver, "profiler", None)
        if profiler is not None and delta:
            profiler.tally_buffer(delta)
        self._accounted_bytes = value

    def _release_resources(self) -> None:
        self._normal_a = None
        self._normal_b = None
        self._set_accounted_bytes(0)

    def ensure_resources(self) -> None:
        capacity = int(self.solver.grid.voxel_capacity)
        if self._normal_a is not None and len(self._normal_a) == capacity:
            return
        self._release_resources()
        # Allocate transactionally: a failure while creating the second buffer
        # must not leave one unaccounted device allocation attached to the solver.
        normal_a = wp.zeros(capacity, dtype=wp.vec3, device=self.solver.device)
        normal_b = wp.zeros(capacity, dtype=wp.vec3, device=self.solver.device)
        self._normal_a = normal_a
        self._normal_b = normal_b
        self._set_accounted_bytes(capacity * 2 * 3 * 4)

    def reset(self) -> None:
        self._last_status = "DISABLED"
        self._last_message = ""
        self._submission_seconds = 0.0
        self._active_count.zero_()
        self._frame_max_active_count.zero_()
        self._max_abs_curvature.zero_()
        self._max_acceleration.zero_()
        if self._enabled():
            self.ensure_resources()
            self._normal_a.zero_()
            self._normal_b.zero_()
        else:
            # Release the 24-byte/voxel optional scratch allocation when the
            # feature is disabled through a solver reset.
            self._release_resources()

    def release(self) -> None:
        """Release optional buffers and break the owning solver reference."""

        self._release_resources()
        self.solver = None

    def apply(self, dt: float) -> bool:
        settings = self.solver.settings
        if not self._enabled():
            # Coefficient is a live setting. Dropping it to zero must release
            # the optional per-voxel scratch arrays immediately rather than
            # retaining both memory and profiler accounting until Reset.
            self._release_resources()
            self._last_status = "DISABLED"
            self._last_message = ""
            return False
        if self.solver.particles.host_count <= 0:
            self._last_status = "NO_LIQUID"
            self._last_message = "No active liquid particles"
            self._active_count.zero_()
            return False
        if not self.solver.surface.ensure_solver_sdf_current():
            self._last_status = "SDF_UNAVAILABLE"
            self._last_message = self.solver.surface.solver_sdf_error
            raise RuntimeError(self._last_message)

        self.ensure_resources()
        assert self._normal_a is not None and self._normal_b is not None
        grid = self.solver.grid
        half_width = max(
            float(settings.surface_tension_band_voxels), 0.5
        ) * float(grid.voxel_size)
        smoothing_iterations = max(
            0, int(settings.surface_tension_smoothing_iterations))
        # Curvature samples one cell beyond the force band. Each smoothing pass
        # adds another stencil radius, so compute enough normal support to avoid
        # truncating the final field at the force-band boundary.
        normal_band = normal_support_voxels(
            float(settings.surface_tension_band_voxels),
            smoothing_iterations,
        ) * float(grid.voxel_size)
        curvature_limit = (
            DEFAULT_CURVATURE_CLAMP_VOXELS / float(grid.voxel_size)
        )
        start = time.perf_counter()
        wp.launch(
            k_surface_tension_normals,
            dim=self.solver._nvox,
            inputs=[
                self.solver.surface_sdf.data,
                self.solver.collision_sdf.data,
                grid.leaf_table,
                grid.leaf_coords,
                grid.voxels_dim,
                grid.leaves_dim,
                grid.num_leaves,
                grid.voxel_size,
                normal_band,
            ],
            outputs=[self._normal_a],
            device=self.solver.device,
        )
        current = self._normal_a
        other = self._normal_b
        for _ in range(smoothing_iterations):
            wp.launch(
                k_surface_tension_smooth_normals,
                dim=self.solver._nvox,
                inputs=[
                    current,
                    self.solver.surface_sdf.data,
                    self.solver.collision_sdf.data,
                    grid.leaf_table,
                    grid.leaf_coords,
                    grid.voxels_dim,
                    grid.leaves_dim,
                    grid.num_leaves,
                    normal_band,
                ],
                outputs=[other],
                device=self.solver.device,
            )
            current, other = other, current

        # ``other`` is no longer needed as a normal field and becomes the
        # acceleration field.  Device resets are ordered before the kernel;
        # the final frame readback therefore reflects the latest substep.
        self._active_count.zero_()
        wp.launch(
            k_surface_tension_acceleration,
            dim=self.solver._nvox,
            inputs=[
                current,
                self.solver.surface_sdf.data,
                self.solver.collision_sdf.data,
                grid.leaf_table,
                grid.leaf_coords,
                grid.voxels_dim,
                grid.leaves_dim,
                grid.num_leaves,
                grid.voxel_size,
                half_width,
                float(settings.surface_tension_coefficient),
                float(settings.surface_tension_density),
                curvature_limit,
            ],
            outputs=[
                other,
                self._active_count,
                self._max_abs_curvature,
                self._max_acceleration,
            ],
            device=self.solver.device,
        )
        wp.launch(
            k_surface_tension_accumulate_frame_active_count,
            dim=1,
            inputs=[self._active_count],
            outputs=[self._frame_max_active_count],
            device=self.solver.device,
        )
        wp.launch(
            k_apply_surface_tension_to_mac,
            dim=self.solver._nvox,
            inputs=[
                other,
                self.solver.collision_sdf.data,
                grid.leaf_table,
                grid.leaf_coords,
                grid.voxels_dim,
                grid.leaves_dim,
                grid.num_leaves,
                float(dt),
                self.solver.vel.u,
                self.solver.vel.v,
                self.solver.vel.w,
                self.solver.vel.wu,
                self.solver.vel.wv,
                self.solver.vel.ww,
            ],
            device=self.solver.device,
        )
        self._submission_seconds += time.perf_counter() - start
        self._last_status = "ACTIVE"
        self._last_message = ""
        return True

    def diagnostics(self) -> SurfaceTensionDiagnostics:
        if not self._enabled():
            return SurfaceTensionDiagnostics(status="DISABLED")
        return SurfaceTensionDiagnostics(
            active_voxels=int(self._frame_max_active_count.numpy()[0]),
            max_abs_curvature=float(self._max_abs_curvature.numpy()[0]),
            max_acceleration=float(self._max_acceleration.numpy()[0]),
            temporary_bytes=self.temporary_bytes,
            submission_seconds=float(self._submission_seconds),
            status=self._last_status,
            message=self._last_message,
        )

    def begin_frame(self) -> None:
        self._submission_seconds = 0.0
        self._active_count.zero_()
        self._frame_max_active_count.zero_()
        self._max_abs_curvature.zero_()
        self._max_acceleration.zero_()
        if self._enabled():
            self._last_status = "PENDING"
            self._last_message = ""
        else:
            self._last_status = "DISABLED"
            self._last_message = ""
