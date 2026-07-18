"""FluidSim solver orchestration: GPU-resident FLIP/PIC pipeline.

All simulation state lives on the GPU (wp.array); the host only sequences
kernel launches, reads back small statistics for the UI/profiler, and
receives surface mesh / particle data for Blender display.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .enums import (
    ParticleSampleType,
    TraceMode,
    ProjectNonDivergentSolver,
    ViscositySolver as ViscositySolverEnum,
    RelaxMode,
    SdfFilterType,
    SurfaceResolutionMode,
    SurfaceUpdateCadence,
    CustomKernelOrder,
)
from .fields import (
    SparseGrid,
    ScalarField,
    MacVelocityField,
    voxel_coord_of_flat,
    voxel_center_world,
)
from .particles import ParticleSystem, ParticleBins
from .pressure import PressureSolver
from .viscosity import ViscositySolver
from .whitewater import (
    WhitewaterSystem,
    k_compute_churn,
    k_emit_whitewater,
    k_advect_whitewater,
    k_whitewater_set_grid_data,
)
from .surface import (
    k_surface_sdf_reset,
    k_build_surface_sdf,
    k_smooth_surface_sdf_dilate,
    k_smooth_surface_sdf_erode,
    k_smooth_surface_sdf_box,
    k_smooth_surface_sdf_mean_curvature,
    k_sdf_to_axis_collision,
    k_expand_shift_axis_velocity_in_collider,
    SurfacePipeline,
)
from .sourcing import HeightSourceDesc, SourcePipeline, SourceShapeDesc
from .kernels_core import (
    k_particle_to_grid_collect,
    k_particle_to_grid_average,
    k_max_vec3_magnitude,
)
from .kernels_fluid import (
    k_particle_set_grid_data,
    k_particle_set_grid_data_spacetime,
    k_spacetime_weights_to_phase_active,
    k_normalize_mac,
    k_particle_to_count,
    k_particle_to_grid_active,
    k_staggered_field_to_particle,
    k_apply_collider_material_to_grid,
    k_advect_particles,
    k_advect_particles_spacetime,
    k_resync_particle_positions_spacetime,
    k_forces_gravity,
    k_forces_speed_limit,
    k_particle_to_voxel_count_and_update_index,
    voxel_count_to_starts,
    k_scatter_particles_to_bins,
    k_particle_age,
    k_particle_lifespan,
    k_particle_in_sink_check,
    k_particles_relax,
    k_clear_just_initialized_tracked,
)
from .custom import SnippetManager


# ---------------------------------------------------------------------------
# ST-FLIP host-side calibration helpers
# ---------------------------------------------------------------------------


def _st_poly6_1d_host(r: float) -> float:
    q = 1.0 - r * r
    if q <= 0.0:
        return 0.0
    return q * q * q


def _st_spatial_weight_1d_host(r: float) -> float:
    return (35.0 / 32.0) * _st_poly6_1d_host(r)


def _st_temporal_weight_host(tau: float) -> float:
    if tau > 0.5:
        return 0.0
    return (35.0 / 16.0) * _st_poly6_1d_host(tau - 0.5)


def _motion_leaf_radius(max_speed: float, frame_dt: float,
                        gravity_magnitude: float, voxel_size: float) -> int:
    """Conservative leaf radius for ballistic motion over one frame."""
    dt = max(0.0, float(frame_dt))
    travel = (max(0.0, float(max_speed)) * dt
              + 0.5 * max(0.0, float(gravity_magnitude)) * dt * dt)
    if travel <= 0.0:
        return 0
    leaf_world_size = max(float(voxel_size) * C.LEAF_DIM, 1.0e-8)
    return int(math.ceil(travel / leaf_world_size)) + 1


# ---------------------------------------------------------------------------
# Small helper kernels owned by the orchestrator
# ---------------------------------------------------------------------------


@wp.kernel
def k_fill_float(data: wp.array(dtype=float), value: float):
    data[wp.tid()] = value


@wp.kernel
def k_mark_domain_walls(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    collision_sdf: wp.array(dtype=float),
):
    """Make closed-domain walls solid in the collision SDF.

    The sparse grid pads voxels_dim up to leaf multiples, so allocated
    voxels can exist beyond the configured domain box. Without this, the
    pressure solve treats that padding as open space and fluid compresses
    against the particle clamp at the domain boundary.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    wc = voxel_center_world(ijk, origin, voxel_size)
    # Signed distance to the domain-exterior solid: negative outside the box.
    dx = wp.max(domain_min[0] - wc[0], wc[0] - domain_max[0])
    dy = wp.max(domain_min[1] - wc[1], wc[1] - domain_max[1])
    dz = wp.max(domain_min[2] - wc[2], wc[2] - domain_max[2])
    outside = wp.max(dx, wp.max(dy, dz))
    phi = -outside
    if phi < collision_sdf[tid]:
        collision_sdf[tid] = phi


@wp.kernel
def k_remap_leaf_field(
    old_data: wp.array(dtype=float),
    old_leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    new_data: wp.array(dtype=float),
):
    """Carry a persistent voxel field across a topology rebuild."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    leaf = tid // C.LEAF_VOLUME
    local = tid - leaf * C.LEAF_VOLUME
    lc = leaf_coords[leaf]
    dense = (lc[2] * leaves_dim[1] + lc[1]) * leaves_dim[0] + lc[0]
    old_slot = old_leaf_table[dense]
    if old_slot < 0:
        new_data[tid] = 0.0
        return
    new_data[tid] = old_data[old_slot * C.LEAF_VOLUME + local]


@wp.kernel
def k_rasterize_collider_material(
    mesh_id: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    band: float,
    linear_velocity: wp.vec3,
    angular_velocity: wp.vec3,
    center: wp.vec3,
    friction: float,
    restitution: float,
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    lc = leaf_coords[tid // C.LEAF_VOLUME]
    local = tid - (tid // C.LEAF_VOLUME) * C.LEAF_VOLUME
    li = local & C.LEAF_MASK
    lj = (local >> C.LEAF_SHIFT) & C.LEAF_MASK
    lk = local >> (2 * C.LEAF_SHIFT)
    ijk = wp.vec3i(lc[0] * C.LEAF_DIM + li, lc[1] * C.LEAF_DIM + lj, lc[2] * C.LEAF_DIM + lk)
    p = origin + wp.vec3(
        (float(ijk[0]) + 0.5) * voxel_size,
        (float(ijk[1]) + 0.5) * voxel_size,
        (float(ijk[2]) + 0.5) * voxel_size,
    )
    q = wp.mesh_query_point_sign_normal(mesh_id, p, band)
    if not q.result:
        return
    cp = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
    distance = wp.length(p - cp) * q.sign
    if distance < collision_sdf[tid]:
        collision_sdf[tid] = distance
        collision_vel[tid] = linear_velocity + wp.cross(angular_velocity, p - center)
        collision_friction[tid] = friction
        collision_restitution[tid] = restitution


@wp.kernel
def k_copy_float(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def k_sanitize_mac_velocity(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    max_speed: float,
):
    """Sanitize MAC velocity faces: kill inf/nan and clamp magnitude.

    Guards against rare pressure-projection blowups on thin fluid slivers
    adjacent to domain walls; without this, FLIP transfer propagates a
    single corrupt face into many particles and the sim explodes.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return

    val = u[tid]
    if wp.isnan(val) or wp.isinf(val):
        u[tid] = 0.0
        if wu[tid] > 1.0e-8:
            wu[tid] = 1.0e-6
    elif wu[tid] > 1.0e-8:
        u[tid] = wp.clamp(val, -max_speed, max_speed)

    val = v[tid]
    if wp.isnan(val) or wp.isinf(val):
        v[tid] = 0.0
        if wv[tid] > 1.0e-8:
            wv[tid] = 1.0e-6
    elif wv[tid] > 1.0e-8:
        v[tid] = wp.clamp(val, -max_speed, max_speed)

    val = w[tid]
    if wp.isnan(val) or wp.isinf(val):
        w[tid] = 0.0
        if ww[tid] > 1.0e-8:
            ww[tid] = 1.0e-6
    elif ww[tid] > 1.0e-8:
        w[tid] = wp.clamp(val, -max_speed, max_speed)


@wp.kernel
def k_sanitize_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    max_speed: float,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
):
    """Kill particles with non-finite state; clamp finite speeds."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if alive[tid] == 0:
        return
    p = positions[tid]
    v = velocities[tid]
    bad = False
    for a in range(3):
        if wp.isnan(p[a]) or wp.isinf(p[a]) or wp.isnan(v[a]) or wp.isinf(v[a]):
            bad = True
    if bad:
        alive[tid] = 0
        return
    speed = wp.length(v)
    if speed > max_speed:
        velocities[tid] = v * (max_speed / speed)


# ---------------------------------------------------------------------------
# Host-side source / collider descriptors (filled by the Blender bridge)
# ---------------------------------------------------------------------------


@dataclass
class ColliderDesc:
    mesh: object = None            # wp.Mesh with world-space points
    linear_velocity: tuple = (0.0, 0.0, 0.0)
    angular_velocity: tuple = (0.0, 0.0, 0.0)
    center: tuple = (0.0, 0.0, 0.0)
    aabb_min: tuple = (0.0, 0.0, 0.0)
    aabb_max: tuple = (0.0, 0.0, 0.0)
    friction: float = 0.0
    restitution: float = 0.0


@dataclass
class SolverSettings:
    domain_min: tuple = (-2.0, -2.0, 0.0)
    domain_max: tuple = (2.0, 2.0, 4.0)
    voxel_size: float = C.DEFAULT_VOXEL_SIZE
    leaf_capacity: int = C.DEFAULT_LEAF_CAPACITY
    max_particles: int = C.DEFAULT_MAX_PARTICLES
    max_whitewater: int = C.DEFAULT_MAX_WHITEWATER

    frame_rate: float = 24.0
    time_scale: float = 1.0
    cfl: float = C.DEFAULT_CFL
    min_substeps: int = 1
    max_substeps: int = 4

    gravity: tuple = C.DEFAULT_GRAVITY
    sample_type: int = int(ParticleSampleType.FLIP)
    flip_ratio: float = C.DEFAULT_FLIP_RATIO
    trace_mode: int = int(TraceMode.RK3)
    max_speed: float = 0.0                    # 0 disables the speed limit
    speed_limit_behavior: int = 0
    speed_limit_percent: float = 0.1

    particles_per_voxel: float = float(C.DEFAULT_PARTICLES_PER_VOXEL)
    count_stiffness: float = 0.0

    # Spatiotemporal FLIP (ST-FLIP, Braun et al. 2026).
    spacetime_enabled: bool = False
    spacetime_jitter_strength: float = 1.0
    spacetime_adaptive_jitter: bool = True
    spacetime_phase_eta: float = 0.5
    spacetime_phase_threshold: float = 0.5
    spacetime_reference_mass: float = 0.0  # <=0 derives from particles_per_voxel
    spacetime_variable_projection: bool = True
    spacetime_liquid_density: float = 1.0
    spacetime_gas_density: float = 0.0
    spacetime_min_density: float = 0.05
    spacetime_local_cfl: float = 1.0
    spacetime_max_local_substeps: int = 64
    spacetime_resync_surface: bool = True
    spacetime_seed: int = 1729

    relax_enabled: bool = True
    relax_strength: float = 0.5
    relax_mode: int = int(RelaxMode.LOCAL_VOXEL)
    open_boundaries: bool = False
    domain_sdf_walls: bool = False
    pressure_solver: int = int(ProjectNonDivergentSolver.PCG)
    pressure_iterations: int = C.DEFAULT_PRESSURE_ITERATIONS
    pressure_tolerance: float = C.DEFAULT_PRESSURE_TOLERANCE
    pressure_threshold: float = 0.0

    viscosity_enabled: bool = False
    viscosity_solver: int = int(ViscositySolverEnum.PCG)
    viscosity_constant: float = 0.0
    viscosity_iterations: int = C.DEFAULT_VISCOSITY_ITERATIONS
    viscosity_tolerance: float = C.DEFAULT_VISCOSITY_TOLERANCE
    viscosity_from_particles: bool = False

    surface_enabled: bool = True
    particle_radius_scale: float = 1.2
    surface_splat_radius: int = 2
    # Render surfacing is intentionally decoupled from the simulation grid.
    # The simulation SDF remains on the solver grid for whitewater/snippets,
    # while the render SDF can use a higher-resolution grid for viewport/export.
    surface_resolution_mode: int = int(SurfaceResolutionMode.SIMULATION)
    surface_resolution_multiplier: float = 2.0
    surface_voxel_size: float = 0.0
    surface_leaf_capacity_scale: float = 8.0
    surface_max_dense_dim: int = 512
    surface_update_cadence: int = int(SurfaceUpdateCadence.FRAME)
    live_surface_publish_cadence: int = 4
    surface_build_lowres_sdf: bool = True
    surface_filter_type: int = int(SdfFilterType.BOX)
    surface_filter_iterations: int = 0
    surface_filter_strength: float = 0.7
    surface_dilate_iterations: int = 0
    surface_dilate_strength: float = 1.0
    surface_erode_iterations: int = 0
    surface_erode_strength: float = 1.0
    surface_smooth_type: int = int(SdfFilterType.BOX)
    surface_smooth_iterations: int = 1
    surface_smooth_strength: float = 0.7

    whitewater_enabled: bool = False
    churn_decay: float = 1.0
    churn_gain: float = 1.0
    ww_churn_threshold: float = 0.5
    ww_emission_rate: float = 2.0
    ww_surface_band: float = 2.0     # in voxels
    ww_min_speed: float = 0.5
    ww_life_min: float = 1.0
    ww_life_max: float = 3.0
    ww_foam_band: float = 1.0        # in voxels
    ww_bubble_drag: float = 8.0
    ww_spray_drag: float = 0.5

    topology_expand: int = 1
    device: str = "cuda:0"


class FrameStats:
    """Per-frame statistics for the profiler/UI (host mirror)."""

    def __init__(self):
        self.step_time = 0.0
        self.active_voxels = 0
        self.initialized_voxels = 0
        self.total_voxels = 0
        self.active_particles = 0
        self.total_particles = 0
        self.whitewater_particles = 0
        self.surface_voxel_size = 0.0
        self.surface_active_voxels = 0
        self.surface_active_leaves = 0
        self.substeps = 0
        self.pressure_iterations = 0
        self.pressure_residual = 0.0
        self.timers = {}

    def as_dict(self):
        vps = self.active_voxels / self.step_time if self.step_time > 0 else 0.0
        pps = self.active_particles / self.step_time if self.step_time > 0 else 0.0
        return {
            "stepTime": self.step_time,
            "activeVoxels": self.active_voxels,
            "initializedVoxels": self.initialized_voxels,
            "totalVoxels": self.total_voxels,
            "activeParticles": self.active_particles,
            "totalParticles": self.total_particles,
            "whitewaterParticles": self.whitewater_particles,
            "surfaceVoxelSize": self.surface_voxel_size,
            "surfaceActiveVoxels": self.surface_active_voxels,
            "surfaceActiveLeaves": self.surface_active_leaves,
            "substeps": self.substeps,
            "pressureIterations": self.pressure_iterations,
            "pressureResidual": self.pressure_residual,
            "voxelsPerSecond": vps,
            "voxelsPerFrame": self.active_voxels,
            "particlesPerSecond": pps,
            "particlesPerFrame": self.active_particles,
            "usedParticleAllocation": self.active_particles / max(self.total_particles, 1),
            "usedVoxelAllocation": self.active_voxels / max(self.initialized_voxels, 1),
            "activeInitializedVoxelRatio": self.active_voxels / max(self.initialized_voxels, 1),
            "particleInitializedVoxelRatio": self.active_particles / max(self.initialized_voxels, 1),
            "particleVoxelFillRatio": self.active_particles / max(self.active_voxels, 1),
            "timers": dict(self.timers),
        }


class FluidSolver:
    """GPU-resident FLIP/PIC liquid solver."""

    def __init__(self, settings: SolverSettings, profiler=None):
        self.settings = settings
        self.device = settings.device
        self.profiler = profiler

        dmin = settings.domain_min
        dmax = settings.domain_max
        size = tuple(dmax[i] - dmin[i] for i in range(3))
        self.grid = SparseGrid(
            origin=dmin, size=size, voxel_size=settings.voxel_size,
            leaf_capacity=settings.leaf_capacity, device=self.device,
        )
        g = self.grid
        vc = g.voxel_capacity

        self.vel = MacVelocityField(g)
        self.vel_saved_u = wp.zeros(vc, dtype=float, device=self.device)
        self.vel_saved_v = wp.zeros(vc, dtype=float, device=self.device)
        self.vel_saved_w = wp.zeros(vc, dtype=float, device=self.device)
        self.tmp_u = wp.zeros(vc, dtype=float, device=self.device)
        self.tmp_v = wp.zeros(vc, dtype=float, device=self.device)
        self.tmp_w = wp.zeros(vc, dtype=float, device=self.device)
        self.collision_impact_u = wp.zeros(vc, dtype=float, device=self.device)
        self.collision_impact_v = wp.zeros(vc, dtype=float, device=self.device)
        self.collision_impact_w = wp.zeros(vc, dtype=float, device=self.device)

        self.voxel_type = ScalarField(g, dtype=wp.int32, name="voxel_type")
        self.voxel_counts = ScalarField(g, dtype=wp.int32, name="voxel_counts")
        self.temperature = ScalarField(g, name="temperature")
        self.viscosity_field = ScalarField(g, name="viscosity")
        self.density = ScalarField(g, name="density")
        self.collision_sdf = ScalarField(g, name="collision_sdf")
        self.collision_vel = wp.zeros(vc, dtype=wp.vec3, device=self.device)
        self.collision_friction = wp.zeros(vc, dtype=float, device=self.device)
        self.collision_restitution = wp.zeros(vc, dtype=float, device=self.device)
        self.churn = ScalarField(g, name="churn")
        self.churn_tmp = ScalarField(g, name="churn_tmp")
        self.attr_weight = ScalarField(g, name="attr_weight")
        self.ww_density = ScalarField(g, name="ww_density")

        self.old_leaf_table = wp.full(
            g.num_leaf_cells, -1, dtype=wp.int32, device=self.device
        )

        self.particles = ParticleSystem(settings.max_particles, self.device)
        self.bins = ParticleBins(settings.max_particles, vc, self.device)
        self.whitewater = WhitewaterSystem(settings.max_whitewater, self.device)

        self.pressure_solver = PressureSolver(g, self.device)
        self.viscosity_solver = ViscositySolver(g, self.device)
        self.sourcing = SourcePipeline(self)
        self.surface = SurfacePipeline(self)
        self._host_frame_cache = None
        self.snippets = SnippetManager()

        self.colliders: list[ColliderDesc] = []

        self.max_speed_dev = wp.zeros(1, dtype=float, device=self.device)

        self.frame = 0
        self.sim_time = 0.0
        self.stats = FrameStats()
        self._whitewater_emit_seed = 12345
        self._st_step_seed = int(settings.spacetime_seed) & 0x7FFFFFFF
        self._st_previous_dt = 0.0
        reference_mass = float(getattr(settings, "spacetime_reference_mass", 0.0))
        self._st_reference_mass_cached = (
            reference_mass if reference_mass > 0.0
            else self._calibrate_st_reference_mass()
            if settings.spacetime_enabled else 0.0
        )
        # Active-work launch bound for per-voxel kernels, refreshed after
        # every topology rebuild. Kernels guard on the device-side count, so
        # any dim >= the active voxel count is safe; dispatching only the
        # active region instead of the full allocation keeps kernel cost
        # proportional to live work (see TASK-POLICY-DRIFT-002).
        self._nvox = 1

    # -- host configuration -------------------------------------------------

    def set_sources(self, sources):
        self.sourcing.set_sources(sources)

    def set_colliders(self, colliders):
        self.colliders = list(colliders)

    def set_height_source(self, desc):
        self.sourcing.set_height_source(desc)

    @property
    def sources(self):
        return self.sourcing.sources

    @property
    def height_source(self):
        return self.sourcing.height_source

    def set_custom_snippets(self, snippets):
        self.snippets.set_snippets(snippets)
        return self.snippets.errors

    def reset(self):
        self._host_frame_cache = None
        self.particles.clear()
        self.whitewater.clear()
        self.churn.zero()
        self.frame = 0
        self.sim_time = 0.0
        self._st_previous_dt = 0.0
        settings = getattr(self, "settings", None)
        if settings is not None:
            self._st_step_seed = int(settings.spacetime_seed) & 0x7FFFFFFF
            reference_mass = float(getattr(settings, "spacetime_reference_mass", 0.0))
            self._st_reference_mass_cached = (
                reference_mass if reference_mass > 0.0
                else self._calibrate_st_reference_mass()
                if settings.spacetime_enabled else 0.0
            )
        self.grid.begin_topology()
        self.grid.finalize_topology()
        self._nvox = max(1, self.grid.active_voxel_count)
        if settings is not None:
            self.surface.reset()

    def release(self):
        """Break GPU-owning backreferences before the session drops this solver."""
        host_frame = self._host_frame_cache
        if host_frame is not None:
            host_frame.solver = None
        self._host_frame_cache = None

        surface = self.surface
        if surface is not None:
            surface.extractor.reset_buffers()
            if surface.render_extractor is not surface.extractor:
                surface.render_extractor.reset_buffers()
            surface.solver = None
        self.surface = None

        sourcing = getattr(self, "sourcing", None)
        if sourcing is not None:
            sourcing.owner = None
        self.sourcing = None

    # -- active-work launch bounds -------------------------------------------

    def _np_dim(self):
        """Live liquid-particle launch bound (host mirror upper bound).

        ``particles.host_count`` is maintained as an upper bound on the
        device count: it is refreshed exactly after emission, injection and
        end-of-frame stats, and compaction can only shrink the device count
        below it. Particle kernels guard on ``count[0]``.
        """
        return max(1, min(self.particles.capacity, self.particles.host_count))

    def _nww_dim(self):
        """Live whitewater launch bound (same contract as ``_np_dim``)."""
        return max(1, min(self.whitewater.capacity, self.whitewater.host_count))

    # -- render surfacing compatibility -------------------------------------

    def _render_surface_voxel_size(self):
        return (self.surface.voxel_size() if hasattr(self, "surface")
                else SurfacePipeline.voxel_size_for(self.settings))

    @property
    def surface_sdf(self): return self.surface.sdf
    @property
    def surface_tmp(self): return self.surface.tmp
    @property
    def surface_extractor(self): return self.surface.extractor
    @property
    def render_surface_grid(self): return self.surface.grid
    @property
    def render_surface_sdf(self): return self.surface.render_sdf
    @property
    def render_surface_tmp(self): return self.surface.render_tmp
    @property
    def render_surface_extractor(self): return self.surface.render_extractor

    def _surface_positions(self):
        s = self.settings
        p = self.particles
        if s.spacetime_enabled and s.spacetime_resync_surface:
            g = self.grid
            wp.launch(
                k_resync_particle_positions_spacetime, dim=self._np_dim(),
                inputs=[p.position, p.velocity, p.time_offset, p.count,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        self.collision_sdf.data, self.collision_vel,
                        self.collision_friction, self.collision_restitution,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size, int(s.trace_mode),
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max)],
                outputs=[p._position2], device=self.device,
            )
            return p._position2
        return p.position

    # -- profiling helper ----------------------------------------------------

    def _phase(self, name):
        if self.profiler is not None:
            return self.profiler.track(name)
        import contextlib
        return contextlib.nullcontext()

    # -- topology ------------------------------------------------------------

    def _rebuild_topology(self, expand=None):
        g = self.grid
        s = self.settings
        wp.copy(self.old_leaf_table, g.leaf_table)
        g.begin_topology()
        g.mark_points(self.particles.position, self.particles.count,
                      self._np_dim())
        for src in self.sources:
            if not src.enabled:
                continue
            lo, hi = src.aabb()
            g.mark_world_box(lo, hi)
        for col in self.colliders:
            g.mark_world_box(col.aabb_min, col.aabb_max)
        hs = self.height_source
        if hs.enabled:
            dmin = s.domain_min
            dmax = s.domain_max
            g.mark_world_box(dmin, (dmax[0], dmax[1], hs.base_height + 4.0 * s.voxel_size))
        # Gate on the setting, not the exact count: the mark kernel guards on
        # count[0] and host_count is an upper bound on it.
        if s.whitewater_enabled:
            g.mark_points(self.whitewater.position, self.whitewater.count,
                          self._nww_dim())
        g.expand(s.topology_expand if expand is None else max(0, int(expand)))
        # One 4-byte readback per rebuild: the fresh active leaf count bounds
        # every subsequent per-voxel launch this substep, which shrinks
        # dispatch from allocation capacity (possibly millions of voxels) to
        # the live topology. This also surfaces leaf-capacity overflow at the
        # rebuild instead of at end-of-frame.
        g.finalize_topology(sync=True)
        self._nvox = max(1, g.active_voxel_count)

        # Carry churn across the rebuild (only persistent grid field).
        wp.launch(
            k_remap_leaf_field, dim=self._nvox,
            inputs=[self.churn.data, self.old_leaf_table, g.leaf_coords,
                    g.num_leaves, g.leaves_dim],
            outputs=[self.churn_tmp.data], device=self.device,
        )
        self.churn.data, self.churn_tmp.data = self.churn_tmp.data, self.churn.data
        return g.active_leaf_count

    # -- per-substep phases ----------------------------------------------------

    def _clear_grid_fields(self, clear_collision=True):
        # Clear only the active slot range: the topology rebuild that
        # precedes this call defines [0, nvox) as the only region any kernel
        # reads or writes this substep (all launches are bounded by nvox and
        # guarded by num_leaves on device).
        nv = self._nvox
        arrays = (*self.vel.arrays(), self.voxel_type.data,
                  self.voxel_counts.data, self.temperature.data,
                  self.viscosity_field.data, self.density.data)
        if clear_collision:
            arrays += (self.collision_vel, self.collision_friction,
                       self.collision_restitution)
        for a in arrays:
            a[:nv].zero_()
        if not clear_collision:
            return
        wp.launch(k_fill_float, dim=nv,
                  inputs=[self.collision_sdf.data, 1.0e6], device=self.device)
        s = self.settings
        if s.domain_sdf_walls and not s.open_boundaries:
            g = self.grid
            wp.launch(
                k_mark_domain_walls, dim=nv,
                inputs=[g.leaf_coords, g.num_leaves, g.origin, g.voxel_size,
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                        self.collision_sdf.data],
                device=self.device,
            )

    def _apply_collisions(self):
        g = self.grid
        s = self.settings
        for col in self.colliders:
            if col.mesh is None:
                continue
            extent = tuple(col.aabb_max[i] - col.aabb_min[i] for i in range(3))
            # Cover the entire resolved collider interior. A narrow query band
            # turns sufficiently deep penetration into the +infinity default.
            band = max(8.0 * s.voxel_size,
                       math.sqrt(sum(value * value for value in extent)))
            lv = col.linear_velocity
            av = col.angular_velocity
            wp.launch(
                k_rasterize_collider_material, dim=self._nvox,
                inputs=[col.mesh.id, g.leaf_coords, g.num_leaves, g.origin,
                        s.voxel_size, band, wp.vec3(*lv), wp.vec3(*av),
                        wp.vec3(*col.center), col.friction, col.restitution],
                outputs=[self.collision_sdf.data, self.collision_vel,
                         self.collision_friction, self.collision_restitution],
                device=self.device,
            )

    def _index_particles(self):
        g = self.grid
        nv = self._nvox
        np_dim = self._np_dim()
        self.voxel_counts.data[:nv].zero_()
        wp.launch(
            k_particle_to_voxel_count_and_update_index,
            dim=np_dim,
            inputs=[self.particles.position, self.particles.count,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size],
            outputs=[self.voxel_counts.data, self.particles.voxel_index],
            device=self.device,
        )
        voxel_count_to_starts(self.voxel_counts.data[:nv], self.bins.starts[:nv])
        self.bins.cursor[:nv].zero_()
        wp.launch(
            k_scatter_particles_to_bins, dim=np_dim,
            inputs=[self.particles.voxel_index, self.particles.count,
                    self.bins.starts],
            outputs=[self.bins.cursor, self.bins.indices], device=self.device,
        )

    def _source(self, dt):
        return self.sourcing.apply(dt)

    def _calibrate_st_reference_mass(self):
        """Monte Carlo estimate of the ST phase reference mass m0.

        The paper defines m0 as the expected space-time mass estimator at a
        face under a uniform particle distribution with Nppc particles per
        cell.  Use the exact kernels used by the Warp P2G path on a small
        particle-filled patch with tau uniformly sampled in [-1/2, 1/2].
        This keeps phase thresholds tied to the implemented kernel constants,
        support convention, and particle-per-cell density rather than assuming
        a closed-form normalization.
        """
        s = self.settings
        manual = float(getattr(s, "spacetime_reference_mass", 0.0))
        if manual > 0.0 and math.isfinite(manual):
            return manual
        ppc = max(1.0e-6, float(s.particles_per_voxel))
        particles_per_cell = max(1, int(math.ceil(ppc)))
        particle_mass = ppc / float(particles_per_cell)
        # Deterministic and intentionally small: 3 MAC face orientations,
        # 27 cells around the face support, 32 random realizations.
        rng = np.random.default_rng(int(getattr(s, "spacetime_seed", 1729)) & 0xFFFFFFFF)
        faces = (
            np.array((0.0, 0.5, 0.5), dtype=np.float64),
            np.array((0.5, 0.0, 0.5), dtype=np.float64),
            np.array((0.5, 0.5, 0.0), dtype=np.float64),
        )
        total = 0.0
        samples = 0
        for face in faces:
            lo = np.floor(face - 1.0).astype(int)
            hi = np.floor(face + 1.0).astype(int)
            for _ in range(32):
                accum = 0.0
                for ix in range(int(lo[0]), int(hi[0]) + 1):
                    for iy in range(int(lo[1]), int(hi[1]) + 1):
                        for iz in range(int(lo[2]), int(hi[2]) + 1):
                            cell = np.array((ix, iy, iz), dtype=np.float64)
                            for _p in range(particles_per_cell):
                                pos = cell + rng.random(3)
                                r = face - pos
                                ws = (
                                    _st_spatial_weight_1d_host(float(r[0]))
                                    * _st_spatial_weight_1d_host(float(r[1]))
                                    * _st_spatial_weight_1d_host(float(r[2]))
                                )
                                if ws <= 0.0:
                                    continue
                                tau = rng.random() - 0.5
                                accum += particle_mass * ws * _st_temporal_weight_host(float(tau))
                total += accum
                samples += 1
        if samples <= 0:
            return max(1.0, ppc)
        estimate = total / float(samples)
        if not math.isfinite(estimate) or estimate <= 1.0e-8:
            return max(1.0, ppc)
        return float(estimate)

    def _st_reference_mass(self):
        s = self.settings
        ref = float(getattr(s, "spacetime_reference_mass", 0.0))
        if ref > 0.0 and math.isfinite(ref):
            return ref
        cached = float(getattr(self, "_st_reference_mass_cached", 0.0))
        if cached > 0.0 and math.isfinite(cached):
            return cached
        return max(1.0, float(s.particles_per_voxel))

    def _p2g(self, dt=None):
        g = self.grid
        p = self.particles
        st = self.settings
        if st.spacetime_enabled:
            prev_dt = self._st_previous_dt if self._st_previous_dt > 0.0 else float(dt or (st.time_scale / max(st.frame_rate, 1.0)))
            wp.launch(
                k_particle_set_grid_data_spacetime, dim=self._np_dim(),
                inputs=[p.position, p.velocity, p.time_offset, p.count,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size, prev_dt],
                outputs=[self.vel.u, self.vel.v, self.vel.w,
                         self.vel.wu, self.vel.wv, self.vel.ww],
                device=self.device,
            )
        else:
            wp.launch(
                k_particle_set_grid_data, dim=self._np_dim(),
                inputs=[p.position, p.velocity, p.count,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size],
                outputs=[self.vel.u, self.vel.v, self.vel.w,
                         self.vel.wu, self.vel.wv, self.vel.ww],
                device=self.device,
            )
        wp.launch(
            k_normalize_mac, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    g.num_leaves],
            device=self.device,
        )
        # Particle/phase accumulators -> voxel activity (fluid mask).
        if st.spacetime_enabled:
            wp.launch(
                k_spacetime_weights_to_phase_active, dim=self._nvox,
                inputs=[self.vel.wu, self.vel.wv, self.vel.ww,
                        self.voxel_type.data, g.leaf_table, g.leaf_coords,
                        g.voxels_dim, g.leaves_dim, g.num_leaves,
                        self._st_reference_mass(), st.spacetime_phase_eta,
                        st.spacetime_phase_threshold],
                outputs=[self.density.data], device=self.device,
            )
        else:
            wp.launch(
                k_particle_to_grid_active, dim=self._nvox,
                inputs=[self.voxel_counts.data, g.num_leaves],
                outputs=[self.voxel_type.data], device=self.device,
            )
        # Splat viscosity attribute when requested.
        st = self.settings
        if st.viscosity_enabled and st.viscosity_from_particles:
            self.attr_weight.data[:self._nvox].zero_()
            self.viscosity_field.data[:self._nvox].zero_()
            wp.launch(
                k_particle_to_grid_collect, dim=self._np_dim(),
                inputs=[p.position, p.attr("viscosity"), p.count,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size],
                outputs=[self.viscosity_field.data, self.attr_weight.data],
                device=self.device,
            )
            wp.launch(
                k_particle_to_grid_average, dim=self._nvox,
                inputs=[self.viscosity_field.data, self.attr_weight.data,
                        g.num_leaves],
                device=self.device,
            )

    def _save_velocities(self):
        nv = self._nvox
        wp.copy(self.vel_saved_u, self.vel.u, count=nv)
        wp.copy(self.vel_saved_v, self.vel.v, count=nv)
        wp.copy(self.vel_saved_w, self.vel.w, count=nv)

    def _forces(self, dt):
        g = self.grid
        s = self.settings
        wp.launch(
            k_forces_gravity, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    g.num_leaves, wp.vec3(*s.gravity), dt],
            device=self.device,
        )
        if s.max_speed > 0.0:
            wp.launch(
                k_forces_speed_limit, dim=self._nvox,
                inputs=[self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        g.leaf_table, g.leaf_coords, g.voxels_dim,
                        g.leaves_dim, g.num_leaves,
                        s.max_speed, int(s.speed_limit_behavior),
                        s.speed_limit_percent, dt],
                device=self.device,
            )

    def _viscosity(self, dt):
        s = self.settings
        if not s.viscosity_enabled:
            return
        if not s.viscosity_from_particles and s.viscosity_constant > 0.0:
            wp.launch(
                k_fill_float, dim=self._nvox,
                inputs=[self.viscosity_field.data, s.viscosity_constant],
                device=self.device,
            )
        if int(s.viscosity_solver) == int(ViscositySolverEnum.JACOBI):
            self.viscosity_solver.smooth(
                self.vel, self.viscosity_field.data, self.voxel_type.data, dt,
                iterations=max(1, s.viscosity_iterations // 10),
            )
        else:
            self.viscosity_solver.solve(
                self.vel, self.viscosity_field.data, self.voxel_type.data, dt,
                use_pcg=True, iterations=s.viscosity_iterations,
                tolerance=s.viscosity_tolerance,
            )

    def _collision_faces(self):
        g = self.grid
        wp.launch(
            k_sdf_to_axis_collision, dim=self._nvox,
            inputs=[self.collision_sdf.data, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, 0.0],
            outputs=[self.voxel_type.data], device=self.device,
        )
        wp.launch(
            k_expand_shift_axis_velocity_in_collider, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w, self.voxel_type.data,
                    self.collision_vel, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves],
            outputs=[self.tmp_u, self.tmp_v, self.tmp_w], device=self.device,
        )
        self.vel.u, self.tmp_u = self.tmp_u, self.vel.u
        self.vel.v, self.tmp_v = self.tmp_v, self.vel.v
        self.vel.w, self.tmp_w = self.tmp_w, self.vel.w

    def _cfl_speed_limit(self, dt):
        safe_dt = max(float(dt), 1.0e-8)
        return max(float(self.settings.cfl) * self.grid.voxel_size / safe_dt,
                   1.0e-6)

    def _pressure(self, dt):
        s = self.settings
        st_projection = bool(s.spacetime_enabled and s.spacetime_variable_projection)
        status = self.pressure_solver.project_frame(
            self.vel,
            self.voxel_type.data,
            self.collision_vel,
            self.voxel_counts.data,
            s,
            max_gradient=self._cfl_speed_limit(dt),
            reference_mass=(self._st_reference_mass()
                            if st_projection else None),
        )
        self.stats.pressure_iterations = status["iterations"]
        self.stats.pressure_residual = status["residual"]
        return bool(status["project_velocity"])

    def _collider_material_response(self):
        g = self.grid
        wp.launch(
            k_apply_collider_material_to_grid, dim=self._nvox,
            inputs=[self.voxel_type.data, self.collision_sdf.data,
                    self.collision_vel, self.collision_friction,
                    self.collision_restitution, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, g.voxel_size,
                    self.collision_impact_u, self.collision_impact_v,
                    self.collision_impact_w,
                    self.vel.u, self.vel.v, self.vel.w],
            device=self.device,
        )

    def _sanitize(self, dt):
        """Clamp MAC velocity to a CFL-safe bound and kill inf/nan."""
        g = self.grid
        max_speed = self._cfl_speed_limit(dt)
        wp.launch(
            k_sanitize_mac_velocity, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    g.num_leaves, max_speed],
            device=self.device,
        )

    def _sanitize_particles(self, dt):
        """Kill non-finite particles, clamp particle speeds."""
        s = self.settings
        g = self.grid
        max_speed = max(s.cfl * g.voxel_size / dt * 2.0, 50.0)
        p = self.particles
        wp.launch(
            k_sanitize_particles, dim=self._np_dim(),
            inputs=[p.position, p.velocity, p.alive, p.count, max_speed,
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max)],
            device=self.device,
        )

    def _g2p_advect(self, dt):
        g = self.grid
        s = self.settings
        p = self.particles
        wp.launch(
            k_staggered_field_to_particle, dim=self._np_dim(),
            inputs=[p.position, p.count,
                    self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    self.vel_saved_u, self.vel_saved_v, self.vel_saved_w,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size,
                    int(s.sample_type), s.flip_ratio],
            outputs=[p.velocity], device=self.device,
        )
        if s.spacetime_enabled:
            self._st_step_seed = (self._st_step_seed * 1664525 + 1013904223) & 0x7FFFFFFF
            wp.launch(
                k_advect_particles_spacetime, dim=self._np_dim(),
                inputs=[p.position, p.velocity, p.time_offset, p.count,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        self.collision_sdf.data, self.collision_vel,
                        self.collision_friction, self.collision_restitution,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size, dt, int(s.trace_mode),
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                        1 if s.open_boundaries else 0,
                        self._st_step_seed, s.spacetime_jitter_strength,
                        1 if s.spacetime_adaptive_jitter else 0,
                        s.spacetime_local_cfl, int(s.spacetime_max_local_substeps)],
                outputs=[p.alive], device=self.device,
            )
            self._st_previous_dt = float(dt)
        else:
            wp.launch(
                k_advect_particles, dim=self._np_dim(),
                inputs=[p.position, p.velocity, p.count,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        self.collision_sdf.data, self.collision_vel,
                        self.collision_friction, self.collision_restitution,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size, dt, int(s.trace_mode),
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                        1 if s.open_boundaries else 0],
                outputs=[p.alive], device=self.device,
            )

    def _lifecycle(self, dt):
        g = self.grid
        p = self.particles
        np_dim = self._np_dim()
        wp.launch(k_particle_age, dim=np_dim,
                  inputs=[p.attr("age"), p.count, dt], device=self.device)
        wp.launch(k_particle_lifespan, dim=np_dim,
                  inputs=[p.attr("age"), p.attr("life"), p.count, 1],
                  outputs=[p.alive], device=self.device)
        wp.launch(
            k_particle_in_sink_check, dim=np_dim,
            inputs=[p.position, p.count, self.voxel_type.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size],
            outputs=[p.alive], device=self.device,
        )
        p.compact(count_hint=np_dim)

    def _relax(self):
        s = self.settings
        if not s.relax_enabled:
            return
        g = self.grid
        p = self.particles
        self._index_particles()
        separation = 0.5 * s.voxel_size * (1.0 / max(s.particles_per_voxel, 1.0)) ** (1.0 / 3.0)
        wp.launch(
            k_particles_relax, dim=self._np_dim(),
            inputs=[p.position, p.count, p.voxel_index,
                    self.voxel_counts.data, self.bins.starts, self.bins.indices,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size,
                    separation, s.relax_strength, int(s.relax_mode),
                    self.collision_sdf.data,
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max)],
            outputs=[p._position2], device=self.device,
        )
        p.position, p._position2 = p._position2, p.position

    def _surface(self, build_render: bool = True, build_lowres: bool | None = None):
        return self.surface.build(build_render, build_lowres)

    def rebuild_surface(self):
        """Rebuild current-frame surface data after realtime setting changes."""
        self.surface.rebuild()

    def _whitewater(self, dt):
        s = self.settings
        if not s.whitewater_enabled:
            return
        g = self.grid
        p = self.particles
        ww = self.whitewater
        wp.launch(
            k_compute_churn, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w, self.voxel_type.data,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, s.voxel_size, dt, s.churn_decay, s.churn_gain],
            outputs=[self.churn.data], device=self.device,
        )
        self._whitewater_emit_seed = (
            self._whitewater_emit_seed * 1664525 + 1013904223
        ) & 0x7FFFFFFF
        wp.launch(
            k_emit_whitewater, dim=self._np_dim(),
            inputs=[p.position, p.velocity, p.count,
                    self.churn.data, self.surface_sdf.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, s.voxel_size, dt,
                    s.ww_churn_threshold, s.ww_emission_rate,
                    s.ww_surface_band * s.voxel_size, s.ww_min_speed,
                    s.ww_life_min, s.ww_life_max,
                    self._whitewater_emit_seed, ww.capacity],
            outputs=[ww.position, ww.velocity, ww.age, ww.life,
                     ww.alive, ww.count, ww.free_slots, ww.free_count],
            device=self.device,
        )
        # Kernels guard on the device-resident count, avoiding a stream barrier
        # and scalar readback between substeps.
        nww = ww.capacity
        wp.launch(
            k_advect_whitewater, dim=nww,
            inputs=[ww.position, ww.velocity, ww.age, ww.life,
                    ww.alive, ww.count,
                    self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    self.surface_sdf.data, self.collision_sdf.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, s.voxel_size, dt, wp.vec3(*s.gravity),
                    s.ww_foam_band * s.voxel_size, s.ww_bubble_drag,
                    s.ww_spray_drag,
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                    ww.capacity, ww.free_slots, ww.free_count],
            device=self.device,
        )

    def _finalize_whitewater(self):
        """Compact and publish whitewater density once per display frame."""
        if not self.settings.whitewater_enabled:
            return
        g = self.grid
        ww = self.whitewater
        ww.compact()
        self.ww_density.data[:self._nvox].zero_()
        wp.launch(
            k_whitewater_set_grid_data, dim=ww.capacity,
            inputs=[ww.position, ww.alive, ww.count,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, self.settings.voxel_size],
            outputs=[self.ww_density.data], device=self.device,
        )

    # -- CFL ----------------------------------------------------------------

    def _compute_substeps(self, frame_dt):
        s = self.settings
        min_substeps = max(1, int(s.min_substeps))
        max_substeps = max(min_substeps, int(s.max_substeps))
        # host_count is kept current by end-of-frame stats, inject_particles
        # and clear(); avoid an extra device sync here.
        if self.particles.host_count == 0:
            return min_substeps
        if min_substeps == max_substeps:
            return min_substeps
        self.max_speed_dev.zero_()
        wp.launch(
            k_max_vec3_magnitude, dim=self.particles.host_count,
            inputs=[self.particles.velocity, self.particles.count],
            outputs=[self.max_speed_dev], device=self.device,
        )
        vmax = float(self.max_speed_dev.numpy()[0])
        gmag = math.sqrt(sum(x * x for x in s.gravity))
        vmax = vmax + gmag * frame_dt
        if vmax < 1.0e-6:
            return min_substeps
        dt_cfl = s.cfl * s.voxel_size / vmax
        n = int(math.ceil(frame_dt / max(dt_cfl, 1.0e-6)))
        return max(min_substeps, min(max_substeps, n))

    def _whitewater_topology_expand(self, frame_dt):
        """Return a leaf radius covering existing whitewater for one frame."""
        s = self.settings
        ww = self.whitewater
        if not s.whitewater_enabled or ww.host_count <= 0:
            return 0
        self.max_speed_dev.zero_()
        wp.launch(
            k_max_vec3_magnitude, dim=ww.host_count,
            inputs=[ww.velocity, ww.count],
            outputs=[self.max_speed_dev], device=self.device,
        )
        vmax = max(0.0, float(self.max_speed_dev.numpy()[0]))
        gmag = math.sqrt(sum(x * x for x in s.gravity))
        return _motion_leaf_radius(vmax, frame_dt, gmag, s.voxel_size)

    # -- main entry -----------------------------------------------------------

    def step_frame(self):
        """Advance one display frame (interactive playback entry point)."""
        s = self.settings
        t_start = _time.perf_counter()
        frame_dt = s.time_scale / s.frame_rate
        if self.profiler is not None:
            self.profiler.begin_frame()

        substeps = self._compute_substeps(frame_dt)
        dt = frame_dt / substeps

        time_ctx = {
            "frame": self.frame, "time": self.sim_time,
            "real_time_step": frame_dt, "solver_time_step": dt,
            "time_scale": s.time_scale, "substeps": substeps,
        }

        rebuild_each_substep = self.snippets.has_any()
        frame_topology_expand = max(
            int(s.topology_expand),
            int(math.ceil(substeps * max(float(s.cfl), 0.0) / C.LEAF_DIM)),
            self._whitewater_topology_expand(frame_dt),
        )
        for substep_index in range(substeps):
            self.snippets.run(CustomKernelOrder.FIRST_STEP, self, time_ctx)
            rebuild = substep_index == 0 or rebuild_each_substep
            with self._phase("misc"):
                if rebuild:
                    self._rebuild_topology(
                        frame_topology_expand if not rebuild_each_substep else None)
                self._clear_grid_fields(clear_collision=rebuild)
            with self._phase("collision"):
                if rebuild:
                    self._apply_collisions()
            self.snippets.run(CustomKernelOrder.BEFORE_SOURCING, self, time_ctx)
            with self._phase("indexing"):
                self._index_particles()
            with self._phase("sourcing"):
                emitted = self._source(dt)
                # Re-count after emission so P2G activity sees new particles.
                # Skipped when no emitter ran: indexing is already current.
                if emitted:
                    self._index_particles()
            with self._phase("p2g"):
                self._p2g(dt)
            self._save_velocities()
            self.snippets.run(CustomKernelOrder.BEFORE_FORCES, self, time_ctx)
            with self._phase("forces"):
                self._forces(dt)
            with self._phase("viscosity"):
                self._viscosity(dt)
            wp.copy(self.collision_impact_u, self.vel.u)
            wp.copy(self.collision_impact_v, self.vel.v)
            wp.copy(self.collision_impact_w, self.vel.w)
            with self._phase("collision"):
                self._collision_faces()
            self.snippets.run(
                CustomKernelOrder.BEFORE_PROJECT_NON_DIVERGENT_AND_ADVECTION,
                self, time_ctx)
            with self._phase("pressure"):
                self._pressure(dt)
                self._collider_material_response()
            self._sanitize(dt)
            with self._phase("g2p"):
                self._g2p_advect(dt)
            self._sanitize_particles(dt)
            with self._phase("lifecycle"):
                self._lifecycle(dt)
            with self._phase("relax"):
                self._relax()
            self.surface.after_substep()
            with self._phase("whitewater"):
                self._whitewater(dt)
            self.snippets.run(CustomKernelOrder.FINAL_STEP, self, time_ctx)
            self.sim_time += dt
            time_ctx["time"] = self.sim_time

        self.frame += 1
        self._host_frame_cache = None
        with self._phase("whitewater"):
            self._finalize_whitewater()
        self.surface.after_frame()

        wp.synchronize_device(self.device)
        self.grid.refresh_topology_counts()

        st = self.stats
        st.step_time = _time.perf_counter() - t_start
        st.substeps = substeps
        st.active_voxels = self.grid.active_voxel_count
        st.initialized_voxels = self.grid.voxel_capacity
        st.total_voxels = self.grid.total_voxels()
        st.active_particles = self.particles.refresh_host_count()
        st.total_particles = self.particles.capacity
        st.whitewater_particles = self.whitewater.refresh_host_count()
        surface_metadata = self.surface.metadata()
        st.surface_voxel_size = float(surface_metadata["voxel_size"])
        st.surface_active_leaves = surface_metadata["active_leaves"]
        st.surface_active_voxels = surface_metadata["active_voxels"]
        if self.profiler is not None:
            st.timers = self.profiler.collect()
        return st

    # -- direct particle injection ---------------------------------------------

    def inject_particles(self, positions, velocity=(0.0, 0.0, 0.0),
                         velocities=None, temperature=0.0, viscosity=0.0,
                         density=1.0, life=0.0, use_lifespan=False):
        """Append external world-space points as liquid particles.

        ``positions``: (N,3) array-like. Optional per-particle ``velocities``
        override the constant ``velocity``. Uses fluidSimSourceParticles.
        """
        return self.sourcing.inject_particles(
            positions,
            velocity=velocity,
            velocities=velocities,
            temperature=temperature,
            viscosity=viscosity,
            density=density,
            life=life,
            use_lifespan=use_lifespan,
        )

    # -- output ---------------------------------------------------------------

    def host_frame(self):
        """Return the lazy host snapshot for the current solver frame."""
        if (self._host_frame_cache is None or
                self._host_frame_cache.frame != self.frame):
            from .frame_output import HostFrame
            self._host_frame_cache = HostFrame(self)
        return self._host_frame_cache

    def surface_mesh(self):
        return self.surface.mesh()

    def particle_positions(self):
        n = self.particles.refresh_host_count()
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # Slice on the device side so only the live range crosses the PCIe bus.
        return self.particles.position[:n].numpy()

    def particle_velocities(self):
        n = self.particles.host_count
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.particles.velocity[:n].numpy()

    def whitewater_positions(self):
        n = self.whitewater.refresh_host_count()
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.whitewater.position[:n].numpy()
