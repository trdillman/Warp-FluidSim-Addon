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
    MultiGrid,
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
    SurfaceExtractor,
)
from .sourcing import (
    SourceShape,
    k_source_shapes,
    k_source_shapes_particles,
    k_source_height,
    k_source_resolve_type,
)
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
class SourceShapeDesc:
    enabled: bool = True
    shape: int = 0            # SourceShapeShape
    stype: int = 0            # SourceShapeType
    mode: int = 1             # SourceShapeSourceMode (default Particles)
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)   # quaternion xyzw
    half_extents: tuple = (0.5, 0.5, 0.5)
    radius: float = 0.5
    height: float = 0.5
    feather: float = 0.0
    velocity: tuple = (0.0, 0.0, 0.0)
    linear_velocity: tuple = (0.0, 0.0, 0.0)
    angular_velocity: tuple = (0.0, 0.0, 0.0)
    blend_mode: int = 1
    blend_amount: float = 1.0
    force_type: int = 0
    force_strength: float = 0.0
    force_dir: tuple = (0.0, 0.0, 1.0)
    emit_rate: float = 8.0
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False

    def aabb(self):
        """Conservative world AABB used for topology marking."""
        r = max(self.radius, self.height,
                self.half_extents[0], self.half_extents[1], self.half_extents[2])
        r = r + max(self.feather, 0.0)
        p = self.position
        return ((p[0] - r, p[1] - r, p[2] - r), (p[0] + r, p[1] + r, p[2] + r))

    def to_struct(self):
        s = SourceShape()
        s.enabled = 1 if self.enabled else 0
        s.shape = int(self.shape)
        s.stype = int(self.stype)
        s.mode = int(self.mode)
        q = wp.quat(*[float(x) for x in self.rotation])
        s.xform = wp.transform(wp.vec3(*[float(x) for x in self.position]), q)
        s.inv_xform = wp.transform_inverse(s.xform)
        s.half_extents = wp.vec3(*[float(x) for x in self.half_extents])
        s.radius = float(self.radius)
        s.height = float(self.height)
        s.feather = float(self.feather)
        s.velocity = wp.vec3(*[float(x) for x in self.velocity])
        s.linear_velocity = wp.vec3(*[float(x) for x in self.linear_velocity])
        s.angular_velocity = wp.vec3(*[float(x) for x in self.angular_velocity])
        s.blend_mode = int(self.blend_mode)
        s.blend_amount = float(self.blend_amount)
        s.force_type = int(self.force_type)
        s.force_strength = float(self.force_strength)
        s.force_dir = wp.vec3(*[float(x) for x in self.force_dir])
        s.emit_rate = float(self.emit_rate)
        s.temperature = float(self.temperature)
        s.viscosity = float(self.viscosity)
        s.density = float(self.density)
        s.life_min = float(self.life_min)
        s.life_max = float(self.life_max)
        s.use_lifespan = 1 if self.use_lifespan else 0
        return s


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
class HeightSourceDesc:
    enabled: bool = False
    base_height: float = 0.0
    back_mode: int = 1             # SourceHeightBackMode
    velocity: tuple = (0.0, 0.0, 0.0)
    profile: object = None         # optional 2D numpy array
    emit_particles: bool = True
    emit_rate: float = 8.0


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
    pressure_multigrid: int = int(MultiGrid.OFF)
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
        self.surface_sdf = ScalarField(g, name="surface_sdf")
        self.surface_tmp = ScalarField(g, name="surface_tmp")
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
        self.surface_extractor = SurfaceExtractor(g, self.device)
        self.render_surface_grid = g
        self.render_surface_sdf = self.surface_sdf
        self.render_surface_tmp = self.surface_tmp
        self.render_surface_extractor = self.surface_extractor
        self._render_surface_grid_key = None
        self._render_surface_nvox = self._nvox if hasattr(self, "_nvox") else max(1, g.voxel_capacity)
        self._surface_topology_generation = 0
        self._surface_mesh_cache = None
        self._surface_mesh_key = None
        self._render_surface_built_key = None
        self.snippets = SnippetManager()

        self.sources: list[SourceShapeDesc] = []
        self.colliders: list[ColliderDesc] = []
        self.height_source = HeightSourceDesc()
        self._shape_array = None
        # Cached particle-emission shape array (sources + derived height box).
        # Rebuilt only when sources/height/domain change instead of once per
        # substep (host alloc + H2D upload).
        self._emit_shape_array = None
        self._emit_shape_key = None
        self._height_profile = wp.zeros((2, 2), dtype=float, device=self.device)
        self._height_profile_res = (2, 2)

        self.max_speed_dev = wp.zeros(1, dtype=float, device=self.device)
        self.first_slot_dev = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.frame = 0
        self.sim_time = 0.0
        self.stats = FrameStats()
        self._emit_seed = 12345
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
        self.sources = list(sources)
        self._shape_array = None
        self._emit_shape_array = None

    def set_colliders(self, colliders):
        self.colliders = list(colliders)

    def set_height_source(self, desc):
        if desc is None:
            desc = HeightSourceDesc()  # disabled by default
        self.height_source = desc
        self._emit_shape_array = None
        if desc.profile is not None:
            arr = np.ascontiguousarray(desc.profile, dtype=np.float32)
            self._height_profile = wp.array(arr, dtype=float, device=self.device)
            self._height_profile_res = arr.shape
        else:
            self._height_profile_res = (2, 2)

    def set_custom_snippets(self, snippets):
        self.snippets.set_snippets(snippets)
        return self.snippets.errors

    def reset(self):
        self.particles.clear()
        self.whitewater.clear()
        self.churn.zero()
        self.frame = 0
        self.sim_time = 0.0
        self._surface_mesh_cache = None
        self._surface_mesh_key = None
        self._render_surface_built_key = None
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
        self.surface_extractor.reset_buffers()
        self._render_surface_grid_key = None
        self.grid.begin_topology()
        self.grid.finalize_topology()
        self._nvox = max(1, self.grid.active_voxel_count)
        if settings is not None:
            self._ensure_render_surface_resources(force=True)

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

    # -- render surfacing resources -----------------------------------------

    def _surface_mode(self) -> int:
        mode = int(getattr(self.settings, "surface_resolution_mode", int(SurfaceResolutionMode.SIMULATION)))
        if mode not in {int(SurfaceResolutionMode.SIMULATION), int(SurfaceResolutionMode.MULTIPLIER), int(SurfaceResolutionMode.CUSTOM)}:
            mode = int(SurfaceResolutionMode.SIMULATION)
        return mode

    def _render_surface_voxel_size(self) -> float:
        s = self.settings
        sim_h = max(float(s.voxel_size), 1.0e-9)
        mode = self._surface_mode()
        if mode == int(SurfaceResolutionMode.CUSTOM):
            custom_h = float(getattr(s, "surface_voxel_size", 0.0))
            if custom_h > 0.0:
                return max(custom_h, sim_h / 16.0)
        if mode == int(SurfaceResolutionMode.MULTIPLIER):
            mult = float(getattr(s, "surface_resolution_multiplier", 1.0))
            mult = max(0.25, min(mult, 8.0))
            return sim_h / mult
        return sim_h

    def _render_surface_leaf_capacity(self, h: float) -> int:
        s = self.settings
        dmin = np.asarray(s.domain_min, dtype=np.float64)
        dmax = np.asarray(s.domain_max, dtype=np.float64)
        size = np.maximum(dmax - dmin, h)
        nx = max(C.LEAF_DIM, int(np.ceil(size[0] / h)))
        ny = max(C.LEAF_DIM, int(np.ceil(size[1] / h)))
        nz = max(C.LEAF_DIM, int(np.ceil(size[2] / h)))
        nx = (nx + C.LEAF_MASK) & ~C.LEAF_MASK
        ny = (ny + C.LEAF_MASK) & ~C.LEAF_MASK
        nz = (nz + C.LEAF_MASK) & ~C.LEAF_MASK
        total_leaf_cells = (nx >> C.LEAF_SHIFT) * (ny >> C.LEAF_SHIFT) * (nz >> C.LEAF_SHIFT)
        if abs(h - float(s.voxel_size)) <= 1.0e-8:
            return min(int(self.grid.capacity), int(total_leaf_cells))
        scale = max(1.0, float(getattr(s, "surface_leaf_capacity_scale", 8.0)))
        return int(min(total_leaf_cells, max(1, math.ceil(float(s.leaf_capacity) * scale))))

    def _render_surface_grid_key_from_settings(self):
        s = self.settings
        h = float(self._render_surface_voxel_size())
        capacity = self._render_surface_leaf_capacity(h)
        max_dense = max(8, int(getattr(s, "surface_max_dense_dim", 512)))
        return (
            round(h, 12), int(capacity), int(max_dense),
            tuple(float(x) for x in s.domain_min),
            tuple(float(x) for x in s.domain_max),
        )

    def _surface_filter_kernel(self, filter_type: int):
        return {
            int(SdfFilterType.DIALATE): k_smooth_surface_sdf_dilate,
            int(SdfFilterType.ERODE): k_smooth_surface_sdf_erode,
            int(SdfFilterType.BOX): k_smooth_surface_sdf_box,
            int(SdfFilterType.MEAN_CURVATURE_FLOW): k_smooth_surface_sdf_mean_curvature,
        }[int(filter_type)]

    def _surface_settings_signature(self):
        s = self.settings
        return (
            int(getattr(s, "surface_resolution_mode", 0)),
            round(float(getattr(s, "surface_resolution_multiplier", 1.0)), 6),
            round(float(getattr(s, "surface_voxel_size", 0.0)), 9),
            round(float(getattr(s, "surface_leaf_capacity_scale", 1.0)), 6),
            int(getattr(s, "surface_max_dense_dim", 512)),
            int(getattr(s, "surface_update_cadence", 1)),
            bool(getattr(s, "surface_build_lowres_sdf", True)),
            round(float(s.particle_radius_scale), 6),
            int(s.surface_splat_radius),
            int(s.surface_filter_type), int(s.surface_filter_iterations), round(float(s.surface_filter_strength), 6),
            int(s.surface_dilate_iterations), round(float(s.surface_dilate_strength), 6),
            int(s.surface_erode_iterations), round(float(s.surface_erode_strength), 6),
            int(s.surface_smooth_type), int(s.surface_smooth_iterations), round(float(s.surface_smooth_strength), 6),
            bool(getattr(s, "spacetime_enabled", False)),
            bool(getattr(s, "spacetime_resync_surface", True)),
            int(self._surface_topology_generation),
        )

    def _invalidate_surface_mesh(self):
        self._surface_mesh_cache = None
        self._surface_mesh_key = None

    def _render_surface_build_key(self):
        return (self.frame, round(float(self.sim_time), 12), self._surface_settings_signature())

    def _ensure_render_surface_resources(self, force: bool = False):
        """Ensure the selected render-surface grid/SDF/extractor exists.

        The simulation-grid surface_sdf is preserved for whitewater, snippets
        and legacy behavior. When the requested surface voxel size matches the
        simulation grid, render resources alias the legacy resources. Otherwise
        a dedicated high-resolution sparse grid is allocated for viewport/export
        surfacing.
        """
        s = self.settings
        h = self._render_surface_voxel_size()
        max_dense = max(8, int(getattr(s, "surface_max_dense_dim", 512)))
        if abs(h - float(self.grid.voxel_size)) <= 1.0e-8:
            self.render_surface_grid = self.grid
            self.render_surface_sdf = self.surface_sdf
            self.render_surface_tmp = self.surface_tmp
            self.render_surface_extractor = self.surface_extractor
            self.render_surface_extractor.max_dense_dim = max_dense
            if force:
                self.render_surface_extractor.reset_buffers()
            self._render_surface_grid_key = ("simulation", max_dense)
            self._render_surface_nvox = self._nvox if hasattr(self, "_nvox") else max(1, self.grid.active_voxel_count)
            self._invalidate_surface_mesh()
            return

        key = self._render_surface_grid_key_from_settings()
        if not force and self._render_surface_grid_key == key:
            # Max-dense can be changed without recreating the sparse fields.
            self.render_surface_extractor.max_dense_dim = max_dense
            return

        dmin = np.asarray(s.domain_min, dtype=np.float64)
        dmax = np.asarray(s.domain_max, dtype=np.float64)
        size = np.maximum(dmax - dmin, h)
        capacity = self._render_surface_leaf_capacity(h)
        rg = SparseGrid(origin=dmin, size=size, voxel_size=h,
                        leaf_capacity=capacity, device=self.device)
        self.render_surface_grid = rg
        self.render_surface_sdf = ScalarField(rg, name="render_surface_sdf")
        self.render_surface_tmp = ScalarField(rg, name="render_surface_tmp")
        self.render_surface_extractor = SurfaceExtractor(rg, self.device, max_dense_dim=max_dense)
        self._render_surface_grid_key = key
        self._render_surface_nvox = max(1, rg.active_voxel_count)
        self._surface_topology_generation += 1
        self._invalidate_surface_mesh()

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

    def _build_surface_sdf_on_grid(self, grid, sdf, tmp, nvox: int, surface_positions, surface_voxel_size: float):
        s = self.settings
        h = float(surface_voxel_size)
        nvox = max(1, int(nvox))
        wp.launch(k_surface_sdf_reset, dim=nvox,
                  inputs=[sdf.data, grid.num_leaves, 4.0 * h],
                  device=self.device)
        radius = 0.5 * float(s.voxel_size) * float(s.particle_radius_scale)
        splat_radius = max(1, int(s.surface_splat_radius), int(math.ceil(radius / h + 1.0)))
        wp.launch(
            k_build_surface_sdf, dim=self._np_dim(),
            inputs=[surface_positions, self.particles.count,
                    grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                    grid.origin, h, radius, splat_radius],
            outputs=[sdf.data], device=self.device,
        )

        def apply_filter(kernel, iterations, strength):
            for _ in range(max(0, int(iterations))):
                wp.launch(
                    kernel, dim=nvox,
                    inputs=[sdf.data, grid.leaf_table, grid.leaf_coords,
                            grid.voxels_dim, grid.leaves_dim, grid.num_leaves,
                            h, float(strength)],
                    outputs=[tmp.data], device=self.device,
                )
                sdf.data, tmp.data = tmp.data, sdf.data

        # Explicit pre-filter pass. This setting existed previously but was not
        # consumed by the solver path; default iterations are now zero to
        # preserve the prior effective default.
        apply_filter(self._surface_filter_kernel(int(s.surface_filter_type)),
                     s.surface_filter_iterations, s.surface_filter_strength)
        # Houdini-style fixed stack: grow/close, shrink/open, then smooth.
        apply_filter(k_smooth_surface_sdf_dilate,
                     s.surface_dilate_iterations, s.surface_dilate_strength)
        apply_filter(k_smooth_surface_sdf_erode,
                     s.surface_erode_iterations, s.surface_erode_strength)
        apply_filter(self._surface_filter_kernel(int(s.surface_smooth_type)),
                     s.surface_smooth_iterations, s.surface_smooth_strength)
        return sdf

    def _rebuild_render_surface_topology(self, surface_positions):
        rg = self.render_surface_grid
        if rg is self.grid:
            self._render_surface_nvox = self._nvox
            return self._render_surface_nvox
        rg.begin_topology()
        rg.mark_points(surface_positions, self.particles.count, self._np_dim())
        rg.expand(max(1, int(getattr(self.settings, "topology_expand", 1))))
        rg.finalize_topology(sync=True)
        self._render_surface_nvox = max(1, rg.active_voxel_count)
        self._surface_topology_generation += 1
        return self._render_surface_nvox

    def _shapes(self):
        if self._shape_array is None:
            structs = [s.to_struct() for s in self.sources] or [SourceShapeDesc(enabled=False).to_struct()]
            self._shape_array = wp.array(structs, dtype=SourceShape, device=self.device)
        return self._shape_array

    # -- profiling helper ----------------------------------------------------

    def _phase(self, name):
        if self.profiler is not None:
            return self.profiler.track(name)
        import contextlib
        return contextlib.nullcontext()

    # -- topology ------------------------------------------------------------

    def _rebuild_topology(self):
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
        g.expand(s.topology_expand)
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

    def _clear_grid_fields(self):
        # Clear only the active slot range: the topology rebuild that
        # precedes this call defines [0, nvox) as the only region any kernel
        # reads or writes this substep (all launches are bounded by nvox and
        # guarded by num_leaves on device).
        nv = self._nvox
        for a in (*self.vel.arrays(), self.voxel_type.data,
                  self.voxel_counts.data, self.temperature.data,
                  self.viscosity_field.data, self.density.data,
                  self.collision_vel, self.collision_friction,
                  self.collision_restitution):
            a[:nv].zero_()
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
        band = 4.0 * s.voxel_size
        for col in self.colliders:
            if col.mesh is None:
                continue
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
        g = self.grid
        s = self.settings
        shapes = self._shapes()
        n_shapes = len(self.sources) if self.sources else 0

        if n_shapes:
            wp.launch(
                k_source_shapes, dim=self._nvox,
                inputs=[shapes, n_shapes, g.leaf_coords, g.num_leaves,
                        g.origin, s.voxel_size, dt],
                outputs=[self.voxel_type.data, self.collision_sdf.data,
                         self.collision_vel, self.vel.u, self.vel.v, self.vel.w,
                         self.temperature.data, self.viscosity_field.data],
                device=self.device,
            )

        hs = self.height_source
        if hs.enabled:
            prof_res = wp.vec2i(int(self._height_profile_res[0]),
                                int(self._height_profile_res[1]))
            wp.launch(
                k_source_height, dim=self._nvox,
                inputs=[g.leaf_coords, g.num_leaves, g.origin, s.voxel_size,
                        hs.base_height, int(hs.back_mode),
                        self._height_profile, prof_res,
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                        wp.vec3(*hs.velocity)],
                outputs=[self.voxel_type.data, self.vel.u, self.vel.v, self.vel.w],
                device=self.device,
            )

        wp.launch(
            k_source_resolve_type, dim=self._nvox,
            inputs=[self.voxel_type.data, g.num_leaves], device=self.device,
        )

        # Particle emission (top-up to emit_rate per voxel inside shapes).
        self._emit_seed = (self._emit_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        emit_shapes = n_shapes > 0 and any(
            src.enabled and src.stype == 0 and src.mode == 1 for src in self.sources
        )
        height_particles = hs.enabled and hs.emit_particles
        if emit_shapes or height_particles:
            n_emit = n_shapes
            if height_particles:
                dmin = s.domain_min
                dmax = s.domain_max
                key = (n_shapes, hs.base_height, tuple(hs.velocity),
                       hs.emit_rate, tuple(dmin), tuple(dmax))
                if self._emit_shape_array is None or self._emit_shape_key != key:
                    cz = (dmin[2] + hs.base_height) * 0.5
                    hz = max((hs.base_height - dmin[2]) * 0.5, 0.0)
                    all_shapes = list(self.sources)
                    all_shapes.append(SourceShapeDesc(
                        enabled=True, shape=2, stype=0, mode=1,
                        position=((dmin[0] + dmax[0]) * 0.5,
                                  (dmin[1] + dmax[1]) * 0.5, cz),
                        half_extents=((dmax[0] - dmin[0]) * 0.5,
                                      (dmax[1] - dmin[1]) * 0.5, hz),
                        velocity=tuple(hs.velocity), emit_rate=hs.emit_rate,
                    ))
                    self._emit_shape_array = wp.array(
                        [sh.to_struct() for sh in all_shapes],
                        dtype=SourceShape, device=self.device)
                    self._emit_shape_key = key
                shapes = self._emit_shape_array
                n_emit = n_shapes + 1
            p = self.particles
            wp.launch(
                k_source_shapes_particles, dim=self._nvox,
                inputs=[shapes, n_emit, g.leaf_coords, g.num_leaves,
                        g.origin, s.voxel_size, self.voxel_counts.data,
                        self._emit_seed, p.capacity],
                outputs=[p.position, p.velocity, p.attr("age"), p.attr("life"),
                         p.attr("temperature"), p.attr("viscosity"),
                         p.attr("density"), p.time_offset, p.alive, p.count],
                device=self.device,
            )
            # Refresh the host-side particle bound after device-side growth
            # (one 4-byte readback per emitting substep) so subsequent
            # particle launches cover the newly emitted range without
            # dispatching the full allocation capacity.
            p.refresh_host_count()
            return True
        return False

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
        ps = self.pressure_solver
        ps.build_rhs(self.vel, self.voxel_type.data, self.collision_vel,
                     self.voxel_counts.data,
                     rest_count=s.particles_per_voxel,
                     count_stiffness=s.count_stiffness)
        st_projection = bool(s.spacetime_enabled and s.spacetime_variable_projection)
        if st_projection:
            ps.build_preconditioner_st(
                self.voxel_type.data, self.vel,
                reference_mass=self._st_reference_mass(),
                eta_phi=s.spacetime_phase_eta,
                rho_liquid=s.spacetime_liquid_density,
                rho_gas=s.spacetime_gas_density,
                eps_rho=s.spacetime_min_density,
            )
        else:
            ps.build_preconditioner(self.voxel_type.data)
        use_mg = int(s.pressure_multigrid) != int(MultiGrid.OFF) and not st_projection
        if int(s.pressure_solver) == int(ProjectNonDivergentSolver.JACOBI):
            if st_projection:
                ps.solve_jacobi_st(
                    self.voxel_type.data, s.pressure_iterations, self.vel,
                    reference_mass=self._st_reference_mass(),
                    eta_phi=s.spacetime_phase_eta,
                    rho_liquid=s.spacetime_liquid_density,
                    rho_gas=s.spacetime_gas_density,
                    eps_rho=s.spacetime_min_density,
                )
            else:
                if use_mg:
                    ps.coarse_initial_guess(self.voxel_type.data)
                ps.solve_jacobi(self.voxel_type.data, s.pressure_iterations)
        else:
            if st_projection:
                ps.solve_pcg_st(
                    self.voxel_type.data, s.pressure_iterations,
                    s.pressure_tolerance, self.vel,
                    reference_mass=self._st_reference_mass(),
                    eta_phi=s.spacetime_phase_eta,
                    rho_liquid=s.spacetime_liquid_density,
                    rho_gas=s.spacetime_gas_density,
                    eps_rho=s.spacetime_min_density,
                )
            else:
                ps.solve_pcg(self.voxel_type.data, s.pressure_iterations,
                             s.pressure_tolerance, use_coarse_guess=use_mg)
        if not ps.should_project_pressure():
            self.stats.pressure_iterations = ps.last_iterations
            self.stats.pressure_residual = ps.last_residual
            return False
        if s.pressure_threshold > 0.0:
            ps.apply_threshold(s.pressure_threshold)
        if st_projection:
            ps.project_st(
                self.vel, self.voxel_type.data, self.collision_vel,
                max_gradient=self._cfl_speed_limit(dt),
                reference_mass=self._st_reference_mass(),
                eta_phi=s.spacetime_phase_eta,
                rho_liquid=s.spacetime_liquid_density,
                rho_gas=s.spacetime_gas_density,
                eps_rho=s.spacetime_min_density,
            )
        else:
            ps.project(self.vel, self.voxel_type.data, self.collision_vel,
                       max_gradient=self._cfl_speed_limit(dt))
        self.stats.pressure_iterations = ps.last_iterations
        self.stats.pressure_residual = ps.last_residual
        return True

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
        """Build simulation-grid and/or render-grid surface SDFs.

        ``surface_sdf`` remains on the simulation grid for legacy fields,
        whitewater and snippets. ``render_surface_sdf`` may live on a separate
        higher-resolution grid and is the source for viewport/cache/export mesh
        extraction.
        """
        s = self.settings
        if not s.surface_enabled:
            return
        if build_lowres is None:
            build_lowres = bool(getattr(s, "surface_build_lowres_sdf", True))
        surface_positions = self._surface_positions()

        if build_lowres:
            self._build_surface_sdf_on_grid(
                self.grid, self.surface_sdf, self.surface_tmp, self._nvox,
                surface_positions, float(s.voxel_size),
            )

        if not build_render:
            return
        self._ensure_render_surface_resources()
        rg = self.render_surface_grid
        if rg is self.grid:
            if not build_lowres:
                self._build_surface_sdf_on_grid(
                    self.grid, self.surface_sdf, self.surface_tmp, self._nvox,
                    surface_positions, float(s.voxel_size),
                )
            self._render_surface_nvox = self._nvox
        else:
            rnvox = self._rebuild_render_surface_topology(surface_positions)
            self._build_surface_sdf_on_grid(
                rg, self.render_surface_sdf, self.render_surface_tmp, rnvox,
                surface_positions, float(rg.voxel_size),
            )
        self._render_surface_built_key = self._render_surface_build_key()
        self._invalidate_surface_mesh()

    def rebuild_surface(self):
        """Rebuild current-frame surface data after realtime setting changes."""
        self._render_surface_built_key = None
        self._invalidate_surface_mesh()
        self._surface()

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
        self._emit_seed = (self._emit_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        wp.launch(
            k_emit_whitewater, dim=self._np_dim(),
            inputs=[p.position, p.velocity, p.count,
                    self.churn.data, self.surface_sdf.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, s.voxel_size, dt,
                    s.ww_churn_threshold, s.ww_emission_rate,
                    s.ww_surface_band * s.voxel_size, s.ww_min_speed,
                    s.ww_life_min, s.ww_life_max,
                    self._emit_seed, ww.capacity],
            outputs=[ww.position, ww.velocity, ww.age, ww.life,
                     ww.alive, ww.count],
            device=self.device,
        )
        # Refresh the whitewater bound after device-side emission so the
        # advect/deposit launches cover new particles without dispatching
        # full capacity (one readback per substep, whitewater only). Live
        # playback needs an explicit stream barrier here: otherwise the
        # device-to-host count copy can race the high-volume atomic emission.
        wp.synchronize_device(self.device)
        ww.refresh_host_count()
        nww = self._nww_dim()
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
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max)],
            device=self.device,
        )
        self.ww_density.data[:self._nvox].zero_()
        wp.launch(
            k_whitewater_set_grid_data, dim=nww,
            inputs=[ww.position, ww.alive, ww.count,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, s.voxel_size],
            outputs=[self.ww_density.data], device=self.device,
        )
        ww.compact(count_hint=nww)

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

        for _ in range(substeps):
            self.snippets.run(CustomKernelOrder.FIRST_STEP, self, time_ctx)
            with self._phase("misc"):
                self._rebuild_topology()
                self._clear_grid_fields()
            with self._phase("collision"):
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
            with self._phase("surface"):
                cadence = int(getattr(s, "surface_update_cadence", int(SurfaceUpdateCadence.FRAME)))
                if cadence == int(SurfaceUpdateCadence.SUBSTEP):
                    self._surface(build_render=True, build_lowres=True)
                elif s.whitewater_enabled:
                    # Whitewater uses the legacy simulation-grid surface SDF.
                    self._surface(build_render=False, build_lowres=True)
            with self._phase("whitewater"):
                self._whitewater(dt)
            self.snippets.run(CustomKernelOrder.FINAL_STEP, self, time_ctx)
            self.sim_time += dt
            time_ctx["time"] = self.sim_time

        self.frame += 1
        cadence = int(getattr(s, "surface_update_cadence", int(SurfaceUpdateCadence.FRAME)))
        if s.surface_enabled and cadence == int(SurfaceUpdateCadence.FRAME):
            with self._phase("surface"):
                self._surface(build_render=True,
                              build_lowres=bool(getattr(s, "surface_build_lowres_sdf", True)))

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
        st.surface_voxel_size = float(self._render_surface_voxel_size())
        st.surface_active_leaves = int(getattr(self.render_surface_grid, "active_leaf_count", 0))
        st.surface_active_voxels = int(st.surface_active_leaves * C.LEAF_VOLUME)
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
        from .sourcing import k_source_particles

        pts = np.ascontiguousarray(positions, dtype=np.float32)
        n = len(pts)
        if n == 0:
            return 0
        src_pos = wp.array(pts, dtype=wp.vec3, device=self.device)
        if velocities is not None:
            vels = np.ascontiguousarray(velocities, dtype=np.float32)
        else:
            vels = np.tile(np.asarray(velocity, dtype=np.float32), (n, 1))
        src_vel = wp.array(vels, dtype=wp.vec3, device=self.device)
        p = self.particles
        wp.launch(
            k_source_particles, dim=n,
            inputs=[src_pos, src_vel, n, float(life),
                    1 if use_lifespan else 0, float(temperature),
                    float(viscosity), float(density), p.capacity,
                    p.position, p.velocity, p.attr("age"), p.attr("life"),
                    p.attr("temperature"), p.attr("viscosity"),
                    p.attr("density"), p.time_offset, p.alive, p.count, self.first_slot_dev],
            device=self.device,
        )
        return p.refresh_host_count()

    # -- output ---------------------------------------------------------------

    def surface_mesh(self):
        """Extract the selected render surface as (verts Nx3, tris Mx3).

        The selected mesh source may be a separate high-resolution render
        surface grid. Meshes are memoized per frame and surfacing settings.
        """
        if self.settings.surface_enabled:
            # In MANUAL cadence or after a settings change the render SDF may
            # be stale/missing. Rebuilding here keeps viewport/export calls
            # deterministic without changing particle simulation state.
            desired = self._render_surface_build_key()
            if self._render_surface_built_key != desired:
                self._surface(build_render=True, build_lowres=False)
        key = self._render_surface_build_key()
        if self._surface_mesh_cache is not None and self._surface_mesh_key == key:
            return self._surface_mesh_cache
        mesh = self.render_surface_extractor.extract(self.render_surface_sdf)
        self._surface_mesh_key = key
        self._surface_mesh_cache = mesh
        return mesh

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
