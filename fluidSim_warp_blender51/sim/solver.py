"""FluidSim solver orchestration: GPU-resident FLIP/PIC pipeline.

All simulation state lives on the GPU (wp.array); the host only sequences
kernel launches, reads back small statistics for the UI/profiler, and
receives surface mesh / particle data for Blender display.
"""

from __future__ import annotations

import gc
import math
import time as _time
from dataclasses import dataclass, field as dc_field

import numpy as np
import warp as wp
from . import fastlaunch
from .fastlaunch import launch as fast_launch

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
    SourceEvaluationCadence,
)
from .fields import (
    SparseGrid,
    ScalarField,
    MacVelocityField,
    voxel_coord_of_flat,
    voxel_center_world,
)
from .particles import ParticleSystem, ParticleBins
from .attributes import persistent_particle_channels
from .checkpoint_state import CheckpointSchema
from .color_mixing import ColorMixTransport
from .frame_features import FrameFeaturePhase, FrameFeatures
from .frame_runtime import FrameStats, SubstepPolicy, select_substeps
from .quality_contract import resolve_quality
from .pressure import PressureSolver
from .viscosity import ViscositySolver
from .whitewater import (
    WhitewaterSystem,
    normalized_life_range,
    k_compute_churn,
    k_advect_whitewater,
    k_whitewater_set_grid_data,
)
from .whitewater_types import (
    WHITEWATER_TYPE_SCHEMA_VERSION,
    classification_policy,
    validate_type_values,
)
from .surface_tension import (
    SurfaceTensionSolver,
    required_sdf_splat_radius_voxels,
    required_topology_expand_leaves,
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
    ForceFieldProxy,
    k_gather_grid_data,
    k_gather_grid_data_apic,
    k_normalize_mac,
    k_particle_to_count,
    k_particle_to_grid_active,
    k_staggered_field_to_particle,
    k_staggered_field_to_particle_apic,
    k_apply_collider_material_to_grid,
    k_advect_particles,
    k_forces_gravity,
    k_forces_proxy_fields,
    k_forces_directional,
    k_forces_omnidirectional,
    k_forces_diffusion,
    k_dissipate_particle_attribute,
    k_attract_sdf_gradient_direction,
    k_forces_speed_limit,
    k_particle_to_voxel_count_and_update_index,
    voxel_count_to_starts,
    scatter_particles_to_bins,
    k_particle_age,
    k_particle_lifespan,
    k_particle_in_sink_check,
    k_particles_relax,
    k_clear_just_initialized_tracked,
)


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


# Small helper kernels owned by the orchestrator


@wp.kernel
def k_fill_float(data: wp.array(dtype=float), value: float):
    data[wp.tid()] = value


@wp.kernel
def k_zero_fluid_grid(
    u: wp.array(dtype=float), v: wp.array(dtype=float),
    w: wp.array(dtype=float), wu: wp.array(dtype=float),
    wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    temperature: wp.array(dtype=float), viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float), voxel_type: wp.array(dtype=wp.int32),
    voxel_counts: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    u[i] = 0.0
    v[i] = 0.0
    w[i] = 0.0
    wu[i] = 0.0
    wv[i] = 0.0
    ww[i] = 0.0
    temperature[i] = 0.0
    viscosity[i] = 0.0
    density[i] = 0.0
    voxel_type[i] = 0
    voxel_counts[i] = 0


@wp.kernel
def k_reset_collision_grid(
    velocity: wp.array(dtype=wp.vec3), friction: wp.array(dtype=float),
    restitution: wp.array(dtype=float), sdf: wp.array(dtype=float),
    far: float,
):
    i = wp.tid()
    velocity[i] = wp.vec3(0.0, 0.0, 0.0)
    friction[i] = 0.0
    restitution[i] = 0.0
    sdf[i] = far


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
def k_transform_collider_points(
    base_points: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    axis: wp.vec3,
    angle: float,
    linear_disp: wp.vec3,
    out_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    q = wp.quat_from_axis_angle(axis, angle)
    out_points[tid] = center + wp.quat_rotate(q, base_points[tid] - center) + linear_disp


@wp.kernel
def k_copy_float(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def k_copy3_float(
    src_a: wp.array(dtype=float), src_b: wp.array(dtype=float),
    src_c: wp.array(dtype=float), dst_a: wp.array(dtype=float),
    dst_b: wp.array(dtype=float), dst_c: wp.array(dtype=float),
):
    tid = wp.tid()
    dst_a[tid] = src_a[tid]
    dst_b[tid] = src_b[tid]
    dst_c[tid] = src_c[tid]


@wp.func
def _sanitize_mac_face(vel: wp.array(dtype=float), weight: wp.array(dtype=float),
                       tid: int, max_speed: float):
    wt = weight[tid]
    # REVIEW-014: every comparison with a NaN weight is False, so a NaN
    # weight used to skip both branches below and poison the next normalize.
    # A face whose weight is not finite carries no trustworthy velocity.
    if wp.isnan(wt) or wp.isinf(wt):
        weight[tid] = 0.0
        vel[tid] = 0.0
        return
    val = vel[tid]
    if wp.isnan(val) or wp.isinf(val):
        vel[tid] = 0.0
        if wt > 1.0e-8:
            weight[tid] = 1.0e-6
    elif wt > 1.0e-8:
        vel[tid] = wp.clamp(val, -max_speed, max_speed)


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
    _sanitize_mac_face(u, wu, tid, max_speed)
    _sanitize_mac_face(v, wv, tid, max_speed)
    _sanitize_mac_face(w, ww, tid, max_speed)


@wp.kernel
def k_apic_clamp_affine(
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    max_speed: float,
):
    """APIC companion of the speed clamp in :func:`k_sanitize_particles`.

    Runs just before it, on the same unclamped velocity, and rescales the
    affine matrix by the factor the clamp will apply to the velocity so a
    clamped particle keeps its direction of motion and its affine shape in
    proportion.  A non-finite matrix is zeroed (zero affine is always valid).
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    if alive[tid] == 0:
        return
    c = affine[tid]
    finite = True
    for i in range(3):
        for j in range(3):
            if wp.isnan(c[i, j]) or wp.isinf(c[i, j]):
                finite = False
    if not finite:
        affine[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return
    speed = wp.length(velocities[tid])
    if speed > max_speed:
        affine[tid] = c * (max_speed / speed)


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


# Host-side source / collider descriptors (filled by the Blender bridge)


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
    # Solver-internal: frame-start copy of mesh.points, set by set_colliders()
    # only when this collider is moving. Non-None marks it eligible for
    # per-substep re-transform; the Blender bridge never sets this field.
    base_points: object = None
    # Solver-internal: mesh.points currently hold an extrapolated sub-frame
    # pose that _restore_collider_poses must undo.
    reposed: bool = False


@dataclass
class ForceFieldDesc:
    """GPU force driven by a native Blender force-field object."""

    force_type: int = 0
    shape: int = 0
    texture_mode: int = 0
    flags: int = 0
    seed: int = 0
    center: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, 1.0)
    source_velocity: tuple = (0.0, 0.0, 1.0)
    guide_points: tuple = ()
    strength: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    falloff_power: float = 0.0
    flow: float = 0.0
    noise: float = 0.0
    size: float = 1.0
    rest_length: float = 0.0
    damping: float = 0.0
    linear_drag: float = 0.0
    quadratic_drag: float = 0.0
    guide_minimum: float = 0.0
    guide_free: float = 0.0
    guide_clump: float = 0.0

    def to_struct(self):
        field = ForceFieldProxy()
        field.force_type = int(self.force_type)
        field.shape = int(self.shape)
        field.texture_mode = int(self.texture_mode)
        field.flags = int(self.flags)
        field.seed = int(self.seed)
        field.center = wp.vec3(*[float(value) for value in self.center])
        field.direction = wp.vec3(*[float(value) for value in self.direction])
        field.source_velocity = wp.vec3(
            *[float(value) for value in self.source_velocity])
        guide_points = list(self.guide_points[:4])
        guide_points.extend([(0.0, 0.0, 0.0)] * (4 - len(guide_points)))
        field.guide_p0 = wp.vec3(*guide_points[0])
        field.guide_p1 = wp.vec3(*guide_points[1])
        field.guide_p2 = wp.vec3(*guide_points[2])
        field.guide_p3 = wp.vec3(*guide_points[3])
        field.guide_count = min(len(self.guide_points), 4)
        field.strength = float(self.strength)
        field.min_distance = float(self.min_distance)
        field.max_distance = float(self.max_distance)
        field.falloff_power = float(self.falloff_power)
        field.flow = float(self.flow)
        field.noise = float(self.noise)
        field.size = float(self.size)
        field.rest_length = float(self.rest_length)
        field.damping = float(self.damping)
        field.linear_drag = float(self.linear_drag)
        field.quadratic_drag = float(self.quadratic_drag)
        field.guide_minimum = float(self.guide_minimum)
        field.guide_free = float(self.guide_free)
        field.guide_clump = float(self.guide_clump)
        return field


from .solver_settings_model import SolverSettings  # noqa: F401 (re-export)


_CHECKPOINT_FIELD_NAMES = (
    "vel.u", "vel.v", "vel.w", "vel.wu", "vel.wv", "vel.ww",
    "vel_saved_u", "vel_saved_v", "vel_saved_w",
    "voxel_type.data", "temperature.data", "viscosity_field.data",
    "density.data", "churn.data",
)


_NARROW_BAND_CHECKPOINT_FIELDS = (
    "narrow_band.phi", "narrow_band.nb_u", "narrow_band.nb_v",
    "narrow_band.nb_w",
)


def _resolve_attr(root, dotted_name: str):
    value = root
    for component in dotted_name.split("."):
        value = getattr(value, component)
    return value


class FluidSolver:
    """GPU-resident FLIP/PIC liquid solver."""

    # Class default so solver stand-ins built without __init__ read as "mode off".
    narrow_band = None

    def __init__(self, settings: SolverSettings, profiler=None):
        self.settings = settings
        self.quality = resolve_quality(
            settings, (settings.domain_min, settings.domain_max))
        settings.resolved_quality = self.quality
        self.device = settings.device
        self.profiler = profiler
        # The PCG pressure path (the default; see SolverSettings.pressure_solver)
        # unconditionally captures a CUDA graph (_launch_pcg_batch ->
        # wp.ScopedCapture) with no CPU fallback, and this add-on is GPU-only
        # by design (AGENTS.md: "Keep simulation GPU-resident"). A bad/CPU
        # device string previously surfaced as a bare
        # "RuntimeError: Must be a CUDA device" from deep inside the first
        # pressure solve. Fail fast with an actionable message instead.
        try:
            resolved_device = wp.get_device(self.device)
            is_cuda = bool(resolved_device.is_cuda)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"FluidSim: invalid Warp device {self.device!r} "
                f"({exc}). Set a valid CUDA device (e.g. 'cuda:0') in "
                "Add-on Preferences."
            ) from exc
        if not is_cuda:
            raise RuntimeError(
                f"FluidSim requires a CUDA device; got {self.device!r}. "
                "This solver has no CPU-resident pressure-projection path. "
                "Set a valid CUDA device (e.g. 'cuda:0') in Add-on Preferences."
            )

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

        self.particles = ParticleSystem(
            settings.max_particles, self.device,
            persistent_particle_channels(settings))
        self.bins = ParticleBins(settings.max_particles, vc, self.device)
        self._sync_affine_mode()
        self.whitewater = WhitewaterSystem(
            settings.max_whitewater, self.device,
            candidate_capacity=settings.max_particles)

        self.pressure_solver = PressureSolver(g, self.device)
        self.viscosity_solver = ViscositySolver(g, self.device)
        self.sourcing = SourcePipeline(
            device=self.device,
            grid=self.grid,
            particles=self.particles,
            settings=self.settings,
            voxel_type=self.voxel_type,
            collision_sdf=self.collision_sdf,
            collision_velocity=self.collision_vel,
            velocity=self.vel,
            temperature=self.temperature,
            viscosity_field=self.viscosity_field,
            density=self.density,
            voxel_counts=self.voxel_counts,
            pressure_solver=self.pressure_solver,
            active_voxel_count=lambda: getattr(self, "_nvox", 1),
            current_frame=lambda: getattr(self, "frame", 0),
            ensure_particle_capacity=self.ensure_particle_capacity,
            invalidate_host_cache=lambda: setattr(
                self, "_host_frame_cache", None),
            invalidate_particle_surface=lambda: (
                self.surface.invalidate_particle_state()
                if getattr(self, "surface", None) is not None else None),
        )
        self.surface = SurfacePipeline(self)
        self.surface_tension = SurfaceTensionSolver(self)
        self.color_mixing = self._new_color_mixing()
        self._host_frame_cache = None
        self.surface_attribute_transfer = None
        self.last_particle_growth = None
        self.features = FrameFeatures(device=self.device)
        # Compatibility alias for diagnostics that read this established name.
        self.effects = self.features.effects

        self.colliders: list[ColliderDesc] = []
        self.force_fields: list[ForceFieldDesc] = []
        self._force_fields_dev = None

        self.max_speed_dev = wp.zeros(1, dtype=float, device=self.device)

        self.frame = 0
        self.sim_time = 0.0
        self.stats = FrameStats()
        self._whitewater_emit_seed = 12345
        # Active-work launch bound for per-voxel kernels, refreshed after
        # every topology rebuild. Kernels guard on the device-side count, so
        # any dim >= the active voxel count is safe; dispatching only the
        # active region instead of the full allocation keeps kernel cost
        # proportional to live work (see TASK-POLICY-DRIFT-002).
        self._nvox = 1
        # Narrow-band FLIP state (None unless the mode is on; see narrow_band_state).
        self.narrow_band = None
        if getattr(settings, "narrow_band_enabled", False):
            from .narrow_band_state import NarrowBand
            self.narrow_band = NarrowBand(self)

    # Host configuration

    def set_source_batch(self, legacy_descriptors=(), native_plans=()):
        """Atomically install the complete runtime source snapshot.

        This is the only way to install sources, and it replaces everything
        installed before: legacy descriptors and native plans always arrive
        together, so neither representation can be dropped by a later call.
        """
        from .sourcing import SourceBatch
        self.sourcing.set_source_batch(SourceBatch(
            tuple(legacy_descriptors), tuple(native_plans)))

    def set_colliders(self, colliders):
        # The meshes are the scene's cached collider meshes; hand back any we
        # re-posed before letting go of them (also covers a step that raised).
        self._restore_collider_poses()
        self.colliders = list(colliders)
        for col in self.colliders:
            moving = col.mesh is not None and (
                any(abs(v) > 1.0e-9 for v in col.linear_velocity)
                or any(abs(v) > 1.0e-9 for v in col.angular_velocity))
            col.base_points = wp.clone(col.mesh.points) if moving else None

    def set_force_fields(self, force_fields):
        self.force_fields = list(force_fields)
        self._force_fields_dev = None

    def _force_field_data(self):
        if self._force_fields_dev is None:
            self._force_fields_dev = wp.array(
                [field.to_struct() for field in self.force_fields],
                dtype=ForceFieldProxy, device=self.device)
        return self._force_fields_dev

    def set_height_source(self, desc):
        self.sourcing.set_height_source(desc)

    @property
    def sources(self):
        return self.sourcing.sources

    @property
    def height_source(self):
        return self.sourcing.height_source

    def set_custom_effect_candidate(self, candidate):
        """Atomically install a fully compiled and sandboxed effect stack."""
        return self.features.swap_candidate(candidate)

    def update_custom_effect_runtime_bindings(self, instances):
        """Update launch-time Custom Effect values without replacing solver state."""
        return self.features.update_runtime_bindings(instances)

    def _apic_active(self) -> bool:
        return int(getattr(self.settings, "sample_type", ParticleSampleType.FLIP)
                   ) == int(ParticleSampleType.APIC)

    def _sync_affine_mode(self):
        """Make the affine channel match the live transfer mode.

        ``sample_type`` is a live setting.  APIC allocates the channel lazily
        (zeroed, which is a valid state); leaving APIC frees it, so nothing
        stale can leak back in when APIC is selected again.
        """
        if self._apic_active():
            self.particles.ensure_affine()
        elif getattr(self.particles, "affine", None) is not None:
            self.particles.release_affine()

    def ensure_particle_capacity(self, required: int):
        """Atomically grow particle/binner storage while preserving solver state."""
        from .memory_safety import (
            affine_bytes_per_particle, release_cuda_memory, vram_snapshot,
        )
        from .particle_capacity import (
            ParticleGrowthResult, plan_particle_growth,
        )

        current = int(self.particles.capacity)
        required = max(0, int(required))
        if required <= current:
            result = ParticleGrowthResult(
                "SUFFICIENT", current, current, required, current)
            self.last_particle_growth = result
            return result
        if not bool(getattr(self.settings, "auto_grow_particles", True)):
            result = ParticleGrowthResult(
                "DISABLED", current, current, required, current,
                reason="automatic particle growth is disabled")
            self.last_particle_growth = result
            return result

        resources = vram_snapshot(self.device)
        plan = plan_particle_growth(
            current, required,
            gpu_total_bytes=int(resources.get("gpu_total_bytes", 0)),
            gpu_free_bytes=int(resources.get("gpu_free_bytes", 0)),
            extra_particle_bytes=affine_bytes_per_particle(self.settings),
        )
        if not plan.allowed:
            result = ParticleGrowthResult(
                "BLOCKED", current, current, required, plan.target,
                plan.peak_allocation_bytes, plan.budget_bytes, plan.reason)
            self.last_particle_growth = result
            return result

        grown_particles = None
        grown_bins = None
        grown_whitewater_workspace = None
        try:
            # The old pools remain authoritative until every allocation and
            # copy, including deterministic whitewater emission scratch, has
            # completed successfully.
            grown_particles = self.particles.resized_copy(plan.target)
            grown_bins = ParticleBins(
                plan.target, self.grid.voxel_capacity, self.device)
            grown_whitewater_workspace = (
                self.whitewater.allocate_candidate_workspace(plan.target))
            wp.synchronize_device(self.device)
        except Exception as exc:  # noqa: BLE001 - CUDA allocation surface
            grown_particles = None
            grown_bins = None
            grown_whitewater_workspace = None
            gc.collect()
            release_cuda_memory(self.device)
            result = ParticleGrowthResult(
                "FAILED", current, current, required, plan.target,
                plan.peak_allocation_bytes, plan.budget_bytes,
                f"CUDA particle growth failed before swap: {exc}")
            self.last_particle_growth = result
            return result

        old_particles, old_bins = self.particles, self.bins
        self.particles = grown_particles
        self.bins = grown_bins
        self.whitewater.swap_candidate_workspace(grown_whitewater_workspace)
        self.settings.max_particles = int(plan.target)
        self._host_frame_cache = None
        sourcing = getattr(self, "sourcing", None)
        if sourcing is not None:
            sourcing.replace_particle_storage(self.particles)
        del old_particles, old_bins
        gc.collect()
        release_cuda_memory(self.device)
        result = ParticleGrowthResult(
            "GROWN", current, plan.target, required, plan.target,
            plan.peak_allocation_bytes, plan.budget_bytes)
        self.last_particle_growth = result
        return result

    def reset(self):
        self._host_frame_cache = None
        self.surface_attribute_transfer = None
        sourcing = getattr(self, "sourcing", None)
        if sourcing is not None:
            sourcing.reset()
        expected_channels = persistent_particle_channels(self.settings)
        if tuple(self.particles.optional_channels) != tuple(expected_channels):
            # Reset is the explicit transaction boundary for persistent output
            # inventory changes.  No live state is copied across this swap.
            old_particles = self.particles
            self.particles = ParticleSystem(
                old_particles.capacity, self.device, expected_channels)
            if sourcing is not None:
                sourcing.replace_particle_storage(self.particles)
            del old_particles
        self.particles.clear()
        self._sync_affine_mode()
        self.color_mixing = self._new_color_mixing()
        self.whitewater.clear()
        if self.narrow_band is not None:
            self.narrow_band.reset()
        self.churn.zero()
        ww_density = getattr(self, "ww_density", None)
        if ww_density is not None:
            ww_density.zero()
        self.frame = 0
        self.sim_time = 0.0
        settings = getattr(self, "settings", None)
        self.grid.begin_topology()
        self.grid.finalize_topology()
        self._nvox = max(1, self.grid.active_voxel_count)
        if settings is not None:
            surface = getattr(self, "surface", None)
            if surface is not None:
                surface.reset()
            surface_tension = getattr(self, "surface_tension", None)
            if surface_tension is not None:
                surface_tension.reset()
        self._host_frame_cache = None

    def _checkpoint_field_names(self):
        """Persistent grid fields; narrow-band adds its own only when on."""
        if self.narrow_band is None:
            return _CHECKPOINT_FIELD_NAMES
        return _CHECKPOINT_FIELD_NAMES + _NARROW_BAND_CHECKPOINT_FIELDS

    def export_checkpoint_state(self):
        """Synchronize once and return host-owned deterministic restart state.

        Scratch pressure, collision, binning, surface-extraction, and temporary
        reduction arrays are omitted because ``import_checkpoint_state`` clears
        or rebuilds them before the next source/P2G transaction.
        """
        wp.synchronize_device(self.device)
        leaf_count = int(self.grid.refresh_topology_counts())
        particle_count = int(self.particles.refresh_host_count())
        whitewater_count = int(self.whitewater.refresh_host_count())
        active_voxels = leaf_count * C.LEAF_VOLUME

        arrays = {
            "grid_leaf_coords": np.ascontiguousarray(
                self.grid.leaf_coords.numpy()[:leaf_count], dtype=np.int32),
            "grid_old_leaf_table": np.ascontiguousarray(
                self.old_leaf_table.numpy(), dtype=np.int32),
            "particle_position": np.ascontiguousarray(
                self.particles.position.numpy()[:particle_count], dtype=np.float32),
            "particle_velocity": np.ascontiguousarray(
                self.particles.velocity.numpy()[:particle_count], dtype=np.float32),
            "particle_alive": np.ascontiguousarray(
                self.particles.alive.numpy()[:particle_count], dtype=np.int32),
            "particle_time_offset": np.ascontiguousarray(
                self.particles.time_offset.numpy()[:particle_count], dtype=np.float32),
            "whitewater_position": np.ascontiguousarray(
                self.whitewater.position.numpy()[:whitewater_count], dtype=np.float32),
            "whitewater_velocity": np.ascontiguousarray(
                self.whitewater.velocity.numpy()[:whitewater_count], dtype=np.float32),
            "whitewater_age": np.ascontiguousarray(
                self.whitewater.age.numpy()[:whitewater_count], dtype=np.float32),
            "whitewater_life": np.ascontiguousarray(
                self.whitewater.life.numpy()[:whitewater_count], dtype=np.float32),
            "whitewater_alive": np.ascontiguousarray(
                self.whitewater.alive.numpy()[:whitewater_count], dtype=np.int32),
            "whitewater_type": np.ascontiguousarray(
                self.whitewater.type.numpy()[:whitewater_count], dtype=np.int32),
        }
        for name in self.particles.ATTR_NAMES:
            arrays[f"particle_attr_{name}"] = np.ascontiguousarray(
                self.particles.attrs[name].numpy()[:particle_count],
                dtype=np.float32)
        arrays.update(self.particles.checkpoint_arrays(particle_count))
        for dotted_name in self._checkpoint_field_names():
            arrays[f"field_{dotted_name.replace('.', '_')}"] = np.ascontiguousarray(
                _resolve_attr(self, dotted_name).numpy()[:active_voxels])

        if self.narrow_band is not None:
            arrays["narrow_band_seed_state"] = np.asarray(
                [self.narrow_band.reseed_epoch, int(self.narrow_band.owed.numpy()[0])],
                dtype=np.uint32)
        return {
            "arrays": arrays,
            "scalars": {
                "frame": int(self.frame),
                "sim_time": float(self.sim_time),
                "leaf_count": leaf_count,
                "particle_count": particle_count,
                "whitewater_count": whitewater_count,
                "leaf_capacity": int(self.grid.capacity),
                "particle_capacity": int(self.particles.capacity),
                "whitewater_capacity": int(self.whitewater.capacity),
                "whitewater_emit_seed": int(self._whitewater_emit_seed),
                "next_particle_uid": int(self.particles.next_uid),
                "particle_optional_channels": ",".join(
                    self.particles.optional_channels),
                "whitewater_type_schema_version": WHITEWATER_TYPE_SCHEMA_VERSION,
            },
            "source_state": self.sourcing.export_checkpoint_state(),
        }

    def _validate_checkpoint_state(self, state):
        schema = CheckpointSchema(
            leaf_capacity=int(self.grid.capacity),
            leaf_table_cells=int(self.grid.num_leaf_cells),
            whitewater_capacity=int(self.whitewater.capacity),
            particle_attributes=tuple(self.particles.ATTR_NAMES),
            optional_particle_channels=tuple(
                self.particles.optional_channels),
            field_names=self._checkpoint_field_names(),
            whitewater_type_schema_version=WHITEWATER_TYPE_SCHEMA_VERSION,
        )
        return schema.validate(state)

    def import_checkpoint_state(self, state):
        """Restore into a newly initialized, signature-compatible solver."""
        arrays, scalars, counts = self._validate_checkpoint_state(state)
        particle_count = counts["particle_count"]
        self._sync_affine_mode()
        if particle_count > self.particles.capacity:
            growth = self.ensure_particle_capacity(particle_count)
            if str(getattr(growth, "status", "")) not in {
                    "GROWN", "SUFFICIENT"}:
                raise ValueError(
                    "checkpoint particle count cannot be admitted: "
                    f"{getattr(growth, 'reason', growth)}")

        self.grid.restore_topology(arrays["grid_leaf_coords"])
        active_voxels = counts["leaf_count"] * C.LEAF_VOLUME
        for dotted_name in self._checkpoint_field_names():
            target = _resolve_attr(self, dotted_name)
            target.zero_()
            if active_voxels:
                source = np.ascontiguousarray(
                    arrays[f"field_{dotted_name.replace('.', '_')}"])
                dtype = wp.int32 if source.dtype.kind in "iu" else float
                wp.copy(
                    target,
                    wp.array(source, dtype=dtype, device=self.device),
                    count=active_voxels)
        old_table = np.ascontiguousarray(
            arrays["grid_old_leaf_table"], dtype=np.int32)
        wp.copy(
            self.old_leaf_table,
            wp.array(old_table, dtype=wp.int32, device=self.device))
        self.particles.restore_checkpoint_prefix(arrays, particle_count)
        if "uid" in self.particles.optional_channels:
            self.particles.next_uid = int(scalars["next_particle_uid"])
            if particle_count:
                maximum_uid = int(np.max(np.asarray(
                    arrays["particle_attr_uid"], dtype=np.int64)))
                if self.particles.next_uid <= maximum_uid:
                    raise ValueError(
                        "next_particle_uid must exceed every restored UID")
        else:
            self.particles.next_uid = int(scalars.get("next_particle_uid", 1))
        self.whitewater.restore_checkpoint_prefix(
            arrays, counts["whitewater_count"])
        self.sourcing.import_checkpoint_state(state["source_state"])

        self.frame = int(scalars["frame"])
        self.sim_time = float(scalars["sim_time"])
        self._whitewater_emit_seed = int(scalars["whitewater_emit_seed"])
        if self.narrow_band is not None:
            seed_state = np.asarray(
                arrays.get("narrow_band_seed_state", ()), dtype=np.uint32)
            self.narrow_band.restore_seed_state(seed_state)
        self._rebuild_transient_state_after_restore()
        self.assert_checkpoint_invariants()

    def _rebuild_transient_state_after_restore(self):
        """Clear/rebuild every intentionally omitted scratch subsystem."""
        self.bins.zero()
        self.voxel_counts.zero()
        for value in (
                self.tmp_u, self.tmp_v, self.tmp_w,
                self.collision_impact_u, self.collision_impact_v,
                self.collision_impact_w, self.collision_vel,
                self.collision_friction, self.collision_restitution):
            value.zero_()
        self.collision_sdf.zero()
        self.churn_tmp.zero()
        self.attr_weight.zero()
        self.ww_density.zero()
        self._force_fields_dev = None
        self._host_frame_cache = None
        self._nvox = max(1, self.grid.active_voxel_count)
        # Not checkpointed, but not scratch either: the next frame's foam
        # ageing reads it. Checkpoints are taken after _finalize_whitewater,
        # so re-splatting the restored whitewater reproduces it exactly.
        if self.settings.whitewater_enabled:
            self._splat_whitewater_density(self.whitewater.refresh_host_count())
        self.surface.reset()
        self.surface_tension.reset()
        self.whitewater.reset_frame_diagnostics()
        self.stats = FrameStats()
        wp.synchronize_device(self.device)

    def assert_checkpoint_invariants(self):
        leaf_count = int(self.grid.refresh_topology_counts())
        particle_count = int(self.particles.refresh_host_count())
        whitewater_count = int(self.whitewater.refresh_host_count())
        if not 0 <= leaf_count <= self.grid.capacity:
            raise ValueError("restored leaf count violates capacity")
        if not 0 <= particle_count <= self.particles.capacity:
            raise ValueError("restored particle count violates capacity")
        if not 0 <= whitewater_count <= self.whitewater.capacity:
            raise ValueError("restored whitewater count violates capacity")
        for name, value in (
                ("particle_position", self.particles.position.numpy()[:particle_count]),
                ("particle_velocity", self.particles.velocity.numpy()[:particle_count]),
                ("whitewater_position", self.whitewater.position.numpy()[:whitewater_count]),
                ("whitewater_velocity", self.whitewater.velocity.numpy()[:whitewater_count])):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"restored {name} contains non-finite values")
        validate_type_values(
            self.whitewater.type.numpy()[:whitewater_count],
            allow_unclassified=False,
        )
        if "uid" in self.particles.optional_channels and particle_count:
            uid = self.particles.int_attrs["uid"].numpy()[:particle_count]
            if np.any(uid <= 0) or len(np.unique(uid)) != particle_count:
                raise ValueError("restored particle UIDs are invalid")
            if int(self.particles.next_uid) <= int(np.max(uid)):
                raise ValueError("restored next_particle_uid is not monotonic")
        if not math.isfinite(self.sim_time):
            raise ValueError("restored timing state is non-finite")
        return True

    def release(self):
        """Break GPU-owning backreferences before the session drops this solver."""
        fastlaunch.clear()
        pressure_solver = getattr(self, "pressure_solver", None)
        if pressure_solver is not None:
            pressure_solver.release()

        viscosity_solver = getattr(self, "viscosity_solver", None)
        if viscosity_solver is not None:
            viscosity_solver.release()

        host_frame = self._host_frame_cache
        if host_frame is not None:
            host_frame.solver = None
        self._host_frame_cache = None
        self.surface_attribute_transfer = None

        surface = getattr(self, "surface", None)
        if surface is not None:
            surface.extractor.reset_buffers()
            if surface.render_extractor is not surface.extractor:
                surface.render_extractor.reset_buffers()
            surface.solver = None
        # Drop the attribute rather than nulling it: _render_surface_voxel_size
        # and io.cache._frame_payload both gate on hasattr(..., "surface"), and
        # a None left in place turns that guard into an AttributeError that
        # masks the solver error being handled.
        surface_tension = getattr(self, "surface_tension", None)
        if surface_tension is not None:
            surface_tension.release()
        if hasattr(self, "surface_tension"):
            del self.surface_tension
        if hasattr(self, "surface"):
            del self.surface

        sourcing = getattr(self, "sourcing", None)
        if sourcing is not None:
            release = getattr(sourcing, "release", None)
            if release is not None:
                release()
            else:
                sourcing.owner = None
        self.sourcing = None
        self.force_fields = []
        self._force_fields_dev = None

    # Active-work launch bounds

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

    # Render surfacing compatibility

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
        return self.particles.position


    def _phase(self, name):
        if self.profiler is not None:
            return self.profiler.track(name)
        import contextlib
        return contextlib.nullcontext()


    def _rebuild_topology(self, expand=None):
        g = self.grid
        s = self.settings
        wp.copy(self.old_leaf_table, g.leaf_table)
        g.begin_topology()
        g.mark_points(self.particles.position, self.particles.count,
                      self._np_dim())
        if self.narrow_band is not None:
            self.narrow_band.mark_leaves()
        for src in self.sources:
            if not self.sourcing.legacy_descriptor_requires_topology(src):
                continue
            lo, hi = src.aabb()
            g.mark_world_box(lo, hi)
        if self.sourcing.plans:
            # Compile before topology so imported-volume active leaves and all
            # other normalized regions allocate through the same flat table.
            dynamic_sources = any(
                plan.cadence == SourceEvaluationCadence.EVERY_SIM_FRAME
                for plan in self.sourcing.plans)
            batch = self.sourcing._compiled_batch
            if batch is None or dynamic_sources:
                batch = self.sourcing.compile(frame=self.frame, substep=0)
            self.sourcing.mark_source_topology(g, batch)
        for col in self.colliders:
            # A collider with no domain overlap (placed outside, or animated
            # out) has no influence this frame, matching the capacity planner;
            # strict rejection stays for sources, where it is an authoring error.
            if g.overlaps_world_box(col.aabb_min, col.aabb_max):
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

        # Carry the persistent grid fields across the rebuild: churn, and the
        # end-of-frame whitewater density that foam ageing reads next frame.
        # churn_tmp is free scratch again once the churn swap is done.
        fast_launch(
            k_remap_leaf_field, dim=self._nvox,
            inputs=[self.churn.data, self.old_leaf_table, g.leaf_coords,
                    g.num_leaves, g.leaves_dim],
            outputs=[self.churn_tmp.data], device=self.device,
        )
        self.churn.data, self.churn_tmp.data = self.churn_tmp.data, self.churn.data
        if s.whitewater_enabled:
            fast_launch(
                k_remap_leaf_field, dim=self._nvox,
                inputs=[self.ww_density.data, self.old_leaf_table, g.leaf_coords,
                        g.num_leaves, g.leaves_dim],
                outputs=[self.churn_tmp.data], device=self.device,
            )
            self.ww_density.data, self.churn_tmp.data = (
                self.churn_tmp.data, self.ww_density.data)
        if self.narrow_band is not None:
            self.narrow_band.remap(self.old_leaf_table)
        return g.active_leaf_count

    # Per-substep phases

    def _clear_grid_fields(self, clear_collision=True):
        # Clear only the active slot range: the topology rebuild that
        # precedes this call defines [0, nvox) as the only region any kernel
        # reads or writes this substep (all launches are bounded by nvox and
        # guarded by num_leaves on device).
        nv = self._nvox
        # Two fused kernels replace twelve memsets and a fill: one launch
        # per group instead of 13 slice + memset pairs.
        fast_launch(
            k_zero_fluid_grid, dim=nv,
            inputs=[*self.vel.arrays(), self.temperature.data,
                    self.viscosity_field.data, self.density.data,
                    self.voxel_type.data, self.voxel_counts.data],
            device=self.device,
        )
        if not clear_collision:
            return
        fast_launch(
            k_reset_collision_grid, dim=nv,
            inputs=[self.collision_vel, self.collision_friction,
                    self.collision_restitution, self.collision_sdf.data,
                    1.0e6],
            device=self.device,
        )
        s = self.settings
        if s.domain_sdf_walls and not s.open_boundaries:
            g = self.grid
            fast_launch(
                k_mark_domain_walls, dim=nv,
                inputs=[g.leaf_coords, g.num_leaves, g.origin, g.voxel_size,
                        wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                        self.collision_sdf.data],
                device=self.device,
            )

    def _update_moving_colliders(self, elapsed):
        """Re-pose moving colliders to their sub-frame transform and refit.

        Extrapolates each collider's frame-start pose with the same constant
        linear/angular velocity already used for collision_vel, so the SDF a
        fast-spinning collider presents to the fluid stops lagging the whole
        frame behind its true position between rebuilds.
        """
        for col in self.colliders:
            if col.base_points is None:
                continue
            av = col.angular_velocity
            mag = math.sqrt(sum(v * v for v in av))
            axis = tuple(v / mag for v in av) if mag > 1.0e-9 else (0.0, 0.0, 1.0)
            angle = mag * elapsed
            linear_disp = tuple(v * elapsed for v in col.linear_velocity)
            fast_launch(
                k_transform_collider_points, dim=col.base_points.shape[0],
                inputs=[col.base_points, wp.vec3(*col.center), wp.vec3(*axis),
                        angle, wp.vec3(*linear_disp)],
                outputs=[col.mesh.points], device=self.device,
            )
            col.mesh.refit()
            col.reposed = True

    def _restore_collider_poses(self):
        """Put re-posed collider meshes back at their authored frame pose.

        ``_update_moving_colliders`` writes extrapolated points into the
        scene-owned cached mesh. Left there, a collider that then stops keeps
        its extrapolated pose: the scene reuses the unchanged cached mesh and
        zero velocity never re-poses it.
        """
        for col in getattr(self, "colliders", ()):
            if col.base_points is None or not getattr(col, "reposed", False):
                continue
            wp.copy(col.mesh.points, col.base_points)
            col.mesh.refit()
            col.reposed = False

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
            fast_launch(
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
        fast_launch(
            k_particle_to_voxel_count_and_update_index,
            dim=np_dim,
            inputs=[self.particles.position, self.particles.count,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size],
            outputs=[self.voxel_counts.data, self.particles.voxel_index,
                     self.bins.arrival],
            device=self.device,
        )
        voxel_count_to_starts(self.voxel_counts.data[:nv], self.bins.starts[:nv])
        scatter_particles_to_bins(
            self.particles.voxel_index, self.bins.arrival,
            self.particles.count, np_dim, self.voxel_counts.data,
            self.bins.starts, self.bins.unordered, self.bins.indices,
        )

    def _source(self, dt):
        return self.sourcing.apply(dt)

    def _p2g(self):
        g = self.grid
        p = self.particles
        # Gather, not scatter: one thread per active voxel pulls its own
        # contributions out of the sorted bins built by _index_particles.
        # The scatter kernels stay registered as the faithful reference, but
        # their float wp.atomic_add sums in thread-arrival order and makes the
        # whole bake irreproducible; see k_gather_grid_data.
        nv = self._nvox
        # PIC/FLIP keep the original kernel untouched (bit-identical); APIC has
        # its own variant so no APIC arithmetic can reach the shared path.
        if self._apic_active():
            gather_kernel = k_gather_grid_data_apic
            head = [p.position, p.velocity, p.affine_for_kernels(),
                    int(self.settings.sample_type)]
        else:
            gather_kernel = k_gather_grid_data
            head = [p.position, p.velocity]
        fast_launch(
            gather_kernel, dim=nv,
            inputs=[*head,
                    g.leaf_table, g.leaf_coords, g.num_leaves,
                    g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size,
                    self.voxel_counts.data, self.bins.starts,
                    self.bins.indices],
            outputs=[self.vel.u, self.vel.v, self.vel.w,
                     self.vel.wu, self.vel.wv, self.vel.ww],
            device=self.device,
        )
        fast_launch(
            k_normalize_mac, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    g.num_leaves],
            device=self.device,
        )
        # Particle/phase accumulators -> voxel activity (fluid mask).
        fast_launch(
            k_particle_to_grid_active, dim=self._nvox,
            inputs=[self.voxel_counts.data, g.num_leaves],
            outputs=[self.voxel_type.data], device=self.device,
        )
        # Splat viscosity attribute when requested.
        st = self.settings
        if st.viscosity_enabled and st.viscosity_from_particles:
            self.attr_weight.data[:self._nvox].zero_()
            self.viscosity_field.data[:self._nvox].zero_()
            fast_launch(
                k_particle_to_grid_collect, dim=self._np_dim(),
                inputs=[p.position, p.attr("viscosity"), p.count,
                        g.leaf_table, g.voxels_dim, g.leaves_dim,
                        g.origin, g.voxel_size],
                outputs=[self.viscosity_field.data, self.attr_weight.data],
                device=self.device,
            )
            fast_launch(
                k_particle_to_grid_average, dim=self._nvox,
                inputs=[self.viscosity_field.data, self.attr_weight.data,
                        g.num_leaves],
                device=self.device,
            )

    def _save_velocities(self):
        fast_launch(
            k_copy3_float, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel_saved_u, self.vel_saved_v, self.vel_saved_w],
            device=self.device,
        )

    def _forces(self, dt):
        g = self.grid
        s = self.settings
        # Advanced field sources must follow P2G; applying them in the sourcing
        # phase would be overwritten by particle transfer.
        post_p2g = getattr(self.sourcing, "apply_post_p2g_fields", None)
        if post_p2g is not None:
            post_p2g(dt)
        fast_launch(
            k_forces_gravity, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    g.num_leaves, wp.vec3(*s.gravity), dt],
            device=self.device,
        )
        # Analytic objects with role=FORCE are source-shape ABI descriptors,
        # but their influence belongs here, after P2G established valid MAC
        # face weights. They must never source liquid or write unweighted MAC
        # values in SourcePipeline.apply().
        self.sourcing.apply_forces(dt)
        if any(component != 0.0 for component in s.directional_force):
            fast_launch(
                k_forces_directional, dim=self._nvox,
                inputs=[self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        g.num_leaves, wp.vec3(*s.directional_force), dt],
                device=self.device,
            )
        if (s.omnidirectional_force_strength != 0.0
                and s.omnidirectional_force_radius > 0.0):
            fast_launch(
                k_forces_omnidirectional, dim=self._nvox,
                inputs=[g.leaf_coords, g.num_leaves,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        g.origin, g.voxel_size,
                        wp.vec3(*s.omnidirectional_force_center),
                        s.omnidirectional_force_strength,
                        s.omnidirectional_force_radius, dt],
                device=self.device,
            )
        if self.force_fields:
            fast_launch(
                k_forces_proxy_fields, dim=self._nvox,
                inputs=[self._force_field_data(), len(self.force_fields),
                        g.leaf_coords, g.num_leaves, g.origin, g.voxel_size,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        self.sim_time, dt],
                device=self.device,
            )
        diffusion_amount = max(0.0, min(float(s.force_diffusion), 1.0))
        if diffusion_amount > 0.0:
            fast_launch(
                k_forces_diffusion, dim=self._nvox,
                inputs=[self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        g.leaf_table, g.leaf_coords, g.voxels_dim,
                        g.leaves_dim, g.num_leaves, diffusion_amount],
                outputs=[self.tmp_u, self.tmp_v, self.tmp_w],
                device=self.device,
            )
            self.vel.u, self.tmp_u = self.tmp_u, self.vel.u
            self.vel.v, self.tmp_v = self.tmp_v, self.vel.v
            self.vel.w, self.tmp_w = self.tmp_w, self.vel.w
        if s.dissipation_enabled and s.dissipation_rate > 0.0:
            fast_launch(
                k_dissipate_particle_attribute, dim=self._np_dim(),
                inputs=[self.particles.attr(s.dissipation_attribute),
                        self.particles.count, s.dissipation_rate,
                        s.dissipation_target, dt],
                device=self.device,
            )
        if (s.attraction_enabled and s.attraction_strength != 0.0
                and s.attraction_max_distance > 0.0):
            fast_launch(
                k_attract_sdf_gradient_direction, dim=self._nvox,
                inputs=[self.collision_sdf.data, g.leaf_table, g.leaf_coords,
                        g.voxels_dim, g.leaves_dim, g.num_leaves,
                        g.voxel_size, s.attraction_strength,
                        s.attraction_max_distance, dt,
                        self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww],
                device=self.device,
            )
        # Surface tension is a physical grid force.  It uses the current
        # particle SDF after P2G and before implicit viscosity/pressure.
        self.surface_tension.apply(dt)
        if s.max_speed > 0.0:
            fast_launch(
                k_forces_speed_limit, dim=self._nvox,
                inputs=[self.vel.u, self.vel.v, self.vel.w,
                        self.vel.wu, self.vel.wv, self.vel.ww,
                        g.leaf_table, g.leaf_coords, g.voxels_dim,
                        g.leaves_dim, g.num_leaves,
                        s.max_speed, int(s.speed_limit_behavior),
                        s.speed_limit_percent, dt],
                outputs=[self.tmp_u, self.tmp_v, self.tmp_w],
                device=self.device,
            )
            self.vel.u, self.tmp_u = self.tmp_u, self.vel.u
            self.vel.v, self.tmp_v = self.tmp_v, self.vel.v
            self.vel.w, self.tmp_w = self.tmp_w, self.vel.w

    def _viscosity(self, dt):
        s = self.settings
        if not s.viscosity_enabled:
            return
        # The grid clear zeroes voxel_type every substep and the collision bits
        # are otherwise re-derived only in _collision_faces, after this solve,
        # so viscosity's VOXEL_COLLISION guards would never fire for mesh
        # colliders or domain walls. collision_sdf is final by now; the later
        # pass re-derives the same bits.
        self._classify_collision_voxels()
        if not s.viscosity_from_particles and s.viscosity_constant > 0.0:
            fast_launch(
                k_fill_float, dim=self._nvox,
                inputs=[self.viscosity_field.data, s.viscosity_constant],
                device=self.device,
            )
        if int(s.viscosity_solver) == int(ViscositySolverEnum.JACOBI):
            status = self.viscosity_solver.smooth(
                self.vel, self.viscosity_field.data, self.voxel_type.data, dt,
                iterations=max(1, s.viscosity_iterations // 10),
            )
        else:
            status = self.viscosity_solver.solve(
                self.vel, self.viscosity_field.data, self.voxel_type.data, dt,
                use_pcg=True, iterations=s.viscosity_iterations,
                tolerance=s.viscosity_tolerance,
            )
        self.stats.viscosity_status = status["reason"]
        self.stats.viscosity_converged = bool(status["converged"])
        self.stats.viscosity_iterations = int(status["iterations"])
        self.stats.viscosity_residual = float(status["residual"])
        self.stats.native_solver_message = self._native_solver_message()

    def _classify_collision_voxels(self):
        g = self.grid
        fast_launch(
            k_sdf_to_axis_collision, dim=self._nvox,
            inputs=[self.collision_sdf.data, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, 0.0],
            outputs=[self.voxel_type.data], device=self.device,
        )

    def _collision_faces(self):
        g = self.grid
        self._classify_collision_voxels()
        fast_launch(
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
        # Live edits mutate self.settings in place; re-read before every solve.
        self.pressure_solver.preconditioner = s.pressure_preconditioner
        status = self.pressure_solver.project_frame(
            self.vel,
            self.voxel_type.data,
            self.collision_vel,
            self.voxel_counts.data,
            s,
            max_gradient=self._cfl_speed_limit(dt),
            pressure_source=lambda: self.sourcing.apply_pressure_sources(dt),
        )
        self.stats.pressure_iterations = status["iterations"]
        self.stats.pressure_residual = status["residual"]
        self.stats.native_solver_message = self._native_solver_message()
        return bool(status["project_velocity"])

    def _native_solver_message(self):
        """Say which solves lost their CUDA-graph path, and why."""
        parts = []
        for label, owner in (
                ("Pressure", self.pressure_solver),
                ("Viscosity", getattr(self, "viscosity_solver", None))):
            error = str(getattr(owner, "last_pcg_native_error", "") or "")
            if error:
                parts.append(f"{label} solve fell back from CUDA graphs "
                             f"to a slower host loop: {error}")
        return "; ".join(parts)

    def _collider_material_response(self, dt):
        g = self.grid
        fast_launch(
            k_apply_collider_material_to_grid, dim=self._nvox,
            inputs=[self.voxel_type.data, self.collision_sdf.data,
                    self.collision_vel, self.collision_friction,
                    self.collision_restitution, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, g.voxel_size,
                    float(dt) / C.COLLIDER_FRICTION_REFERENCE_TIME,
                    self.collision_impact_u, self.collision_impact_v,
                    self.collision_impact_w,
                    self.vel.u, self.vel.v, self.vel.w],
            device=self.device,
        )

    def _sanitize(self, dt):
        """Clamp MAC velocity to a CFL-safe bound and kill inf/nan."""
        g = self.grid
        max_speed = self._cfl_speed_limit(dt)
        fast_launch(
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
        # Time Scale = 0 is an authored pause mode. Keep its no-op substep
        # finite rather than dividing by zero during post-advection sanitation.
        safe_dt = max(float(dt), 1.0e-8)
        max_speed = max(s.cfl * g.voxel_size / safe_dt * 2.0, 50.0)
        p = self.particles
        if self._apic_active() and p.affine_allocated:
            fast_launch(
                k_apic_clamp_affine, dim=self._np_dim(),
                inputs=[p.velocity, p.affine, p.alive, p.count, max_speed],
                device=self.device,
            )
        fast_launch(
            k_sanitize_particles, dim=self._np_dim(),
            inputs=[p.position, p.velocity, p.alive, p.count, max_speed,
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max)],
            device=self.device,
        )

    def _g2p_transfer(self):
        """Grid -> particle velocity update (PIC / FLIP / APIC); no advection."""
        g = self.grid
        s = self.settings
        p = self.particles
        apic = self._apic_active()
        fast_launch(
            k_staggered_field_to_particle_apic if apic
            else k_staggered_field_to_particle,
            dim=self._np_dim(),
            inputs=[p.position, p.count,
                    self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    self.vel_saved_u, self.vel_saved_v, self.vel_saved_w,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size,
                    int(s.sample_type), s.flip_ratio],
            outputs=([p.velocity, p.affine_for_kernels()] if apic
                     else [p.velocity]),
            device=self.device,
        )

    def _g2p_advect(self, dt):
        g = self.grid
        s = self.settings
        p = self.particles
        self._g2p_transfer()

        fast_launch(
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

    def _new_color_mixing(self):
        if not bool(getattr(self.settings, "species_enabled", False)):
            return None
        return ColorMixTransport(self.grid, self.device)

    def _mix_particle_colors(self, dt):
        transport = self.color_mixing
        if transport is None:
            return
        transport.mix(
            self.particles,
            self.grid,
            dt,
            float(getattr(self.settings, "species_mixing_rate", 8.0)),
            self._nvox,
        )

    def _lifecycle(self, dt):
        g = self.grid
        p = self.particles
        np_dim = self._np_dim()
        fast_launch(k_particle_age, dim=np_dim,
                  inputs=[p.attr("age"), p.count, dt], device=self.device)
        fast_launch(k_particle_lifespan, dim=np_dim,
                  inputs=[p.attr("age"), p.attr("life"), p.count, 1],
                  outputs=[p.alive], device=self.device)
        fast_launch(
            k_particle_in_sink_check, dim=np_dim,
            inputs=[p.position, p.count, self.voxel_type.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, g.voxel_size],
            outputs=[p.alive], device=self.device,
        )
        nb = self.narrow_band
        if nb is not None:
            nb.update_phi(dt)
            nb.kill_interior()
        p.compact(count_hint=np_dim)
        if nb is not None:
            self._index_particles()
            nb.reseed(nb._substep_index)

    def _relax(self):
        s = self.settings
        if not s.relax_enabled:
            return
        g = self.grid
        p = self.particles
        self._index_particles()
        separation = 0.5 * self.quality.nominal_particle_spacing
        fast_launch(
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

    def _whitewater(self, dt, sdf_ready=True):
        s = self.settings
        if not s.whitewater_enabled:
            return
        if not sdf_ready:
            self.whitewater.mark_sdf_unavailable(
                getattr(self.surface, "solver_sdf_error", ""))
            return
        g = self.grid
        p = self.particles
        ww = self.whitewater
        fast_launch(
            k_compute_churn, dim=self._nvox,
            inputs=[self.vel.u, self.vel.v, self.vel.w, self.voxel_type.data,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, s.voxel_size, dt, s.churn_decay, s.churn_gain],
            outputs=[self.churn.data], device=self.device,
        )
        self._whitewater_emit_seed = (
            self._whitewater_emit_seed * 1664525 + 1013904223
        ) & 0x7FFFFFFF
        life_min, life_max = normalized_life_range(
            s.ww_life_min, s.ww_life_max)
        type_policy = classification_policy(s)
        ww.emit(
            candidate_count=self._np_dim(),
            liquid_positions=p.position,
            liquid_velocities=p.velocity,
            liquid_count=p.count,
            churn=self.churn.data,
            surface_sdf=self.surface_sdf.data,
            leaf_table=g.leaf_table,
            voxels_dim=g.voxels_dim,
            leaves_dim=g.leaves_dim,
            origin=g.origin,
            voxel_size=s.voxel_size,
            dt=dt,
            churn_threshold=s.ww_churn_threshold,
            emission_rate=s.ww_emission_rate,
            surface_band=s.ww_surface_band * s.voxel_size,
            min_speed=s.ww_min_speed,
            foam_band=type_policy.foam_band,
            min_spray_speed=type_policy.min_spray_speed,
            life_min=life_min,
            life_max=life_max,
            foam_life_multiplier=s.ww_foam_life_multiplier,
            bubble_life_multiplier=s.ww_bubble_life_multiplier,
            spray_life_multiplier=s.ww_spray_life_multiplier,
            seed=self._whitewater_emit_seed,
        )
        # Bound advection by the prior live prefix plus this substep's maximum
        # possible admissions. Kernels still guard on the device count, so no
        # scalar readback or stream barrier is introduced.
        self._whitewater_launch_bound = min(
            ww.capacity,
            getattr(self, "_whitewater_launch_bound", ww.host_count)
            + self._np_dim())
        nww = max(1, self._whitewater_launch_bound)
        fast_launch(
            k_advect_whitewater, dim=nww,
            inputs=[ww.position, ww.velocity, ww.age, ww.life,
                    ww.alive, ww.type, ww.count,
                    self.vel.u, self.vel.v, self.vel.w,
                    self.vel.wu, self.vel.wv, self.vel.ww,
                    self.surface_sdf.data, self.collision_sdf.data,
                    self.ww_density.data,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, s.voxel_size, dt, wp.vec3(*s.gravity),
                    type_policy.spray_enter_height,
                    type_policy.spray_exit_height,
                    type_policy.bubble_enter_depth,
                    type_policy.bubble_exit_depth,
                    type_policy.min_spray_speed,
                    type_policy.spray_exit_speed,
                    s.ww_foam_advection,
                    s.ww_foam_preservation,
                    s.ww_foam_density_influence,
                    s.ww_bubble_drag,
                    s.ww_bubble_buoyancy,
                    s.ww_spray_drag,
                    s.ww_spray_gravity_scale,
                    s.ww_foam_life_multiplier,
                    s.ww_bubble_life_multiplier,
                    s.ww_spray_life_multiplier,
                    wp.vec3(*s.domain_min), wp.vec3(*s.domain_max),
                    ww.capacity],
            outputs=[ww.dead],
            device=self.device,
        )
        ww.recycle_dead_slots(nww)

    def _finalize_whitewater(self):
        """Compact, publish, and return the live whitewater count per frame."""
        ww = self.whitewater
        active_whitewater = ww.refresh_host_count()
        if not self.settings.whitewater_enabled:
            ww.update_type_counts(active_whitewater)
            return active_whitewater

        ww.compact(active_whitewater)
        active_whitewater = ww.refresh_host_count()
        self._splat_whitewater_density(active_whitewater)
        return active_whitewater

    def _splat_whitewater_density(self, active_whitewater):
        """Rebuild the end-of-frame whitewater density from whitewater positions."""
        g = self.grid
        ww = self.whitewater
        self.ww_density.data[:self._nvox].zero_()
        fast_launch(
            k_whitewater_set_grid_data, dim=max(1, active_whitewater),
            inputs=[ww.position, ww.alive, ww.count,
                    g.leaf_table, g.voxels_dim, g.leaves_dim,
                    g.origin, self.settings.voxel_size],
            outputs=[self.ww_density.data], device=self.device,
        )

    # CFL

    def _collider_max_speed(self):
        """Max collider surface speed this frame: |linear| + |angular| * tip radius.

        Fluid-only CFL is blind to a fast collider meeting still fluid: the
        substep count is sized before any collision response has happened,
        so a rotating paddle can leave the pool's frame-start velocity near
        zero while its blade tip moves several m/s. Folding collider tip
        speed into the CFL vmax keeps the substep dt fine enough that the
        segment-based collision resolve (kernels_fluid.py resolve_collision)
        can actually contain the impact, instead of a single oversized
        substep overshooting the domain.
        """
        vmax = 0.0
        for col in self.colliders:
            lin = math.sqrt(sum(v * v for v in col.linear_velocity))
            av = col.angular_velocity
            amag = math.sqrt(sum(v * v for v in av))
            tip_speed = lin
            if amag > 0.0:
                cx, cy, cz = col.center
                radius = 0.0
                for x in (col.aabb_min[0], col.aabb_max[0]):
                    for y in (col.aabb_min[1], col.aabb_max[1]):
                        for z in (col.aabb_min[2], col.aabb_max[2]):
                            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                            radius = max(radius, d)
                tip_speed += amag * radius
            vmax = max(vmax, tip_speed)
        return vmax

    def _compute_substeps(self, frame_dt):
        """Measure live speed and delegate pure schedule selection."""
        policy = SubstepPolicy.from_settings(self.settings)
        has_particles = self.particles.host_count > 0
        fluid_max_speed = 0.0
        collider_max_speed = 0.0
        # A fixed-substep scene cannot adapt to CFL, so avoid the synchronous
        # device reduction/readback entirely. Pure capillary demand is still
        # calculated by select_substeps so clamp diagnostics remain truthful.
        if policy.adaptive and has_particles:
            self.max_speed_dev.zero_()
            fast_launch(
                k_max_vec3_magnitude, dim=self.particles.host_count,
                inputs=[self.particles.velocity, self.particles.count],
                outputs=[self.max_speed_dev], device=self.device,
            )
            fluid_max_speed = float(self.max_speed_dev.numpy()[0])
            collider_max_speed = self._collider_max_speed()
        schedule = select_substeps(
            policy,
            frame_dt,
            has_particles=has_particles,
            fluid_max_speed=fluid_max_speed,
            collider_max_speed=collider_max_speed,
        )
        self._last_substep_diagnostics = schedule.diagnostics()
        return schedule.substeps, schedule.cfl_clamped

    def _surface_tension_topology_expand(self):
        """Return the leaf halo required by the physical capillary SDF."""

        s = self.settings
        if (not bool(getattr(s, "surface_tension_enabled", False))
                or float(getattr(s, "surface_tension_coefficient", 0.0)) <= 0.0):
            return 0
        sdf_filter_iterations = sum(max(0, int(value)) for value in (
            s.surface_filter_iterations,
            s.surface_dilate_iterations,
            s.surface_erode_iterations,
            s.surface_smooth_iterations,
        ))
        splat_radius = max(
            int(s.surface_splat_radius),
            required_sdf_splat_radius_voxels(
                0.5 * float(s.particle_radius_scale),
                float(s.surface_tension_band_voxels),
                int(s.surface_tension_smoothing_iterations),
                sdf_filter_iterations,
            ),
        )
        return required_topology_expand_leaves(splat_radius, C.LEAF_DIM)

    def _whitewater_topology_expand(self, frame_dt):
        """Return a leaf radius covering existing whitewater for one frame."""
        s = self.settings
        ww = self.whitewater
        if not s.whitewater_enabled or ww.host_count <= 0:
            return 0
        self.max_speed_dev.zero_()
        fast_launch(
            k_max_vec3_magnitude, dim=ww.host_count,
            inputs=[ww.velocity, ww.count],
            outputs=[self.max_speed_dev], device=self.device,
        )
        vmax = max(0.0, float(self.max_speed_dev.numpy()[0]))
        gmag = math.sqrt(sum(x * x for x in s.gravity))
        return _motion_leaf_radius(vmax, frame_dt, gmag, s.voxel_size)


    def step_frame(self):
        """Advance one display frame (interactive playback entry point)."""
        s = self.settings
        t_start = _time.perf_counter()
        self._sync_affine_mode()
        frame_dt = s.time_scale / s.frame_rate
        if self.profiler is not None:
            self.profiler.begin_frame()
        self.features.begin_frame()
        self.surface_tension.begin_frame()

        substeps, cfl_clamped = self._compute_substeps(frame_dt)
        dt = frame_dt / substeps

        # The Solver Session sets effect_frame to the timeline frame this step
        # produces (decision D6); a solver stepped directly uses its counter.
        effect_frame = getattr(self, "effect_frame", None)
        self.effect_frame = None
        time_ctx = {
            "frame": self.frame if effect_frame is None else int(effect_frame),
            "time": self.sim_time,
            "real_time_step": frame_dt, "solver_time_step": dt,
            "time_scale": s.time_scale, "substeps": substeps,
        }

        rebuild_each_substep = self.features.rebuild_topology_each_substep
        moving_colliders = any(
            col.base_points is not None for col in self.colliders)
        frame_topology_expand = max(
            int(s.topology_expand),
            int(math.ceil(substeps * max(float(s.cfl), 0.0) / C.LEAF_DIM)),
            self._whitewater_topology_expand(frame_dt),
            self._surface_tension_topology_expand(),
        )
        self.whitewater.reset_frame_diagnostics()
        self._whitewater_launch_bound = self.whitewater.host_count
        for substep_index in range(substeps):
            time_ctx["substep_index"] = substep_index
            if self.narrow_band is not None:
                self.narrow_band._substep_index = substep_index
            self.features.run(
                FrameFeaturePhase.BEFORE_EMISSION, self, time_ctx)
            rebuild = substep_index == 0 or rebuild_each_substep
            collision_rebuild = rebuild or moving_colliders
            with self._phase("misc"):
                if rebuild:
                    self._rebuild_topology(
                        frame_topology_expand if not rebuild_each_substep else None)
                self._clear_grid_fields(clear_collision=collision_rebuild)
            with self._phase("collision"):
                if collision_rebuild:
                    if moving_colliders and substep_index > 0:
                        self._update_moving_colliders(substep_index * dt)
                    self._apply_collisions()
            self.features.run(
                FrameFeaturePhase.BEFORE_SOURCING, self, time_ctx)
            with self._phase("indexing"):
                self._index_particles()
            with self._phase("sourcing"):
                if self.narrow_band is not None:
                    self.narrow_band.credit_interior_counts()
                bins_before = self.bins
                emitted = self._source(dt)
                if self.narrow_band is not None:
                    self.narrow_band.release_interior_counts()
                # A successful pool growth swaps in fresh, empty bins and
                # defers the emission, so the index must be rebuilt even
                # though nothing was emitted.
                if not emitted and self.bins is not bins_before:
                    self._index_particles()
                # Re-count after emission so P2G activity sees new particles.
                # Skipped when no emitter ran: indexing is already current.
                if emitted:
                    self._index_particles()
                    # The previous frame may have produced a current-key SDF
                    # during render surfacing. Newly emitted particles change
                    # the physical interface without advancing frame/time, so
                    # invalidate explicitly before the capillary force reads it.
                    self.surface.invalidate_solver_sdf()
            with self._phase("p2g"):
                self._p2g()
            if self.narrow_band is not None:
                self.narrow_band.prepare_grid(dt)
            self._save_velocities()
            self.features.run(
                FrameFeaturePhase.AFTER_PARTICLE_TRANSFER, self, time_ctx)
            with self._phase("forces"):
                self._forces(dt)
            with self._phase("viscosity"):
                self._viscosity(dt)
            fast_launch(
                k_copy3_float, dim=self._nvox,
                inputs=[self.vel.u, self.vel.v, self.vel.w,
                        self.collision_impact_u, self.collision_impact_v,
                        self.collision_impact_w],
                device=self.device,
            )
            with self._phase("collision"):
                self._collision_faces()
            self.features.run(
                FrameFeaturePhase.BEFORE_PRESSURE, self, time_ctx)
            with self._phase("pressure"):
                self._pressure(dt)
                self._collider_material_response(dt)
            self._sanitize(dt)
            with self._phase("g2p"):
                self._g2p_advect(dt)
            self._sanitize_particles(dt)
            with self._phase("lifecycle"):
                self._lifecycle(dt)
            with self._phase("relax"):
                self._relax()
            # Surface tension may have built the SDF before G2P/advection.  The
            # particle positions now describe a new interface; invalidate that
            # pre-advection key so whitewater and substep render surfacing can
            # never consume stale geometry.
            self.surface.invalidate_solver_sdf()
            sdf_ready = self.surface.after_substep()
            with self._phase("whitewater"):
                self._whitewater(dt, sdf_ready=sdf_ready)
            self.features.run(
                FrameFeaturePhase.AFTER_PARTICLE_UPDATE, self, time_ctx)
            self.sim_time += dt
            time_ctx["time"] = self.sim_time

        # Color is an artistic display attribute, so exchange it once per
        # display frame.  Exponential rate scaling keeps the result tied to
        # elapsed simulation time without multiplying work by CFL substeps.
        with self._phase("g2p"):
            self._mix_particle_colors(frame_dt)
        self._restore_collider_poses()

        self.frame += 1
        self._host_frame_cache = None
        with self._phase("whitewater"):
            whitewater_particles = self._finalize_whitewater()
        self.surface.after_frame()

        wp.synchronize_device(self.device)
        self.grid.refresh_topology_counts()
        whitewater_diagnostics = self.whitewater.refresh_diagnostics()
        whitewater_types = self.whitewater.refresh_type_counts()

        st = self.stats
        st.step_time = _time.perf_counter() - t_start
        st.substeps = substeps
        st.cfl_substeps_clamped = cfl_clamped
        limiter = getattr(self, "_last_substep_diagnostics", {})
        st.capillary_dt = float(limiter.get("capillary_dt", float("inf")))
        st.capillary_substeps_required = int(
            limiter.get("capillary_substeps_required", 1))
        st.capillary_substeps_clamped = bool(
            limiter.get("capillary_substeps_clamped", False))
        st.substep_limiter = str(limiter.get("substep_limiter", "MINIMUM"))
        tension = self.surface_tension.diagnostics()
        st.surface_tension_active_voxels = tension.active_voxels
        st.surface_tension_max_curvature = tension.max_abs_curvature
        st.surface_tension_max_acceleration = tension.max_acceleration
        st.surface_tension_temporary_bytes = tension.temporary_bytes
        st.surface_tension_submission_seconds = tension.submission_seconds
        st.surface_tension_status = tension.status
        st.surface_tension_message = tension.message
        st.active_voxels = self.grid.active_voxel_count
        st.initialized_voxels = self.grid.voxel_capacity
        st.total_voxels = self.grid.total_voxels()
        st.active_particles = self.particles.refresh_host_count()
        if self.narrow_band is not None:
            st.liquid_voxels = self.narrow_band.end_frame()
            st.narrow_band_reseed_overflow = self.narrow_band.reseed_overflow
        st.total_particles = self.particles.capacity
        st.whitewater_particles = whitewater_particles
        st.whitewater_foam = whitewater_types["foam"]
        st.whitewater_bubbles = whitewater_types["bubble"]
        st.whitewater_spray = whitewater_types["spray"]
        st.whitewater_invalid_types = whitewater_types["invalid"]
        st.whitewater_required = whitewater_diagnostics["required"]
        st.whitewater_available = whitewater_diagnostics["available"]
        st.whitewater_admitted = whitewater_diagnostics["admitted"]
        st.whitewater_dropped = whitewater_diagnostics["dropped"]
        st.whitewater_policy = whitewater_diagnostics["policy"]
        st.whitewater_status = whitewater_diagnostics["status"]
        st.whitewater_message = whitewater_diagnostics["message"]
        surface_metadata = self.surface.metadata()
        st.surface_voxel_size = float(surface_metadata["voxel_size"])
        st.surface_active_leaves = surface_metadata["active_leaves"]
        st.surface_active_voxels = surface_metadata["active_voxels"]
        st.custom_effects = self.features.frame_stats_snapshot()
        if self.profiler is not None:
            st.timers = self.profiler.collect()
        return st

    # Direct particle injection

    def inject_particle_packets_exact(self, packets):
        """Internal exact all-or-zero packet transaction."""
        return self.sourcing.inject_particle_packets_exact(packets)

    def inject_particles_exact(self, positions, velocity=(0.0, 0.0, 0.0),
                               velocities=None, temperature=0.0, viscosity=0.0,
                               density=1.0, color=(0.1, 0.45, 1.0, 1.0),
                               source_id=0, life=0.0, use_lifespan=False):
        """Compatibility alias for :meth:`inject_particles`."""
        return self.inject_particles(
            positions, velocity=velocity, velocities=velocities,
            temperature=temperature, viscosity=viscosity, density=density,
            color=color, source_id=source_id,
            life=life, use_lifespan=use_lifespan)

    def inject_particles(self, positions, velocity=(0.0, 0.0, 0.0),
                         velocities=None, temperature=0.0, viscosity=0.0,
                         density=1.0, color=(0.1, 0.45, 1.0, 1.0),
                         source_id=0, life=0.0, use_lifespan=False):
        """Append external world-space points and return exact admission truth.

        ``positions``: (N,3) array-like. Supply either constant ``velocity`` or
        per-particle ``velocities``; supplying both is rejected. Uses
        fluidSimSourceParticles.
        """
        return self.sourcing.inject_particles(
            positions,
            velocity=velocity,
            velocities=velocities,
            temperature=temperature,
            viscosity=viscosity,
            density=density,
            color=color,
            source_id=source_id,
            life=life,
            use_lifespan=use_lifespan,
        )

    def inject_particles_legacy_total(self, positions, velocity=(0.0, 0.0, 0.0),
                                      velocities=None, temperature=0.0, viscosity=0.0,
                                      density=1.0, color=(0.1, 0.45, 1.0, 1.0),
                                      source_id=0, life=0.0, use_lifespan=False):
        """Return the legacy lossy total count; prefer :meth:`inject_particles`."""
        return self.sourcing.inject_particles_legacy_total(
            positions,
            velocity=velocity,
            velocities=velocities,
            temperature=temperature,
            viscosity=viscosity,
            density=density,
            color=color,
            source_id=source_id,
            life=life,
            use_lifespan=use_lifespan,
        )


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
