"""The solver's host-side settings record.

Warp-free on purpose: the cache settings signature builds one of these during
playback, which must work where ``import warp`` fails. ``sim.solver``
re-exports ``SolverSettings`` so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

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
)


@dataclass
class SolverSettings:
    domain_min: tuple = (-2.0, -2.0, 0.0)
    domain_max: tuple = (2.0, 2.0, 4.0)
    voxel_size: float = C.DEFAULT_VOXEL_SIZE
    leaf_capacity: int = C.DEFAULT_LEAF_CAPACITY
    max_particles: int = C.DEFAULT_MAX_PARTICLES
    auto_grow_particles: bool = True
    particle_growth_failure_policy: str = "DEFER"
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
    speed_limit_percent: float = 0.5
    directional_force: tuple = (0.0, 0.0, 0.0)
    omnidirectional_force_center: tuple = (0.0, 0.0, 0.0)
    omnidirectional_force_strength: float = 0.0
    omnidirectional_force_radius: float = 1.0
    force_diffusion: float = 0.0
    dissipation_enabled: bool = False
    dissipation_attribute: str = "temperature"
    dissipation_rate: float = 0.0
    dissipation_target: float = 0.0
    attraction_enabled: bool = False
    attraction_strength: float = 0.0
    attraction_max_distance: float = 1.0

    particles_per_voxel: float = float(C.DEFAULT_PARTICLES_PER_VOXEL)
    count_stiffness: float = 0.0
    resolved_quality: object = dc_field(default=None, repr=False, compare=False)

    relax_enabled: bool = True
    relax_strength: float = 0.5
    relax_mode: int = int(RelaxMode.LOCAL_VOXEL)
    open_boundaries: bool = False
    # Closed domains (open_boundaries=False) need actual collision-SDF walls
    # or the pressure solve treats sparse-grid padding beyond the domain box
    # as open space (see k_mark_domain_walls docstring). Default True so a
    # fresh closed-domain scene is not silently unbounded.
    domain_sdf_walls: bool = True
    pressure_solver: int = int(ProjectNonDivergentSolver.PCG)
    pressure_iterations: int = C.DEFAULT_PRESSURE_ITERATIONS
    pressure_tolerance: float = C.DEFAULT_PRESSURE_TOLERANCE
    pressure_threshold: float = 0.0
    # "JACOBI" or "MULTIGRID" (sim.pressure.PRECONDITIONER_*); PCG only.
    pressure_preconditioner: str = "JACOBI"

    viscosity_enabled: bool = False
    viscosity_solver: int = int(ViscositySolverEnum.PCG)
    viscosity_constant: float = 0.05
    viscosity_iterations: int = C.DEFAULT_VISCOSITY_ITERATIONS
    viscosity_tolerance: float = C.DEFAULT_VISCOSITY_TOLERANCE
    viscosity_from_particles: bool = False

    # Continuum-surface-force surface tension.  Coefficient is N/m, density
    # is kg/m^3, and the narrow-band width is authored in simulation voxels.
    surface_tension_enabled: bool = False
    surface_tension_coefficient: float = 0.072
    surface_tension_density: float = 1000.0
    surface_tension_band_voxels: float = 1.5
    surface_tension_safety: float = 0.5
    surface_tension_smoothing_iterations: int = 1

    # GPU-resident artistic mixing of the existing per-particle color channel.
    species_enabled: bool = False
    species_mixing_rate: float = 8.0

    surface_enabled: bool = True
    particle_radius_scale: float = 1.2
    surface_splat_radius: int = 2
    # Render surfacing is intentionally decoupled from the simulation grid.
    # The optional simulation SDF is retained only when requested or required
    # by whitewater, while the render SDF can use a separate resolution.
    surface_resolution_mode: int = int(SurfaceResolutionMode.SIMULATION)
    surface_resolution_multiplier: float = 2.0
    surface_voxel_size: float = 0.0
    surface_leaf_capacity_scale: float = 8.0
    surface_max_dense_dim: int = 512
    surface_update_cadence: int = int(SurfaceUpdateCadence.FRAME)
    live_surface_publish_cadence: int = 1
    surface_build_lowres_sdf: bool = False
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

    # Artist-ready output inventory.  These fields are signature-bound and
    # selected before initialization so optional particle channels stay opt-in.
    export_attributes: bool = False
    surface_attribute_velocity: bool = True
    surface_attribute_speed: bool = False
    surface_attribute_vorticity: bool = False
    surface_attribute_age: bool = False
    surface_attribute_lifetime: bool = False
    surface_attribute_color: bool = False
    surface_attribute_source_id: bool = False
    surface_attribute_viscosity: bool = False
    surface_attribute_density: bool = False
    surface_attribute_motion: bool = True
    particles_attribute_velocity: bool = False
    particles_attribute_speed: bool = False
    particles_attribute_vorticity: bool = False
    particles_attribute_age: bool = True
    particles_attribute_lifetime: bool = True
    particles_attribute_color: bool = False
    particles_attribute_source_id: bool = False
    particles_attribute_uid: bool = False
    particles_attribute_viscosity: bool = False
    particles_attribute_density: bool = False
    whitewater_attribute_velocity: bool = False
    whitewater_attribute_speed: bool = False
    whitewater_attribute_age: bool = True
    whitewater_attribute_lifetime: bool = True
    whitewater_attribute_type: bool = True

    whitewater_enabled: bool = False
    churn_decay: float = 2.0
    churn_gain: float = 1.0
    ww_churn_threshold: float = 0.5
    ww_emission_rate: float = 2.0
    ww_surface_band: float = 2.0     # in voxels
    ww_min_speed: float = 0.5
    ww_life_min: float = 1.0
    ww_life_max: float = 3.0
    ww_foam_band: float = 1.0        # in voxels
    ww_spray_min_speed: float = 0.5
    ww_type_hysteresis: float = 0.25
    ww_foam_advection: float = 1.0
    ww_foam_preservation: float = 0.0
    ww_foam_density_influence: float = 0.0
    ww_foam_life_multiplier: float = 1.0
    ww_bubble_drag: float = 4.0
    ww_bubble_buoyancy: float = 1.0
    ww_bubble_life_multiplier: float = 1.0
    ww_spray_drag: float = 0.2
    ww_spray_gravity_scale: float = 1.0
    ww_spray_life_multiplier: float = 1.0

    topology_expand: int = 1
    device: str = "cuda:0"
