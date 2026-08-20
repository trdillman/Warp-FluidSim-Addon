"""Projection of Blender-authored values into owned solver settings.

This module owns value conversion only. Lifecycle classification is owned by
``pending_settings``.
"""

from __future__ import annotations


DIRECT_SETTING_FIELDS = (
    "voxel_size", "leaf_capacity", "topology_expand", "open_boundaries",
    "domain_sdf_walls",
    "max_particles", "auto_grow_particles",
    "particle_growth_failure_policy", "max_whitewater", "time_scale", "cfl",
    "min_substeps", "max_substeps", "flip_ratio", "max_speed",
    "limit_percent", "directional_force", "omnidirectional_force_center",
    "omnidirectional_force_strength", "omnidirectional_force_radius",
    "force_diffusion", "dissipation_enabled", "dissipation_attribute",
    "dissipation_rate", "dissipation_target", "attraction_enabled",
    "attraction_strength", "attraction_max_distance",
    "particles_per_voxel", "count_stiffness",
    "spacetime_enabled", "spacetime_jitter_strength",
    "spacetime_adaptive_jitter", "spacetime_phase_eta",
    "spacetime_phase_threshold", "spacetime_reference_mass",
    "spacetime_variable_projection", "spacetime_liquid_density",
    "spacetime_gas_density", "spacetime_min_density",
    "spacetime_local_cfl", "spacetime_max_local_substeps",
    "spacetime_resync_surface", "spacetime_seed",
    "relax_enabled", "relax_strength",
    "pressure_iterations", "pressure_tolerance", "pressure_threshold",
    "viscosity_enabled", "viscosity_constant", "viscosity_from_particles",
    "viscosity_iterations", "viscosity_tolerance",
    "surface_tension_enabled", "surface_tension_coefficient",
    "surface_tension_density", "surface_tension_band_voxels",
    "surface_tension_safety", "surface_tension_smoothing_iterations",
    "surface_enabled",
    "particle_radius_scale", "surface_splat_radius",
    "surface_resolution_multiplier", "surface_voxel_size",
    "surface_leaf_capacity_scale", "surface_max_dense_dim",
    "live_surface_publish_cadence",
    "surface_build_lowres_sdf", "surface_filter_iterations",
    "surface_filter_strength", "surface_dilate_iterations",
    "surface_dilate_strength", "surface_erode_iterations",
    "surface_erode_strength", "surface_smooth_iterations",
    "surface_smooth_strength", "whitewater_enabled", "ww_churn_decay",
    "ww_churn_gain", "ww_churn_threshold", "ww_emission_rate",
    "ww_surface_band", "ww_min_speed", "ww_life_min", "ww_life_max",
    "ww_foam_band", "ww_spray_min_speed", "ww_type_hysteresis",
    "ww_foam_advection", "ww_foam_preservation",
    "ww_foam_density_influence", "ww_foam_life_multiplier",
    "ww_bubble_drag", "ww_bubble_buoyancy", "ww_bubble_life_multiplier",
    "ww_spray_drag", "ww_spray_gravity_scale", "ww_spray_life_multiplier",
    "surface_attribute_velocity", "surface_attribute_speed",
    "surface_attribute_vorticity", "surface_attribute_age",
    "surface_attribute_lifetime", "surface_attribute_color",
    "surface_attribute_source_id", "surface_attribute_viscosity",
    "surface_attribute_density", "surface_attribute_motion",
    "particles_attribute_velocity", "particles_attribute_speed",
    "particles_attribute_vorticity", "particles_attribute_age",
    "particles_attribute_lifetime", "particles_attribute_color",
    "particles_attribute_source_id", "particles_attribute_uid",
    "particles_attribute_viscosity", "particles_attribute_density",
    "whitewater_attribute_velocity", "whitewater_attribute_speed",
    "whitewater_attribute_age", "whitewater_attribute_lifetime",
    "whitewater_attribute_type",
)

VECTOR_SETTING_FIELDS = frozenset({
    "directional_force",
    "omnidirectional_force_center",
})

SETTING_ALIASES = {
    "limit_percent": "speed_limit_percent",
    "limit_behavior": "speed_limit_behavior",
    "ww_churn_decay": "churn_decay",
    "ww_churn_gain": "churn_gain",
}

_ENUM_SETTING_FIELDS = None
_SETTING_REGISTRY_VALIDATED = False


def domain_bounds(scene) -> tuple:
    """Return plain host tuples for the authored domain's world AABB."""
    obj = scene.fluidsim.domain_object
    if obj is None:
        return (-2.0, -2.0, 0.0), (2.0, 2.0, 4.0)
    from mathutils import Vector

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = tuple(float(min(getattr(corner, axis) for corner in corners))
               for axis in ("x", "y", "z"))
    hi = tuple(float(max(getattr(corner, axis) for corner in corners))
               for axis in ("x", "y", "z"))
    return lo, hi


def domain_transform(scene):
    """Capture the domain transform as owned scalar tuples."""
    obj = getattr(scene.fluidsim, "domain_object", None)
    matrix = getattr(obj, "matrix_world", None)
    if matrix is None:
        return None
    return tuple(tuple(float(matrix[row][column]) for column in range(4))
                 for row in range(4))


def enum_setting_fields():
    global _ENUM_SETTING_FIELDS
    if _ENUM_SETTING_FIELDS is not None:
        return _ENUM_SETTING_FIELDS
    from ..sim.enums import (
        LimitBehavior,
        ParticleSampleType,
        ProjectNonDivergentSolver,
        RelaxMode,
        SdfFilterType,
        SurfaceResolutionMode,
        SurfaceUpdateCadence,
        TraceMode,
        ViscositySolver,
    )
    _ENUM_SETTING_FIELDS = {
        "sample_type": ("sample_type", ParticleSampleType),
        "trace_mode": ("trace_mode", TraceMode),
        "limit_behavior": ("limit_behavior", LimitBehavior),
        "relax_mode": ("relax_mode", RelaxMode),
        "pressure_solver": ("pressure_solver", ProjectNonDivergentSolver),
        "viscosity_solver": ("viscosity_solver", ViscositySolver),
        "surface_filter_type": ("surface_filter_type", SdfFilterType),
        "surface_smooth_type": ("surface_smooth_type", SdfFilterType),
        "surface_resolution_mode": (
            "surface_resolution_mode", SurfaceResolutionMode),
        "surface_update_cadence": (
            "surface_update_cadence", SurfaceUpdateCadence),
    }
    return _ENUM_SETTING_FIELDS


def validate_setting_registry():
    global _SETTING_REGISTRY_VALIDATED
    if _SETTING_REGISTRY_VALIDATED:
        return
    from ..sim.solver import SolverSettings

    targets = {SETTING_ALIASES.get(name, name) for name in DIRECT_SETTING_FIELDS}
    targets.update(SETTING_ALIASES.get(target, target)
                   for target in enum_setting_fields())
    unknown = sorted(targets - set(SolverSettings.__dataclass_fields__))
    if unknown:
        raise ValueError(
            "Scene setting targets not declared by SolverSettings: "
            + ", ".join(unknown))
    _SETTING_REGISTRY_VALIDATED = True


def apply_scene_settings(fs, settings, fields=None):
    """Apply selected authored fields using only owned Python values."""
    validate_setting_registry()
    selected = None if fields is None else set(fields)
    for source in DIRECT_SETTING_FIELDS:
        target = SETTING_ALIASES.get(source, source)
        if (selected is None or source in selected) and hasattr(fs, source):
            value = getattr(fs, source)
            if source in VECTOR_SETTING_FIELDS:
                value = tuple(float(component) for component in value)
            setattr(settings, target, value)
    for target, (source, enum_type) in enum_setting_fields().items():
        declared_target = SETTING_ALIASES.get(target, target)
        if (selected is None or source in selected) and hasattr(fs, source):
            setattr(settings, declared_target, enum_type[getattr(fs, source)])
    return settings


def apply_gravity(scene, settings):
    fs = scene.fluidsim
    if not hasattr(fs, "gravity"):
        return settings
    if (getattr(fs, "use_scene_gravity", False)
            and getattr(scene, "use_gravity", False)):
        settings.gravity = tuple(float(value) for value in scene.gravity)
    else:
        settings.gravity = tuple(float(value) for value in fs.gravity)
    return settings


def build_settings(scene):
    """Create SolverSettings without retaining Blender RNA/mathutils proxies."""
    from ..preferences import configured_device
    from ..sim.solver import SolverSettings

    fs = scene.fluidsim
    settings = SolverSettings()
    settings.device = str(configured_device())
    settings.domain_min, settings.domain_max = domain_bounds(scene)
    settings.frame_rate = float(scene.render.fps) / float(scene.render.fps_base)
    apply_gravity(scene, settings)
    apply_scene_settings(fs, settings)
    return settings
