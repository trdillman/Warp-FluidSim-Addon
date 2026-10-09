"""Projection of Blender-authored values into owned solver settings.

This module owns value conversion only. Lifecycle classification is owned by
``pending_settings``.
"""

from __future__ import annotations

import sys

from .scene_access import active_domain_object, active_domain_settings


# Canonical fallback domain used when no domain object is designated. The
# migration create-branch builds an object spanning exactly this box so that
# a previously-unset scene keeps identical bounds (and thus identical solver
# output) after a domain object is auto-created.
DEFAULT_DOMAIN_MIN = (-2.0, -2.0, 0.0)
DEFAULT_DOMAIN_MAX = (2.0, 2.0, 4.0)


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
    "relax_enabled", "relax_strength",
    "pressure_iterations", "pressure_tolerance", "pressure_threshold",
    "pressure_preconditioner",
    "viscosity_enabled", "viscosity_constant", "viscosity_from_particles",
    "viscosity_iterations", "viscosity_tolerance",
    "surface_tension_enabled", "surface_tension_coefficient",
    "surface_tension_density", "surface_tension_band_voxels",
    "surface_tension_safety", "surface_tension_smoothing_iterations",
    "species_enabled", "species_mixing_rate",
    "narrow_band_enabled", "narrow_band_voxels",
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
    "export_attributes",
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

# The only settings-group attributes that may be read with a default
# (``getattr(group, name, default)``). Every other settings field is declared
# by the group that owns it, so a read of it is direct attribute access and a
# read from the wrong group raises instead of quietly yielding a default (the
# way defects #1, #3 and #5 happened). ``tests/runtime/test_settings_read_allowlist.py``
# fails on any defaulted read of a settings field outside this list.
#   schema_version / active_snippet / snippets -- legacy snippet-era carriers.
#       Neither settings group declares them; a pre-1.1 file may still hold them
#       as stored IDProperties (see ``blender/file_compat.py``).
#   stat_*  -- runtime statistics read during cache playback, where the solver
#       that normally fills them has never run.
OPTIONAL_SETTING_READS = frozenset({"schema_version", "active_snippet", "snippets"})
OPTIONAL_SETTING_READ_PREFIXES = ("stat_",)


def is_optional_setting_read(name: str) -> bool:
    """True when ``name`` may be read from a settings group with a default."""
    return name in OPTIONAL_SETTING_READS or name.startswith(
        OPTIONAL_SETTING_READ_PREFIXES)


def domain_bounds(scene) -> tuple:
    """Return plain host tuples for the authored domain's world AABB."""
    obj = active_domain_object(scene)
    if obj is None:
        return DEFAULT_DOMAIN_MIN, DEFAULT_DOMAIN_MAX
    from mathutils import Vector

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = tuple(float(min(getattr(corner, axis) for corner in corners))
               for axis in ("x", "y", "z"))
    hi = tuple(float(max(getattr(corner, axis) for corner in corners))
               for axis in ("x", "y", "z"))
    return lo, hi


def domain_transform(scene):
    """Capture the domain transform as owned scalar tuples."""
    obj = active_domain_object(scene)
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
    from ..sim.solver_settings_model import SolverSettings

    targets = {SETTING_ALIASES.get(name, name) for name in DIRECT_SETTING_FIELDS}
    targets.update(SETTING_ALIASES.get(target, target)
                   for target in enum_setting_fields())
    unknown = sorted(targets - set(SolverSettings.__dataclass_fields__))
    if unknown:
        raise ValueError(
            "Scene setting targets not declared by SolverSettings: "
            + ", ".join(unknown))
    _SETTING_REGISTRY_VALIDATED = True


def scene_quality_fidelity(scene) -> float:
    """Fidelity multiplier authored by the scene-level Quality tier.

    ``quality`` is a scene-level presentation dial (drawn on ``scene.fluidsim``
    beside ``ui_mode``), so it is read here rather than from the per-domain
    settings the solver otherwise builds from. This keeps the read owner and the
    UI write owner identical on both migrated and pre-migration scenes.
    """
    from ..sim.quality_contract import quality_tier_fidelity

    fs_scene = getattr(scene, "fluidsim", None)
    # No scene group at all resolves to the neutral tier; a group that lacks
    # ``quality`` is a bug and raises rather than reading as PREVIEW.
    return quality_tier_fidelity(None if fs_scene is None else fs_scene.quality)


def _require_domain_group(fs):
    """Refuse to project anything but a domain settings group.

    ``apply_scene_settings`` skips every field ``fs`` does not declare, so the
    scene group -- which declares none of them -- would otherwise yield a fully
    defaulted ``SolverSettings`` without a word. Class identity is the exact
    test; no field-presence rule can tell the scene group apart from a minimal
    settings double.
    """
    if fs is None:
        raise ValueError(
            "FluidSim has no domain object designated for this scene; "
            "designate one before simulating")
    # Looked up, not imported: an instance of the scene group implies its module
    # is loaded, and importing it here would drag bpy into bpy-free callers.
    properties = sys.modules.get(__package__.rpartition(".")[0] + ".properties")
    scene_cls = getattr(properties, "FLUIDSIM_PG_scene", None)
    if scene_cls is not None and isinstance(fs, scene_cls):
        raise TypeError(
            "solver settings must be read from the domain object's "
            "fluidsim_domain group, not scene.fluidsim")


def apply_scene_settings(fs, settings, fields=None, *, quality_fidelity=1.0):
    """Apply selected authored fields using only owned Python values.

    ``quality_fidelity`` rescales the Quality-tier knobs (voxel size, substeps,
    pressure iterations, particles per voxel) as they are projected. Scaling
    always reads the freshly-copied authored value from ``fs``, never a prior
    solver value, so repeated application is idempotent and cannot compound.
    """
    _require_domain_group(fs)
    validate_setting_registry()
    from ..sim.quality_contract import QUALITY_SCALED_FIELDS, scale_quality_value

    scale = float(quality_fidelity)
    selected = None if fields is None else set(fields)
    for source in DIRECT_SETTING_FIELDS:
        target = SETTING_ALIASES.get(source, source)
        if selected is None or source in selected:
            value = getattr(fs, source)
            if source in VECTOR_SETTING_FIELDS:
                value = tuple(float(component) for component in value)
            elif scale != 1.0 and source in QUALITY_SCALED_FIELDS:
                value = scale_quality_value(source, value, scale)
            setattr(settings, target, value)
    for target, (source, enum_type) in enum_setting_fields().items():
        declared_target = SETTING_ALIASES.get(target, target)
        if selected is None or source in selected:
            setattr(settings, declared_target, enum_type[getattr(fs, source)])
    return settings


def apply_gravity(scene, settings):
    fs = active_domain_settings(scene)
    _require_domain_group(fs)
    if fs.use_scene_gravity and getattr(scene, "use_gravity", False):
        settings.gravity = tuple(float(value) for value in scene.gravity)
    else:
        settings.gravity = tuple(float(value) for value in fs.gravity)
    return settings


def normalize_domain_boundary_settings(settings):
    """Enforce the solver invariant required by closed particle bounds."""
    if not settings.open_boundaries:
        settings.domain_sdf_walls = True
    return settings


def build_settings(scene):
    """Create SolverSettings without retaining Blender RNA/mathutils proxies."""
    from ..preferences import configured_device
    from ..sim.solver_settings_model import SolverSettings

    fs = active_domain_settings(scene)
    settings = SolverSettings()
    settings.device = str(configured_device())
    settings.domain_min, settings.domain_max = domain_bounds(scene)
    settings.frame_rate = float(scene.render.fps) / float(scene.render.fps_base)
    apply_gravity(scene, settings)
    apply_scene_settings(fs, settings, quality_fidelity=scene_quality_fidelity(scene))
    # Closed particle bounds require matching pressure/collision walls. Old
    # files may store the internal switch as false, so normalize the solver
    # projection without mutating Blender data during read-only preflight.
    return normalize_domain_boundary_settings(settings)
