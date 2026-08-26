"""Pending Settings lifecycle policy and Blender interaction state.

Owns transition classification, applied snapshots/signatures, per-scene pending
state, live/reset/output application, debounce, revert/apply transitions, and
persistent depsgraph/undo/load handlers. UI, properties, playback, and the Solver
Session are adapters to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import bpy

from .solver_settings import (
    DIRECT_SETTING_FIELDS as _DIRECT_SETTING_FIELDS,
    SETTING_ALIASES as _SETTING_ALIASES,
    apply_gravity as _project_gravity,
    apply_scene_settings as _apply_scene_settings,
    domain_bounds as _domain_bounds,
    enum_setting_fields as _enum_setting_fields,
    normalize_domain_boundary_settings as _normalize_domain_boundary_settings,
    validate_setting_registry as _validate_solver_setting_registry,
)

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)


def session():
    from .scene_bridge import session as solver_session
    return solver_session()


def _scene_get(scene, key: str, default=""):
    try:
        return scene.get(key, default)
    except AttributeError:
        return getattr(scene, key, default)


def _scene_set(scene, key: str, value):
    try:
        scene[key] = value
    except TypeError:
        setattr(scene, key, value)


class SceneUpdate(str, Enum):
    """Kinds of scene edits that can affect an active solver session."""

    REBUILD = "REBUILD"
    RESET = "RESET"
    LIVE = "LIVE"
    SYNC = "SYNC"
    OUTPUT = "OUTPUT"
    SURFACE = "SURFACE"
    WHITEWATER = "WHITEWATER"


class SettingTier(str, Enum):
    """Exhaustive lifecycle classification for scene FluidSim properties."""

    LIVE = "LIVE"
    SYNC = "SYNC"
    OUTPUT = "OUTPUT"
    SURFACE = "SURFACE"
    WHITEWATER = "WHITEWATER"
    RESET = "RESET"
    REINITIALIZE = "REINITIALIZE"
    AUTHORED_NEXT_RUN = "AUTHORED_NEXT_RUN"
    ADAPTER = "ADAPTER"
    PRESENTATION = "PRESENTATION"


_LIVE_REGISTRY_FIELDS = frozenset({
    "time_scale", "cfl", "min_substeps", "max_substeps", "gravity",
    "use_scene_gravity", "sample_type", "flip_ratio",
    "trace_mode", "relax_enabled", "relax_strength", "relax_mode",
    "max_speed", "limit_behavior", "limit_percent", "directional_force",
    "omnidirectional_force_center", "omnidirectional_force_strength",
    "omnidirectional_force_radius", "force_diffusion", "dissipation_enabled",
    "dissipation_attribute", "dissipation_rate", "dissipation_target",
    "attraction_enabled", "attraction_strength", "attraction_max_distance",
    "count_stiffness", "spacetime_jitter_strength",
    "spacetime_adaptive_jitter", "spacetime_phase_eta",
    "spacetime_phase_threshold", "spacetime_variable_projection",
    "spacetime_liquid_density", "spacetime_gas_density",
    "spacetime_min_density", "spacetime_local_cfl",
    "spacetime_max_local_substeps", "pressure_solver",
    "pressure_iterations", "pressure_tolerance", "pressure_threshold",
    "viscosity_enabled", "viscosity_solver", "viscosity_constant",
    "viscosity_from_particles", "viscosity_iterations",
    "viscosity_tolerance", "surface_tension_coefficient",
    "surface_tension_density", "surface_tension_band_voxels",
    "surface_tension_safety", "surface_tension_smoothing_iterations",
    "species_mixing_rate",
    "ww_churn_decay", "ww_churn_gain",
    "ww_churn_threshold", "ww_emission_rate", "ww_min_speed",
    "ww_life_min", "ww_life_max", "ww_bubble_drag", "ww_spray_drag",
    "ww_surface_band", "ww_foam_band", "ww_spray_min_speed",
    "ww_type_hysteresis", "ww_foam_advection", "ww_foam_preservation",
    "ww_foam_density_influence", "ww_foam_life_multiplier",
    "ww_bubble_buoyancy", "ww_bubble_life_multiplier",
    "ww_spray_gravity_scale", "ww_spray_life_multiplier",
    "auto_grow_particles", "particle_growth_failure_policy",
})
_OUTPUT_REGISTRY_FIELDS = frozenset({
    "liquid_view_mode", "show_surface", "show_particles", "show_whitewater",
    "particle_display_fraction", "particle_display_scale",
    "surface_attribute_mode", "particle_attribute_mode",
    "whitewater_attribute_mode", "whitewater_display_scale",
    "whitewater_output_mode",
    "surface_material", "particle_material", "whitewater_material",
    "foam_material", "bubble_material", "spray_material",
    "output_collection_name", "show_overlay",
})
_SURFACE_REGISTRY_FIELDS = frozenset({
    "surface_enabled", "particle_radius_scale",
    "surface_splat_radius", "surface_resolution_mode",
    "surface_resolution_multiplier", "surface_voxel_size",
    "surface_leaf_capacity_scale", "surface_max_dense_dim",
    "surface_update_cadence", "live_surface_publish_cadence",
    "surface_build_lowres_sdf", "surface_filter_type",
    "surface_filter_iterations", "surface_filter_strength",
    "surface_dilate_iterations", "surface_dilate_strength",
    "surface_erode_iterations", "surface_erode_strength",
    "surface_smooth_type", "surface_smooth_iterations",
    "surface_smooth_strength",
})
_RESET_REGISTRY_FIELDS = frozenset({
    "particles_per_voxel", "topology_expand", "open_boundaries", "domain_sdf_walls",
    "spacetime_enabled", "spacetime_reference_mass", "spacetime_seed",
    "surface_tension_enabled", "species_enabled",
    "height_enabled", "height_base", "height_back_mode", "height_profile_image",
    "height_profile_scale", "height_emission_mode", "height_velocity",
    "height_emit_particles",
    "surface_attribute_velocity", "surface_attribute_speed", "surface_attribute_vorticity", "surface_attribute_age", "surface_attribute_lifetime", "surface_attribute_color", "surface_attribute_source_id", "surface_attribute_viscosity", "surface_attribute_density", "surface_attribute_motion", "particles_attribute_velocity", "particles_attribute_speed", "particles_attribute_vorticity", "particles_attribute_age", "particles_attribute_lifetime", "particles_attribute_color", "particles_attribute_source_id", "particles_attribute_uid", "particles_attribute_viscosity", "particles_attribute_density", "whitewater_attribute_velocity", "whitewater_attribute_speed", "whitewater_attribute_age", "whitewater_attribute_lifetime", "whitewater_attribute_type",
})
_REINITIALIZE_REGISTRY_FIELDS = frozenset({
    "domain_object", "voxel_size", "leaf_capacity", "max_particles", "max_whitewater",
    "profiling_enabled", "profiling_sync",
})
_AUTHORED_NEXT_RUN_FIELDS = frozenset({
    "snippets", "active_snippet", "snippet_library_selection",
    "snippet_editor_lines", "shortcut_field", "shortcut_constant_name",
    "shortcut_constant_value", "report_path", "benchmark_frames",
    "benchmark_baseline", "spacetime_resync_surface", "cache_enabled",
    "cache_path", "cache_compress",
    "cache_use_custom_range", "cache_frame_start", "cache_frame_end",
    "active_preset_id", "product_preset", "setup_mode",
    "bake_start", "bake_end", "cache_format", "checkpoint_interval",
})
_ADAPTER_FIELDS = frozenset({
    "live", "cache_playback_enabled", "max_frame_catchup",
})
_PRESENTATION_FIELDS = frozenset({
    "playback_advanced_expanded", "capacity_show_alternatives",
    "surface_advanced_expanded", "height_error", "status", "stat_particles",
    "stat_voxels", "stat_whitewater", "stat_step_ms", "stat_substeps",
    "stat_cfl_clamped", "stat_whitewater_admitted",
    "stat_whitewater_dropped", "stat_whitewater_policy",
    "stat_whitewater_required", "stat_whitewater_status",
    "stat_whitewater_foam", "stat_whitewater_bubbles",
    "stat_whitewater_spray", "stat_whitewater_invalid_types",
    "stat_capillary_dt", "stat_capillary_substeps_required",
    "stat_capillary_clamped", "stat_substep_limiter",
    "stat_surface_tension_active_voxels",
    "stat_surface_tension_max_curvature",
    "stat_surface_tension_max_acceleration",
    "stat_surface_tension_temporary_bytes",
    "stat_surface_tension_submission_ms", "stat_surface_tension_status",
    "stat_surface_tension_message", "bake_status", "bake_progress",
    "bake_message",
})

_TIER_FIELDS = {
    SettingTier.LIVE: _LIVE_REGISTRY_FIELDS,
    SettingTier.OUTPUT: _OUTPUT_REGISTRY_FIELDS,
    SettingTier.SURFACE: _SURFACE_REGISTRY_FIELDS,
    SettingTier.WHITEWATER: frozenset({"whitewater_enabled"}),
    SettingTier.RESET: _RESET_REGISTRY_FIELDS,
    SettingTier.REINITIALIZE: _REINITIALIZE_REGISTRY_FIELDS,
    SettingTier.AUTHORED_NEXT_RUN: _AUTHORED_NEXT_RUN_FIELDS,
    SettingTier.ADAPTER: _ADAPTER_FIELDS,
    SettingTier.PRESENTATION: _PRESENTATION_FIELDS,
    SettingTier.SYNC: frozenset(),
}
SETTING_TRANSITION_REGISTRY = {
    name: tier for tier, names in _TIER_FIELDS.items() for name in names
}


def validate_transition_registry(declared_fields=None):
    """Fail closed on overlap, omission, or undeclared lifecycle policy."""
    seen = set()
    overlap = set()
    for names in _TIER_FIELDS.values():
        overlap.update(seen & set(names))
        seen.update(names)
    if overlap:
        raise ValueError("Conflicting Pending Settings tiers: "
                         + ", ".join(sorted(overlap)))
    if declared_fields is not None:
        declared = set(declared_fields)
        missing = declared - seen
        unknown = seen - declared
        if missing or unknown:
            raise ValueError(
                f"Pending Settings registry mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}")
    _validate_solver_setting_registry()
    return True


_SCENE_UPDATE_REASONS = {
    SceneUpdate.REBUILD: "Solver allocation changed",
    SceneUpdate.RESET: "Simulation settings changed",
    SceneUpdate.LIVE: "Live setting stored; applies on initialization",
    SceneUpdate.SYNC: "Scene source settings changed",
    SceneUpdate.OUTPUT: "Viewport display settings changed",
    SceneUpdate.SURFACE: "Surface display settings changed",
    SceneUpdate.WHITEWATER: "Whitewater display settings changed",
}


_HOT_SETTING_FIELDS = frozenset({
    "time_scale", "cfl", "min_substeps", "max_substeps",
    "sample_type", "flip_ratio", "trace_mode", "relax_enabled",
    "relax_strength", "relax_mode", "max_speed", "limit_behavior",
    "limit_percent", "directional_force", "omnidirectional_force_center",
    "omnidirectional_force_strength", "omnidirectional_force_radius",
    "force_diffusion", "dissipation_enabled", "dissipation_attribute",
    "dissipation_rate", "dissipation_target", "attraction_enabled",
    "attraction_strength", "attraction_max_distance", "count_stiffness",
    "spacetime_jitter_strength", "spacetime_adaptive_jitter",
    "spacetime_phase_eta", "spacetime_phase_threshold",
    "spacetime_variable_projection", "spacetime_liquid_density",
    "spacetime_gas_density", "spacetime_min_density",
    "spacetime_local_cfl", "spacetime_max_local_substeps",
    "pressure_solver", "pressure_iterations", "pressure_tolerance",
    "pressure_threshold", "viscosity_enabled", "viscosity_solver",
    "viscosity_constant", "viscosity_from_particles",
    "viscosity_iterations", "viscosity_tolerance",
    "ww_churn_decay", "ww_churn_gain", "ww_churn_threshold",
    "ww_emission_rate", "ww_min_speed", "ww_life_min", "ww_life_max",
    "ww_bubble_drag", "ww_spray_drag", "ww_surface_band", "ww_foam_band",
    "ww_spray_min_speed", "ww_type_hysteresis", "ww_foam_advection",
    "ww_foam_preservation", "ww_foam_density_influence",
    "ww_foam_life_multiplier", "ww_bubble_buoyancy",
    "ww_bubble_life_multiplier", "ww_spray_gravity_scale",
    "ww_spray_life_multiplier",
})

_LIVE_REFRESH_SETTING_FIELDS = frozenset()
_LIVE_SETTING_FIELDS = _HOT_SETTING_FIELDS | _LIVE_REFRESH_SETTING_FIELDS

_RESET_SETTING_FIELDS = frozenset({
    "particles_per_voxel", "topology_expand", "open_boundaries", "domain_sdf_walls",
    "spacetime_enabled", "spacetime_reference_mass", "spacetime_seed",
    "surface_attribute_velocity", "surface_attribute_speed", "surface_attribute_vorticity", "surface_attribute_age", "surface_attribute_lifetime", "surface_attribute_color", "surface_attribute_source_id", "surface_attribute_viscosity", "surface_attribute_density", "surface_attribute_motion", "particles_attribute_velocity", "particles_attribute_speed", "particles_attribute_vorticity", "particles_attribute_age", "particles_attribute_lifetime", "particles_attribute_color", "particles_attribute_source_id", "particles_attribute_uid", "particles_attribute_viscosity", "particles_attribute_density", "whitewater_attribute_velocity", "whitewater_attribute_speed", "whitewater_attribute_age", "whitewater_attribute_lifetime", "whitewater_attribute_type",
})

_SURFACE_SETTING_FIELDS = frozenset({
    "surface_enabled", "particle_radius_scale", "surface_splat_radius",
    "surface_resolution_mode", "surface_resolution_multiplier",
    "surface_voxel_size", "surface_leaf_capacity_scale",
    "surface_max_dense_dim", "surface_update_cadence",
    "live_surface_publish_cadence",
    "surface_build_lowres_sdf", "surface_filter_type",
    "surface_filter_iterations", "surface_filter_strength",
    "surface_dilate_iterations", "surface_dilate_strength",
    "surface_erode_iterations", "surface_erode_strength",
    "surface_smooth_type", "surface_smooth_iterations",
    "surface_smooth_strength",
})


_REINIT_SETTING_FIELDS = frozenset({
    "domain_object", "voxel_size", "leaf_capacity", "max_particles",
    "max_whitewater", "profiling_enabled", "profiling_sync",
})
_RESET_SCENE_FIELDS = _RESET_SETTING_FIELDS | frozenset({
    "height_enabled", "height_base", "height_back_mode",
    "height_profile_image", "height_profile_scale", "height_emission_mode",
    "height_velocity", "height_emit_particles",
})
_RESET_OBJECT_FIELDS = frozenset({
    "role", "point_source_representation", "mesh_evaluation_mode",
    "emission_mode", "auto_inject_seed",
})


@dataclass
class SceneInteractionState:
    """Per-scene applied snapshots and cached pending presentation."""

    scene: object
    applied_reset_signature: tuple | None = None
    applied_reinit_signature: tuple | None = None
    current_reset_signature: tuple | None = None
    current_reinit_signature: tuple | None = None
    reset_snapshot: dict = field(default_factory=dict)
    object_reset_snapshot: tuple = ()
    reinit_snapshot: dict = field(default_factory=dict)
    domain_matrix_snapshot: object = None
    applied_device: str = ""
    pending_action: SceneUpdate | None = None
    pending_reasons: tuple = ()
    blocked_reason: str = ""
    suppress_callbacks: bool = False
    timer_scheduled: bool = False
    capacity_plan_timer_scheduled: bool = False


_INTERACTION_STATES = {}
_ST_REFERENCE_MASS_CACHE = {}
_ST_REFERENCE_MASS_CACHE_LIMIT = 64


def _scene_identity(scene):
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
            if value:
                return value
        except (TypeError, ValueError, ReferenceError):
            pass
    return id(scene)


def _interaction_state(scene):
    key = _scene_identity(scene)
    state = _INTERACTION_STATES.get(key)
    if state is None or state.scene is not scene:
        state = SceneInteractionState(scene=scene)
        _INTERACTION_STATES[key] = state
    return state


def _stable_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return ("ID", int(pointer()))
        except (TypeError, ValueError, ReferenceError):
            pass
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return ("ID", name, id(value))
    try:
        return tuple(_stable_value(item) for item in value)
    except TypeError:
        return repr(value)


def _snapshot_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(getattr(value, "as_pointer", None)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _configured_device():
    from ..preferences import configured_device

    return str(configured_device())


def _reset_signature(scene):
    fs = scene.fluidsim
    values = [(name, _stable_value(getattr(fs, name, None)))
              for name in sorted(_RESET_SCENE_FIELDS)]
    objects = []
    for obj in getattr(scene, "objects", ()) or ():
        pg = getattr(obj, "fluidsim", None)
        if pg is None:
            continue
        object_id = _stable_value(obj)
        fields = tuple((name, _stable_value(getattr(pg, name, None)))
                       for name in sorted(_RESET_OBJECT_FIELDS))
        objects.append((object_id, fields))
    values.append(("objects", tuple(sorted(objects, key=repr))))
    return tuple(values)


def _reinit_signature(scene):
    fs = scene.fluidsim
    values = [(name, _stable_value(getattr(fs, name, None)))
              for name in sorted(_REINIT_SETTING_FIELDS)]
    domain = getattr(fs, "domain_object", None)
    values.append(("domain_identity", _stable_value(domain)))
    try:
        values.append(("domain_bounds", _stable_value(_domain_bounds(scene))))
    except Exception as exc:  # preserve a deterministic, visible pending reason
        values.append(("domain_bounds_error", repr(exc)))
    values.append(("device", _configured_device()))
    return tuple(values)


def current_interaction_signatures(scene):
    """Return canonical reset/reinitialize signatures for undo/script checks."""
    return _reset_signature(scene), _reinit_signature(scene)


def _signature_changes(applied, current):
    if applied is None:
        # A missing snapshot is unknown, not proof that the values match.
        return tuple(sorted(name for name, _value in current))
    old = dict(applied)
    new = dict(current)
    return tuple(sorted(name for name in set(old) | set(new)
                        if old.get(name) != new.get(name)))


def _pending_label(action):
    if action is SceneUpdate.REBUILD:
        return "REINIT_PENDING"
    if action is SceneUpdate.RESET:
        return "RESET_PENDING"
    return "NONE"


def _cache_pending_presentation(scene, state):
    action = _pending_label(state.pending_action)
    _scene_set(scene, "fluidsim_pending_action", action)
    _scene_set(scene, "fluidsim_pending_count", len(state.pending_reasons))
    _scene_set(scene, "fluidsim_pending_reasons", "\n".join(state.pending_reasons))
    _scene_set(scene, "fluidsim_pending_blocked", state.blocked_reason)


def pending_presentation(scene):
    """Return draw-safe cached pending data without inspecting RNA or geometry."""
    state = _INTERACTION_STATES.get(_scene_identity(scene))
    if state is None:
        reasons = tuple(filter(None, str(_scene_get(
            scene, "fluidsim_pending_reasons", "")).splitlines()))
        return {
            "action": str(_scene_get(scene, "fluidsim_pending_action", "NONE")),
            "count": int(_scene_get(scene, "fluidsim_pending_count", len(reasons))),
            "reasons": reasons,
            "blocked_reason": str(_scene_get(
                scene, "fluidsim_pending_blocked", "")),
        }
    return {
        "action": _pending_label(state.pending_action),
        "count": len(state.pending_reasons),
        "reasons": state.pending_reasons,
        "blocked_reason": state.blocked_reason,
    }


def refresh_pending_state(scene, sess=None, *, use_cached=False):
    state = _interaction_state(scene)
    if state.applied_reset_signature is None:
        state.pending_action = None
        state.pending_reasons = ()
        state.blocked_reason = ""
        _cache_pending_presentation(scene, state)
        return state
    if (use_cached and state.current_reset_signature is not None
            and state.current_reinit_signature is not None):
        reset_now = state.current_reset_signature
        reinit_now = state.current_reinit_signature
    else:
        reset_now, reinit_now = current_interaction_signatures(scene)
        state.current_reset_signature = reset_now
        state.current_reinit_signature = reinit_now
    reinit_changes = _signature_changes(state.applied_reinit_signature, reinit_now)
    reset_changes = _signature_changes(state.applied_reset_signature, reset_now)
    if reinit_changes:
        state.pending_action = SceneUpdate.REBUILD
        state.pending_reasons = tuple(f"Reinitialize: {name}" for name in reinit_changes)
    elif reset_changes:
        state.pending_action = SceneUpdate.RESET
        state.pending_reasons = tuple(f"Reset: {name}" for name in reset_changes)
    else:
        state.pending_action = None
        state.pending_reasons = ()
        state.blocked_reason = ""
    _cache_pending_presentation(scene, state)
    return state


def _snapshot_object_reset_values(scene):
    result = []
    for obj in getattr(scene, "objects", ()) or ():
        pg = getattr(obj, "fluidsim", None)
        if pg is not None:
            result.append((obj, {name: _snapshot_value(getattr(pg, name, None))
                                 for name in _RESET_OBJECT_FIELDS}))
    return tuple(result)


def capture_applied_state(scene, sess=None, *, reset_only=False):
    state = _interaction_state(scene)
    reset_signature, reinit_signature = current_interaction_signatures(scene)
    fs = scene.fluidsim
    state.applied_reset_signature = reset_signature
    state.current_reset_signature = reset_signature
    state.current_reinit_signature = reinit_signature
    state.reset_snapshot = {name: _snapshot_value(getattr(fs, name, None))
                            for name in _RESET_SCENE_FIELDS}
    state.object_reset_snapshot = _snapshot_object_reset_values(scene)
    if not reset_only:
        state.applied_reinit_signature = reinit_signature
        state.reinit_snapshot = {
            name: _snapshot_value(getattr(fs, name, None))
            for name in _REINIT_SETTING_FIELDS
        }
        domain = getattr(fs, "domain_object", None)
        matrix = getattr(domain, "matrix_world", None)
        state.domain_matrix_snapshot = (
            matrix.copy() if matrix is not None and hasattr(matrix, "copy") else matrix)
        state.applied_device = _configured_device()
    state.pending_action = None
    state.pending_reasons = ()
    state.blocked_reason = ""
    _cache_pending_presentation(scene, state)
    if reset_only:
        return refresh_pending_state(scene, sess or session())
    return state


def _tag_view3d_redraw():
    screen = getattr(bpy.context, "screen", None)
    for area in getattr(screen, "areas", []) or []:
        if getattr(area, "type", "") == "VIEW_3D":
            area.tag_redraw()


def _apply_gravity(scene, settings):
    return _project_gravity(scene, settings)


def _setting_targets(fields):
    targets = {
        _SETTING_ALIASES.get(source, source)
        for source in _DIRECT_SETTING_FIELDS if source in fields
    }
    targets.update(
        _SETTING_ALIASES.get(target, target)
        for target, (source, _enum_type) in _enum_setting_fields().items()
        if source in fields
    )
    return targets


def _remember_st_reference_mass(settings, reference_mass):
    """Cache the density-normalized automatic ST calibration for one seed."""
    ppc = max(1.0e-6, float(settings.particles_per_voxel))
    reference_mass = float(reference_mass or 0.0)
    if reference_mass <= 0.0:
        return
    key = int(getattr(settings, "spacetime_seed", 1729))
    if len(_ST_REFERENCE_MASS_CACHE) >= _ST_REFERENCE_MASS_CACHE_LIMIT:
        _ST_REFERENCE_MASS_CACHE.pop(next(iter(_ST_REFERENCE_MASS_CACHE)))
    _ST_REFERENCE_MASS_CACHE[key] = reference_mass / ppc


def _cached_st_reference_mass(solver):
    """Return a calibration that scales linearly with particle density.

    The estimator's expected mass is linear in particles-per-voxel.  Solver
    initialization already paid for a real calibration, so remember its
    density-normalized coefficient and avoid an ~86 ms Monte-Carlo call for
    every distinct slider/keyframe value.
    """
    settings = solver.settings
    ppc = max(1.0e-6, float(settings.particles_per_voxel))
    key = int(getattr(settings, "spacetime_seed", 1729))
    coefficient = _ST_REFERENCE_MASS_CACHE.get(key)
    if coefficient is None:
        calibrated = solver._calibrate_st_reference_mass()
        coefficient = float(calibrated) / ppc
        if len(_ST_REFERENCE_MASS_CACHE) >= _ST_REFERENCE_MASS_CACHE_LIMIT:
            _ST_REFERENCE_MASS_CACHE.pop(next(iter(_ST_REFERENCE_MASS_CACHE)))
        _ST_REFERENCE_MASS_CACHE[key] = coefficient
    return coefficient * ppc


def _apply_realtime_solver_settings(scene, sess: Session | None = None):
    """Atomically push live settings and refresh ST-FLIP derived state."""
    sess = sess or session()
    if not sess.owns_scene(scene):
        return False
    fs = scene.fluidsim
    solver = sess.solver
    settings = getattr(solver, "settings", None)
    if settings is None:
        return False
    targets = _setting_targets(_LIVE_SETTING_FIELDS) | {"gravity"}
    previous = {
        target: getattr(settings, target)
        for target in targets if hasattr(settings, target)
    }
    old_ppv = getattr(settings, "particles_per_voxel", None)
    old_reference = getattr(solver, "_st_reference_mass_cached", None)
    if (old_reference is not None
            and getattr(settings, "spacetime_enabled", False)
            and float(getattr(settings, "spacetime_reference_mass", 0.0)) <= 0.0):
        _remember_st_reference_mass(settings, old_reference)
    try:
        _apply_scene_settings(fs, settings, _LIVE_SETTING_FIELDS)
        _apply_gravity(scene, settings)
        new_ppv = getattr(settings, "particles_per_voxel", None)
        if (old_ppv != new_ppv
                and getattr(settings, "spacetime_enabled", False)
                and float(getattr(settings, "spacetime_reference_mass", 0.0)) <= 0.0
                and hasattr(solver, "_calibrate_st_reference_mass")):
            solver._st_reference_mass_cached = _cached_st_reference_mass(solver)
    except Exception:
        for target, value in previous.items():
            setattr(settings, target, value)
        if old_reference is not None:
            solver._st_reference_mass_cached = old_reference
        raise
    return True


def apply_live_settings_safely(scene, sess: Session | None = None):
    """Contain recoverable RNA translation errors without dropping the solver."""
    sess = sess or session()
    try:
        return _apply_realtime_solver_settings(scene, sess)
    except Exception as exc:  # noqa: BLE001 - RNA/handler boundary must not raise
        sess.set_error(scene, repr(exc))
        try:
            scene.fluidsim.status = f"Live settings error: {exc}"
        except Exception:  # noqa: BLE001
            pass
        _tag_view3d_redraw()
        return False


def apply_reset_solver_settings(scene, sess: Session | None = None):
    """Apply hot and reset values in one bounded pass before state reset."""
    sess = sess or session()
    if not sess.owns_scene(scene):
        return False
    _apply_scene_settings(
        scene.fluidsim, sess.solver.settings,
        _LIVE_SETTING_FIELDS | _RESET_SETTING_FIELDS)
    _apply_gravity(scene, sess.solver.settings)
    _normalize_domain_boundary_settings(sess.solver.settings)
    return True


def _apply_realtime_surface_settings(scene, sess: Session | None = None):
    """Push current surface UI settings into an active solver frame."""
    sess = sess or session()
    if not sess.owns_scene(scene):
        return False
    fs = scene.fluidsim
    solver = sess.solver
    _apply_scene_settings(fs, solver.settings, _SURFACE_SETTING_FIELDS)
    solver.rebuild_surface()
    return True


def _apply_realtime_whitewater_settings(scene, sess: Session | None = None):
    sess = sess or session()
    if not sess.owns_scene(scene):
        return False
    sess.solver.settings.whitewater_enabled = scene.fluidsim.whitewater_enabled
    return True


def _is_disposable_start(sess):
    if not sess.active:
        return True
    solver_frame = int(getattr(sess.solver, "frame", 0) or 0)
    return solver_frame <= 0 and sess.last_frame <= sess.start_frame


def _apply_pending(scene, requested=None, *, depsgraph=None):
    """Apply a scene's pending transition, clearing it only after success."""
    sess = session()
    state = refresh_pending_state(scene, sess)
    action = state.pending_action
    if action is None:
        return False
    if requested is SceneUpdate.REBUILD:
        action = SceneUpdate.REBUILD
    if requested is SceneUpdate.RESET and action is SceneUpdate.REBUILD:
        state.blocked_reason = "Reinitialize required; reset cannot apply allocation changes"
        _cache_pending_presentation(scene, state)
        raise RuntimeError(state.blocked_reason)
    if requested is SceneUpdate.RESET and not sess.owns_scene(scene):
        state.blocked_reason = "This scene has no active solver to reset"
        _cache_pending_presentation(scene, state)
        raise RuntimeError(state.blocked_reason)
    if (requested is SceneUpdate.REBUILD and sess.active
            and not sess.owns_scene(scene)):
        state.blocked_reason = "Another scene owns the active solver; release it first"
        _cache_pending_presentation(scene, state)
        raise RuntimeError(state.blocked_reason)
    try:
        if action is SceneUpdate.REBUILD:
            sess.initialize(scene, depsgraph=depsgraph)
        else:
            sess.reset(scene, depsgraph=depsgraph)
        refresh_pending_state(scene, sess)
        return True
    except Exception as exc:
        state.pending_action = action
        if not state.pending_reasons:
            state.pending_reasons = ("Pending settings",)
        state.blocked_reason = str(exc)
        _cache_pending_presentation(scene, state)
        raise


def reinitialize_eligibility(scene):
    if getattr(scene, "fluidsim", None) is None:
        return False, "No FluidSim scene is active"
    if pending_presentation(scene)["action"] == "NONE":
        return False, "No pending settings to apply"
    sess = session()
    if sess.active and not sess.owns_scene(scene):
        return False, "Another scene owns the active solver; release it first"
    return True, ""


def reset_eligibility(scene):
    sess = session()
    if not sess.active:
        return False, "Initialize the solver first"
    if getattr(scene, "fluidsim", None) is None:
        return False, "No FluidSim scene is active"
    if pending_presentation(scene)["action"] != "RESET_PENDING":
        return False, "No reset-tier settings are pending"
    if not sess.owns_scene(scene):
        return False, "The active solver belongs to another scene"
    return True, ""


def revert_eligibility(scene):
    if getattr(scene, "fluidsim", None) is None:
        return False, "No FluidSim scene is active"
    if pending_presentation(scene)["count"] <= 0:
        return False, "No pending settings to revert"
    return True, ""


def apply_pending_reinitialize(scene):
    return _apply_pending(scene, SceneUpdate.REBUILD)


def apply_pending_reset(scene):
    return _apply_pending(scene, SceneUpdate.RESET)


def apply_pending_on_rewind(scene, sess, *, depsgraph=None):
    state = refresh_pending_state(scene, sess)
    if state.pending_action is None:
        return False
    _apply_pending(scene, state.pending_action, depsgraph=depsgraph)
    return True


def revert_pending_values(scene):
    """Restore the successfully applied RNA snapshot without saving prefs."""
    state = _interaction_state(scene)
    if state.applied_reset_signature is None:
        return False
    state.suppress_callbacks = True
    try:
        fs = scene.fluidsim
        for name, value in state.reset_snapshot.items():
            if hasattr(fs, name):
                setattr(fs, name, value)
        for obj, values in state.object_reset_snapshot:
            pg = getattr(obj, "fluidsim", None)
            if pg is not None:
                for name, value in values.items():
                    if hasattr(pg, name):
                        setattr(pg, name, value)
        for name, value in state.reinit_snapshot.items():
            if hasattr(fs, name):
                setattr(fs, name, value)
        domain = getattr(fs, "domain_object", None)
        if domain is not None and state.domain_matrix_snapshot is not None:
            domain.matrix_world = state.domain_matrix_snapshot
    finally:
        state.suppress_callbacks = False
    refresh_pending_state(scene, session())
    if (_configured_device() != state.applied_device
            and state.pending_action is SceneUpdate.REBUILD):
        state.blocked_reason = (
            "Device preference differs from the applied run; change it explicitly "
            "or reinitialize (preferences are never changed automatically)")
        _cache_pending_presentation(scene, state)
    return state.pending_action is None


def _schedule_disposable_apply(scene):
    """Debounce a safe start-state transition with the edited scene captured."""
    state = _interaction_state(scene)
    if state.timer_scheduled:
        return
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is None or not hasattr(timers, "register"):
        _apply_pending(scene)
        return
    state.timer_scheduled = True

    def flush_scene_pending():
        state.timer_scheduled = False
        sess = session()
        if sess.owns_scene(scene) and _is_disposable_start(sess):
            try:
                _apply_pending(scene)
            except Exception as exc:  # noqa: BLE001 - timer boundary
                sess.set_error(scene, repr(exc))
                scene.fluidsim.status = f"Pending settings apply failed: {exc}"
        _tag_view3d_redraw()
        return None

    timers.register(flush_scene_pending, first_interval=0.05)


def _schedule_capacity_plan_refresh(scene):
    """Debounce heavyweight source/GPU planning outside RNA callbacks."""
    state = _interaction_state(scene)
    if state.capacity_plan_timer_scheduled:
        return
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is None or not hasattr(timers, "register"):
        return
    state.capacity_plan_timer_scheduled = True

    def flush_capacity_plan():
        state.capacity_plan_timer_scheduled = False
        sess = session()
        if sess.owns_scene(scene):
            try:
                sess.refresh_capacity_plan(scene)
            except Exception as exc:  # noqa: BLE001 - timer boundary
                sess.set_error(scene, repr(exc))
                scene.fluidsim.status = f"Capacity plan refresh failed: {exc}"
        _tag_view3d_redraw()
        return None

    timers.register(flush_capacity_plan, first_interval=0.10)


def update_particle_capacity_policy(scene):
    """Apply growth policy live and keep capacity plan/pending truth scene-owned."""
    sess = session()
    fs = scene.fluidsim
    if sess.owns_scene(scene):
        sess.solver.settings.auto_grow_particles = bool(fs.auto_grow_particles)
        sess.solver.settings.particle_growth_failure_policy = str(
            fs.particle_growth_failure_policy)
        sess.particle_capacity_notice_scene_identity = _scene_identity(scene)
        fs.status = sess.particle_capacity_notice_for_scene(scene)
        had_scene_plan = (
            sess.capacity_plan is not None
            and sess.capacity_plan_scene_identity == _scene_identity(scene))
        sess.invalidate_capacity_plan_for_scene(scene)
        refresh_pending_state(scene, sess)
        if had_scene_plan:
            _schedule_capacity_plan_refresh(scene)
    else:
        sess.invalidate_capacity_plan_for_scene(scene)
        sess.particle_capacity_notice_scene_identity = _scene_identity(scene)
        fs.status = sess.particle_capacity_notice_for_scene(scene)
    _tag_view3d_redraw()


def request_scene_update(scene, action: SceneUpdate):
    """Apply property changes immediately where the live session allows it."""
    from . import output_publication

    sess = session()
    if not isinstance(action, SceneUpdate):
        action = SceneUpdate(action)
    state = _interaction_state(scene)
    if state.suppress_callbacks:
        return
    reason = _SCENE_UPDATE_REASONS[action]
    if not sess.owns_scene(scene):
        try:
            if action in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
                state = refresh_pending_state(scene, sess)
                if state.pending_action is not None:
                    scene.fluidsim.status = (
                        f"{_pending_label(state.pending_action).replace('_', ' ').title()}: "
                        + ", ".join(state.pending_reasons))
            elif action in {
                SceneUpdate.OUTPUT,
                SceneUpdate.SURFACE,
                SceneUpdate.WHITEWATER,
            }:
                # Cache/export playback leaves Blender output objects behind
                # even when no live solver session exists. Keep their material
                # and visibility settings responsive without inventing geometry.
                output_publication.refresh_existing_output_settings(scene)
                scene.fluidsim.status = reason
            elif action not in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
                scene.fluidsim.status = reason
        finally:
            _tag_view3d_redraw()
        return

    if action in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
        state = refresh_pending_state(scene, sess)
        if state.pending_action is None:
            return
        if _is_disposable_start(sess):
            _schedule_disposable_apply(scene)
        else:
            scene.fluidsim.status = (
                f"{_pending_label(state.pending_action).replace('_', ' ').title()}: "
                + ", ".join(state.pending_reasons))
            _tag_view3d_redraw()
        return
    if action is SceneUpdate.LIVE:
        # Successful slider ticks preserve the useful frame-statistics status;
        # only a contained error replaces it with an actionable message.
        apply_live_settings_safely(scene, sess)
    elif action is SceneUpdate.SYNC:
        sess.sync_scene(scene)
    elif action is SceneUpdate.SURFACE:
        _apply_realtime_surface_settings(scene, sess)
        output_publication.prepare_outputs(scene)
        output_publication.update_outputs(scene, sess, force_surface=True)
    elif action is SceneUpdate.WHITEWATER:
        _apply_realtime_whitewater_settings(scene, sess)
        output_publication.prepare_outputs(scene)
        output_publication.update_outputs(scene, sess)
    elif action is SceneUpdate.OUTPUT:
        output_publication.prepare_outputs(scene)
        output_publication.update_outputs(scene, sess, force_surface=True)
    _tag_view3d_redraw()


# ---------------------------------------------------------------------------
# Pending Settings handlers
# ---------------------------------------------------------------------------


@persistent
def _on_interaction_state_update(*args):
    """Refresh cached pending UI state after undo, drivers, or scripts."""
    scene = args[0] if args and hasattr(args[0], "fluidsim") else getattr(
        bpy.context, "scene", None)
    if scene is None:
        return
    state = _INTERACTION_STATES.get(_scene_identity(scene))
    if state is not None:
        refresh_pending_state(scene, session())


@persistent
def _on_load_pre(_filepath):
    from . import playback

    playback.clear_cache_cursors()
    session().force_release()
    _INTERACTION_STATES.clear()


def register_handlers():
    """Register the persistent handlers owned by Pending Settings."""
    _remove_handlers()
    if hasattr(bpy.app.handlers, "depsgraph_update_post"):
        bpy.app.handlers.depsgraph_update_post.append(_on_interaction_state_update)
    if hasattr(bpy.app.handlers, "undo_post"):
        bpy.app.handlers.undo_post.append(_on_interaction_state_update)
    if hasattr(bpy.app.handlers, "load_pre"):
        bpy.app.handlers.load_pre.append(_on_load_pre)


def _remove_handlers():
    for name in ("depsgraph_update_post", "undo_post", "load_pre"):
        handlers = getattr(bpy.app.handlers, name, None)
        if handlers is None:
            continue
        for fn in list(handlers):
            if getattr(fn, "__module__", "") == __name__:
                handlers.remove(fn)


def unregister_handlers():
    _remove_handlers()
    _INTERACTION_STATES.clear()
