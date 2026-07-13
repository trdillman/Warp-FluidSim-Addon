"""Scene <-> solver bridge.

Owns the live solver session (module-level, cannot live in ID properties),
translates Blender scene/object data into solver descriptors, and steps the
simulation during playback via frame-change handlers.
"""

from __future__ import annotations

import traceback

import bpy
from mathutils import Vector

from .scene_inputs import SceneInputs

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class Session:
    """Live GPU solver session for one scene."""

    def __init__(self):
        self.solver = None
        self.profiler = None
        self.last_frame = -1
        self.start_frame = 1
        self.frame_stats = []
        self.validation_results = []
        self.benchmark_result = None
        self.benchmark_compare = None
        self.error = ""
        self.device = None
        self.memory_assessment = {}
        self.scene_inputs = SceneInputs()

    @property
    def active(self):
        return self.solver is not None

    @property
    def collider_mesh_cache_stats(self):
        return self.scene_inputs.collider_mesh_cache_stats

    def _clear_runtime(self):
        solver = self.solver
        device = getattr(solver, "device", None) or self.device
        self.solver = None
        if solver is not None:
            if hasattr(solver, "release"):
                solver.release()
        self.profiler = None
        self.start_frame = 1
        self.last_frame = -1
        self.frame_stats = []
        self.scene_inputs.clear()
        self.device = None
        del solver
        from ..sim.memory_safety import release_cuda_memory
        release_cuda_memory(device)

    def set_error(self, scene, message):
        self.error = str(message)
        _scene_set(scene, ERROR_SCENE_KEY, self.error)

    def clear_error(self, scene=None):
        self.error = ""
        self.memory_assessment = {}
        if scene is not None:
            _scene_set(scene, ERROR_SCENE_KEY, "")

    def current_error(self, scene):
        return str(_scene_get(scene, ERROR_SCENE_KEY, "") or self.error or "")

    def release(self, scene=None, *, preserve_error=False):
        if scene is None:
            scene = getattr(bpy.context, "scene", None)
        self._clear_runtime()
        if not preserve_error:
            self.clear_error(scene)

    def initialize(self, scene, *, ignore_memory_safeguard=False):
        """(Re)create the solver from current scene state."""
        from ..diagnostics.profiler import Profiler
        from ..sim.solver import FluidSolver
        from . import sources

        self.release(scene)
        try:
            fs = scene.fluidsim
            settings = build_settings(scene)
            self.device = settings.device
            from ..sim.memory_safety import assess_configuration, resource_snapshot

            self.memory_assessment = assess_configuration(
                settings, resource_snapshot(settings.device)
            )
            if not self.memory_assessment["safe"] and not ignore_memory_safeguard:
                estimate_gib = self.memory_assessment["estimated_bytes"] / 1024**3
                budget_gib = self.memory_assessment["budget_bytes"] / 1024**3
                raise RuntimeError(
                    "FluidSim memory safeguard stopped initialization before CUDA "
                    f"allocation (estimated {estimate_gib:.1f} GiB, safe live budget "
                    f"{budget_gib:.1f} GiB). Apply Safe Settings & Retry."
                )
            profiler = None
            if fs.profiling_enabled:
                profiler = Profiler(
                    device=settings.device, sync=fs.profiling_sync)
            self.solver = FluidSolver(settings, profiler=profiler)
            self.profiler = profiler
            self.start_frame = scene.frame_start
            self.sync_scene(scene)
            sources.inject_auto_seeds(scene, self)
            errors = self.solver.set_custom_snippets(build_snippets(scene))
            for name, error in errors:
                for pg in fs.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
            fs.status = "Solver ready"
            self.memory_assessment = {}
            return self
        except Exception as exc:
            self.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            self.release(scene, preserve_error=True)
            raise

    def set_live_enabled(self, scene, enabled):
        fs = scene.fluidsim
        if enabled:
            if getattr(fs, "cache_playback_enabled", False):
                fs.status = "Cache playback enabled - press play"
            elif not self.active:
                self.initialize(scene)
                fs.status = "Live simulation enabled - press play"
            else:
                fs.status = "Live simulation enabled - press play"
        else:
            fs.status = "Live simulation disabled"

    def set_cache_playback_enabled(self, scene, enabled):
        fs = scene.fluidsim
        if enabled:
            self.release(scene)
            if getattr(fs, "live", False):
                fs.status = "Cache playback enabled - press play"
        elif getattr(fs, "live", False):
            self.set_live_enabled(scene, True)

    def sync_scene(self, scene, depsgraph=None):
        if not self.active:
            return
        self.scene_inputs.device = self.solver.device
        self.solver.set_sources(self.scene_inputs.gather_sources(scene))
        self.solver.set_colliders(self.scene_inputs.gather_colliders(scene, depsgraph))
        self.solver.set_height_source(self.scene_inputs.gather_height_source(scene))

    def step(self, scene, *, write_cache=True):
        from . import viewport

        if not self.active:
            return None
        fs = scene.fluidsim
        stats = self.solver.step_frame()
        self.frame_stats.append(stats.as_dict())
        if len(self.frame_stats) > 1000:
            self.frame_stats.pop(0)
        fs.stat_particles = stats.active_particles
        fs.stat_voxels = stats.active_voxels
        fs.stat_whitewater = stats.whitewater_particles
        fs.stat_step_ms = stats.step_time * 1000.0
        fs.stat_substeps = stats.substeps
        fs.status = (
            f"Frame {self.solver.frame}: {stats.active_particles:,} particles, "
            f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
        )
        viewport.update_outputs(scene, self)
        if write_cache and fs.cache_enabled:
            from ..io import cache

            try:
                cache.write_frame(scene, self)
            except Exception as exc:  # noqa: BLE001 - cache is supplemental
                print(f"fluidSim cache write failed: {exc!r}")
        return stats

    def play_cache_frame(self, scene, frame=None):
        from . import viewport
        from ..io import cache

        fs = scene.fluidsim
        frame = int(scene.frame_current if frame is None else frame)
        if getattr(fs, "cache_use_custom_range", False):
            start, end = cache.resolve_frame_range(
                scene,
                use_custom=True,
                frame_start=fs.cache_frame_start,
                frame_end=fs.cache_frame_end,
            )
            if frame < start or frame > end:
                raise FileNotFoundError(
                    f"Cache frame {frame} is outside playback range {start}-{end}"
                )
        base = bpy.path.abspath(fs.cache_path)
        frame_data = cache.read_frame(base, frame)
        if frame_data is None:
            raise FileNotFoundError(
                f"Cache frame {frame} is missing or corrupt in {base}")
        viewport.update_outputs_from_cache_frame(scene, frame_data)
        fs.stat_particles = len(frame_data.get("positions", ()))
        fs.stat_whitewater = len(frame_data.get("whitewater", ()))
        fs.stat_voxels = 0
        fs.stat_step_ms = 0.0
        fs.stat_substeps = 0
        fs.status = f"Cache frame {frame}: {fs.stat_particles:,} particles"
        self.clear_error(scene)
        return frame_data

    def reset(self, scene):
        from . import sources, viewport

        if not self.active:
            return
        self.solver.reset()
        self.start_frame = scene.frame_start
        self.last_frame = -1
        self.frame_stats = []
        self.clear_error(scene)
        self.scene_inputs.clear()
        self.sync_scene(scene)
        sources.inject_auto_seeds(scene, self)
        viewport.update_outputs(scene, self)
        scene.fluidsim.status = "Simulation reset"

    def export_cache_sequence(self, scene, start, end):
        from ..io import cache

        fs = scene.fluidsim
        base = bpy.path.abspath(fs.cache_path)
        original_frame = scene.frame_current
        original_live = fs.live
        frame_set = getattr(scene, "frame_set", None)
        fs.live = False
        try:
            self.initialize(scene)
            self.start_frame = start
            for timeline_frame in range(start, end + 1):
                if frame_set is not None:
                    frame_set(timeline_frame)
                self.solver.frame = timeline_frame - 1
                self.sync_scene(scene)
                self.step(scene, write_cache=False)
                self.solver.frame = timeline_frame
                cache.write_frame(scene, self)
            cache.write_sequence_metadata(base, start, end, source="simulation")
        finally:
            if frame_set is not None:
                try:
                    frame_set(original_frame)
                except Exception:  # noqa: BLE001 - preserve original failure
                    pass
            fs.live = original_live
        return base

    def handle_frame_change(self, scene, depsgraph):
        fs = getattr(scene, "fluidsim", None)
        if fs is None or not fs.live:
            return
        frame = scene.frame_current
        if frame == self.last_frame:
            return
        if getattr(fs, "cache_playback_enabled", False):
            self.last_frame = frame
            try:
                self.play_cache_frame(scene, frame)
            except Exception as exc:  # noqa: BLE001 - never break playback
                self.set_error(scene, repr(exc))
                fs.status = f"Cache playback error: {exc}"
                print(f"fluidSim cache playback failed: {exc!r}")
            return
        if not self.active:
            try:
                self.initialize(scene)
            except Exception as exc:  # noqa: BLE001 - never break playback
                self.set_error(scene, repr(exc))
                fs.status = f"Solver initialization error: {exc}"
                print(f"FluidSim playback initialization failed: {exc!r}")
                return
        if frame <= self.start_frame or frame < self.last_frame:
            self.reset(scene)
            self.last_frame = frame
            return
        self.last_frame = frame
        try:
            self.sync_scene(scene, depsgraph)
            self.step(scene)
        except Exception as exc:  # noqa: BLE001 - never break playback
            self.set_error(scene, repr(exc))
            fs.status = f"Solver error: {exc}"
            fs.live = False
            self.release(preserve_error=True)
            print(f"fluidSim step failed: {exc!r}")


_SESSION = Session()
ERROR_SCENE_KEY = "fluidsim_last_error"


def session() -> Session:
    return _SESSION


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


def set_error(scene, message: str):
    """Persist the latest simulation error for panel-visible recovery."""
    session().set_error(scene, message)


def clear_error(scene):
    session().clear_error(scene)


def current_error(scene) -> str:
    return session().current_error(scene)


# ---------------------------------------------------------------------------
# Settings translation
# ---------------------------------------------------------------------------


def _domain_bounds(scene) -> tuple:
    fs = scene.fluidsim
    obj = fs.domain_object
    if obj is not None:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo = (min(c.x for c in corners), min(c.y for c in corners),
              min(c.z for c in corners))
        hi = (max(c.x for c in corners), max(c.y for c in corners),
              max(c.z for c in corners))
        return lo, hi
    return (-2.0, -2.0, 0.0), (2.0, 2.0, 4.0)


_DIRECT_SETTING_FIELDS = (
    "voxel_size", "leaf_capacity", "topology_expand", "open_boundaries",
    "domain_sdf_walls",
    "max_particles", "max_whitewater", "time_scale", "cfl",
    "min_substeps", "max_substeps", "flip_ratio", "max_speed",
    "limit_percent", "particles_per_voxel", "count_stiffness",
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
    "viscosity_iterations", "viscosity_tolerance", "surface_enabled",
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
    "ww_min_speed", "ww_life_min", "ww_life_max", "ww_bubble_drag",
    "ww_spray_drag",
)

_SETTING_ALIASES = {
    "limit_percent": "speed_limit_percent",
    "limit_behavior": "speed_limit_behavior",
    "ww_churn_decay": "churn_decay",
    "ww_churn_gain": "churn_gain",
}

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


def _apply_scene_settings(fs, settings, fields=None):
    from ..sim.solver import SolverSettings
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

    selected = None if fields is None else set(fields)
    enum_fields = {
        "sample_type": ("sample_type", ParticleSampleType),
        "trace_mode": ("trace_mode", TraceMode),
        "limit_behavior": ("limit_behavior", LimitBehavior),
        "relax_mode": ("relax_mode", RelaxMode),
        "pressure_solver": ("pressure_solver", ProjectNonDivergentSolver),
        "viscosity_solver": ("viscosity_solver", ViscositySolver),
        "surface_filter_type": ("surface_filter_type", SdfFilterType),
        "surface_smooth_type": ("surface_smooth_type", SdfFilterType),
        "surface_resolution_mode": ("surface_resolution_mode", SurfaceResolutionMode),
        "surface_update_cadence": ("surface_update_cadence", SurfaceUpdateCadence),
    }
    targets = {_SETTING_ALIASES.get(name, name) for name in _DIRECT_SETTING_FIELDS}
    targets.update(_SETTING_ALIASES.get(target, target) for target in enum_fields)
    declared = set(SolverSettings.__dataclass_fields__)
    unknown = sorted(targets - declared)
    if unknown:
        raise ValueError(f"Scene setting targets not declared by SolverSettings: {', '.join(unknown)}")

    for source in _DIRECT_SETTING_FIELDS:
        target = _SETTING_ALIASES.get(source, source)
        if (selected is None or source in selected) and hasattr(fs, source):
            setattr(settings, target, getattr(fs, source))
    for target, (source, enum_type) in enum_fields.items():
        declared_target = _SETTING_ALIASES.get(target, target)
        if (selected is None or source in selected) and hasattr(fs, source):
            setattr(settings, declared_target, enum_type[getattr(fs, source)])


def build_settings(scene):
    """Create SolverSettings from scene properties (imports Warp lazily)."""
    from ..sim.solver import SolverSettings

    fs = scene.fluidsim
    s = SolverSettings()
    try:
        prefs = bpy.context.preferences.addons["fluidSim_warp_blender51"].preferences
        if prefs.device:
            s.device = prefs.device
    except (KeyError, AttributeError):
        pass
    lo, hi = _domain_bounds(scene)
    s.domain_min = lo
    s.domain_max = hi
    s.frame_rate = scene.render.fps / scene.render.fps_base
    if fs.use_scene_gravity and scene.use_gravity:
        s.gravity = tuple(scene.gravity)
    else:
        s.gravity = tuple(fs.gravity)
    _apply_scene_settings(fs, s)
    return s


def snippet_argument_names(pg):
    """Names a snippet's declared arguments expose to its body."""
    names = []
    for apg in pg.arguments:
        names.append(apg.field if apg.mode == "FIELD" else apg.constant_name)
    return names


def validate_snippet_pg(pg):
    """Statically validate one snippet property group and mirror the result.

    Updates ``is_valid`` / ``validated_hash`` / ``validation_report`` /
    ``last_error`` on the property group and returns the list of issues.
    """
    from ..sim import snippet_library

    code = pg.text_block.as_string() if pg.text_block else ""
    issues = snippet_library.validate_snippet(
        code, pg.run_over, snippet_argument_names(pg))
    errors = [i for i in issues if i.level == "ERROR"]
    pg.is_valid = not errors
    pg.validated_hash = snippet_library.code_hash(code)
    pg.validation_report = "\n".join(str(i) for i in issues)[:1024]
    if errors:
        pg.last_error = str(errors[0])[:256]
    elif pg.last_error.startswith("ERROR:"):
        pg.last_error = ""
    return issues


def build_snippets(scene):
    """Convert snippet property groups into sim CustomSnippet objects.

    This is the single choke point through which snippets reach the solver
    (both solver init and Apply). Enabled snippets are statically validated
    here; snippets with errors are gated out so only valid code can run in
    the scene.
    """
    result = []
    for pg in scene.fluidsim.snippets:
        if pg.enabled:
            validate_snippet_pg(pg)
            if not pg.is_valid:
                continue
        result.append(snippet_from_pg(pg))
    return result


def snippet_from_pg(pg):
    """Convert one snippet property group into a sim CustomSnippet."""
    from ..sim.custom import CustomArgument, CustomSnippet
    from ..sim.enums import (
        CustomKernelArgumentMode,
        CustomKernelConstantType,
        CustomKernelOrder,
        CustomKernelRunOver,
    )

    code = pg.text_block.as_string() if pg.text_block else ""
    args = []
    for apg in pg.arguments:
        args.append(CustomArgument(
            mode=CustomKernelArgumentMode[apg.mode],
            field=apg.field,
            constant_type=CustomKernelConstantType[apg.constant_type],
            constant_name=apg.constant_name,
            constant_value=apg.constant_value,
        ))
    return CustomSnippet(
        enabled=pg.enabled,
        kernel_name=pg.kernel_name,
        order=CustomKernelOrder[pg.order],
        run_over=CustomKernelRunOver[pg.run_over],
        code=code,
        arguments=args,
        frame_start=pg.frame_start,
        frame_end=pg.frame_end,
        frame_step=pg.frame_step,
    )


# ---------------------------------------------------------------------------
# Object -> descriptor translation
# ---------------------------------------------------------------------------


def gather_sources(scene, sess: Session):
    """Compatibility wrapper for source descriptor translation."""
    from . import sources

    return sources.gather_sources(scene, sess)


def gather_colliders(scene, sess: Session, depsgraph=None):
    """Compatibility wrapper for collider descriptor translation."""
    from . import colliders

    return colliders.gather_colliders(scene, sess, depsgraph)


def gather_height_source(scene):
    """Compatibility wrapper for height-source descriptor translation."""
    from . import sources

    return sources.gather_height_source(scene)
# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def initialize_session(scene, *, ignore_memory_safeguard=False) -> Session:
    """(Re)create the solver from current scene state."""
    return session().initialize(
        scene, ignore_memory_safeguard=ignore_memory_safeguard)


def release_session(scene=None):
    session().release(scene)


def set_live_enabled(scene, enabled: bool):
    """Own the scene.fluidsim.live enable/disable lifecycle."""
    session().set_live_enabled(scene, enabled)


def set_cache_playback_enabled(scene, enabled: bool):
    """Own cache playback transition without requiring a live solver."""
    session().set_cache_playback_enabled(scene, enabled)


def sync_scene_to_solver(scene, sess: Session | None = None, depsgraph=None):
    """Push current scene objects (sources/colliders/height) to the solver."""
    (sess or session()).sync_scene(scene, depsgraph)


def step_frame(scene, sess: Session | None = None, *, write_cache: bool = True):
    """Advance the solver one frame and update output objects."""
    return (sess or session()).step(scene, write_cache=write_cache)


def play_cache_frame(scene, frame: int | None = None):
    """Load one cached timeline frame into the regular output objects."""
    return session().play_cache_frame(scene, frame)


def reset_simulation(scene):
    """Reset solver state and clear outputs (used at playback start)."""
    session().reset(scene)


def _tag_view3d_redraw():
    screen = getattr(bpy.context, "screen", None)
    for area in getattr(screen, "areas", []) or []:
        if getattr(area, "type", "") == "VIEW_3D":
            area.tag_redraw()


def _apply_realtime_surface_settings(scene, sess: Session | None = None):
    """Push current surface UI settings into an active solver frame."""
    sess = sess or session()
    if not sess.active:
        return False
    fs = scene.fluidsim
    solver = sess.solver
    _apply_scene_settings(fs, solver.settings, _SURFACE_SETTING_FIELDS)
    solver.rebuild_surface()
    return True


def _apply_realtime_whitewater_settings(scene, sess: Session | None = None):
    sess = sess or session()
    if not sess.active:
        return False
    sess.solver.settings.whitewater_enabled = scene.fluidsim.whitewater_enabled
    return True


def request_scene_update(scene, action: str, reason: str = ""):
    """Apply property changes immediately where the live session allows it."""
    from . import viewport

    sess = session()
    action = str(action or "").upper()
    if not sess.active:
        try:
            if action in {"OUTPUT", "SURFACE", "WHITEWATER"}:
                # Cache/export playback leaves Blender output objects behind
                # even when no live solver session exists. Keep their material
                # and visibility settings responsive without inventing geometry.
                viewport.refresh_existing_output_settings(scene)
            scene.fluidsim.status = reason or "Settings changed"
        finally:
            _tag_view3d_redraw()
        return

    if action == "REBUILD":
        initialize_session(scene)
    elif action == "RESET":
        reset_simulation(scene)
    elif action == "SYNC":
        sync_scene_to_solver(scene, sess)
    elif action == "SURFACE":
        _apply_realtime_surface_settings(scene, sess)
        viewport.update_outputs(scene, sess, force_surface=True)
    elif action == "WHITEWATER":
        _apply_realtime_whitewater_settings(scene, sess)
        viewport.update_outputs(scene, sess)
    elif action == "OUTPUT":
        viewport.update_outputs(scene, sess, force_surface=True)
    else:
        scene.fluidsim.status = reason or "Settings changed"
    _tag_view3d_redraw()


# ---------------------------------------------------------------------------
# Handlers (interactive playback)
# ---------------------------------------------------------------------------


@persistent
def _on_frame_change(scene, depsgraph):
    session().handle_frame_change(scene, depsgraph)


@persistent
def _on_load_pre(_filepath):
    session().release()


def register_handlers():
    unregister_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if hasattr(bpy.app.handlers, "load_pre"):
        bpy.app.handlers.load_pre.append(_on_load_pre)


def unregister_handlers():
    handler_lists = [bpy.app.handlers.frame_change_post]
    if hasattr(bpy.app.handlers, "load_pre"):
        handler_lists.append(bpy.app.handlers.load_pre)
    for handlers in handler_lists:
        for fn in list(handlers):
            if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51"):
                handlers.remove(fn)
