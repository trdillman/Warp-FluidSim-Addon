"""Scene <-> solver bridge.

Owns the live solver session (module-level, cannot live in ID properties),
translates Blender scene/object data into solver descriptors, and steps the
simulation during playback via frame-change handlers.
"""

from __future__ import annotations

from enum import Enum
import traceback

import bpy
from mathutils import Vector

from .scene_inputs import SceneInputs

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

# True while _on_frame_change is on the stack. The handler runs inside the
# depsgraph evaluation it was handed, so anything it writes to scene.fluidsim
# can fire an RNA update callback that re-enters this module from inside that
# window -- fs.live = False on the error path is the standing example.
# Same idiom as properties._LIQUID_VIEW_SYNCING: module bool, early return,
# try/finally reset.
_IN_FRAME_HANDLER = False

# Frame Blender will restore once a render job ends; armed by render.py so the
# handler can ignore that one frame change. See arm_render_frame_restore.
_RENDER_RESTORE_FRAME = None

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
        self.cache_written_start = None
        self.cache_written_end = None
        self.frame_stats = []
        self.validation_results = []
        self.benchmark_result = None
        self.benchmark_compare = None
        self.error = ""
        self.device = None
        self.memory_assessment = {}
        self.scene_inputs = SceneInputs()
        # Identity of the scene this session bakes from; stamped into every
        # cache frame and into the sequence metadata.
        self.scene_key = ""

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
        self.cache_written_start = None
        self.cache_written_end = None
        self.last_frame = -1
        self.frame_stats = []
        self.scene_inputs.clear()
        self.device = None
        self.scene_key = ""
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
            from ..sim.memory_safety import assess_configuration, vram_snapshot

            # vram_snapshot, not resource_snapshot: initialize() is reachable
            # from the frame-change handler, and the telemetry fields cost a
            # 50 ms sleep plus an nvidia-smi launch that the capacity decision
            # never reads.
            self.memory_assessment = assess_configuration(
                settings, vram_snapshot(settings.device)
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
            from ..io import cache

            self.scene_key = cache.scene_key(scene)
            self.start_frame = scene.frame_start
            self.sync_scene(scene, capture_first_mesh=False)
            sources.inject_auto_seeds(scene, self)
            errors = self.solver.set_custom_snippets(build_snippets(scene))
            for name, error in errors:
                for pg in fs.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
            # Allocate the output objects/materials here rather than letting
            # the first live frame create them from inside frame_change_post.
            from . import viewport

            viewport.ensure_outputs(scene)
            fs.status = "Solver ready"
            self.memory_assessment = {}
            return self
        except Exception as exc:
            self.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            self.release(scene, preserve_error=True)
            raise

    def set_live_enabled(self, scene, enabled):
        if _IN_FRAME_HANDLER:
            # Reached from the frame-change handler writing fs.live, which
            # fires properties._on_live_changed synchronously. The handler
            # owns its own teardown; re-entering here would run initialize()
            # -- a full GPU allocation -- or touch Blender IDs while the
            # depsgraph the handler was handed is still mid-evaluation.
            return
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

    def sync_scene(self, scene, depsgraph=None, *, capture_first_mesh=True):
        if not self.active:
            return
        self.scene_inputs.device = self.solver.device
        sync_generation = self.scene_inputs.begin_sync()
        self.solver.set_sources(
            self.scene_inputs.gather_sources(
                scene, depsgraph, capture_first_mesh=capture_first_mesh,
                sync_generation=sync_generation))
        extra_plans = self.scene_inputs.gather_source_plans(scene, depsgraph)
        if extra_plans:
            self.solver.set_source_plans(
                tuple(self.solver.sourcing.plans) + tuple(extra_plans))
        self.solver.set_colliders(self.scene_inputs.gather_colliders(
            scene, depsgraph, sync_generation=sync_generation))
        self.solver.set_force_fields(
            self.scene_inputs.gather_force_fields(scene))
        self.solver.set_height_source(self.scene_inputs.gather_height_source(scene))

    def step(self, scene, *, write_cache=True, timeline_frame=None):
        from . import viewport

        if not self.active:
            return None
        fs = scene.fluidsim
        stats = self.solver.step_frame()
        sourcing = getattr(self.solver, "sourcing", None)
        if sourcing is not None:
            self.scene_inputs.update_source_status(
                scene, sourcing.last_diagnostics, sourcing.last_apply_result)
        self.frame_stats.append(stats.as_dict())
        if len(self.frame_stats) > 1000:
            self.frame_stats.pop(0)
        fs.stat_particles = stats.active_particles
        fs.stat_voxels = stats.active_voxels
        fs.stat_whitewater = stats.whitewater_particles
        fs.stat_step_ms = stats.step_time * 1000.0
        fs.stat_substeps = stats.substeps
        fs.stat_cfl_clamped = stats.cfl_substeps_clamped
        fs.status = (
            f"Frame {self.solver.frame}: {stats.active_particles:,} particles, "
            f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
        )
        viewport.update_outputs(scene, self)
        if write_cache and fs.cache_enabled:
            from ..io import cache

            try:
                # Catch-up steps several solver frames inside one frame change,
                # so the caller names the timeline index each one belongs at;
                # keying them all on scene.frame_current would overwrite one
                # index and leave the skipped ones as holes.
                timeline_frame = int(
                    scene.frame_current if timeline_frame is None
                    else timeline_frame)
                cache.write_frame(scene, self, timeline_frame)
                self.cache_written_start = (
                    timeline_frame if self.cache_written_start is None
                    else min(self.cache_written_start, timeline_frame))
                self.cache_written_end = (
                    timeline_frame if self.cache_written_end is None
                    else max(self.cache_written_end, timeline_frame))
                cache.write_sequence_metadata(
                    bpy.path.abspath(fs.cache_path),
                    self.cache_written_start,
                    self.cache_written_end,
                    source="live_simulation",
                    scene_key_value=self.scene_key,
                )
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
        _check_cache_scene_key(scene, base, frame_data)
        viewport.update_outputs_from_cache_frame(scene, frame_data)
        fs.stat_particles = len(frame_data.get("positions", ()))
        fs.stat_whitewater = len(frame_data.get("whitewater", ()))
        fs.stat_voxels = 0
        fs.stat_step_ms = 0.0
        fs.stat_substeps = 0
        # Cached frames carry no substep record, so a stale True here would
        # report a CFL violation against a frame this session never integrated.
        fs.stat_cfl_clamped = False
        fs.status = f"Cache frame {frame}: {fs.stat_particles:,} particles"
        self.clear_error(scene)
        return frame_data

    def reset(self, scene):
        from . import sources, viewport

        if not self.active:
            return
        self.solver.reset()
        self.start_frame = scene.frame_start
        self.cache_written_start = None
        self.cache_written_end = None
        self.last_frame = -1
        self.frame_stats = []
        self.clear_error(scene)
        self.scene_inputs.clear()
        self.sync_scene(scene, capture_first_mesh=False)
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
            cache.write_sequence_metadata(
                base, start, end, source="simulation",
                scene_key_value=self.scene_key)
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
        # Blender drops frames during playback, so a forward jump larger than
        # one is normal rather than exceptional. Taking a single step per frame
        # change ran the solver in slow motion against the timeline and left
        # holes in the cache. Step the whole gap instead -- but only up to a
        # bound, so a long scrub reports a desync instead of hanging Blender
        # inside a frame-change handler.
        gap = (frame - self.last_frame
               if self.last_frame >= self.start_frame else 1)
        bound = max(1, int(getattr(fs, "max_frame_catchup", 8)))
        if gap > bound:
            self.last_frame = frame
            message = (
                f"Timeline jumped {gap} frames, past the {bound}-frame catch-up "
                f"bound. The solver would be out of sync with the timeline. "
                f"Rewind to restart, or raise Max Frame Catch-up.")
            self.set_error(scene, message)
            fs.status = f"Solver desync: {message}"
            return
        self.last_frame = frame
        try:
            for offset in range(gap - 1, -1, -1):
                self.sync_scene(scene, depsgraph)
                self.step(scene, timeline_frame=frame - offset)
        except Exception as exc:  # noqa: BLE001 - never break playback
            self.set_error(scene, repr(exc))
            # The property callback this fires is suppressed while the handler
            # is on the stack (see set_live_enabled), so the status written
            # below is the one the user sees.
            fs.live = False
            fs.status = f"Solver error: {exc}"
            self.release(preserve_error=True)
            print(f"fluidSim step failed: {exc!r}")


_SESSION = Session()
ERROR_SCENE_KEY = "fluidsim_last_error"


def session() -> Session:
    return _SESSION


_CACHE_SCENE_KEY_MESSAGE = (
    "Cache frame {frame} was baked from different scene settings "
    "(cache {baked}, scene {current}). Re-bake the cache, or restore the "
    "settings it was baked with."
)


def _check_cache_scene_key(scene, base, frame_data):
    """Refuse to play a cache baked from a materially different scene.

    Caches written before this check carry no key; they still play, so an
    existing bake is not invalidated by upgrading. Both the sequence metadata
    and the frame's own stamp are checked, so a frame file moved out of its
    sequence is caught too.
    """
    import numpy as np

    from ..io import cache

    frame = int(frame_data.get("frame", -1))
    metadata_key = str((cache.read_sequence_metadata(base) or {}).get("scene_key", ""))
    frame_key = (str(np.asarray(frame_data["scene_key"]).item())
                 if "scene_key" in frame_data else "")
    baked = metadata_key or frame_key
    if not baked:
        return
    current = cache.scene_key(scene)
    for candidate in (metadata_key, frame_key):
        if candidate and candidate != current:
            raise ValueError(_CACHE_SCENE_KEY_MESSAGE.format(
                frame=frame, baked=candidate[:12], current=current[:12]))


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


class SceneUpdate(str, Enum):
    """Kinds of scene edits that can affect an active solver session."""

    REBUILD = "REBUILD"
    RESET = "RESET"
    SYNC = "SYNC"
    OUTPUT = "OUTPUT"
    SURFACE = "SURFACE"
    WHITEWATER = "WHITEWATER"


_SCENE_UPDATE_REASONS = {
    SceneUpdate.REBUILD: "Solver allocation changed",
    SceneUpdate.RESET: "Simulation settings changed",
    SceneUpdate.SYNC: "Scene source settings changed",
    SceneUpdate.OUTPUT: "Viewport display settings changed",
    SceneUpdate.SURFACE: "Surface display settings changed",
    SceneUpdate.WHITEWATER: "Whitewater display settings changed",
}


_DIRECT_SETTING_FIELDS = (
    "voxel_size", "leaf_capacity", "topology_expand", "open_boundaries",
    "domain_sdf_walls",
    "max_particles", "max_whitewater", "time_scale", "cfl",
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

_VECTOR_SETTING_FIELDS = frozenset({
    "directional_force",
    "omnidirectional_force_center",
})

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
            value = getattr(fs, source)
            if source in _VECTOR_SETTING_FIELDS:
                # FloatVectorProperty returns a live RNA/mathutils proxy whose
                # IDProperty owner can be invalidated by later scene edits.
                # SolverSettings must own plain host values across frames.
                value = tuple(float(component) for component in value)
            setattr(settings, target, value)
    for target, (source, enum_type) in enum_fields.items():
        declared_target = _SETTING_ALIASES.get(target, target)
        if (selected is None or source in selected) and hasattr(fs, source):
            setattr(settings, declared_target, enum_type[getattr(fs, source)])


def build_settings(scene):
    """Create SolverSettings from scene properties (imports Warp lazily)."""
    from ..sim.solver import SolverSettings

    fs = scene.fluidsim
    s = SolverSettings()
    from ..preferences import configured_device

    s.device = configured_device()
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
    from ..sim import custom

    code = pg.text_block.as_string() if pg.text_block else ""
    issues = custom.validate_snippet(
        code, pg.run_over, snippet_argument_names(pg))
    errors = [i for i in issues if i.level == "ERROR"]
    pg.is_valid = not errors
    pg.validated_hash = custom.code_hash(code)
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


# REBUILD (full GPU solver reallocation) and RESET (full simulation-state
# discard) are the two expensive request_scene_update actions: a live UI
# slider drag emits one property-update callback per drag tick, and applying
# either action synchronously on every tick measured 174-191ms/tick for
# REBUILD (voxel_size) in a real GUI session with a live solver -- a
# multi-second freeze over a normal drag gesture. Coalesce them onto a single
# one-shot bpy.app.timers callback instead of a bpy.app.timers import at
# module scope, so plain unit tests that stub bpy without app.timers still
# import this module.
_PENDING_REBUILD_RANK = {
    SceneUpdate.RESET: 1,
    SceneUpdate.REBUILD: 2,
}
_pending_rebuild_action = None
_rebuild_flush_scheduled = False


def _flush_pending_rebuild():
    """One-shot timer callback: apply the highest-priority coalesced action."""
    global _pending_rebuild_action, _rebuild_flush_scheduled
    action = _pending_rebuild_action
    _pending_rebuild_action = None
    _rebuild_flush_scheduled = False
    if action is None:
        return None
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return None
    sess = session()
    if not sess.active:
        return None
    try:
        if action is SceneUpdate.REBUILD:
            initialize_session(scene)
        else:
            reset_simulation(scene)
    except Exception as exc:  # noqa: BLE001 - a deferred timer must not raise
        set_error(scene, repr(exc))
        try:
            scene.fluidsim.status = f"Deferred update failed: {exc}"
        except Exception:  # noqa: BLE001
            pass
    _tag_view3d_redraw()
    return None


def _schedule_rebuild(scene, action):
    """Coalesce REBUILD/RESET onto one deferred flush; keep the worst action."""
    global _pending_rebuild_action, _rebuild_flush_scheduled
    current_rank = (
        _PENDING_REBUILD_RANK[_pending_rebuild_action]
        if _pending_rebuild_action is not None else -1)
    if _PENDING_REBUILD_RANK[action] >= current_rank:
        _pending_rebuild_action = action
    scene.fluidsim.status = _SCENE_UPDATE_REASONS[action]
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is None or not hasattr(timers, "register"):
        # No real Blender timer API available (e.g. a bare unit-test stub):
        # fall back to the previous synchronous behavior rather than lose
        # the update entirely.
        _flush_pending_rebuild()
        return
    if not _rebuild_flush_scheduled:
        _rebuild_flush_scheduled = True
        timers.register(_flush_pending_rebuild, first_interval=0.05)


def request_scene_update(scene, action: SceneUpdate):
    """Apply property changes immediately where the live session allows it."""
    from . import viewport

    sess = session()
    if not isinstance(action, SceneUpdate):
        action = SceneUpdate(action)
    reason = _SCENE_UPDATE_REASONS[action]
    if not sess.active:
        try:
            if action in {
                SceneUpdate.OUTPUT,
                SceneUpdate.SURFACE,
                SceneUpdate.WHITEWATER,
            }:
                # Cache/export playback leaves Blender output objects behind
                # even when no live solver session exists. Keep their material
                # and visibility settings responsive without inventing geometry.
                viewport.refresh_existing_output_settings(scene)
            scene.fluidsim.status = reason
        finally:
            _tag_view3d_redraw()
        return

    if action in _PENDING_REBUILD_RANK:
        _schedule_rebuild(scene, action)
        return
    if action is SceneUpdate.SYNC:
        sync_scene_to_solver(scene, sess)
    elif action is SceneUpdate.SURFACE:
        _apply_realtime_surface_settings(scene, sess)
        viewport.update_outputs(scene, sess, force_surface=True)
    elif action is SceneUpdate.WHITEWATER:
        _apply_realtime_whitewater_settings(scene, sess)
        viewport.update_outputs(scene, sess)
    elif action is SceneUpdate.OUTPUT:
        viewport.update_outputs(scene, sess, force_surface=True)
    _tag_view3d_redraw()


# ---------------------------------------------------------------------------
# Handlers (interactive playback)
# ---------------------------------------------------------------------------


def ensure_session_before_frames(scene, reason):
    """Build the solver before the first frame change of a playback or render.

    Without this the first live frame reaches Session.initialize() from inside
    frame_change_post -- a full CUDA allocation plus object/collection creation
    against a depsgraph that is still being evaluated. animation_playback_pre
    and render_init both fire once, ahead of that frame change, and outside the
    frame-evaluation cycle.

    Deliberately a no-op on a bare timeline scrub: nothing fires ahead of
    frame_change_pre there, so the lazy-init path in handle_frame_change stays
    the fallback and keeps working exactly as before.
    """
    fs = getattr(scene, "fluidsim", None) if scene is not None else None
    if fs is None or not getattr(fs, "live", False):
        # Not live -- must not allocate VRAM just because playback started.
        return
    if getattr(fs, "cache_playback_enabled", False):
        # Cache playback never builds a solver; handle_frame_change returns
        # ahead of the lazy-init path for it too.
        return
    sess = session()
    if sess.active:
        return
    try:
        sess.initialize(scene)
    except Exception as exc:  # noqa: BLE001 - never break playback or a render
        # Same wording as the lazy-init path below: panels.py string-matches the
        # message to offer the memory-safeguard recovery buttons.
        set_error(scene, repr(exc))
        fs.status = f"Solver initialization error: {exc}"
        print(f"FluidSim {reason} initialization failed: {exc!r}")


@persistent
def _on_playback_pre(scene, *_args):
    ensure_session_before_frames(scene, "playback")


# The render-side call lives in render._on_render_init: that module clears every
# add-on callback off bpy.app.handlers.render_init when it registers, so a second
# handler here would be silently removed depending on registration order.


def arm_render_frame_restore(frame):
    """Ignore the frame change Blender fires to restore the pre-render frame.

    Blender jumps back to the frame the render started from once the job ends.
    That frame change is Blender's own bookkeeping, not a user gesture, but it
    reaches the frame handler like any other: after a 1-4 animation render from
    frame 5 it stepped the solver one more time and wrote that state into the
    cache at index 5 -- a frame the user never simulated. One-shot and matched
    on the exact frame, so a user scrub that happens to come next is untouched.
    """
    global _RENDER_RESTORE_FRAME
    _RENDER_RESTORE_FRAME = int(frame)


@persistent
def _on_frame_change(scene, depsgraph):
    global _IN_FRAME_HANDLER
    global _RENDER_RESTORE_FRAME
    restore_frame, _RENDER_RESTORE_FRAME = _RENDER_RESTORE_FRAME, None
    if (restore_frame is not None and scene is not None
            and scene.frame_current == restore_frame):
        # Keep the bookkeeping level with the timeline so the next real frame
        # change steps once per frame instead of catching up over the render.
        session().last_frame = restore_frame
        return
    if _IN_FRAME_HANDLER:
        # A nested frame change (scene.frame_set from inside the handler, or
        # another add-on's handler driving the timeline) would step the solver
        # recursively against a depsgraph the outer call still owns.
        return
    _IN_FRAME_HANDLER = True
    try:
        session().handle_frame_change(scene, depsgraph)
    finally:
        _IN_FRAME_HANDLER = False


@persistent
def _on_load_pre(_filepath):
    session().release()


def register_handlers():
    # Deduplicate only -- registering must not release a live session.
    _remove_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if hasattr(bpy.app.handlers, "load_pre"):
        bpy.app.handlers.load_pre.append(_on_load_pre)
    # Build the solver ahead of the first frame change instead of from inside
    # frame_change_post. The render side is handled in render._on_render_init.
    if hasattr(bpy.app.handlers, "animation_playback_pre"):
        bpy.app.handlers.animation_playback_pre.append(_on_playback_pre)


def _remove_handlers():
    handler_lists = [bpy.app.handlers.frame_change_post]
    # render_init is deliberately absent: render.py owns that list, and clearing
    # it here would drop render._on_render_init depending on register order.
    for name in ("load_pre", "animation_playback_pre"):
        if hasattr(bpy.app.handlers, name):
            handler_lists.append(getattr(bpy.app.handlers, name))
    for handlers in handler_lists:
        for fn in list(handlers):
            if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51"):
                handlers.remove(fn)


def unregister_handlers():
    # Reload Scripts reaches this without going through unregister(), so
    # without the release the whole GPU allocation is orphaned with no owner
    # left to free it short of restarting Blender. Handlers come off first:
    # releasing while _on_frame_change is still registered would let a frame
    # change re-enter a half-torn-down session.
    _remove_handlers()
    session().release()
