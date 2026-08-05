"""Scene <-> solver bridge and Solver Session owner.

The module owns the live Solver Session, direct scene synchronization and
direct stepping. Blender timeline interpretation lives in ``playback`` and
deterministic production cache generation lives in ``cache_bake``.
"""

from __future__ import annotations

from types import SimpleNamespace

import bpy

from .scene_inputs import SceneInputs
from .simulation_admission import capacity_blocking_message

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
        self.cache_written_start = None
        self.cache_written_end = None
        self.cache_settings_records = {}
        self.frame_stats = []
        self.validation_results = []
        self.benchmark_result = None
        self.benchmark_compare = None
        self.error = ""
        self.device = None
        self.memory_assessment = {}
        self.capacity_plan = None
        self.capacity_plan_memory_bypassed = False
        self.last_reported_growth_capacity = 0
        self.scene_inputs = SceneInputs()
        self.scene_identity = None
        # Content digest stamped into cache frames/metadata.
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
        self.cache_settings_records = {}
        self.last_frame = -1
        self.frame_stats = []
        self.scene_inputs.clear()
        self.device = None
        self.last_reported_growth_capacity = 0
        self.scene_identity = None
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

    def _admission(self):
        from . import pending_settings, snippets
        from .simulation_admission import SimulationAdmission

        return SimulationAdmission(
            self, scene_identity=_scene_identity,
            snippet_builder=snippets.build_snippets,
            applied_state_capture=pending_settings.capture_applied_state,
            reset_settings_apply=pending_settings.apply_reset_solver_settings,
        )

    def refresh_capacity_plan(self, scene, depsgraph=None):
        """Recompute admission options without mutating authored settings."""
        return self._admission().refresh_capacity_plan(scene, depsgraph)

    def initialize(self, scene, *, depsgraph=None,
                   ignore_memory_safeguard=False):
        """Admit one coherent Blender scene into a running Solver Session."""
        return self._admission().initialize(
            scene, depsgraph=depsgraph,
            ignore_memory_safeguard=ignore_memory_safeguard)

    def owns_scene(self, scene):
        return self.active and (
            self.scene_identity is None
            or self.scene_identity == _scene_identity(scene))

    def sync_scene(self, scene, depsgraph=None, *, capture_first_mesh=True):
        if not self.owns_scene(scene):
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

    def prepare_step(self, scene, *, refresh_pending=False):
        """Reconcile pending signatures and live settings before every step."""
        if not self.owns_scene(scene):
            return False
        # Playback reuses snapshots refreshed by RNA/depsgraph/undo handlers.
        # Direct scripting callers default to a conservative fresh signature
        # through step_frame(refresh_pending=True), so an edit followed by an
        # immediate step cannot slip past pending detection.
        from . import pending_settings

        pending_settings.refresh_pending_state(
            scene, self, use_cached=not refresh_pending)
        return pending_settings.apply_live_settings_safely(scene, self)

    def step(self, scene, *, write_cache=True, timeline_frame=None,
             publish_outputs=True):

        if not self.owns_scene(scene):
            return None
        fs = scene.fluidsim
        stats = self.solver.step_frame()
        sourcing = getattr(self.solver, "sourcing", None)
        apply_result = (getattr(sourcing, "last_apply_result", {})
                        if sourcing is not None else {})
        capacity_notice = ""
        if apply_result and not bool(apply_result.get("admitted", True)):
            if bool(apply_result.get("deferred", False)):
                growth_status = str(apply_result.get("growth_status", "BLOCKED"))
                required_total = (
                    int(self.solver.particles.host_count)
                    + int(apply_result.get("required") or 0))
                reason = str(apply_result.get("growth_reason") or "")
                if bool(apply_result.get("pause_requested", False)):
                    fs.live = False
                    capacity_notice = (
                        f"Particle growth {growth_status.lower()}; Live paused, "
                        f"solver preserved (need {required_total:,})")
                elif growth_status == "GROWN":
                    capacity_notice = (
                        f"Particle capacity grew to "
                        f"{int(apply_result.get('growth_current') or 0):,}; "
                        "source retries next substep")
                else:
                    capacity_notice = (
                        f"Source emission deferred; solver continued "
                        f"(need {required_total:,}). {reason}")
            else:
                blocker = SimpleNamespace(
                    code="PARTICLE_CAPACITY",
                    required=apply_result.get("required"),
                    available=apply_result.get("available"),
                )
                message = capacity_blocking_message((blocker,))
                self.set_error(scene, message)
                fs.status = f"Source capacity blocked: {message}"
                raise RuntimeError(message)
        if sourcing is not None:
            self.scene_inputs.update_source_status(
                scene, getattr(sourcing, "last_diagnostics", ()), apply_result)
        self.frame_stats.append(stats.as_dict())
        if len(self.frame_stats) > 1000:
            self.frame_stats.pop(0)
        fs.stat_particles = stats.active_particles
        fs.stat_voxels = stats.active_voxels
        fs.stat_whitewater = stats.whitewater_particles
        fs.stat_step_ms = stats.step_time * 1000.0
        fs.stat_substeps = stats.substeps
        fs.stat_cfl_clamped = stats.cfl_substeps_clamped
        growth = getattr(self.solver, "last_particle_growth", None)
        if (growth is not None and getattr(growth, "grew", False)
                and int(growth.current) != self.last_reported_growth_capacity):
            self.last_reported_growth_capacity = int(growth.current)
            capacity_notice = (
                f"Particle capacity grew {int(growth.previous):,} → "
                f"{int(growth.current):,}")
        fs.status = (
            f"Frame {self.solver.frame}: {stats.active_particles:,} particles, "
            f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
            + (f" — {capacity_notice}" if capacity_notice else "")
        )
        if publish_outputs:
            from . import output_publication

            output_publication.update_outputs(scene, self)
        if write_cache and fs.cache_enabled:
            from ..io import cache

            try:
                timeline_frame = int(
                    scene.frame_current if timeline_frame is None
                    else timeline_frame)
                cache.write_frame(scene, self, timeline_frame)
                record_settings = getattr(cache, "settings_record", None)
                if callable(record_settings):
                    self.cache_settings_records[timeline_frame] = record_settings(
                        self.solver.settings)
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
                    settings_records=self.cache_settings_records,
                    scene_key_value=self.scene_key,
                )
            except Exception as exc:  # noqa: BLE001 - cache is supplemental
                print(f"fluidSim cache write failed: {exc!r}")
        return stats

    def reset(self, scene, *, depsgraph=None):
        """Reset through the same coherent admission preflight."""
        return self._admission().reset(scene, depsgraph=depsgraph)

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


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def refresh_capacity_plan(scene, depsgraph=None):
    """Refresh actionable capacity options without mutating scene settings."""
    return session().refresh_capacity_plan(scene, depsgraph)


def initialize_session(
        scene, *, depsgraph=None, ignore_memory_safeguard=False) -> Session:
    """(Re)create the solver from current scene state."""
    return session().initialize(
        scene, depsgraph=depsgraph,
        ignore_memory_safeguard=ignore_memory_safeguard)


def release_session(scene=None):
    session().release(scene)


def sync_scene_to_solver(scene, sess: Session | None = None, depsgraph=None):
    """Push current scene objects (sources/colliders/height) to the solver."""
    (sess or session()).sync_scene(scene, depsgraph)


def step_frame(
        scene, sess: Session | None = None, *, write_cache: bool = True,
        refresh_pending: bool = True, timeline_frame=None,
        publish_outputs: bool = True):
    """Reconcile live RNA, then advance through the authoritative step seam.

    Direct callers conservatively refresh pending signatures. Interactive
    playback passes ``refresh_pending=False`` because persistent RNA/depsgraph
    handlers already refreshed the per-scene snapshot.
    """
    sess = sess or session()
    reconciled = sess.prepare_step(scene, refresh_pending=refresh_pending)
    result = sess.step(
        scene, write_cache=write_cache, timeline_frame=timeline_frame,
        publish_outputs=publish_outputs)
    if reconciled is False:
        error = sess.current_error(scene)
        if error:
            scene.fluidsim.status = f"Live settings error: {error}"
    return result


def reset_simulation(scene, depsgraph=None):
    """Reset solver state and clear outputs (used at playback start)."""
    session().reset(scene, depsgraph=depsgraph)
