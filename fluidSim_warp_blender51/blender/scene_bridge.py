"""Scene <-> solver bridge and Solver Session owner.

The module owns the live Solver Session, direct scene synchronization and
direct stepping. Blender timeline interpretation lives in ``playback`` and
deterministic production cache generation lives in ``cache_bake``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import bpy

from .scene_access import (
    active_domain_settings,
    scene_get,
    scene_identity,
    scene_set,
)
from ..sim.effects.runtime import EffectExecutionError
from .scene_inputs import SceneInputs
from .simulation_admission import capacity_blocking_message, narrow_band_refusal

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)


# Session state


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
        # The narrow-band refusal the last scene sync reported, so the sync
        # that finds the conflict removed can retire exactly that error.
        self.sync_refusal = ""
        self.device = None
        self.capacity_plan = None
        self.capacity_plan_memory_bypassed = False
        self.capacity_plan_operation = None
        self.capacity_plan_scene_identity = None
        self.capacity_plan_scene_signature = ""
        self.particle_capacity_notice_scene_identity = None
        self.last_reported_growth_capacity = 0
        self.admitted_effects = ()
        self.scene_inputs = SceneInputs()
        self.scene_identity = None
        self._admission_owner = None
        # Content digest stamped into cache frames/metadata.
        self.scene_key = ""

    @property
    def active(self):
        return self.solver is not None

    @property
    def collider_mesh_cache_stats(self):
        return self.scene_inputs.collider_mesh_cache_stats

    def _clear_capacity_plan(self):
        self.capacity_plan = None
        self.capacity_plan_memory_bypassed = False
        self.capacity_plan_operation = None
        self.capacity_plan_scene_identity = None
        self.capacity_plan_scene_signature = ""

    @staticmethod
    def _capacity_plan_binding_signature(scene, plan, operation):
        plan_signature = str(getattr(plan, "scene_signature", ""))
        binding = f"{plan_signature}|operation={operation}"
        # ``max_particles`` is domain-owned: on a migrated scene the scene
        # group's copy is the frozen pre-migration ghost, so a re-authored
        # reserve would never change the signature and the plan would never
        # rebind.
        fs = active_domain_settings(scene)
        authored_reserve = getattr(fs, "max_particles", None)
        if authored_reserve is None:
            return binding
        return f"{binding}|initial-reserve={int(authored_reserve)}"

    def set_capacity_plan(
            self, scene, plan, *, operation=None, memory_bypassed=False):
        if operation is None:
            operation = "REINITIALIZE" if self.owns_scene(scene) else "INITIALIZE"
        self.capacity_plan = plan
        self.capacity_plan_memory_bypassed = bool(memory_bypassed)
        self.capacity_plan_operation = str(operation)
        self.capacity_plan_scene_identity = scene_identity(scene)
        self.capacity_plan_scene_signature = self._capacity_plan_binding_signature(
            scene, plan, self.capacity_plan_operation)
        return plan

    def capacity_plan_for_scene(self, scene):
        plan = self.capacity_plan
        if plan is None or scene is None:
            return None
        if self.capacity_plan_scene_identity != scene_identity(scene):
            return None
        if self.capacity_plan_scene_signature != self._capacity_plan_binding_signature(
                scene, plan, self.capacity_plan_operation):
            return None
        return plan

    def capacity_plan_signature_for_scene(self, scene):
        if self.capacity_plan_for_scene(scene) is None:
            return ""
        return self.capacity_plan_scene_signature

    def invalidate_capacity_plan_for_scene(self, scene):
        if (scene is not None and self.capacity_plan_scene_identity
                == scene_identity(scene)):
            self._clear_capacity_plan()

    def particle_capacity_notice_for_scene(self, scene):
        if (scene is None or self.particle_capacity_notice_scene_identity
                != scene_identity(scene)):
            return ""
        # Domain-owned: the notice must compare the runtime capacity against
        # the reserve the user actually authored, not the migrated ghost.
        fs = active_domain_settings(scene)
        authored = getattr(fs, "max_particles", None)
        if authored is None:
            return ""
        authored = int(authored)
        if self.owns_scene(scene):
            runtime = int(self.solver.particles.capacity)
            if authored < runtime:
                return (
                    f"Active pool {runtime:,}; Initial Reserve {authored:,}. "
                    "Reset keeps the active pool; Reinitialize applies the smaller "
                    "reserve. The active pool never shrinks automatically.")
            if authored > runtime:
                return (
                    f"Active pool {runtime:,}; Initial Reserve {authored:,}. "
                    "Reset uses the active pool; Reinitialize applies the larger "
                    "reserve.")
            return (
                f"Active pool {runtime:,}; Initial Reserve {authored:,}. "
                "Growth policy applies live; Reinitialize recreates the pool.")
        from ..sim.particle_capacity import MAX_AUTO_PARTICLES
        return (
            f"Initial Reserve {authored:,} will be allocated on Initialize; "
            f"the auto-growth hard limit remains {MAX_AUTO_PARTICLES:,}.")

    def begin_run(self, start_frame=1, *, reported_growth_capacity=None):
        """Start one run's bookkeeping afresh.

        The frame cursor, cache coverage and frame statistics describe a single
        run of the solver. Every path that installs or retires a solver begins
        a new run here, so these fields cannot drift apart. The growth notice
        marker is only restarted when ``reported_growth_capacity`` is given.
        """
        self.start_frame = int(start_frame)
        self.cache_written_start = None
        self.cache_written_end = None
        self.cache_settings_records = {}
        self.last_frame = -1
        self.frame_stats = []
        if reported_growth_capacity is not None:
            self.last_reported_growth_capacity = int(reported_growth_capacity)

    def bind_scene(self, identity, key):
        """Bind the session to the scene identity and content digest it owns."""
        self.scene_identity = identity
        self.scene_key = key

    def install_solver(
            self, candidate, *, profiler, device, scene_inputs, admitted_effects=()):
        """Replacement Cutover: the single swap of the authoritative solver.

        ``candidate`` has already passed admission. The retired solver, its
        device memory and the superseded scene inputs are released here.
        ``admitted_effects`` are the effect instances the candidate was
        admitted with.
        """
        old_solver = self.solver
        old_inputs = self.scene_inputs
        self.solver = candidate
        self.profiler = profiler
        self.device = device
        self.scene_inputs = scene_inputs
        self.admitted_effects = admitted_effects
        if old_inputs is not scene_inputs:
            old_inputs.clear()
        if old_solver is not None:
            retired_device = getattr(old_solver, "device", None)
            release = getattr(old_solver, "release", None)
            if callable(release):
                release()
            from ..sim.memory_safety import release_cuda_memory
            release_cuda_memory(retired_device)

    def _clear_runtime(self):
        solver = self.solver
        device = getattr(solver, "device", None) or self.device
        self.solver = None
        if solver is not None:
            if hasattr(solver, "release"):
                solver.release()
        self.profiler = None
        self.begin_run(reported_growth_capacity=0)
        self.scene_inputs.clear()
        self.device = None
        self.bind_scene(None, "")
        self.admitted_effects = ()
        self.sync_refusal = ""
        self._clear_capacity_plan()
        self.particle_capacity_notice_scene_identity = None
        del solver
        from ..sim.memory_safety import release_cuda_memory
        release_cuda_memory(device)

    def set_error(self, scene, message):
        self.error = str(message)
        scene_set(scene, ERROR_SCENE_KEY, self.error)

    def clear_error(self, scene=None):
        self.error = ""
        if scene is not None:
            scene_set(scene, ERROR_SCENE_KEY, "")

    def current_error(self, scene):
        return str(scene_get(scene, ERROR_SCENE_KEY, "") or self.error or "")

    def release(self, scene, *, preserve_error=False):
        """Release only an inactive session or the runtime owned by ``scene``."""
        if self.active and not self.owns_scene(scene):
            return False
        self._clear_runtime()
        if not preserve_error:
            self.clear_error(scene)
        return True

    def force_release(self, *, preserve_error=False, scene=None):
        """Explicitly tear down the singleton for global or fatal lifecycle paths."""
        self._clear_runtime()
        if not preserve_error:
            self.clear_error(scene)
        return True

    def _admission(self):
        if self._admission_owner is None:
            from . import pending_settings
            from .simulation_admission import SimulationAdmission

            def build_effect_stack(scene):
                # Host-Python contract tests intentionally provide a minimal
                # bpy seam. Custom Effects are a Blender adapter and are only
                # importable when Blender's property API is present.
                if not hasattr(bpy, "props"):
                    return ()
                from . import effects

                return effects.build_effect_stack_for_admission(scene)

            def publish_effect_stack(scene, instances):
                if not hasattr(bpy, "props"):
                    return
                from . import effects

                effects.publish_admitted_effects(scene, instances)

            self._admission_owner = SimulationAdmission(
                self, scene_identity=scene_identity,
                effect_builder=build_effect_stack,
                effect_publisher=publish_effect_stack,
                applied_state_capture=pending_settings.capture_applied_state,
                reset_settings_apply=pending_settings.apply_reset_solver_settings,
            )
        return self._admission_owner

    def refresh_capacity_plan(self, scene, depsgraph=None, *, operation=None):
        """Recompute admission options without mutating authored settings."""
        return self._admission().refresh_capacity_plan(
            scene, depsgraph, operation=operation)

    def initialize(self, scene, *, depsgraph=None,
                   ignore_memory_safeguard=False):
        """Admit one coherent Blender scene into a running Solver Session."""
        return self._admission().initialize(
            scene, depsgraph=depsgraph,
            ignore_memory_safeguard=ignore_memory_safeguard)

    def owns_scene(self, scene):
        return bool(
            self.active
            and scene is not None
            and self.scene_identity is not None
            and self.scene_identity == scene_identity(scene)
        )

    def sync_scene(self, scene, depsgraph=None, *, capture_first_mesh=True):
        if not self.owns_scene(scene):
            return
        self.scene_inputs.device = self.solver.device
        sync_generation = self.scene_inputs.begin_sync()
        # A narrow-band run was admitted without features that keep
        # per-particle interior state. Playback syncs every frame, so a
        # source edit admission would refuse (a Scalar field source) must not
        # reach the solver here: the admitted sources stay installed and the
        # refusal is reported until the conflict is removed or Reset re-admits.
        refusal = narrow_band_refusal(
            self.solver.settings, getattr(scene, "objects", ()))
        if refusal:
            self.sync_refusal = refusal
            self.set_error(scene, refusal)
            scene.fluidsim.status = f"Scene sync refused: {refusal}"
        else:
            if self.sync_refusal:
                if self.current_error(scene) == self.sync_refusal:
                    self.clear_error(scene)
                self.sync_refusal = ""
            self.solver.set_source_batch(
                self.scene_inputs.gather_sources(
                    scene, depsgraph, capture_first_mesh=capture_first_mesh,
                    sync_generation=sync_generation),
                self.scene_inputs.gather_source_plans(scene, depsgraph))
        self.solver.set_colliders(self.scene_inputs.gather_colliders(
            scene, depsgraph, sync_generation=sync_generation))
        self.solver.set_force_fields(
            self.scene_inputs.gather_force_fields(scene))
        self.solver.set_height_source(self.scene_inputs.gather_height_source(scene))

    def prepare_step(self, scene, *, refresh_pending=False, apply_live=True):
        """Reconcile pending signatures and live settings before every step.

        Production Bake passes ``apply_live=False``: its solver keeps the
        settings it was admitted with, which are the cache's identity, so an
        edit made mid-bake cannot change later frames under that identity.
        """
        if not self.owns_scene(scene):
            return False
        # Playback reuses snapshots refreshed by RNA/depsgraph/undo handlers.
        # Direct scripting callers default to a conservative fresh signature
        # through step_frame(refresh_pending=True), so an edit followed by an
        # immediate step cannot slip past pending detection.
        from . import pending_settings

        pending_settings.refresh_pending_state(
            scene, self, use_cached=not refresh_pending)
        if not apply_live:
            return True
        return pending_settings.apply_live_settings_safely(scene, self)

    def step(self, scene, *, write_cache=True, timeline_frame=None,
             publish_outputs=True):

        if not self.owns_scene(scene):
            return None
        fs = scene.fluidsim
        # Custom Effect frame windows are authored in timeline frames; tell the
        # solver which one this step produces so preview and bake (whose solver
        # frame counters start from different origins) agree (decision D6).
        self.solver.effect_frame = int(
            scene.frame_current if timeline_frame is None else timeline_frame)
        effects = None
        if hasattr(bpy, "props") and getattr(scene, "fluidsim_effects", None) is not None:
            from . import effects  # animated effect values (B66, lane E)

            effects.sync_animated_effect_bindings(scene, self)
        try:
            stats = self.solver.step_frame()
        except EffectExecutionError as exc:
            if effects is not None:
                # Before any caller releases the solver and its runtime error.
                effects.publish_runtime_error(scene, exc)
            raise
        if effects is not None:
            effects.publish_runtime_telemetry(
                scene, getattr(stats, "custom_effects", None) or ())
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
        fs.stat_liquid_voxels = int(getattr(stats, "liquid_voxels", 0))
        fs.stat_whitewater = stats.whitewater_particles
        fs.stat_whitewater_foam = int(getattr(stats, "whitewater_foam", 0))
        fs.stat_whitewater_bubbles = int(
            getattr(stats, "whitewater_bubbles", 0))
        fs.stat_whitewater_spray = int(getattr(stats, "whitewater_spray", 0))
        fs.stat_whitewater_invalid_types = int(
            getattr(stats, "whitewater_invalid_types", 0))
        fs.stat_whitewater_required = int(
            getattr(stats, "whitewater_required", 0))
        fs.stat_whitewater_admitted = int(
            getattr(stats, "whitewater_admitted", 0))
        fs.stat_whitewater_dropped = int(
            getattr(stats, "whitewater_dropped", 0))
        fs.stat_whitewater_policy = str(
            getattr(stats, "whitewater_policy", ""))
        fs.stat_whitewater_status = str(
            getattr(stats, "whitewater_status", "OK"))
        fs.stat_step_ms = stats.step_time * 1000.0
        fs.stat_substeps = stats.substeps
        fs.stat_cfl_clamped = stats.cfl_substeps_clamped
        capillary = float(getattr(stats, "capillary_dt", float("inf")))
        fs.stat_capillary_dt = 0.0 if not math.isfinite(capillary) else capillary
        fs.stat_capillary_substeps_required = int(
            getattr(stats, "capillary_substeps_required", 1))
        fs.stat_capillary_clamped = bool(
            getattr(stats, "capillary_substeps_clamped", False))
        fs.stat_substep_limiter = str(
            getattr(stats, "substep_limiter", "MINIMUM"))
        fs.stat_surface_tension_active_voxels = int(
            getattr(stats, "surface_tension_active_voxels", 0))
        fs.stat_surface_tension_max_curvature = float(
            getattr(stats, "surface_tension_max_curvature", 0.0))
        fs.stat_surface_tension_max_acceleration = float(
            getattr(stats, "surface_tension_max_acceleration", 0.0))
        fs.stat_surface_tension_temporary_bytes = int(
            getattr(stats, "surface_tension_temporary_bytes", 0))
        fs.stat_surface_tension_submission_ms = float(
            getattr(stats, "surface_tension_submission_seconds", 0.0)) * 1000.0
        fs.stat_surface_tension_status = str(
            getattr(stats, "surface_tension_status", "DISABLED"))
        fs.stat_surface_tension_message = str(
            getattr(stats, "surface_tension_message", ""))
        fs.stat_native_solver_message = str(
            getattr(stats, "native_solver_message", ""))
        fs.stat_learned_guess_message = str(
            getattr(stats, "learned_guess_message", ""))
        growth = getattr(self.solver, "last_particle_growth", None)
        if (growth is not None and getattr(growth, "grew", False)
                and int(growth.current) != self.last_reported_growth_capacity):
            self.last_reported_growth_capacity = int(growth.current)
            capacity_notice = (
                f"Particle capacity grew {int(growth.previous):,} → "
                f"{int(growth.current):,}")
        whitewater_notice = str(
            getattr(stats, "whitewater_message", "") or "")
        notices = [
            notice for notice in (capacity_notice, whitewater_notice) if notice]
        fs.status = (
            f"Frame {self.solver.frame}: {stats.active_particles:,} particles, "
            f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
            + (f" — {'; '.join(notices)}" if notices else "")
        )
        if publish_outputs:
            from . import output_publication

            publication = output_publication.update_outputs(scene, self)
            warning = str(getattr(publication, "warning", "") or "")
            if warning:
                self.set_error(scene, warning)
                fs.status = f"{fs.status} — {warning}"
        return stats

    def reset(self, scene, *, depsgraph=None):
        """Reset through the same coherent admission preflight."""
        return self._admission().reset(scene, depsgraph=depsgraph)

_SESSION = Session()
ERROR_SCENE_KEY = "fluidsim_last_error"


def session() -> Session:
    return _SESSION


def set_error(scene, message: str):
    """Persist the latest simulation error for panel-visible recovery."""
    session().set_error(scene, message)


def clear_error(scene):
    session().clear_error(scene)


def current_error(scene) -> str:
    return session().current_error(scene)


# Settings translation


# Session lifecycle


def refresh_capacity_plan(scene, depsgraph=None, *, operation=None):
    """Refresh actionable capacity options without mutating scene settings."""
    return session().refresh_capacity_plan(
        scene, depsgraph, operation=operation)


def capacity_plan_for_scene(scene):
    sess = session()
    getter = getattr(sess, "capacity_plan_for_scene", None)
    return getter(scene) if callable(getter) else getattr(sess, "capacity_plan", None)


def capacity_plan_signature_for_scene(scene):
    sess = session()
    getter = getattr(sess, "capacity_plan_signature_for_scene", None)
    if callable(getter):
        return getter(scene)
    plan = getattr(sess, "capacity_plan", None)
    return str(getattr(plan, "scene_signature", ""))


def initialize_session(
        scene, *, depsgraph=None, ignore_memory_safeguard=False) -> Session:
    """(Re)create the solver from current scene state."""
    return session().initialize(
        scene, depsgraph=depsgraph,
        ignore_memory_safeguard=ignore_memory_safeguard)


def release_session(scene=None):
    if scene is None:
        scene = getattr(bpy.context, "scene", None)
    return session().release(scene)


def force_release_session(*, preserve_error=False, scene=None):
    return session().force_release(
        preserve_error=preserve_error, scene=scene)


def sync_scene_to_solver(scene, sess: Session | None = None, depsgraph=None):
    """Push current scene objects (sources/colliders/height) to the solver."""
    (sess or session()).sync_scene(scene, depsgraph)


def step_frame(
        scene, sess: Session | None = None, *, write_cache: bool = True,
        refresh_pending: bool = True, timeline_frame=None,
        publish_outputs: bool = True, apply_live_settings: bool = True):
    """Reconcile live RNA, then advance through the authoritative step seam.

    Direct callers conservatively refresh pending signatures. Interactive
    playback passes ``refresh_pending=False`` because persistent RNA/depsgraph
    handlers already refreshed the per-scene snapshot. Production Bake passes
    ``apply_live_settings=False`` (see ``Session.prepare_step``).
    """
    sess = sess or session()
    reconciled = sess.prepare_step(
        scene, refresh_pending=refresh_pending, apply_live=apply_live_settings)
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
