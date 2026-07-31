"""Scene <-> solver bridge.

Owns the live solver session (module-level, cannot live in ID properties),
translates Blender scene/object data into solver descriptors, and steps the
simulation during playback via frame-change handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from types import SimpleNamespace
import math
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


def capacity_blocking_message(blockers):
    """Render bounded, actionable source-capacity blockers for UI/errors."""
    details = []
    labels = {
        "PARTICLE_CAPACITY": "particles",
        "LEAF_CAPACITY": "leaves",
        "MEMORY_ESTIMATE": "estimated bytes",
        "PARTICLE_DEMAND_UNKNOWN": "particle demand",
    }
    for blocker in blockers:
        code = str(getattr(blocker, "code", "CAPACITY"))
        label = labels.get(code, code.lower().replace("_", " "))
        required = getattr(blocker, "required", None)
        available = getattr(blocker, "available", None)
        if required is None:
            reason = str(getattr(blocker, "message", "") or "").strip()
            suffix = f": {reason}" if reason else ""
            details.append(
                f"{label} is unknown and cannot be admitted safely{suffix}")
        else:
            details.append(f"{label} required {required}, available {available}")
    joined = "; ".join(details) or "source capacity could not be admitted"
    return (
        f"FluidSim source capacity blocked: {joined}. "
        "No partial source emission was committed. Adjust settings or apply "
        "an explicit planner suggestion, then Initialize/Reset and retry."
    )


def reset_memory_blocking_message(capacity_plan):
    """Describe transactional reset peak refusal without implying an OOM."""
    estimate = int(getattr(capacity_plan, "estimated_bytes", 0))
    budget = int(getattr(capacity_plan, "budget_bytes", 0))
    return (
        "FluidSim reset memory safeguard stopped before candidate allocation: "
        "the conservative old-solver + candidate transactional peak is "
        f"{estimate / 1024**3:.1f} GiB, above the safe live budget "
        f"{budget / 1024**3:.1f} GiB. The live solver was preserved. "
        "Apply Safe Settings or Reinitialize explicitly; this estimate is not "
        "an OOM guarantee."
    )


@dataclass
class ScenePlanningSnapshot:
    scene_inputs: SceneInputs
    descriptors: tuple
    extra_plans: tuple
    colliders: tuple
    force_fields: tuple
    height_source: object
    seeds: tuple
    capacity_plan: object


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
        self.capacity_plan_operation = None
        self.capacity_plan_scene_identity = None
        self.capacity_plan_scene_signature = ""
        self.particle_capacity_notice_scene_identity = None
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
        fs = getattr(scene, "fluidsim", None)
        authored_reserve = getattr(fs, "max_particles", None)
        if authored_reserve is None:
            return binding
        return f"{binding}|initial-reserve={int(authored_reserve)}"

    def _set_capacity_plan(
            self, scene, plan, *, operation=None, memory_bypassed=False):
        if operation is None:
            operation = "REINITIALIZE" if self.owns_scene(scene) else "INITIALIZE"
        self.capacity_plan = plan
        self.capacity_plan_memory_bypassed = bool(memory_bypassed)
        self.capacity_plan_operation = str(operation)
        self.capacity_plan_scene_identity = _scene_identity(scene)
        self.capacity_plan_scene_signature = self._capacity_plan_binding_signature(
            scene, plan, self.capacity_plan_operation)
        return plan

    def capacity_plan_for_scene(self, scene):
        """Return only a plan owned by this exact scene and signature."""
        plan = self.capacity_plan
        if plan is None or scene is None:
            return None
        if self.capacity_plan_scene_identity != _scene_identity(scene):
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
                == _scene_identity(scene)):
            self._clear_capacity_plan()

    def particle_capacity_notice_for_scene(self, scene):
        """Render a scene-owned capacity notice from current authoritative state."""
        if (scene is None or self.particle_capacity_notice_scene_identity
                != _scene_identity(scene)):
            return ""
        fs = getattr(scene, "fluidsim", None)
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
        self._clear_capacity_plan()
        self.particle_capacity_notice_scene_identity = None
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

    def _plan_scene(
            self, scene, settings, resources, depsgraph=None, *,
            additional_transaction_bytes=0):
        """Gather one immutable-enough scene snapshot and capacity report."""
        from . import sources
        from ..sim.capacity_planner import (
            DemandQuality, leaf_coords_for_world_bounds, plan_initialization,
            suggest_configurations,
        )
        from ..sim.enums import SourceOperation
        from ..sim.source_compat import (
            descriptors_to_plans, height_descriptor_to_plan,
        )
        from ..sim.source_regions import CompileContext, SourceCompiler

        inputs = SceneInputs()
        inputs.device = settings.device
        sync_generation = inputs.begin_sync()
        descriptors = tuple(inputs.gather_sources(
            scene, depsgraph, capture_first_mesh=True,
            sync_generation=sync_generation))
        extra_plans = tuple(inputs.gather_source_plans(scene, depsgraph))
        colliders = tuple(inputs.gather_colliders(
            scene, depsgraph, sync_generation=sync_generation))
        force_fields = tuple(inputs.gather_force_fields(scene))
        height_source = inputs.gather_height_source(scene)
        seeds = sources.prepare_auto_seeds(scene, depsgraph)

        legacy_plans = tuple(descriptors_to_plans(
            descriptors, session_generation=inputs._session_generation))
        plans = list(extra_plans)
        if getattr(height_source, "enabled", False):
            height_plan = height_descriptor_to_plan(
                height_source, session_generation=inputs._session_generation)
            if not getattr(height_source, "emit_particles", False):
                height_plan = replace(
                    height_plan, operation=SourceOperation.ACTIVATION)
            plans.append(height_plan)

        compiler = SourceCompiler()
        context = CompileContext(
            frame=int(getattr(scene, "frame_current", 0)), substep=0,
            session_generation=inputs._session_generation,
            device=settings.device,
            origin=tuple(float(value) for value in settings.domain_min),
            voxel_size=float(settings.voxel_size),
            domain_min=tuple(float(value) for value in settings.domain_min),
            domain_max=tuple(float(value) for value in settings.domain_max),
            leaf_capacity=int(settings.leaf_capacity),
            particle_capacity=int(settings.max_particles), upload=False,
        )
        batch = compiler.compile(tuple(plans), context)
        footprints = []
        for legacy_plan in legacy_plans:
            legacy_batch = compiler.compile((legacy_plan,), context)
            compiled_footprint = legacy_batch.footprints[0]
            if compiled_footprint.particle_quality is not DemandQuality.UNKNOWN:
                # The shared rasterizer gives a substantially tighter safe
                # upper bound for ordinary analytic/static mesh descriptors.
                footprints.append(compiled_footprint)
                continue
            bounds = getattr(legacy_plan.region, "bounds", None)
            coords = (leaf_coords_for_world_bounds(settings, *bounds())
                      if bounds is not None else ())
            if legacy_plan.operation == SourceOperation.LIQUID:
                target = max(0, min(64, int(math.ceil(
                    float(legacy_plan.payload.target_particles_per_voxel)))))
                if target == 0:
                    coords = ()
                # Animated legacy meshes can intentionally exceed the CPU
                # compiler's staging budget while their accepted runtime path
                # remains GPU-resident. Bound those by clipped AABB leaves;
                # runtime device admission still computes the exact demand.
                particle_demand = len(coords) * 512 * target
                quality = DemandQuality.UPPER_BOUND
            else:
                particle_demand = 0
                quality = DemandQuality.EXACT
            footprints.append(SimpleNamespace(
                stable_id=str(legacy_plan.stable_id),
                particle_demand=particle_demand,
                particle_quality=quality,
                leaf_coords=coords,
                staging_bytes=0,
                reason="legacy device predicate bounded by clipped AABB leaves",
            ))
        footprints.extend(batch.footprints)
        batch_errors = tuple(
            item for item in batch.diagnostics
            if item.severity == "ERROR"
            and item.code not in {"SOURCE_LEAF_CAPACITY"}
            and not any(item in source.region.diagnostics
                        for source in batch.sources))
        if batch_errors:
            footprints.append(SimpleNamespace(
                stable_id="source-batch", particle_demand=None,
                particle_quality=DemandQuality.UNKNOWN, leaf_coords=(),
                staging_bytes=0,
                reason="; ".join(item.message for item in batch_errors),
            ))

        collider_footprints = tuple(SimpleNamespace(
            leaf_coords=leaf_coords_for_world_bounds(
                settings, collider.aabb_min, collider.aabb_max))
            for collider in colliders)
        capacity_plan = plan_initialization(
            settings, tuple(footprints), seeds, collider_footprints, resources,
            additional_transaction_bytes=additional_transaction_bytes)
        capacity_plan = replace(
            capacity_plan,
            suggestions=suggest_configurations(capacity_plan, settings),
        )
        compiler.release()
        return ScenePlanningSnapshot(
            inputs, descriptors, extra_plans, colliders, force_fields,
            height_source, seeds, capacity_plan,
        )

    def refresh_capacity_plan(self, scene, *, operation=None):
        """Recompute the originating operation's warning/options model."""
        from ..sim.memory_safety import estimate_solver_bytes, vram_snapshot

        if operation is None:
            owned_plan = self.capacity_plan_for_scene(scene)
            operation = (
                self.capacity_plan_operation if owned_plan is not None
                else ("REINITIALIZE" if self.owns_scene(scene) else "INITIALIZE")
            )
        settings = build_settings(scene)
        live_solver_bytes = 0
        if self.owns_scene(scene):
            if operation == "RESET":
                runtime_capacity = int(self.solver.particles.capacity)
                settings = replace(settings, max_particles=runtime_capacity)
            # Every owned replacement keeps the authoritative solver resident
            # until its candidate commits, independent of candidate capacity.
            live_solver_bytes = estimate_solver_bytes(self.solver.settings)
        resources = vram_snapshot(settings.device)
        snapshot = self._plan_scene(
            scene, settings, resources,
            additional_transaction_bytes=live_solver_bytes)
        return self._set_capacity_plan(
            scene, snapshot.capacity_plan, operation=operation)

    def _apply_scene_snapshot(
            self, snapshot, *, solver=None, commit_inputs=True):
        solver = solver or self.solver
        if commit_inputs:
            self.scene_inputs = snapshot.scene_inputs
        solver.set_source_batch(snapshot.descriptors, snapshot.extra_plans)
        solver.set_colliders(snapshot.colliders)
        solver.set_force_fields(snapshot.force_fields)
        solver.set_height_source(snapshot.height_source)

    @staticmethod
    def _discard_prepared_solver(solver):
        """Release only an uncommitted candidate, never the live session."""
        if solver is None:
            return
        device = getattr(solver, "device", None)
        release = getattr(solver, "release", None)
        if callable(release):
            release()
        from ..sim.memory_safety import release_cuda_memory
        release_cuda_memory(device)

    def _prepare_solver_replacement(self, scene, settings, snapshot, profiler):
        """Allocate and fully admit an empty solver before live-state mutation."""
        from ..sim.solver import FluidSolver
        from . import sources

        candidate = None
        try:
            candidate = FluidSolver(settings, profiler=profiler)
            self._apply_scene_snapshot(
                snapshot, solver=candidate, commit_inputs=False)
            sources.commit_prepared_seeds(
                SimpleNamespace(solver=candidate), snapshot.seeds)
            seed_result = getattr(
                getattr(candidate, "sourcing", None), "last_apply_result", {})
            if seed_result and not seed_result.get("admitted", True):
                blocker = SimpleNamespace(
                    code="PARTICLE_CAPACITY",
                    required=seed_result.get("required"),
                    available=seed_result.get("available"),
                )
                raise RuntimeError(capacity_blocking_message((blocker,)))
            snippet_errors = candidate.set_custom_snippets(
                build_snippets(scene))
            return candidate, snippet_errors
        except Exception:
            if candidate is not None:
                self._discard_prepared_solver(candidate)
            else:
                # FluidSolver construction can fail after Warp has reserved
                # candidate-owned pooled blocks but before Python receives the
                # object. This path is allocation cleanup, not live-session
                # cleanup; pure planner refusals never reach it.
                from ..sim.memory_safety import release_cuda_memory
                release_cuda_memory(settings.device)
            raise

    def _commit_solver_replacement(
            self, candidate, snapshot, *, profiler, settings):
        """Swap a proven candidate, then retire the formerly authoritative solver."""
        old_solver = self.solver
        old_inputs = self.scene_inputs
        self.solver = candidate
        self.profiler = profiler
        self.device = settings.device
        self.scene_inputs = snapshot.scene_inputs
        if old_inputs is not snapshot.scene_inputs:
            old_inputs.clear()
        if old_solver is not None:
            device = getattr(old_solver, "device", None)
            release = getattr(old_solver, "release", None)
            if callable(release):
                release()
            old_solver = None
            from ..sim.memory_safety import release_cuda_memory
            release_cuda_memory(device)

    @staticmethod
    def _integrity_blockers(capacity_plan):
        return tuple(blocker for blocker in capacity_plan.blockers
                     if blocker.code != "MEMORY_ESTIMATE")

    def initialize(self, scene, *, ignore_memory_safeguard=False):
        """(Re)create the solver without discarding a valid active session."""
        from ..diagnostics.profiler import Profiler

        candidate = None
        candidate_work_started = False
        replacement_committed = False
        snapshot = None
        try:
            fs = scene.fluidsim
            settings = build_settings(scene)
            from ..sim.memory_safety import (
                assess_configuration, estimate_solver_bytes, vram_snapshot,
            )

            # One non-blocking live VRAM sample feeds both the legacy memory
            # assessment and the source-capacity planner. Reinitialization is
            # transactional, so an owned solver remains resident until the
            # fully prepared candidate commits; first initialization adds zero.
            resources = vram_snapshot(settings.device)
            memory_assessment = assess_configuration(settings, resources)
            owns_scene = self.owns_scene(scene)
            plan_operation = "REINITIALIZE" if owns_scene else "INITIALIZE"
            live_solver_bytes = (
                estimate_solver_bytes(self.solver.settings)
                if owns_scene else 0)
            snapshot = self._plan_scene(
                scene, settings, resources,
                additional_transaction_bytes=live_solver_bytes)
            initial_growth = None
            particle_blocker = next((
                blocker for blocker in snapshot.capacity_plan.blockers
                if blocker.code == "PARTICLE_CAPACITY"
                and blocker.required is not None
            ), None)
            if (particle_blocker is not None
                    and bool(getattr(settings, "auto_grow_particles", True))):
                from ..sim.particle_capacity import growth_target

                target = growth_target(
                    int(settings.max_particles), int(particle_blocker.required))
                candidate_settings = replace(settings, max_particles=target)
                candidate_snapshot = self._plan_scene(
                    scene, candidate_settings, resources,
                    additional_transaction_bytes=live_solver_bytes)
                candidate_integrity = self._integrity_blockers(
                    candidate_snapshot.capacity_plan)
                candidate_memory = any(
                    blocker.code == "MEMORY_ESTIMATE"
                    for blocker in candidate_snapshot.capacity_plan.blockers)
                if (not candidate_integrity
                        and (not candidate_memory or ignore_memory_safeguard)):
                    # The explicit operator bypass applies to the selected
                    # grown candidate too; the retained blocked plan remains
                    # marked as bypassed below rather than reported admitted.
                    initial_growth = (int(settings.max_particles), int(target))
                    settings = candidate_settings
                    snapshot = candidate_snapshot
                    memory_assessment = assess_configuration(settings, resources)
            self._set_capacity_plan(
                scene, snapshot.capacity_plan,
                operation=plan_operation,
                memory_bypassed=ignore_memory_safeguard)
            integrity_blockers = self._integrity_blockers(snapshot.capacity_plan)
            if integrity_blockers:
                raise RuntimeError(capacity_blocking_message(integrity_blockers))
            memory_blockers = tuple(
                blocker for blocker in snapshot.capacity_plan.blockers
                if blocker.code == "MEMORY_ESTIMATE")
            if ((not memory_assessment["safe"] or memory_blockers)
                    and not ignore_memory_safeguard):
                estimate = max(
                    int(memory_assessment["estimated_bytes"]),
                    int(snapshot.capacity_plan.estimated_bytes))
                budget = int(snapshot.capacity_plan.budget_bytes)
                estimate_gib = estimate / 1024**3
                budget_gib = budget / 1024**3
                raise RuntimeError(
                    "FluidSim memory safeguard stopped initialization before CUDA "
                    f"allocation (estimated {estimate_gib:.1f} GiB, safe live budget "
                    f"{budget_gib:.1f} GiB). Apply Safe Settings & Retry. "
                    "This estimate is not an OOM guarantee."
                )

            profiler = None
            if fs.profiling_enabled:
                profiler = Profiler(
                    device=settings.device, sync=fs.profiling_sync)
            candidate_work_started = True
            candidate, errors = self._prepare_solver_replacement(
                scene, settings, snapshot, profiler)
            self._commit_solver_replacement(
                candidate, snapshot, profiler=profiler, settings=settings)
            replacement_committed = True
            candidate = None
            self.memory_assessment = memory_assessment
            from ..io import cache

            self.scene_identity = _scene_identity(scene)
            self.scene_key = cache.scene_key(scene)
            self.start_frame = scene.frame_start
            self.cache_written_start = None
            self.cache_written_end = None
            self.cache_settings_records = {}
            self.last_frame = -1
            self.frame_stats = []
            for name, error in errors:
                for pg in fs.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
            from . import viewport

            viewport.ensure_outputs(scene)
            if initial_growth is not None:
                self.last_reported_growth_capacity = int(initial_growth[1])
                fs.status = (
                    f"Solver ready — particle capacity grew "
                    f"{initial_growth[0]:,} → {initial_growth[1]:,}")
            else:
                fs.status = "Solver ready"
            self.memory_assessment = {}
            self.clear_error(scene)
            _capture_applied_state(scene, self)
            return self
        except Exception as exc:
            self.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            if replacement_committed:
                self.release(scene, preserve_error=True)
            elif candidate is not None:
                if self.solver is candidate:
                    self.release(scene, preserve_error=True)
                else:
                    self._discard_prepared_solver(candidate)
            elif candidate_work_started:
                # Constructor exceptions can retain an unassigned candidate
                # cycle through their traceback until frames are cleared above.
                # Collect only because candidate allocation work was attempted.
                import gc
                gc.collect()
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
            elif not self.owns_scene(scene):
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

    def owns_scene(self, scene):
        return self.active and (
            self.scene_identity is None
            or self.scene_identity == _scene_identity(scene))

    def sync_scene(self, scene, depsgraph=None, *, capture_first_mesh=True):
        if not self.owns_scene(scene):
            return
        self.scene_inputs.device = self.solver.device
        sync_generation = self.scene_inputs.begin_sync()
        descriptors = self.scene_inputs.gather_sources(
            scene, depsgraph, capture_first_mesh=capture_first_mesh,
            sync_generation=sync_generation)
        extra_plans = self.scene_inputs.gather_source_plans(scene, depsgraph)
        self.solver.set_source_batch(descriptors, extra_plans)
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
        _refresh_pending_state(scene, self, use_cached=not refresh_pending)
        return _apply_live_settings_safely(scene, self)

    def step(self, scene, *, write_cache=True, timeline_frame=None):
        from . import viewport

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
        fs.stat_whitewater_required = int(getattr(stats, "whitewater_required", 0))
        fs.stat_whitewater_admitted = int(getattr(stats, "whitewater_admitted", 0))
        fs.stat_whitewater_dropped = int(getattr(stats, "whitewater_dropped", 0))
        fs.stat_whitewater_policy = str(getattr(stats, "whitewater_policy", ""))
        fs.stat_whitewater_status = str(getattr(stats, "whitewater_status", "OK"))
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
        whitewater_notice = str(getattr(stats, "whitewater_message", "") or "")
        notices = [notice for notice in (capacity_notice, whitewater_notice) if notice]
        fs.status = (
            f"Frame {self.solver.frame}: {stats.active_particles:,} particles, "
            f"{stats.active_voxels:,} voxels, {stats.step_time * 1000.0:.1f} ms"
            + (f" — {'; '.join(notices)}" if notices else "")
        )
        viewport.update_outputs(scene, self)
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
        scalar = lambda key, default=0: frame_data.get(key, default).item() if hasattr(frame_data.get(key, default), "item") else frame_data.get(key, default)
        fs.stat_whitewater_required = int(scalar("whitewater_required"))
        fs.stat_whitewater_admitted = int(scalar("whitewater_admitted"))
        fs.stat_whitewater_dropped = int(scalar("whitewater_dropped"))
        fs.stat_whitewater_policy = str(scalar("whitewater_policy", "UNAVAILABLE_OLD_CACHE"))
        fs.stat_whitewater_status = str(scalar("whitewater_status", "UNAVAILABLE_OLD_CACHE"))
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
        from . import viewport
        from ..sim.memory_safety import estimate_solver_bytes, vram_snapshot

        if not self.owns_scene(scene):
            return

        # The authored reserve describes the next fresh solver. An active reset
        # must admit against the pool that actually exists, independent of the
        # auto-growth toggle and without shrinking a previously grown pool.
        settings = build_settings(scene)
        runtime_capacity = int(self.solver.particles.capacity)
        settings = replace(settings, max_particles=runtime_capacity)
        resources = vram_snapshot(settings.device)
        # Reset is a replacement transaction: the authoritative solver stays
        # resident until its fully prepared candidate is ready to commit. Feed
        # that old-solver residency through the planner's existing transaction
        # scratch seam so the reported estimate is the actual peak model.
        live_solver_bytes = estimate_solver_bytes(self.solver.settings)
        snapshot = self._plan_scene(
            scene, settings, resources,
            additional_transaction_bytes=live_solver_bytes)
        replacement_growth = None
        particle_blocker = next((
            blocker for blocker in snapshot.capacity_plan.blockers
            if blocker.code == "PARTICLE_CAPACITY"
            and blocker.required is not None
        ), None)
        if (particle_blocker is not None
                and bool(getattr(settings, "auto_grow_particles", True))):
            from ..sim.particle_capacity import growth_target

            target = growth_target(
                runtime_capacity, int(particle_blocker.required))
            candidate_settings = replace(settings, max_particles=target)
            candidate_snapshot = self._plan_scene(
                scene, candidate_settings, resources,
                additional_transaction_bytes=live_solver_bytes)
            candidate_integrity = self._integrity_blockers(
                candidate_snapshot.capacity_plan)
            candidate_memory = any(
                blocker.code == "MEMORY_ESTIMATE"
                for blocker in candidate_snapshot.capacity_plan.blockers)
            if not candidate_integrity and not candidate_memory:
                settings = candidate_settings
                snapshot = candidate_snapshot
                replacement_growth = (runtime_capacity, int(target))

        self._set_capacity_plan(
            scene, snapshot.capacity_plan, operation="RESET")
        integrity_blockers = self._integrity_blockers(snapshot.capacity_plan)
        memory_blockers = tuple(
            blocker for blocker in snapshot.capacity_plan.blockers
            if blocker.code == "MEMORY_ESTIMATE")
        if integrity_blockers or memory_blockers:
            messages = []
            if integrity_blockers:
                messages.append(capacity_blocking_message(integrity_blockers))
            if memory_blockers:
                messages.append(reset_memory_blocking_message(
                    snapshot.capacity_plan))
            message = " ".join(messages)
            self.set_error(scene, message)
            scene.fluidsim.status = "Reset blocked — live solver preserved"
            raise RuntimeError(message)

        candidate = None
        candidate_work_started = False
        replacement_committed = False
        try:
            candidate_work_started = True
            candidate, errors = self._prepare_solver_replacement(
                scene, settings, snapshot, self.profiler)
            self._commit_solver_replacement(
                candidate, snapshot, profiler=self.profiler, settings=settings)
            replacement_committed = True
            candidate = None
            self.start_frame = scene.frame_start
            self.cache_written_start = None
            self.cache_written_end = None
            self.cache_settings_records = {}
            self.last_frame = -1
            self.frame_stats = []
            self.clear_error(scene)
            for name, error in errors:
                for pg in scene.fluidsim.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
            viewport.update_outputs(scene, self)
            _capture_applied_state(scene, self, reset_only=True)
            if replacement_growth is None:
                scene.fluidsim.status = "Simulation reset"
            else:
                self.last_reported_growth_capacity = replacement_growth[1]
                scene.fluidsim.status = (
                    "Simulation reset — particle capacity grew "
                    f"{replacement_growth[0]:,} → {replacement_growth[1]:,}")
        except Exception as exc:
            self.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            if replacement_committed:
                self.release(scene, preserve_error=True)
            elif candidate is not None:
                if self.solver is candidate:
                    self.release(scene, preserve_error=True)
                else:
                    self._discard_prepared_solver(candidate)
            elif candidate_work_started:
                import gc
                gc.collect()
            raise

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
                step_frame(scene, self, write_cache=False)
                self.solver.frame = timeline_frame
                cache.write_frame(scene, self, timeline_frame)
                record_settings = getattr(cache, "settings_record", None)
                if callable(record_settings):
                    self.cache_settings_records[timeline_frame] = record_settings(
                        self.solver.settings)
            cache.write_sequence_metadata(
                base, start, end, source="simulation",
                settings_records=self.cache_settings_records,
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
        if not self.owns_scene(scene):
            try:
                self.initialize(scene)
            except Exception as exc:  # noqa: BLE001 - never break playback
                self.set_error(scene, repr(exc))
                fs.status = f"Solver initialization error: {exc}"
                print(f"FluidSim playback initialization failed: {exc!r}")
                return
        if frame <= self.start_frame or frame < self.last_frame:
            try:
                if not _apply_pending_on_rewind(scene, self):
                    self.reset(scene)
            except Exception as exc:  # noqa: BLE001 - retain pending on failure
                self.set_error(scene, repr(exc))
                # Do not continue stepping an advanced solver from a rewound
                # timeline after its pending reset/reinitialize failed.
                fs.live = False
                fs.status = f"Pending settings apply failed; playback stopped: {exc}"
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
                step_frame(
                    scene, self, refresh_pending=False,
                    timeline_frame=frame - offset)
        except Exception as exc:  # noqa: BLE001 - never break playback
            self.set_error(scene, repr(exc))
            # The property callback this fires is suppressed while the handler
            # is on the stack, so this status remains authoritative.
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
    LIVE = "LIVE"
    SYNC = "SYNC"
    OUTPUT = "OUTPUT"
    SURFACE = "SURFACE"
    WHITEWATER = "WHITEWATER"


_SCENE_UPDATE_REASONS = {
    SceneUpdate.REBUILD: "Solver allocation changed",
    SceneUpdate.RESET: "Simulation settings changed",
    SceneUpdate.LIVE: "Live setting stored; applies on initialization",
    SceneUpdate.SYNC: "Scene source settings changed",
    SceneUpdate.OUTPUT: "Viewport display settings changed",
    SceneUpdate.SURFACE: "Surface display settings changed",
    SceneUpdate.WHITEWATER: "Whitewater display settings changed",
}


_DIRECT_SETTING_FIELDS = (
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
    "ww_bubble_drag", "ww_spray_drag",
})

_LIVE_REFRESH_SETTING_FIELDS = frozenset({"particles_per_voxel"})
_LIVE_SETTING_FIELDS = _HOT_SETTING_FIELDS | _LIVE_REFRESH_SETTING_FIELDS

_RESET_SETTING_FIELDS = frozenset({
    "topology_expand", "open_boundaries", "domain_sdf_walls",
    "spacetime_enabled", "spacetime_reference_mass", "spacetime_seed",
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


def _refresh_pending_state(scene, sess=None, *, use_cached=False):
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


def _capture_applied_state(scene, sess=None, *, reset_only=False):
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
        return _refresh_pending_state(scene, sess or session())
    return state


_SETTING_REGISTRY_VALIDATED = False
_ENUM_SETTING_FIELDS = None


def _enum_setting_fields():
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
        "surface_resolution_mode": ("surface_resolution_mode", SurfaceResolutionMode),
        "surface_update_cadence": ("surface_update_cadence", SurfaceUpdateCadence),
    }
    return _ENUM_SETTING_FIELDS


def _validate_setting_registry():
    global _SETTING_REGISTRY_VALIDATED
    if _SETTING_REGISTRY_VALIDATED:
        return
    from ..sim.solver import SolverSettings

    targets = {_SETTING_ALIASES.get(name, name) for name in _DIRECT_SETTING_FIELDS}
    targets.update(_SETTING_ALIASES.get(target, target)
                   for target in _enum_setting_fields())
    unknown = sorted(targets - set(SolverSettings.__dataclass_fields__))
    if unknown:
        raise ValueError(
            "Scene setting targets not declared by SolverSettings: "
            + ", ".join(unknown))
    _SETTING_REGISTRY_VALIDATED = True


def _apply_scene_settings(fs, settings, fields=None):
    _validate_setting_registry()
    selected = None if fields is None else set(fields)
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
    for target, (source, enum_type) in _enum_setting_fields().items():
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


def refresh_capacity_plan(scene):
    """Refresh actionable capacity options without mutating scene settings."""
    return session().refresh_capacity_plan(scene)


def capacity_plan_for_scene(scene):
    """Return the current scene's owned plan, never another scene's plan."""
    sess = session()
    owned = getattr(sess, "capacity_plan_for_scene", None)
    if callable(owned):
        return owned(scene)
    # Lightweight test adapters predate plan ownership metadata.
    return getattr(sess, "capacity_plan", None)


def capacity_plan_signature_for_scene(scene):
    """Return the scene+authored-settings binding used by capacity dialogs."""
    sess = session()
    signature = getattr(sess, "capacity_plan_signature_for_scene", None)
    if callable(signature):
        return signature(scene)
    plan = getattr(sess, "capacity_plan", None)
    return str(getattr(plan, "scene_signature", ""))


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


def step_frame(
        scene, sess: Session | None = None, *, write_cache: bool = True,
        refresh_pending: bool = True, timeline_frame=None):
    """Reconcile live RNA, then advance through the authoritative step seam.

    Direct callers conservatively refresh pending signatures. Interactive
    playback passes ``refresh_pending=False`` because persistent RNA/depsgraph
    handlers already refreshed the per-scene snapshot.
    """
    sess = sess or session()
    reconciled = sess.prepare_step(scene, refresh_pending=refresh_pending)
    result = sess.step(
        scene, write_cache=write_cache, timeline_frame=timeline_frame)
    if reconciled is False:
        error = sess.current_error(scene)
        if error:
            scene.fluidsim.status = f"Live settings error: {error}"
    return result


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


def _apply_gravity(scene, settings):
    fs = scene.fluidsim
    if not hasattr(fs, "gravity"):
        return
    if (getattr(fs, "use_scene_gravity", False)
            and getattr(scene, "use_gravity", False)):
        settings.gravity = tuple(float(value) for value in scene.gravity)
    else:
        settings.gravity = tuple(float(value) for value in fs.gravity)


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


def _apply_live_settings_safely(scene, sess: Session | None = None):
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


def _apply_pending(scene, requested=None):
    """Apply a scene's pending transition, clearing it only after success."""
    sess = session()
    state = _refresh_pending_state(scene, sess)
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
            sess.initialize(scene)
        else:
            sess.reset(scene)
        _refresh_pending_state(scene, sess)
        return True
    except Exception as exc:
        state.pending_action = action
        if not state.pending_reasons:
            state.pending_reasons = ("Pending settings",)
        state.blocked_reason = str(exc)
        _cache_pending_presentation(scene, state)
        raise


def apply_pending_reinitialize(scene):
    return _apply_pending(scene, SceneUpdate.REBUILD)


def apply_pending_reset(scene):
    return _apply_pending(scene, SceneUpdate.RESET)


def _apply_pending_on_rewind(scene, sess):
    state = _refresh_pending_state(scene, sess)
    if state.pending_action is None:
        return False
    _apply_pending(scene, state.pending_action)
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
    _refresh_pending_state(scene, session())
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
        # Headless/minimal hosts can refresh explicitly later. Never turn an
        # RNA slider callback into synchronous SourceCompiler/GPU work.
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
    """Apply policy live and persist a scene-owned capacity notice marker."""
    sess = session()
    fs = scene.fluidsim
    if sess.owns_scene(scene):
        sess.solver.settings.auto_grow_particles = bool(fs.auto_grow_particles)
        sess.solver.settings.particle_growth_failure_policy = str(
            fs.particle_growth_failure_policy)
        sess.particle_capacity_notice_scene_identity = _scene_identity(scene)
        fs.status = sess.particle_capacity_notice_for_scene(scene)
        # Never leave suggestions computed from the old reserve/policy visible.
        had_scene_plan = (
            sess.capacity_plan is not None
            and sess.capacity_plan_scene_identity == _scene_identity(scene))
        sess.invalidate_capacity_plan_for_scene(scene)
        _refresh_pending_state(scene, sess)
        if had_scene_plan:
            _schedule_capacity_plan_refresh(scene)
    else:
        sess.invalidate_capacity_plan_for_scene(scene)
        sess.particle_capacity_notice_scene_identity = _scene_identity(scene)
        fs.status = sess.particle_capacity_notice_for_scene(scene)
    _tag_view3d_redraw()


def request_scene_update(scene, action: SceneUpdate):
    """Apply property changes immediately where the live session allows it."""
    from . import viewport

    sess = session()
    if not isinstance(action, SceneUpdate):
        action = SceneUpdate(action)
    state = _interaction_state(scene)
    if state.suppress_callbacks:
        return
    if action in {SceneUpdate.RESET, SceneUpdate.REBUILD, SceneUpdate.SYNC}:
        sess.invalidate_capacity_plan_for_scene(scene)
    reason = _SCENE_UPDATE_REASONS[action]
    if not sess.owns_scene(scene):
        try:
            if action in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
                state = _refresh_pending_state(scene, sess)
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
                viewport.refresh_existing_output_settings(scene)
                scene.fluidsim.status = reason
            elif action not in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
                scene.fluidsim.status = reason
        finally:
            _tag_view3d_redraw()
        return

    if action in {SceneUpdate.RESET, SceneUpdate.REBUILD}:
        state = _refresh_pending_state(scene, sess)
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
        _apply_live_settings_safely(scene, sess)
    elif action is SceneUpdate.SYNC:
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
def _on_interaction_state_update(*args):
    """Refresh cached pending UI state after undo, drivers, or scripts."""
    scene = args[0] if args and hasattr(args[0], "fluidsim") else getattr(
        bpy.context, "scene", None)
    if scene is None:
        return
    state = _INTERACTION_STATES.get(_scene_identity(scene))
    if state is not None:
        _refresh_pending_state(scene, session())


@persistent
def _on_load_pre(_filepath):
    session().release()
    _INTERACTION_STATES.clear()


def register_handlers():
    # Deduplicate only -- registering must not release a live session.
    _remove_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if hasattr(bpy.app.handlers, "depsgraph_update_post"):
        bpy.app.handlers.depsgraph_update_post.append(_on_interaction_state_update)
    if hasattr(bpy.app.handlers, "undo_post"):
        bpy.app.handlers.undo_post.append(_on_interaction_state_update)
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
    for name in (
            "depsgraph_update_post", "undo_post", "load_pre",
            "animation_playback_pre"):
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
