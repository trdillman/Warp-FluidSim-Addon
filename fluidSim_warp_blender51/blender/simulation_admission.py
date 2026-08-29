"""Blender-owned authored-scene to running-solver admission transaction.

Admission performs every non-destructive gather, compile, memory, capacity and
integrity check against one :class:`AdmissionSnapshot` before Replacement
Cutover. Replacement candidates are staged while the current solver remains
authoritative, with their peak residency included in admission planning. Only
a failure after the authority swap leaves a clean, inactive Solver Session.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import math
from types import SimpleNamespace
import traceback

from .scene_inputs import SceneInputs
from .solver_settings import build_settings, domain_transform


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
    estimate = int(getattr(capacity_plan, "estimated_bytes", 0))
    budget = int(getattr(capacity_plan, "budget_bytes", 0))
    return (
        "Reset replacement cannot be prepared while the live solver remains "
        f"authoritative (old-solver + candidate estimate {estimate / 1024**3:.1f} "
        f"GiB, safe budget {budget / 1024**3:.1f} GiB). The live solver was "
        "preserved; review capacity options or Reinitialize explicitly."
    )


def _pointer(value):
    as_pointer = getattr(value, "as_pointer", None)
    return int(as_pointer()) if callable(as_pointer) else id(value)


def _depsgraph_generation(depsgraph):
    if depsgraph is None:
        return None
    updates = []
    for update in getattr(depsgraph, "updates", ()) or ():
        updated = getattr(update, "id", None)
        updated = getattr(updated, "original", updated)
        if updated is not None:
            updates.append(_pointer(updated))
    return (_pointer(depsgraph), tuple(sorted(updates)))


def _authoritative_depsgraph(depsgraph):
    """Acquire exactly one evaluated dependency graph for this admission."""
    if depsgraph is not None:
        return depsgraph
    import bpy

    return bpy.context.evaluated_depsgraph_get()


@dataclass
class AdmissionSnapshot:
    """One owned capture used by an entire Simulation Admission attempt."""

    scene_identity: object
    scene_key: str
    frame: int
    frame_start: int
    depsgraph_generation: object
    configured_device: str
    domain_transform: object
    domain_min: tuple
    domain_max: tuple
    settings: object
    profiling_enabled: bool
    profiling_sync: bool
    scene_inputs: SceneInputs | None
    descriptors: tuple
    extra_plans: tuple
    colliders: tuple
    force_fields: tuple
    height_source: object
    seeds: tuple
    footprints: tuple
    collider_footprints: tuple
    resources: object
    capacity_plan: object
    effects: tuple = ()
    depsgraph: object = None

    def assert_evaluation_valid(self, scene):
        if int(getattr(scene, "frame_current", self.frame)) != self.frame:
            raise RuntimeError(
                "FluidSim admission snapshot invalidated while gathering scene")
        if _depsgraph_generation(self.depsgraph) != self.depsgraph_generation:
            raise RuntimeError(
                "FluidSim admission depsgraph generation invalidated while "
                "gathering scene")

    def release_transients(self):
        inputs, self.scene_inputs = self.scene_inputs, None
        if inputs is not None:
            inputs.clear()

    def transfer_scene_inputs(self):
        inputs, self.scene_inputs = self.scene_inputs, None
        return inputs


def _integrity_blockers(capacity_plan):
    return tuple(blocker for blocker in capacity_plan.blockers
                 if blocker.code != "MEMORY_ESTIMATE")


def _plan_capacity(
        settings, footprints, seeds, collider_footprints, resources, *,
        additional_transaction_bytes=0):
    from ..sim.capacity_service import PreparedCapacityInput, assess

    prepared = PreparedCapacityInput(
        tuple(footprints), tuple(seeds), tuple(collider_footprints))
    return assess(
        settings, prepared, resources,
        additional_transaction_bytes=additional_transaction_bytes,
    ).plan


def _legacy_footprint(legacy_plan, compiler, context, settings):
    from ..sim.capacity_planner import DemandQuality, leaf_coords_for_world_bounds
    from ..sim.enums import SourceOperation

    legacy_batch = compiler.compile((legacy_plan,), context)
    compiled = legacy_batch.footprints[0]
    if compiled.particle_quality is not DemandQuality.UNKNOWN:
        return compiled
    bounds = getattr(legacy_plan.region, "bounds", None)
    coords = (leaf_coords_for_world_bounds(settings, *bounds())
              if bounds is not None else ())
    if legacy_plan.operation == SourceOperation.LIQUID:
        target = max(0, min(64, int(math.ceil(
            float(legacy_plan.payload.target_particles_per_voxel)))))
        particle_demand = len(coords) * 512 * target
        quality = DemandQuality.UPPER_BOUND
    else:
        particle_demand = 0
        quality = DemandQuality.EXACT
    return SimpleNamespace(
        stable_id=str(legacy_plan.stable_id),
        particle_demand=particle_demand,
        particle_quality=quality,
        leaf_coords=coords,
        staging_bytes=0,
        reason="legacy device predicate bounded by clipped AABB leaves",
    )


def capture_snapshot(
        scene, settings, resources, *, depsgraph=None, scene_identity=None,
        scene_key="", effect_builder=lambda _scene: (),
        additional_transaction_bytes=0):
    """Gather and compile one evaluated scene exactly once.

    The compiler is always released.  The returned SceneInputs owner is either
    transferred to the running session or cleared by the caller on every exit.
    """
    from . import sources
    from ..sim.capacity_planner import DemandQuality, leaf_coords_for_world_bounds
    from ..sim.enums import SourceOperation
    from ..sim.source_compat import descriptors_to_plans, height_descriptor_to_plan
    from ..sim.source_regions import CompileContext, SourceCompiler

    depsgraph = _authoritative_depsgraph(depsgraph)
    generation = _depsgraph_generation(depsgraph)
    frame = int(getattr(scene, "frame_current", 0))
    captured_domain_transform = domain_transform(scene)
    inputs = SceneInputs()
    # Planning is deliberately CPU-staged.  The configured CUDA device is
    # sampled for capacity only; its first allocation belongs to the committed
    # replacement after Cutover.
    inputs.device = "cpu"
    compiler = SourceCompiler()
    try:
        sync_generation = inputs.begin_sync()
        descriptors = tuple(inputs.gather_sources(
            scene, depsgraph, capture_first_mesh=True,
            sync_generation=sync_generation))
        extra_plans = tuple(inputs.gather_source_plans(scene, depsgraph))
        colliders = tuple(inputs.gather_colliders(
            scene, depsgraph, sync_generation=sync_generation))
        force_fields = tuple(inputs.gather_force_fields(scene, depsgraph))
        height_source = inputs.gather_height_source(scene, depsgraph)
        seeds = tuple(sources.prepare_auto_seeds(scene, depsgraph))
        effects = tuple(effect_builder(scene))
        # Register staged mesh descriptor bindings so every failure path can
        # break those references deterministically.
        inputs.materialize_device_meshes(descriptors, colliders, "cpu")

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

        context = CompileContext(
            frame=frame, substep=0,
            session_generation=inputs._session_generation,
            device="cpu",
            origin=tuple(float(value) for value in settings.domain_min),
            voxel_size=float(settings.voxel_size),
            domain_min=tuple(float(value) for value in settings.domain_min),
            domain_max=tuple(float(value) for value in settings.domain_max),
            leaf_capacity=int(settings.leaf_capacity),
            particle_capacity=int(settings.max_particles), upload=False,
        )
        batch = compiler.compile(tuple(plans), context)
        footprints = [
            _legacy_footprint(plan, compiler, context, settings)
            for plan in legacy_plans
        ]
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
        capacity_plan = _plan_capacity(
            settings, tuple(footprints), seeds, collider_footprints, resources,
            additional_transaction_bytes=additional_transaction_bytes)
        fs = scene.fluidsim
        snapshot = AdmissionSnapshot(
            scene_identity=scene_identity,
            scene_key=str(scene_key),
            frame=frame,
            frame_start=int(getattr(scene, "frame_start", 1)),
            depsgraph_generation=generation,
            configured_device=str(settings.device),
            domain_transform=captured_domain_transform,
            domain_min=tuple(float(value) for value in settings.domain_min),
            domain_max=tuple(float(value) for value in settings.domain_max),
            settings=settings,
            profiling_enabled=bool(getattr(fs, "profiling_enabled", False)),
            profiling_sync=bool(getattr(fs, "profiling_sync", False)),
            scene_inputs=inputs,
            descriptors=descriptors,
            extra_plans=extra_plans,
            colliders=colliders,
            force_fields=force_fields,
            height_source=height_source,
            seeds=seeds,
            effects=effects,
            footprints=tuple(footprints),
            collider_footprints=collider_footprints,
            resources=resources,
            capacity_plan=capacity_plan,
            depsgraph=depsgraph,
        )
        snapshot.assert_evaluation_valid(scene)
        return snapshot
    except Exception:
        inputs.clear()
        raise
    finally:
        compiler.release()


def replan_snapshot(
        snapshot, settings, *, resources=None, additional_transaction_bytes=0):
    """Reassess capacity from the original snapshot without a second gather."""
    snapshot.settings = settings
    snapshot.configured_device = str(settings.device)
    if resources is not None:
        snapshot.resources = resources
    snapshot.capacity_plan = _plan_capacity(
        settings, snapshot.footprints, snapshot.seeds,
        snapshot.collider_footprints, snapshot.resources,
        additional_transaction_bytes=additional_transaction_bytes)
    return snapshot


def discard_prepared_solver(solver):
    """Release only an uncommitted candidate, never the live session."""
    if solver is None:
        return
    device = getattr(solver, "device", None)
    release = getattr(solver, "release", None)
    if callable(release):
        release()
    from ..sim.memory_safety import release_cuda_memory
    release_cuda_memory(device)


class SimulationAdmission:
    """Deep admission owner used by one Solver Session."""

    def __init__(self, session, *, scene_identity,
                 applied_state_capture, reset_settings_apply,
                 effect_builder=None):
        self.session = session
        self._scene_identity = scene_identity
        self._effect_builder = effect_builder or (lambda _scene: ())
        self._capture_applied_state = applied_state_capture
        self._apply_reset_settings = reset_settings_apply

    def _capture(
            self, scene, settings, resources, depsgraph=None, *,
            additional_transaction_bytes=0):
        from . import cache_identity

        return capture_snapshot(
            scene, settings, resources, depsgraph=depsgraph,
            scene_identity=self._scene_identity(scene),
            scene_key=cache_identity.scene_key(scene),
            effect_builder=self._effect_builder,
            additional_transaction_bytes=additional_transaction_bytes,
        )

    def _plan_scene(
            self, scene, settings, resources, depsgraph=None, *,
            additional_transaction_bytes=0):
        return self._capture(
            scene, settings, resources, depsgraph,
            additional_transaction_bytes=additional_transaction_bytes)

    def _call_plan_scene(
            self, scene, settings, resources, *, depsgraph=None,
            additional_transaction_bytes=0):
        kwargs = {
            "additional_transaction_bytes": additional_transaction_bytes,
        }
        if depsgraph is not None:
            kwargs["depsgraph"] = depsgraph
        return self._plan_scene(scene, settings, resources, **kwargs)

    def _replan_snapshot(
            self, snapshot, settings, resources, *, scene=None,
            depsgraph=None, additional_transaction_bytes=0):
        if not hasattr(snapshot, "footprints"):
            if scene is None:
                return snapshot
            return self._call_plan_scene(
                scene, settings, resources, depsgraph=depsgraph,
                additional_transaction_bytes=additional_transaction_bytes)
        return replan_snapshot(
            snapshot, settings, resources=resources,
            additional_transaction_bytes=additional_transaction_bytes)

    def _owns_scene(self, scene):
        owns = getattr(self.session, "owns_scene", None)
        if callable(owns):
            return bool(owns(scene))
        return getattr(self.session, "solver", None) is not None

    def _live_solver_bytes(self):
        if getattr(self.session, "solver", None) is None:
            return 0
        from ..sim.memory_safety import estimate_solver_bytes
        return estimate_solver_bytes(self.session.solver.settings)

    def _preflight(self, scene, *, depsgraph=None, runtime_capacity=None,
                   ignore_memory_safeguard=False,
                   defer_particle_growth=False, operation=None):
        from ..sim.memory_safety import vram_snapshot

        if operation is None:
            operation = (
                "REINITIALIZE" if self._owns_scene(scene)
                else "INITIALIZE")
        settings = build_settings(scene)
        if runtime_capacity is not None:
            settings = replace(settings, max_particles=int(runtime_capacity))
        resources = vram_snapshot(settings.device)
        live_solver_bytes = self._live_solver_bytes()
        snapshot = self._call_plan_scene(
            scene, settings, resources, depsgraph=depsgraph,
            additional_transaction_bytes=live_solver_bytes)
        snapshot.settings = settings
        if not hasattr(snapshot, "frame_start"):
            snapshot.frame_start = int(getattr(scene, "frame_start", 1))
        try:
            initial_growth = None
            blocker = next((
                item for item in snapshot.capacity_plan.blockers
                if item.code == "PARTICLE_CAPACITY" and item.required is not None
            ), None)
            if (blocker is not None
                    and bool(getattr(settings, "auto_grow_particles", True))):
                from ..sim.particle_capacity import growth_target

                original_snapshot = snapshot
                target = growth_target(
                    int(settings.max_particles), int(blocker.required))
                candidate_settings = replace(settings, max_particles=target)
                candidate_snapshot = self._replan_snapshot(
                    snapshot, candidate_settings, resources, scene=scene,
                    depsgraph=depsgraph,
                    additional_transaction_bytes=live_solver_bytes)
                candidate_integrity = _integrity_blockers(
                    candidate_snapshot.capacity_plan)
                candidate_memory = any(
                    item.code == "MEMORY_ESTIMATE"
                    for item in candidate_snapshot.capacity_plan.blockers)
                if (not candidate_integrity
                        and (not candidate_memory or ignore_memory_safeguard)):
                    initial_growth = (int(settings.max_particles), int(target))
                    settings = candidate_settings
                    snapshot = candidate_snapshot
                else:
                    snapshot = original_snapshot
            self.session._set_capacity_plan(
                scene, snapshot.capacity_plan, operation=operation,
                memory_bypassed=ignore_memory_safeguard)
            integrity = _integrity_blockers(snapshot.capacity_plan)
            if defer_particle_growth:
                integrity = tuple(
                    item for item in integrity
                    if not (item.code == "PARTICLE_CAPACITY"
                            and item.required is not None))
            if integrity:
                raise RuntimeError(capacity_blocking_message(integrity))
            memory_blockers = tuple(
                item for item in snapshot.capacity_plan.blockers
                if item.code == "MEMORY_ESTIMATE")
            if memory_blockers and not ignore_memory_safeguard:
                if operation == "RESET":
                    raise RuntimeError(reset_memory_blocking_message(
                        snapshot.capacity_plan))
                estimate = int(snapshot.capacity_plan.estimated_bytes)
                budget = int(snapshot.capacity_plan.budget_bytes)
                raise RuntimeError(
                    "FluidSim memory safeguard stopped initialization before CUDA "
                    f"allocation (estimated {estimate / 1024**3:.1f} GiB, safe live "
                    f"budget {budget / 1024**3:.1f} GiB). Review Capacity Options, "
                    "then retry. This estimate is not an OOM guarantee.")

            resources = vram_snapshot(settings.device)
            from ..sim.capacity_service import refresh_resources
            from ..sim.capacity_planner import CapacityPlan
            if isinstance(snapshot.capacity_plan, CapacityPlan):
                snapshot.resources = resources
                snapshot.capacity_plan = refresh_resources(
                    snapshot.capacity_plan, settings, resources)
            else:
                snapshot = self._replan_snapshot(
                    snapshot, settings, resources, scene=scene,
                    depsgraph=depsgraph,
                    additional_transaction_bytes=live_solver_bytes)
            snapshot.settings = settings
            if not hasattr(snapshot, "frame_start"):
                snapshot.frame_start = int(getattr(scene, "frame_start", 1))
            self.session._set_capacity_plan(
                scene, snapshot.capacity_plan, operation=operation,
                memory_bypassed=ignore_memory_safeguard)
            late_integrity = _integrity_blockers(snapshot.capacity_plan)
            late_memory = tuple(
                item for item in snapshot.capacity_plan.blockers
                if item.code == "MEMORY_ESTIMATE")
            if late_integrity:
                raise RuntimeError(capacity_blocking_message(late_integrity))
            if late_memory and not ignore_memory_safeguard:
                if operation == "RESET":
                    raise RuntimeError(reset_memory_blocking_message(
                        snapshot.capacity_plan))
                raise RuntimeError(capacity_blocking_message(late_memory))
            memory_assessment = {
                "safe": not bool(late_memory),
                "estimated_bytes": int(getattr(
                    snapshot.capacity_plan, "estimated_bytes", 0)),
                "budget_bytes": int(getattr(
                    snapshot.capacity_plan, "budget_bytes", 0)),
                "recommended": tuple(getattr(
                    snapshot.capacity_plan, "suggestions", ())),
                "resources": dict(resources),
            }
            return snapshot, memory_assessment, initial_growth
        except Exception:
            self._release_snapshot(snapshot)
            raise

    def refresh_capacity_plan(self, scene, depsgraph=None, *, operation=None):
        if operation is None:
            owned = self.session.capacity_plan_for_scene(scene)
            operation = (
                self.session.capacity_plan_operation if owned is not None
                else ("REINITIALIZE" if self._owns_scene(scene)
                      else "INITIALIZE"))
        settings = build_settings(scene)
        if operation == "RESET" and self._owns_scene(scene):
            settings = replace(
                settings,
                max_particles=int(self.session.solver.particles.capacity))
        from ..sim.memory_safety import vram_snapshot
        resources = vram_snapshot(settings.device)
        snapshot = self._call_plan_scene(
            scene, settings, resources, depsgraph=depsgraph,
            additional_transaction_bytes=self._live_solver_bytes())
        try:
            return self.session._set_capacity_plan(
                scene, snapshot.capacity_plan, operation=operation)
        finally:
            self._release_snapshot(snapshot)

    def _allocate_solver(self, snapshot, profiler):
        from ..sim.solver import FluidSolver

        return FluidSolver(snapshot.settings, profiler=profiler)

    def _configure_candidate(self, candidate, snapshot, scene):
        inputs = snapshot.scene_inputs
        inputs.materialize_device_meshes(
            snapshot.descriptors, snapshot.colliders,
            snapshot.configured_device)
        candidate.set_sources(snapshot.descriptors)
        if snapshot.extra_plans:
            candidate.set_source_plans(
                tuple(candidate.sourcing.plans) + snapshot.extra_plans)
        candidate.set_colliders(snapshot.colliders)
        candidate.set_force_fields(snapshot.force_fields)
        candidate.set_height_source(snapshot.height_source)

        from ..sim.effects.compiler import build_candidate_effects
        from .effects import trust_store_for_scene

        effect_candidate = build_candidate_effects(
            snapshot.effects,
            device=snapshot.configured_device,
            trust_store=trust_store_for_scene(scene),
        )
        if not effect_candidate.success:
            issue = effect_candidate.issues[0] if effect_candidate.issues else None
            message = issue.message if issue is not None else "effect candidate validation failed"
            raise RuntimeError(f"Custom Effects admission failed closed: {message}")
        if not candidate.set_custom_effect_candidate(effect_candidate):
            raise RuntimeError("Custom Effects candidate could not be installed atomically")
        return ()

    def _prepare_solver_replacement(self, scene, settings, snapshot, profiler):
        from . import sources

        candidate = None
        try:
            candidate = self._allocate_solver(snapshot, profiler)
            errors = self._configure_candidate(candidate, snapshot, scene)
            sources.commit_prepared_seeds(
                SimpleNamespace(solver=candidate), snapshot.seeds)
            result = getattr(
                getattr(candidate, "sourcing", None),
                "last_apply_result", {},
            )
            if result and not result.get("admitted", True):
                raise RuntimeError(capacity_blocking_message((SimpleNamespace(
                    code="PARTICLE_CAPACITY",
                    required=result.get("required"),
                    available=result.get("available"),
                ),)))
            return candidate, errors
        except Exception:
            if candidate is not None:
                discard_prepared_solver(candidate)
            else:
                from ..sim.memory_safety import release_cuda_memory
                release_cuda_memory(settings.device)
            raise

    def _commit_solver_replacement(
            self, candidate, snapshot, *, profiler, settings):
        old_solver = self.session.solver
        old_inputs = self.session.scene_inputs
        transfer = getattr(snapshot, "transfer_scene_inputs", None)
        new_inputs = (
            transfer() if callable(transfer)
            else getattr(snapshot, "scene_inputs", old_inputs))
        self.session.solver = candidate
        self.session.profiler = profiler
        self.session.device = settings.device
        self.session.scene_inputs = new_inputs
        if old_inputs is not new_inputs:
            old_inputs.clear()
        if old_solver is not None:
            device = getattr(old_solver, "device", None)
            release = getattr(old_solver, "release", None)
            if callable(release):
                release()
            from ..sim.memory_safety import release_cuda_memory
            release_cuda_memory(device)

    def _prepare_outputs(self, scene):
        from . import output_publication

        output_publication.prepare_outputs(scene)

    @staticmethod
    def _release_snapshot(snapshot):
        release = getattr(snapshot, "release_transients", None)
        if callable(release):
            release()
            return
        inputs = getattr(snapshot, "scene_inputs", None)
        clear = getattr(inputs, "clear", None)
        if callable(clear):
            clear()

    def initialize(self, scene, *, depsgraph=None, ignore_memory_safeguard=False):
        from ..diagnostics.profiler import Profiler
        from . import cache_identity

        candidate = None
        replacement_committed = False
        snapshot = None
        try:
            operation = (
                "REINITIALIZE" if self._owns_scene(scene)
                else "INITIALIZE")
            snapshot, memory_assessment, initial_growth = self._preflight(
                scene, depsgraph=depsgraph, operation=operation,
                ignore_memory_safeguard=ignore_memory_safeguard)
            validate = getattr(snapshot, "assert_evaluation_valid", None)
            if callable(validate):
                validate(scene)
            settings = snapshot.settings
            profiler = None
            if bool(getattr(snapshot, "profiling_enabled", False)):
                profiler = Profiler(
                    device=settings.device,
                    sync=bool(getattr(snapshot, "profiling_sync", False)))
            candidate, _configuration_warnings = self._prepare_solver_replacement(
                scene, settings, snapshot, profiler)

            # Replacement Cutover: the candidate has passed admission. This is
            # the single authority swap; all candidate failures above preserve
            # the previously authoritative solver.
            self._commit_solver_replacement(
                candidate, snapshot, profiler=profiler, settings=settings)
            replacement_committed = True
            candidate = None
            self.session.memory_assessment = memory_assessment
            self.session.scene_identity = getattr(
                snapshot, "scene_identity", None)
            if self.session.scene_identity is None:
                self.session.scene_identity = self._scene_identity(scene)
            self.session.scene_key = getattr(snapshot, "scene_key", None)
            if self.session.scene_key is None:
                self.session.scene_key = cache_identity.scene_key(scene)
            self.session.start_frame = int(getattr(
                snapshot, "frame_start", getattr(scene, "frame_start", 1)))
            self.session.cache_written_start = None
            self.session.cache_written_end = None
            self.session.cache_settings_records = {}
            self.session.last_frame = -1
            self.session.frame_stats = []
            self._prepare_outputs(scene)
            fs = scene.fluidsim
            if initial_growth is not None:
                self.session.last_reported_growth_capacity = int(initial_growth[1])
                fs.status = (
                    "Solver ready — particle capacity grew "
                    f"{initial_growth[0]:,} → {initial_growth[1]:,}")
            else:
                fs.status = "Solver ready"
            self.session.memory_assessment = {}
            self.session.clear_error(scene)
            self._capture_applied_state(scene, self.session)
            return self.session
        except Exception as exc:
            self.session.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            # Constructor failures can strand self-referential candidate
            # objects in cleared traceback cycles before Session ownership is
            # established. Collect them before returning the failure.
            gc.collect()
            if replacement_committed:
                self.session.release(scene, preserve_error=True)
            elif candidate is not None:
                discard_prepared_solver(candidate)
            raise
        finally:
            if snapshot is not None:
                self._release_snapshot(snapshot)

    def reset(self, scene, *, depsgraph=None):
        if not self._owns_scene(scene):
            return
        runtime_capacity = int(self.session.solver.particles.capacity)
        candidate = None
        replacement_committed = False
        snapshot = None
        try:
            snapshot, _memory, replacement_growth = self._preflight(
                scene, depsgraph=depsgraph,
                runtime_capacity=runtime_capacity, operation="RESET")
            settings = snapshot.settings
            candidate, _configuration_warnings = self._prepare_solver_replacement(
                scene, settings, snapshot, self.session.profiler)
            self._commit_solver_replacement(
                candidate, snapshot, profiler=self.session.profiler,
                settings=settings)
            replacement_committed = True
            candidate = None
            self.session.start_frame = int(getattr(
                snapshot, "frame_start", getattr(scene, "frame_start", 1)))
            self.session.cache_written_start = None
            self.session.cache_written_end = None
            self.session.cache_settings_records = {}
            self.session.last_frame = -1
            self.session.frame_stats = []
            self.session.clear_error(scene)
            from . import output_publication
            output_publication.update_outputs(scene, self.session)
            self._capture_applied_state(scene, self.session, reset_only=True)
            if replacement_growth is None:
                scene.fluidsim.status = "Simulation reset"
            else:
                self.session.last_reported_growth_capacity = int(
                    replacement_growth[1])
                scene.fluidsim.status = (
                    "Simulation reset — particle capacity grew "
                    f"{replacement_growth[0]:,} → {replacement_growth[1]:,}")
        except Exception as exc:
            self.session.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            if not replacement_committed:
                scene.fluidsim.status = "Reset blocked — live solver preserved"
            elif replacement_committed:
                self.session.release(scene, preserve_error=True)
            if candidate is not None:
                discard_prepared_solver(candidate)
            raise
        finally:
            if snapshot is not None:
                self._release_snapshot(snapshot)
