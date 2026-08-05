"""Blender-owned authored-scene to running-solver admission transaction.

Admission performs every non-destructive gather, compile, memory, capacity and
integrity check against one :class:`AdmissionSnapshot` before Replacement
Cutover.  Allocation failures after cutover intentionally leave a clean,
inactive Solver Session; two-solver staging is not used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    snippets: tuple
    footprints: tuple
    collider_footprints: tuple
    resources: object
    capacity_plan: object
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


def _plan_capacity(settings, footprints, seeds, collider_footprints, resources):
    from ..sim.capacity_planner import plan_initialization, suggest_configurations

    capacity_plan = plan_initialization(
        settings, footprints, seeds, collider_footprints, resources)
    return replace(
        capacity_plan,
        suggestions=suggest_configurations(capacity_plan, settings, resources),
    )


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
        scene_key="", snippet_builder=lambda _scene: ()):
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
        snippets = tuple(snippet_builder(scene))
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
            settings, tuple(footprints), seeds, collider_footprints, resources)
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
            snippets=snippets,
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


def replan_snapshot(snapshot, settings):
    """Reassess capacity from the original snapshot without a second gather."""
    snapshot.settings = settings
    snapshot.configured_device = str(settings.device)
    snapshot.capacity_plan = _plan_capacity(
        settings, snapshot.footprints, snapshot.seeds,
        snapshot.collider_footprints, snapshot.resources)
    return snapshot


class SimulationAdmission:
    """Deep admission owner used by one Solver Session."""

    def __init__(self, session, *, scene_identity, snippet_builder,
                 applied_state_capture, reset_settings_apply):
        self.session = session
        self._scene_identity = scene_identity
        self._snippet_builder = snippet_builder
        self._capture_applied_state = applied_state_capture
        self._apply_reset_settings = reset_settings_apply

    def _capture(self, scene, settings, resources, depsgraph=None):
        from ..io import cache

        return capture_snapshot(
            scene, settings, resources, depsgraph=depsgraph,
            scene_identity=self._scene_identity(scene),
            scene_key=cache.scene_key(scene),
            snippet_builder=self._snippet_builder,
        )

    def _preflight(self, scene, *, depsgraph=None, runtime_capacity=None,
                   ignore_memory_safeguard=False,
                   defer_particle_growth=False):
        from ..sim.memory_safety import assess_configuration, vram_snapshot

        settings = build_settings(scene)
        if (runtime_capacity is not None
                and bool(getattr(settings, "auto_grow_particles", True))):
            settings = replace(
                settings,
                max_particles=max(
                    int(settings.max_particles), int(runtime_capacity)),
            )
        resources = vram_snapshot(settings.device)
        memory_assessment = assess_configuration(settings, resources)
        snapshot = self._capture(scene, settings, resources, depsgraph)
        try:
            initial_growth = None
            blocker = next((
                item for item in snapshot.capacity_plan.blockers
                if item.code == "PARTICLE_CAPACITY" and item.required is not None
            ), None)
            if (blocker is not None
                    and bool(getattr(settings, "auto_grow_particles", True))):
                from ..sim.particle_capacity import growth_target

                original_plan = snapshot.capacity_plan
                target = growth_target(
                    int(settings.max_particles), int(blocker.required))
                candidate_settings = replace(settings, max_particles=target)
                replan_snapshot(snapshot, candidate_settings)
                candidate_integrity = _integrity_blockers(
                    snapshot.capacity_plan)
                candidate_memory = any(
                    item.code == "MEMORY_ESTIMATE"
                    for item in snapshot.capacity_plan.blockers)
                if not candidate_integrity and not candidate_memory:
                    initial_growth = (int(settings.max_particles), int(target))
                    settings = candidate_settings
                    memory_assessment = assess_configuration(settings, resources)
                else:
                    snapshot.settings = settings
                    snapshot.configured_device = str(settings.device)
                    snapshot.capacity_plan = original_plan
            self.session.capacity_plan = snapshot.capacity_plan
            self.session.capacity_plan_memory_bypassed = bool(
                ignore_memory_safeguard)
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
            if ((not memory_assessment["safe"] or memory_blockers)
                    and not ignore_memory_safeguard):
                estimate = max(
                    int(memory_assessment["estimated_bytes"]),
                    int(snapshot.capacity_plan.estimated_bytes))
                budget = int(snapshot.capacity_plan.budget_bytes)
                raise RuntimeError(
                    "FluidSim memory safeguard stopped initialization before CUDA "
                    f"allocation (estimated {estimate / 1024**3:.1f} GiB, safe live "
                    f"budget {budget / 1024**3:.1f} GiB). Apply Safe Settings & "
                    "Retry. This estimate is not an OOM guarantee.")
            return snapshot, memory_assessment, initial_growth
        except Exception:
            snapshot.release_transients()
            raise

    def refresh_capacity_plan(self, scene, depsgraph=None):
        from ..sim.memory_safety import vram_snapshot

        settings = build_settings(scene)
        snapshot = self._capture(
            scene, settings, vram_snapshot(settings.device), depsgraph)
        try:
            self.session.capacity_plan = snapshot.capacity_plan
            return snapshot.capacity_plan
        finally:
            snapshot.release_transients()

    def _allocate_solver(self, snapshot, profiler):
        from ..sim.solver import FluidSolver

        return FluidSolver(snapshot.settings, profiler=profiler)

    def _configure_solver(self, snapshot):
        solver = self.session.solver
        previous_inputs = self.session.scene_inputs
        self.session.scene_inputs = snapshot.transfer_scene_inputs()
        if previous_inputs is not self.session.scene_inputs:
            previous_inputs.clear()
        # This is the first mesh/BVH allocation on the configured CUDA device.
        # Ownership is already in the Session, so a materialization or later
        # configuration failure is released by Session.release before its
        # allocator-pool cleanup.
        self.session.scene_inputs.materialize_device_meshes(
            snapshot.descriptors, snapshot.colliders,
            snapshot.configured_device)
        solver.set_sources(snapshot.descriptors)
        if snapshot.extra_plans:
            solver.set_source_plans(
                tuple(solver.sourcing.plans) + snapshot.extra_plans)
        solver.set_colliders(snapshot.colliders)
        solver.set_force_fields(snapshot.force_fields)
        solver.set_height_source(snapshot.height_source)
        errors = solver.set_custom_snippets(snapshot.snippets)
        return errors

    def _commit_seeds(self, snapshot):
        from . import sources

        sources.commit_prepared_seeds(self.session, snapshot.seeds)
        result = getattr(
            getattr(self.session.solver, "sourcing", None),
            "last_apply_result", {},
        )
        if result and not result.get("admitted", True):
            raise RuntimeError(capacity_blocking_message((SimpleNamespace(
                code="PARTICLE_CAPACITY",
                required=result.get("required"),
                available=result.get("available"),
            ),)))

    def _prepare_outputs(self, scene):
        from . import output_publication

        output_publication.prepare_outputs(scene)

    def initialize(self, scene, *, depsgraph=None, ignore_memory_safeguard=False):
        from ..diagnostics.profiler import Profiler

        snapshot = None
        replacement_started = False
        try:
            snapshot, memory_assessment, initial_growth = self._preflight(
                scene, depsgraph=depsgraph,
                ignore_memory_safeguard=ignore_memory_safeguard)
            validate_evaluation = getattr(
                snapshot, "assert_evaluation_valid", None)
            if callable(validate_evaluation):
                validate_evaluation(scene)

            # Replacement Cutover: every authored-input rejection has already
            # happened and the existing GPU solver is preserved until here.
            self.session.release(scene)
            replacement_started = True
            self.session.device = snapshot.configured_device
            self.session.memory_assessment = memory_assessment
            self.session.capacity_plan = snapshot.capacity_plan
            self.session.capacity_plan_memory_bypassed = bool(
                ignore_memory_safeguard)
            profiler = None
            if snapshot.profiling_enabled:
                profiler = Profiler(
                    device=snapshot.configured_device,
                    sync=snapshot.profiling_sync)
            self.session.solver = self._allocate_solver(snapshot, profiler)
            self.session.profiler = profiler
            self.session.scene_identity = snapshot.scene_identity
            self.session.scene_key = snapshot.scene_key
            self.session.start_frame = snapshot.frame_start
            snippet_errors = self._configure_solver(snapshot)
            self._commit_seeds(snapshot)
            for name, error in snippet_errors:
                for pg in scene.fluidsim.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
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
            self._capture_applied_state(scene, self.session)
            return self.session
        except Exception as exc:
            self.session.set_error(scene, repr(exc))
            traceback.clear_frames(exc.__traceback__)
            if replacement_started:
                self.session.release(scene, preserve_error=True)
            raise
        finally:
            if snapshot is not None:
                snapshot.release_transients()

    def reset(self, scene, *, depsgraph=None):
        if not self.session.owns_scene(scene):
            return
        runtime_capacity = getattr(
            getattr(self.session.solver, "particles", None), "capacity", None)
        snapshot = None
        reset_started = False
        try:
            snapshot, _memory, planned_growth = self._preflight(
                scene, depsgraph=depsgraph, runtime_capacity=runtime_capacity,
                ignore_memory_safeguard=True, defer_particle_growth=True)
            required = (
                int(planned_growth[1]) if planned_growth is not None else next((
                    int(item.required) for item in snapshot.capacity_plan.blockers
                    if (item.code == "PARTICLE_CAPACITY"
                        and item.required is not None)
                ), None)
            )
            if required is not None:
                growth = self.session.solver.ensure_particle_capacity(required)
                if not growth.grew and int(growth.current) < required:
                    raise RuntimeError(capacity_blocking_message((SimpleNamespace(
                        code="PARTICLE_CAPACITY", required=required,
                        available=int(growth.current)),)))
                replan_snapshot(
                    snapshot,
                    replace(snapshot.settings, max_particles=int(growth.current)))
            integrity = _integrity_blockers(snapshot.capacity_plan)
            if integrity:
                raise RuntimeError(capacity_blocking_message(integrity))

            self._apply_reset_settings(scene, self.session)
            self.session.solver.reset()
            reset_started = True
            self.session.start_frame = snapshot.frame_start
            self.session.cache_written_start = None
            self.session.cache_written_end = None
            self.session.cache_settings_records = {}
            self.session.last_frame = -1
            self.session.frame_stats = []
            self.session.clear_error(scene)
            snippet_errors = self._configure_solver(snapshot)
            self._commit_seeds(snapshot)
            for name, error in snippet_errors:
                for pg in scene.fluidsim.snippets:
                    if pg.kernel_name == name:
                        pg.last_error = str(error)[:256]
            from . import output_publication

            output_publication.update_outputs(scene, self.session)
            self._capture_applied_state(
                scene, self.session, reset_only=True)
            scene.fluidsim.status = "Simulation reset"
        except Exception as exc:
            self.session.set_error(scene, repr(exc))
            if reset_started:
                self.session.release(scene, preserve_error=True)
            raise
        finally:
            if snapshot is not None:
                snapshot.release_transients()
