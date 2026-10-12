"""Transactional execution and capability-derived scheduling for Custom Effects v2."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

from .enums import EffectCadence, EffectDomain, EffectStage, FieldAccess
from .fields import FIELD_SPECS, requirements_for_bindings
from .fingerprint import effective_stack_instances, execution_stack_fingerprint
from .source import parameter_runtime_inputs


@dataclass(frozen=True)
class EffectRuntimeRequirements:
    """Prerequisites required by the currently applied effect stack."""

    particles: bool = False
    topology: bool = False
    surface_sdf: bool = False
    collision: bool = False
    rebuild_topology_each_substep: bool = False
    # A binding writes particle_position: products derived from particle
    # positions (the solver SDF) are stale after the stage runs.
    moves_particles: bool = False

    def merged(self, other: EffectRuntimeRequirements) -> EffectRuntimeRequirements:
        return EffectRuntimeRequirements(
            particles=self.particles or other.particles,
            topology=self.topology or other.topology,
            surface_sdf=self.surface_sdf or other.surface_sdf,
            collision=self.collision or other.collision,
            rebuild_topology_each_substep=(
                self.rebuild_topology_each_substep or other.rebuild_topology_each_substep
            ),
            moves_particles=self.moves_particles or other.moves_particles,
        )


@dataclass
class EffectRuntimeStats:
    """Accumulated runtime telemetry for one applied kernel."""

    effect_id: str
    instance_id: str
    kernel_id: str
    calls: int = 0
    elements: int = 0
    submission_seconds: float = 0.0
    last_error: str = ""


@dataclass(frozen=True)
class EffectRuntimeError:
    effect_id: str
    instance_id: str
    kernel_id: str
    stage: str
    frame: int
    substep_index: int
    message: str


class EffectExecutionError(RuntimeError):
    """Raised when an applied custom effect stack can no longer run safely."""

    def __init__(self, runtime_error: EffectRuntimeError):
        self.runtime_error = runtime_error
        super().__init__(
            "Custom Effect execution failed for "
            f"{runtime_error.instance_id}/{runtime_error.kernel_id} at "
            f"{runtime_error.stage}, frame {runtime_error.frame}, "
            f"substep {runtime_error.substep_index}: {runtime_error.message}"
        )


class EffectPrerequisiteError(RuntimeError):
    """A solver product an effect reads could not be prepared this stage."""


@dataclass(frozen=True)
class EffectCandidateInstallDiagnostic:
    """Structured context for an unexpected candidate installation failure."""

    phase: str
    exception_type: str
    message: str


class EffectCandidateInstallError(RuntimeError):
    """Raised when a valid candidate cannot be prepared for atomic install."""

    def __init__(
        self,
        diagnostic: EffectCandidateInstallDiagnostic,
        original_exception: Exception,
    ):
        self.diagnostic = diagnostic
        self.original_exception = original_exception
        super().__init__(
            "Custom Effect candidate installation failed during "
            f"{diagnostic.phase}: {diagnostic.exception_type}: "
            f"{diagnostic.message}"
        )


@dataclass(frozen=True)
class EffectRuntimeEntry:
    instance: Any
    compiled_effect: Any
    compiled_kernel: Any
    order: int
    kernel_index: int

    @property
    def key(self) -> str:
        kernel_id = self.compiled_kernel.kernel_definition.kernel_id
        return f"{self.instance.instance_id}:{kernel_id}:{self.kernel_index}"


@dataclass(frozen=True)
class _PreparedRuntime:
    by_stage: dict[EffectStage, tuple[EffectRuntimeEntry, ...]]
    signature: str
    inventory: tuple[dict[str, object], ...]
    requirements: EffectRuntimeRequirements
    stage_requirements: dict[EffectStage, EffectRuntimeRequirements]
    stats: dict[str, EffectRuntimeStats]


def _requirements_for_entry(entry: EffectRuntimeEntry) -> EffectRuntimeRequirements:
    kernel = entry.compiled_kernel.kernel_definition
    names = set(requirements_for_bindings(tuple(kernel.fields)))
    names.update(str(item) for item in getattr(kernel, "prerequisites", ()))
    particles = kernel.domain is EffectDomain.PARTICLES or "PARTICLES" in names
    topology = kernel.domain is EffectDomain.ACTIVE_VOXELS or "TOPOLOGY" in names
    surface_sdf = "SURFACE_SDF" in names
    collision = "COLLISION" in names
    moves_particles = any(
        str(getattr(binding, "field", "")) == "particle_position"
        and getattr(binding, "access", FieldAccess.READ) != FieldAccess.READ
        for binding in kernel.fields)
    return EffectRuntimeRequirements(
        particles=particles,
        topology=topology,
        surface_sdf=surface_sdf,
        collision=collision,
        rebuild_topology_each_substep=topology,
        moves_particles=moves_particles,
    )


def _prepare_candidate(candidate: Any) -> _PreparedRuntime:
    compiled_effects = tuple(candidate.compiled_effects)
    entries: list[EffectRuntimeEntry] = []
    instances = []
    inventory: list[dict[str, object]] = []
    for effect_position, compiled_effect in enumerate(compiled_effects):
        if not bool(getattr(compiled_effect, "success", False)):
            raise ValueError("candidate contains an unsuccessful compiled effect")
        instance = compiled_effect.instance
        instances.append(instance)
        for kernel_position, compiled_kernel in enumerate(compiled_effect.kernels):
            generated = getattr(compiled_kernel, "generated", None)
            kernel_index = int(getattr(generated, "kernel_index", kernel_position))
            entries.append(EffectRuntimeEntry(
                instance=instance,
                compiled_effect=compiled_effect,
                compiled_kernel=compiled_kernel,
                order=int(instance.order),
                kernel_index=kernel_index,
            ))
        inventory.append({
            "instance_id": instance.instance_id,
            "effect_id": instance.definition.effect_id,
            "effect_version": instance.definition.version,
            "order": int(instance.order),
            "kernel_count": len(compiled_effect.kernels),
            "input_position": effect_position,
        })
    entries.sort(key=lambda item: (
        int(item.order),
        next(
            index
            for index, effect in enumerate(compiled_effects)
            if effect is item.compiled_effect
        ),
        int(item.kernel_index),
    ))
    by_stage: dict[EffectStage, list[EffectRuntimeEntry]] = {stage: [] for stage in EffectStage}
    overall = EffectRuntimeRequirements()
    stage_requirements: dict[EffectStage, EffectRuntimeRequirements] = {
        stage: EffectRuntimeRequirements() for stage in EffectStage
    }
    stats: dict[str, EffectRuntimeStats] = {}
    for entry in entries:
        stage = entry.compiled_kernel.kernel_definition.stage
        by_stage[stage].append(entry)
        required = _requirements_for_entry(entry)
        overall = overall.merged(required)
        stage_requirements[stage] = stage_requirements[stage].merged(required)
        stats[entry.key] = EffectRuntimeStats(
            effect_id=entry.instance.definition.effect_id,
            instance_id=entry.instance.instance_id,
            kernel_id=entry.compiled_kernel.kernel_definition.kernel_id,
        )
    return _PreparedRuntime(
        by_stage={stage: tuple(items) for stage, items in by_stage.items()},
        signature=execution_stack_fingerprint(tuple(instances))[0],
        inventory=tuple(inventory),
        requirements=overall,
        stage_requirements=stage_requirements,
        stats=stats,
    )


def _runtime_structure_matches(applied: Any, updated: Any) -> bool:
    """Return whether two instances differ only in launch-time bindings."""

    return (
        applied.instance_id == updated.instance_id
        and applied.definition == updated.definition
        and applied.enabled == updated.enabled
        and applied.muted == updated.muted
        and applied.solo == updated.solo
        and applied.order == updated.order
        and applied.frame_start == updated.frame_start
        and applied.frame_end == updated.frame_end
        and applied.frame_step == updated.frame_step
        and applied.safe_mode == updated.safe_mode
        and applied.expert_acknowledged == updated.expert_acknowledged
    )


class EffectRuntimeManager:
    """Own the last fully tested/applied stack and execute it deterministically."""

    def __init__(self, *, device: str):
        self.device = str(device)
        self.by_stage: dict[EffectStage, tuple[EffectRuntimeEntry, ...]] = {
            stage: () for stage in EffectStage
        }
        self.applied_signature = ""
        self.inventory: tuple[dict[str, object], ...] = ()
        self.requirements = EffectRuntimeRequirements()
        self._stage_requirements: dict[EffectStage, EffectRuntimeRequirements] = {
            stage: EffectRuntimeRequirements() for stage in EffectStage
        }
        self.stats: dict[str, EffectRuntimeStats] = {}
        self.runtime_errors: list[EffectRuntimeError] = []
        self.failed_error: EffectRuntimeError | None = None
        self._once_per_frame: dict[str, int] = {}
        self._initialized: set[str] = set()
        self._frame_baseline: dict[str, tuple[int, int, float]] = {}

    @property
    def entry_count(self) -> int:
        return sum(len(items) for items in self.by_stage.values())

    def has_any(self) -> bool:
        return self.entry_count > 0

    def requirements_for_stage(self, stage: EffectStage | str) -> EffectRuntimeRequirements:
        stage_enum = stage if isinstance(stage, EffectStage) else EffectStage(str(stage))
        return self._stage_requirements[stage_enum]

    def swap_candidate(self, candidate: Any) -> bool:
        """Atomically install a successful candidate; preserve the old stack on failure."""

        if not bool(getattr(candidate, "success", False)):
            return False
        try:
            prepared = _prepare_candidate(candidate)
        except Exception as exc:
            diagnostic = EffectCandidateInstallDiagnostic(
                phase="PREPARE_CANDIDATE",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise EffectCandidateInstallError(diagnostic, exc) from exc
        self.by_stage = prepared.by_stage
        self.applied_signature = prepared.signature
        self.inventory = prepared.inventory
        self.requirements = prepared.requirements
        self._stage_requirements = prepared.stage_requirements
        self.stats = prepared.stats
        self.runtime_errors = []
        self.failed_error = None
        self._once_per_frame = {}
        self._initialized = set()
        self._frame_baseline = {}
        return True

    def applied_instances(self) -> tuple[Any, ...]:
        """The installed effect instances, one per instance id, in stack order."""
        seen: dict[str, Any] = {}
        for entries in self.by_stage.values():
            for entry in entries:
                seen.setdefault(entry.instance.instance_id, entry.instance)
        return tuple(sorted(seen.values(), key=lambda item: int(item.order)))

    def update_runtime_bindings(self, instances: Any) -> bool:
        """Update launch-time values without replacing compiled kernels or solver state.

        The update is accepted only when the effective stack and every
        compile/scheduling field are unchanged, and every changed parameter is
        explicitly runtime-bindable.  Rejection leaves the applied runtime
        untouched so structural edits still require the guarded apply path.
        """

        effective = effective_stack_instances(tuple(instances))
        current: dict[str, Any] = {}
        for entries in self.by_stage.values():
            for entry in entries:
                current.setdefault(entry.instance.instance_id, entry.instance)
        if len(effective) != len(current) or {
            item.instance_id for item in effective
        } != set(current):
            return False
        replacements: dict[str, Any] = {}
        for updated in effective:
            applied = current[updated.instance_id]
            if not _runtime_structure_matches(applied, updated):
                return False
            definitions = applied.definition.parameter_map()
            applied_values = applied.effective_parameter_values()
            updated_values = updated.effective_parameter_values()
            changed = {
                key for key in definitions
                if applied_values[key] != updated_values[key]
                or applied.controller_bindings.get(key) != updated.controller_bindings.get(key)
            }
            if any(not definitions[key].runtime_bindable for key in changed):
                return False
            replacements[updated.instance_id] = updated
        new_by_stage = {
            stage: tuple(
                replace(entry, instance=replacements[entry.instance.instance_id])
                for entry in entries
            )
            for stage, entries in self.by_stage.items()
        }
        self.by_stage = new_by_stage
        self.applied_signature = execution_stack_fingerprint(effective)[0]
        return True

    def clear(self) -> None:
        class _Empty:
            success = True
            compiled_effects = ()

        self.swap_candidate(_Empty())

    def begin_frame(self) -> None:
        """Freeze cumulative counters so the frame report can expose deltas."""
        self._frame_baseline = {
            key: (stats.calls, stats.elements, stats.submission_seconds)
            for key, stats in self.stats.items()
        }

    def frame_stats_snapshot(self) -> list[dict[str, object]]:
        """Return stable per-frame and cumulative telemetry for each kernel."""
        requirements = {
            entry.key: _requirements_for_entry(entry)
            for entries in self.by_stage.values()
            for entry in entries
        }
        records = []
        for key, stats in sorted(self.stats.items()):
            calls, elements, seconds = self._frame_baseline.get(
                key, (0, 0, 0.0))
            required = requirements.get(key, EffectRuntimeRequirements())
            records.append({
                "key": key,
                "effectId": stats.effect_id,
                "instanceId": stats.instance_id,
                "kernelId": stats.kernel_id,
                "calls": stats.calls - calls,
                "elements": stats.elements - elements,
                "submissionSeconds": stats.submission_seconds - seconds,
                "totalCalls": stats.calls,
                "totalElements": stats.elements,
                "totalSubmissionSeconds": stats.submission_seconds,
                "requirements": {
                    "particles": required.particles,
                    "topology": required.topology,
                    "surfaceSdf": required.surface_sdf,
                    "collision": required.collision,
                    "rebuildTopologyEachSubstep": (
                        required.rebuild_topology_each_substep),
                },
                "lastError": stats.last_error,
            })
        return records

    def _should_run(self, entry: EffectRuntimeEntry, frame: int) -> bool:
        instance = entry.instance
        kernel = entry.compiled_kernel.kernel_definition
        return bool(instance.active_on_frame(frame) and kernel.active_on_frame(frame))

    def mark_initialized_through(self, frame: int) -> None:
        """Record INITIALIZATION effects that already ran by ``frame``.

        A checkpoint restores solver state those effects already modified, but
        not this manager's cadence history, so without this a resumed bake
        re-applied them. An entry counts as run when any frame up to ``frame``
        was inside its window.
        """
        frame = int(frame)
        for entries in self.by_stage.values():
            for entry in entries:
                kernel = entry.compiled_kernel.kernel_definition
                if kernel.cadence is not EffectCadence.INITIALIZATION:
                    continue
                first = max(int(entry.instance.frame_start or 0),
                            int(kernel.frame_start or 0))
                if any(self._should_run(entry, f) for f in range(first, frame + 1)):
                    self._initialized.add(entry.key)

    def _cadence_allows(self, entry: EffectRuntimeEntry, frame: int) -> bool:
        cadence = entry.compiled_kernel.kernel_definition.cadence
        key = entry.key
        if cadence is EffectCadence.EVERY_SUBSTEP:
            return True
        if cadence is EffectCadence.ONCE_PER_FRAME:
            if self._once_per_frame.get(key) == frame:
                return False
            self._once_per_frame[key] = frame
            return True
        if cadence is EffectCadence.INITIALIZATION:
            if key in self._initialized:
                return False
            self._initialized.add(key)
            return True
        return False

    def run(
            self,
            stage: EffectStage | str,
            solver: Any,
            time_ctx: dict[str, object],
            *,
            surface_sdf_error: str | None = None,
    ) -> None:
        """Run one stage. ``surface_sdf_error`` set means the solver SDF could
        not be refreshed: any entry that reads it fails closed instead of
        sampling stale or zero-initialized storage."""
        if self.failed_error is not None:
            raise EffectExecutionError(self.failed_error)
        stage_enum = stage if isinstance(stage, EffectStage) else EffectStage(str(stage))
        frame = int(time_ctx.get("frame", 0))
        substep_index = int(time_ctx.get("substep_index", 0))
        for entry in self.by_stage[stage_enum]:
            if not self._should_run(entry, frame):
                continue
            if not self._cadence_allows(entry, frame):
                continue
            stats = self.stats[entry.key]
            try:
                if (surface_sdf_error is not None
                        and _requirements_for_entry(entry).surface_sdf):
                    raise EffectPrerequisiteError(
                        "required surface SDF is unavailable: "
                        f"{surface_sdf_error or 'refresh failed'}")
                elements, duration = launch_compiled_kernel(
                    entry, solver, time_ctx, self.device
                )
            except Exception as exc:  # noqa: BLE001 - captured for UI/reporting
                message = f"{type(exc).__name__}: {exc}"
                stats.last_error = message
                runtime_error = EffectRuntimeError(
                    effect_id=entry.instance.definition.effect_id,
                    instance_id=entry.instance.instance_id,
                    kernel_id=entry.compiled_kernel.kernel_definition.kernel_id,
                    stage=stage_enum.value,
                    frame=frame,
                    substep_index=substep_index,
                    message=message,
                )
                self.runtime_errors.append(runtime_error)
                self.failed_error = runtime_error
                raise EffectExecutionError(runtime_error) from exc
            stats.calls += 1
            stats.elements += max(0, int(elements))
            stats.submission_seconds += max(0.0, float(duration))
            stats.last_error = ""


def _resolve_runtime_accessor(solver: Any, accessor: str):
    if accessor.startswith("particles.attr:"):
        key = accessor.split(":", 1)[1]
        return solver.particles.attrs[key]
    value = solver
    for component in accessor.split("."):
        value = getattr(value, component)
    return value


def _wp_value(value: object):
    import warp as wp

    if isinstance(value, tuple):
        if len(value) == 3:
            return wp.vec3(*value)
        if len(value) == 4:
            return wp.vec4(*value)
    return value


def _common_time_inputs(solver: Any, time_ctx: dict[str, object]) -> list[object]:
    return [
        float(getattr(solver.settings, "voxel_size", time_ctx.get("voxel_size", 0.1))),
        int(time_ctx.get("frame", getattr(solver, "frame", 0))),
        float(time_ctx.get("time", getattr(solver, "sim_time", 0.0))),
        float(time_ctx.get("real_time_step", time_ctx.get("dt", 0.0))),
        float(time_ctx.get("solver_time_step", time_ctx.get("dt", 0.0))),
        float(time_ctx.get("time_scale", 1.0)),
        int(time_ctx.get("substeps", 1)),
    ]


def launch_compiled_kernel(
    entry: EffectRuntimeEntry,
    solver: Any,
    time_ctx: dict[str, object],
    device: str,
) -> tuple[int, float]:
    """Launch one compiled kernel against the live solver with resolved typed inputs."""

    import warp as wp

    kernel_definition = entry.compiled_kernel.kernel_definition
    inputs: list[object]
    if kernel_definition.domain is EffectDomain.PARTICLES:
        particles = solver.particles
        # The launch spans capacity and the generated prelude returns past
        # count[0] on the device, so no host count readback is needed (P13).
        launch_dim = max(0, int(getattr(
            particles, "capacity", getattr(particles, "host_count", 0))))
        inputs = [particles.position, particles.velocity, particles.alive, particles.count]
    else:
        grid = solver.grid
        # Normally free: the solver's topology rebuild already committed the
        # count. It only reads back when a rebuilt candidate is still pending,
        # which it must commit before voxel data is addressed.
        leaf_count = int(grid.refresh_topology_counts()) if hasattr(grid, "refresh_topology_counts") else int(
            getattr(grid, "host_leaf_count", 0)
        )
        launch_dim = max(0, int(leaf_count) * 512)
        origin = getattr(solver.grid, "origin", (0.0, 0.0, 0.0))
        if not isinstance(origin, wp.vec3):
            origin = wp.vec3(*origin)
        inputs = [grid.leaf_coords, grid.num_leaves, solver.voxel_type.data, origin]
    inputs.extend(_common_time_inputs(solver, time_ctx))
    for binding in kernel_definition.fields:
        spec = FIELD_SPECS[binding.field]
        inputs.append(_resolve_runtime_accessor(solver, spec.runtime_accessor))
    params = parameter_runtime_inputs(entry.instance, entry.kernel_index)
    inputs.extend(_wp_value(value) for value in params)
    if not kernel_definition.fields and not kernel_definition.parameter_keys:
        inputs.append(0.0)
    started = time.perf_counter()
    wp.launch(entry.compiled_kernel.kernel, dim=launch_dim, inputs=inputs, device=device)
    if _launch_needs_sync(entry):
        wp.synchronize_device(device)
    duration = max(0.0, time.perf_counter() - started)
    return launch_dim, duration


def _launch_needs_sync(entry: EffectRuntimeEntry) -> bool:
    """Synchronize only after a kernel that can fault (P13).

    Safe-mode kernels touch only the current element of capacity-sized
    arrays behind the device-side count guard, so they cannot fault and
    queue asynchronously like the solver's own kernels. Expert kernels may
    index arbitrarily; a sync right after them makes a device fault surface
    here, attributed to this effect, instead of at a later solver sync.
    """
    from .validation import effective_safe_mode

    kernel = entry.compiled_kernel.kernel_definition
    return not effective_safe_mode(entry.instance, kernel) or any(
        binding.access in {FieldAccess.ATOMIC_SCATTER, FieldAccess.UNSAFE_ARBITRARY}
        for binding in kernel.fields)
