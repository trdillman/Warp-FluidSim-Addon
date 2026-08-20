"""Transactional execution and capability-derived scheduling for Custom Effects v2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .enums import EffectCadence, EffectDomain, EffectStage
from .fields import FIELD_SPECS, requirements_for_bindings
from .fingerprint import execution_stack_fingerprint
from .source import parameter_runtime_inputs


@dataclass(frozen=True)
class EffectRuntimeRequirements:
    """Prerequisites required by the currently applied effect stack."""

    particles: bool = False
    topology: bool = False
    surface_sdf: bool = False
    collision: bool = False
    rebuild_topology_each_substep: bool = False

    def merged(self, other: "EffectRuntimeRequirements") -> "EffectRuntimeRequirements":
        return EffectRuntimeRequirements(
            particles=self.particles or other.particles,
            topology=self.topology or other.topology,
            surface_sdf=self.surface_sdf or other.surface_sdf,
            collision=self.collision or other.collision,
            rebuild_topology_each_substep=(
                self.rebuild_topology_each_substep or other.rebuild_topology_each_substep
            ),
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
    return EffectRuntimeRequirements(
        particles=particles,
        topology=topology,
        surface_sdf=surface_sdf,
        collision=collision,
        rebuild_topology_each_substep=topology,
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
        except Exception:
            return False
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
        return True

    def clear(self) -> None:
        class _Empty:
            success = True
            compiled_effects = ()

        self.swap_candidate(_Empty())

    def _should_run(self, entry: EffectRuntimeEntry, frame: int) -> bool:
        instance = entry.instance
        kernel = entry.compiled_kernel.kernel_definition
        return bool(instance.active_on_frame(frame) and kernel.active_on_frame(frame))

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

    def run(self, stage: EffectStage | str, solver: Any, time_ctx: dict[str, object]) -> None:
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
        count = int(particles.refresh_host_count()) if hasattr(particles, "refresh_host_count") else int(
            getattr(particles, "host_count", 0)
        )
        launch_dim = max(0, int(getattr(particles, "capacity", count)))
        inputs = [particles.position, particles.velocity, particles.alive, particles.count]
    else:
        grid = solver.grid
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
    wp.synchronize_device(device)
    duration = max(0.0, time.perf_counter() - started)
    return launch_dim, duration
