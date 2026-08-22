"""Bounded guard-buffer sandbox launches for compiled custom effects."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .enums import EffectDomain
from .fields import FIELD_SPECS
from .schema import EffectIssue
from .source import parameter_runtime_inputs


@dataclass(frozen=True)
class KernelSandboxReport:
    kernel_id: str
    passed: bool
    canaries_intact: bool
    finite: bool
    duration_seconds: float
    elements: int


@dataclass(frozen=True)
class SandboxReport:
    passed: bool
    kernels: tuple[KernelSandboxReport, ...]
    issues: tuple[EffectIssue, ...]


def _wp_parameter(value):
    import warp as wp
    if isinstance(value, tuple):
        if len(value) == 3:
            return wp.vec3(*value)
        if len(value) == 4:
            return wp.vec4(*value)
    return value


def _float_field(wp, size: int, device: str, *, sentinel: float = 0.25):
    import numpy as np
    return wp.array(np.full((size,), sentinel, dtype=np.float32), dtype=float, device=device)


def _int_field(wp, size: int, device: str, *, sentinel: int = 1):
    import numpy as np
    return wp.array(np.full((size,), sentinel, dtype=np.int32), dtype=wp.int32, device=device)


def _vec3_field(wp, size: int, device: str, *, sentinel=(0.25, -0.5, 0.75)):
    import numpy as np
    return wp.array(np.tile(np.asarray(sentinel, dtype=np.float32), (size, 1)), dtype=wp.vec3, device=device)


def _field_array(wp, spec, size: int, device: str):
    if spec.dtype == "float":
        return _float_field(wp, size, device)
    if spec.dtype == "int":
        return _int_field(wp, size, device)
    if spec.dtype == "vec3":
        return _vec3_field(wp, size, device)
    raise ValueError(f"unsupported sandbox dtype {spec.dtype}")


def _finite_array(array) -> bool:
    import numpy as np
    values = array.numpy()
    return bool(np.isfinite(values).all()) if values.dtype.kind == "f" else True


def _equal_tail(array, before, start: int) -> bool:
    import numpy as np
    return bool(np.array_equal(array.numpy()[start:], before[start:]))


def _launch_particle(compiled_kernel, instance, device: str):
    import numpy as np
    import warp as wp

    size = 8
    active_count = 4
    positions_np = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
        [91.0, 92.0, 93.0], [94.0, 95.0, 96.0], [97.0, 98.0, 99.0], [100.0, 101.0, 102.0],
    ], dtype=np.float32)
    velocities_np = np.array([
        [1.0, 2.0, 0.0], [-2.0, 1.0, 0.0], [-1.0, -2.0, 0.0], [2.0, -1.0, 0.0],
        [81.0, 82.0, 83.0], [84.0, 85.0, 86.0], [87.0, 88.0, 89.0], [90.0, 91.0, 92.0],
    ], dtype=np.float32)
    alive_np = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=np.int32)
    count_np = np.array([active_count], dtype=np.int32)
    positions = wp.array(positions_np, dtype=wp.vec3, device=device)
    velocities = wp.array(velocities_np, dtype=wp.vec3, device=device)
    alive = wp.array(alive_np, dtype=wp.int32, device=device)
    count = wp.array(count_np, dtype=wp.int32, device=device)
    arrays = [positions, velocities]
    field_inputs = []
    for binding in compiled_kernel.kernel_definition.fields:
        spec = FIELD_SPECS[binding.field]
        if binding.field == "particle_position":
            array = positions
        elif binding.field == "particle_velocity":
            array = velocities
        else:
            array = _field_array(wp, spec, size, device)
        field_inputs.append(array)
        arrays.append(array)
    before = {id(array): array.numpy().copy() for array in arrays}
    params = [_wp_parameter(value) for value in parameter_runtime_inputs(instance, compiled_kernel.generated.kernel_index)]
    inputs = [positions, velocities, alive, count, 0.1, 1, 0.0, 1.0 / 24.0, 1.0 / 48.0, 1.0, 2]
    inputs.extend(field_inputs)
    inputs.extend(params or ([0.0] if not compiled_kernel.kernel_definition.fields and not compiled_kernel.kernel_definition.parameter_keys else []))
    started = time.perf_counter()
    wp.launch(compiled_kernel.kernel, dim=size, inputs=inputs, device=device)
    wp.synchronize_device(device)
    duration = max(0.0, time.perf_counter() - started)
    canaries = all(_equal_tail(array, before[id(array)], active_count) for array in arrays)
    finite = all(_finite_array(array) for array in arrays)
    return canaries, finite, duration, size


def _launch_voxel(compiled_kernel, instance, device: str):
    import numpy as np
    import warp as wp

    size = 512
    launch_dim = 16
    leaf_coords = wp.array(np.array([[0, 0, 0]], dtype=np.int32), dtype=wp.vec3i, device=device)
    num_leaves = wp.array(np.array([1], dtype=np.int32), dtype=wp.int32, device=device)
    voxel_type = wp.array(np.zeros((size,), dtype=np.int32), dtype=wp.int32, device=device)
    field_inputs = []
    arrays = []
    for binding in compiled_kernel.kernel_definition.fields:
        spec = FIELD_SPECS[binding.field]
        if binding.field == "voxel_type":
            array = voxel_type
        else:
            array = _field_array(wp, spec, size, device)
        field_inputs.append(array)
        arrays.append(array)
    before = {id(array): array.numpy().copy() for array in arrays}
    params = [_wp_parameter(value) for value in parameter_runtime_inputs(instance, compiled_kernel.generated.kernel_index)]
    inputs = [leaf_coords, num_leaves, voxel_type, wp.vec3(0.0, 0.0, 0.0), 0.1, 1, 0.0, 1.0 / 24.0, 1.0 / 48.0, 1.0, 2]
    inputs.extend(field_inputs)
    inputs.extend(params or ([0.0] if not compiled_kernel.kernel_definition.fields and not compiled_kernel.kernel_definition.parameter_keys else []))
    started = time.perf_counter()
    wp.launch(compiled_kernel.kernel, dim=launch_dim, inputs=inputs, device=device)
    wp.synchronize_device(device)
    duration = max(0.0, time.perf_counter() - started)
    canaries = all(_equal_tail(array, before[id(array)], launch_dim) for array in arrays)
    finite = all(_finite_array(array) for array in arrays)
    return canaries, finite, duration, launch_dim


def sandbox_test(compiled_effect, *, device: str = "cpu") -> SandboxReport:
    if not compiled_effect.success:
        return SandboxReport(False, (), compiled_effect.issues)
    reports: list[KernelSandboxReport] = []
    issues: list[EffectIssue] = []
    for compiled_kernel in compiled_effect.kernels:
        try:
            if compiled_kernel.kernel_definition.domain is EffectDomain.PARTICLES:
                canaries, finite, duration, elements = _launch_particle(
                    compiled_kernel, compiled_effect.instance, device
                )
            else:
                canaries, finite, duration, elements = _launch_voxel(
                    compiled_kernel, compiled_effect.instance, device
                )
            passed = canaries and finite and math.isfinite(duration)
            reports.append(KernelSandboxReport(
                compiled_kernel.kernel_definition.kernel_id,
                passed,
                canaries,
                finite,
                duration,
                elements,
            ))
            if not canaries:
                issues.append(EffectIssue(
                    "SANDBOX_CANARY_CORRUPTION",
                    "Kernel wrote outside the bounded active region.",
                    effect_id=compiled_effect.instance.definition.effect_id,
                    instance_id=compiled_effect.instance.instance_id,
                    kernel_id=compiled_kernel.kernel_definition.kernel_id,
                ))
            if not finite:
                issues.append(EffectIssue(
                    "SANDBOX_NONFINITE_OUTPUT",
                    "Kernel produced NaN or Infinity in sandbox data.",
                    effect_id=compiled_effect.instance.definition.effect_id,
                    instance_id=compiled_effect.instance.instance_id,
                    kernel_id=compiled_kernel.kernel_definition.kernel_id,
                ))
        except Exception as exc:  # noqa: BLE001
            issues.append(EffectIssue(
                "SANDBOX_LAUNCH_FAILED",
                f"{type(exc).__name__}: {exc}",
                effect_id=compiled_effect.instance.definition.effect_id,
                instance_id=compiled_effect.instance.instance_id,
                kernel_id=compiled_kernel.kernel_definition.kernel_id,
            ))
    return SandboxReport(not issues and all(report.passed for report in reports), tuple(reports), tuple(issues))
