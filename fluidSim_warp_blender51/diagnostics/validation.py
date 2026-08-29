"""Product validation orchestration owned by the diagnostics layer."""

from __future__ import annotations

import time

from ..sim import validation as simulation_validation
from ..sim.validation import ValidationResult
from .benchmark import (
    BENCHMARK_FIELDS,
    compare_benchmark,
    default_benchmark_settings,
    run_benchmark,
)
from .profiler import PROFILE_FRAME_FIELDS, Profiler

BENCHMARK_SCHEMA_TEST = "benchmark profile schema compatibility"
VALIDATION_TESTS = [*simulation_validation.VALIDATION_TESTS, BENCHMARK_SCHEMA_TEST]


def test_benchmark_schema(device: str) -> tuple[bool, str]:
    """Exercise benchmark and frame-stat schemas without measuring performance."""
    from ..sim.solver import FluidSolver, SolverSettings

    settings = default_benchmark_settings()
    settings.voxel_size = 0.08
    settings.max_particles = 400_000
    settings.device = device
    result = run_benchmark(frames=4, emit_frames=2, settings=settings)
    data = result.as_dict()
    missing = [field for field in BENCHMARK_FIELDS if field not in data]
    if missing:
        return False, f"missing benchmark fields: {missing}"
    comparison = compare_benchmark(data, data)
    if comparison["result"] != "pass":
        return False, f"self-comparison failed: {comparison['result']}"

    profiler = Profiler(device=device)
    frame_settings = SolverSettings()
    frame_settings.domain_min = (-0.4, -0.4, 0.0)
    frame_settings.domain_max = (0.4, 0.4, 0.8)
    frame_settings.voxel_size = 0.08
    frame_settings.max_particles = 100_000
    frame_settings.device = device
    solver = FluidSolver(frame_settings, profiler=profiler)
    try:
        frame_data = solver.step_frame().as_dict()
    finally:
        solver.release()
    missing_frame_fields = [
        field for field in PROFILE_FRAME_FIELDS if field not in frame_data
    ]
    if missing_frame_fields:
        return False, f"missing frame fields: {missing_frame_fields}"
    return True, (
        f"benchmark fields ok ({len(BENCHMARK_FIELDS)}), "
        f"frame fields ok ({len(PROFILE_FRAME_FIELDS)}), self-compare pass"
    )


def run_all(device: str = "cuda:0", progress=None) -> list[ValidationResult]:
    """Run simulation validations followed by diagnostics-owned contracts."""
    results = simulation_validation.run_all(device=device, progress=progress)
    started = time.perf_counter()
    try:
        passed, detail = test_benchmark_schema(device)
    except Exception as exc:  # noqa: BLE001 - validation reports failures
        passed, detail = False, f"exception: {exc!r}"
    result = ValidationResult(
        BENCHMARK_SCHEMA_TEST,
        passed,
        detail,
        time.perf_counter() - started,
    )
    results.append(result)
    if progress is not None:
        progress(result)
    return results
