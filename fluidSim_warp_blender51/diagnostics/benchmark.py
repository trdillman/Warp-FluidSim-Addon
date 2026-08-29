"""Benchmark runner implementing the synthesized BENCHMARK_FIELDS contract.

Runs a deterministic scenario through the FluidSolver, aggregates per-frame
stats into the benchmark schema, and compares against stored baselines with
exact matching on active-voxel statistics and tolerant matching on
performance fields.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

BENCHMARK_FIELDS = [
    "totalTime",
    "timeStepAverage",
    "timeStepMinimum",
    "timeStepMaximum",
    "voxelsPerSecondAverage",
    "voxelsPerSecondMinimum",
    "voxelsPerSecondMaximum",
    "activeVoxelsAverage",
    "activeVoxelsMinimum",
    "activeVoxelsMaximum",
    "result",
]

# Fields that must match a baseline exactly (determinism check).
EXACT_FIELDS = ["activeVoxelsAverage", "activeVoxelsMinimum", "activeVoxelsMaximum"]
# Fields compared with relative tolerance (performance check).
PERFORMANCE_FIELDS = ["timeStepAverage", "voxelsPerSecondAverage"]
NUMERIC_CONTRACT_FIELDS = [
    name for name in BENCHMARK_FIELDS if name != "result"
]


def _numeric_field_error(record: dict, name: str, side: str) -> str | None:
    if name not in record:
        return f"{side} is missing the contract field"
    value = record[name]
    if type(value) not in (int, float) or not math.isfinite(value):
        return f"{side} must be a finite JSON number"
    return None


@dataclass
class BenchmarkResult:
    total_time: float = 0.0
    step_times: list = field(default_factory=list)
    active_voxels: list = field(default_factory=list)
    voxels_per_second: list = field(default_factory=list)
    frames: int = 0
    result: str = "unknown"

    def as_dict(self) -> dict:
        st = self.step_times or [0.0]
        av = self.active_voxels or [0]
        vps = self.voxels_per_second or [0.0]
        return {
            "totalTime": self.total_time,
            "timeStepAverage": sum(st) / len(st),
            "timeStepMinimum": min(st),
            "timeStepMaximum": max(st),
            "voxelsPerSecondAverage": sum(vps) / len(vps),
            "voxelsPerSecondMinimum": min(vps),
            "voxelsPerSecondMaximum": max(vps),
            "activeVoxelsAverage": sum(av) // len(av),
            "activeVoxelsMinimum": min(av),
            "activeVoxelsMaximum": max(av),
            "result": self.result,
            "frames": self.frames,
        }


def default_benchmark_settings():
    """Deterministic dam-break scenario used for benchmark comparisons."""
    from ..sim.solver import SolverSettings

    settings = SolverSettings()
    settings.domain_min = (-1.0, -1.0, 0.0)
    settings.domain_max = (1.0, 1.0, 2.0)
    settings.voxel_size = 0.05
    settings.max_particles = 1_500_000
    settings.whitewater_enabled = True
    settings.surface_enabled = True
    return settings


def default_benchmark_sources():
    from ..sim.enums import SourceShapeShape, SourceShapeSourceMode, SourceShapeType
    from ..sim.solver import SourceShapeDesc

    box = SourceShapeDesc()
    box.shape = SourceShapeShape.BOX
    box.stype = SourceShapeType.SOURCE
    box.mode = SourceShapeSourceMode.PARTICLES
    box.position = (-0.5, 0.0, 1.0)
    box.half_extents = (0.45, 0.9, 0.9)
    box.emit_rate = 8.0
    return [box]


def run_benchmark(
    frames: int = 24,
    emit_frames: int = 6,
    settings=None,
    sources=None,
    progress=None,
) -> BenchmarkResult:
    """Run the benchmark scenario and return aggregated results."""
    from ..sim.solver import FluidSolver

    settings = settings or default_benchmark_settings()
    sources = sources if sources is not None else default_benchmark_sources()

    solver = FluidSolver(settings)
    try:
        solver.set_sources(sources)

        result = BenchmarkResult()
        t_begin = time.perf_counter()
        for f in range(frames):
            if f == emit_frames:
                solver.set_sources([])
            stats = solver.step_frame()
            result.step_times.append(stats.step_time)
            result.active_voxels.append(stats.active_voxels)
            vps = (
                stats.active_voxels / stats.step_time
                if stats.step_time > 0
                else 0.0
            )
            result.voxels_per_second.append(vps)
            result.frames += 1
            if progress is not None:
                progress(f + 1, frames, stats)
        result.total_time = time.perf_counter() - t_begin
        result.result = "completed"
        return result
    finally:
        solver.release()


def compare_benchmark(current: dict, baseline: dict, perf_tolerance: float = 0.25) -> dict:
    """Compare a benchmark run against a baseline.

    Exact fields must match exactly; performance fields must be within
    ``perf_tolerance`` relative deviation. Returns a comparison report.
    """
    report = {
        "contract": {}, "exact": {}, "performance": {}, "result": "pass",
    }
    if not isinstance(current, dict):
        return {**report, "result": "fail", "error": "current must be a JSON object"}
    if not isinstance(baseline, dict):
        return {**report, "result": "fail", "error": "baseline must be a JSON object"}

    for name in NUMERIC_CONTRACT_FIELDS:
        error = (
            _numeric_field_error(current, name, "current")
            or _numeric_field_error(baseline, name, "baseline")
        )
        detail = {
            "current": current.get(name),
            "baseline": baseline.get(name),
            "valid": error is None,
        }
        if error is not None:
            detail["error"] = error
            report.setdefault("error", f"{name}: {error}")
            report["result"] = "fail"
        report["contract"][name] = detail

    for name in EXACT_FIELDS:
        error = report["contract"][name].get("error")
        detail = {
            "current": current.get(name),
            "baseline": baseline.get(name),
            "match": error is None and current[name] == baseline[name],
        }
        if error is not None:
            detail["error"] = error
            report.setdefault("error", f"{name}: {error}")
        report["exact"][name] = detail
        if not detail["match"]:
            report["result"] = "fail"

    for name in PERFORMANCE_FIELDS:
        error = report["contract"][name].get("error")
        if error is not None:
            report["performance"][name] = {
                "current": current.get(name),
                "baseline": baseline.get(name),
                "relative_delta": None,
                "within_tolerance": False,
                "error": error,
            }
            report.setdefault("error", f"{name}: {error}")
            report["result"] = "fail"
            continue

        cur = float(current[name])
        base = float(baseline[name])
        # A zero baseline only matches another zero.  Previously every current
        # value had a zero relative delta and false performance passes.
        rel = (abs(cur - base) / abs(base) if base
               else (0.0 if cur == 0.0 else float("inf")))
        ok = rel <= perf_tolerance
        report["performance"][name] = {
            "current": cur,
            "baseline": base,
            "relative_delta": rel,
            "within_tolerance": ok,
        }
        if not ok and report["result"] == "pass":
            report["result"] = "performance_regression"
    return report


def save_benchmark(path: str, result: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Output parent directory does not exist: {parent}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def load_benchmark(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
