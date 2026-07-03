"""Benchmark runner implementing the synthesized BENCHMARK_FIELDS contract.

Runs a deterministic scenario through the FluidSolver, aggregates per-frame
stats into the benchmark schema, and compares against stored baselines with
exact matching on active-voxel statistics and tolerant matching on
performance fields.
"""

from __future__ import annotations

import json
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
    solver.set_sources(sources)

    result = BenchmarkResult()
    t_begin = time.perf_counter()
    for f in range(frames):
        if f == emit_frames:
            solver.set_sources([])
        stats = solver.step_frame()
        result.step_times.append(stats.step_time)
        result.active_voxels.append(stats.active_voxels)
        vps = stats.active_voxels / stats.step_time if stats.step_time > 0 else 0.0
        result.voxels_per_second.append(vps)
        result.frames += 1
        if progress is not None:
            progress(f + 1, frames, stats)
    result.total_time = time.perf_counter() - t_begin
    result.result = "completed"
    return result


def compare_benchmark(current: dict, baseline: dict, perf_tolerance: float = 0.25) -> dict:
    """Compare a benchmark run against a baseline.

    Exact fields must match exactly; performance fields must be within
    ``perf_tolerance`` relative deviation. Returns a comparison report.
    """
    report = {"exact": {}, "performance": {}, "result": "pass"}
    for name in EXACT_FIELDS:
        ok = current.get(name) == baseline.get(name)
        report["exact"][name] = {
            "current": current.get(name),
            "baseline": baseline.get(name),
            "match": ok,
        }
        if not ok:
            report["result"] = "fail"
    for name in PERFORMANCE_FIELDS:
        cur = float(current.get(name, 0.0))
        base = float(baseline.get(name, 0.0))
        rel = abs(cur - base) / base if base else 0.0
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
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
