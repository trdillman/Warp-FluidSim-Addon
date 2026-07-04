#!/usr/bin/env python3
"""Profile the GPU solver with phase timings under Blender's Python.

Run from the repository root, for example:

    "/mnt/c/Program Files/Blender Foundation/Blender 5.1/5.1/python/bin/python.exe" \
        tools/profile_gpu_solver.py --device cuda:0 --frames 8 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[idx]


def _fmt_ms(seconds):
    return seconds * 1000.0


def _phase_labels():
    from fluidSim_warp_blender51.diagnostics.profiler import PHASE_SLOTS

    return {slot: phase for phase, slot in PHASE_SLOTS.items()}


def _build_settings(args):
    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_settings

    settings = default_benchmark_settings()
    settings.device = args.device
    settings.voxel_size = args.voxel_size
    settings.max_particles = args.max_particles
    settings.max_whitewater = args.max_whitewater
    settings.min_substeps = args.min_substeps
    settings.max_substeps = args.max_substeps
    settings.surface_enabled = not args.no_surface
    settings.whitewater_enabled = args.whitewater
    return settings


def run_profile(args):
    import warp as wp

    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_sources
    from fluidSim_warp_blender51.diagnostics.profiler import Profiler
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    wp.init()
    if str(args.device).startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA is not available")

    settings = _build_settings(args)
    profiler = Profiler(device=args.device, sync=args.sync, enabled=not args.no_phase_timers)
    solver = FluidSolver(settings, profiler=profiler)
    solver.set_sources(default_benchmark_sources())

    measured = []
    raw_frames = []
    total_frames = args.warmup + args.frames
    t_begin = time.perf_counter()
    for frame in range(total_frames):
        if frame == args.emit_frames:
            solver.set_sources([])
        stats = solver.step_frame()
        if frame == args.warmup - 1:
            profiler.history.clear()
        if frame >= args.warmup:
            record = stats.as_dict()
            record["pressureIterations"] = stats.pressure_iterations
            record["pressureResidual"] = stats.pressure_residual
            record["substeps"] = stats.substeps
            measured.append(stats)
            raw_frames.append(record)
    wall_total = time.perf_counter() - t_begin

    step_times = [s.step_time for s in measured]
    active_voxels = [s.active_voxels for s in measured]
    active_particles = [s.active_particles for s in measured]
    whitewater_particles = [s.whitewater_particles for s in measured]
    substeps = [s.substeps for s in measured]
    pressure_iters = [s.pressure_iterations for s in measured]
    phase_labels = _phase_labels()
    phase_means = {}
    for stats in measured:
        for slot, value in stats.timers.items():
            if slot.startswith("compute_") and value > 0.0:
                phase_means.setdefault(slot, []).append(float(value))
    phase_means = {slot: _mean(values) for slot, values in phase_means.items()}
    top_phases = sorted(phase_means.items(), key=lambda item: item[1], reverse=True)[:5]

    avg_voxels = _mean(active_voxels)
    avg_particles = _mean(active_particles)
    summary = {
        "result": "PROFILE-GPU-SOLVER-PASS",
        "device": str(wp.get_device(args.device).name),
        "warpVersion": getattr(wp, "__version__", "unknown"),
        "frames": args.frames,
        "warmup": args.warmup,
        "wallTotalSeconds": wall_total,
        "stepMsMean": _fmt_ms(_mean(step_times)),
        "stepMsMin": _fmt_ms(min(step_times) if step_times else 0.0),
        "stepMsMax": _fmt_ms(max(step_times) if step_times else 0.0),
        "stepMsP95": _fmt_ms(_p95(step_times)),
        "activeVoxelsMean": avg_voxels,
        "activeParticlesMean": avg_particles,
        "whitewaterParticlesMean": _mean(whitewater_particles),
        "substepsMean": _mean(substeps),
        "pressureIterationsMean": _mean(pressure_iters),
        "pressureResidualLast": measured[-1].pressure_residual if measured else 0.0,
        "msPer100kActiveVoxels": _fmt_ms(_mean(step_times)) / max(avg_voxels / 100000.0, 1.0e-9),
        "msPer100kParticles": _fmt_ms(_mean(step_times)) / max(avg_particles / 100000.0, 1.0e-9),
        "topPhases": [
            {"slot": slot, "phase": phase_labels.get(slot, slot), "meanMs": value}
            for slot, value in top_phases
        ],
    }
    return summary, raw_frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--emit-frames", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--max-particles", type=int, default=400000)
    parser.add_argument("--max-whitewater", type=int, default=200000)
    parser.add_argument("--min-substeps", type=int, default=1)
    parser.add_argument("--max-substeps", type=int, default=4)
    parser.add_argument("--whitewater", action="store_true")
    parser.add_argument("--no-surface", action="store_true")
    parser.add_argument("--sync", action="store_true", help="CUDA-sync every profiled phase for accurate phase timings.")
    parser.add_argument("--no-phase-timers", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    summary, raw_frames = run_profile(args)
    print(summary["result"])
    print(f"device: {summary['device']}")
    print(f"warp: {summary['warpVersion']}")
    print(f"frames: {summary['frames']} measured, {summary['warmup']} warmup")
    print(f"wall_total_s: {summary['wallTotalSeconds']:.3f}")
    print(
        "step_ms: "
        f"mean={summary['stepMsMean']:.3f} "
        f"min={summary['stepMsMin']:.3f} "
        f"max={summary['stepMsMax']:.3f} "
        f"p95={summary['stepMsP95']:.3f}"
    )
    print(
        "means: "
        f"voxels={summary['activeVoxelsMean']:.1f} "
        f"particles={summary['activeParticlesMean']:.1f} "
        f"whitewater={summary['whitewaterParticlesMean']:.1f} "
        f"substeps={summary['substepsMean']:.2f} "
        f"pressure_iters={summary['pressureIterationsMean']:.2f}"
    )
    print(
        "normalized: "
        f"ms_per_100k_voxels={summary['msPer100kActiveVoxels']:.3f} "
        f"ms_per_100k_particles={summary['msPer100kParticles']:.3f}"
    )
    if summary["topPhases"]:
        print("top_phases:")
        for phase in summary["topPhases"]:
            print(f"  {phase['phase']} ({phase['slot']}): {phase['meanMs']:.3f} ms")

    if args.json_out:
        path = Path(args.json_out)
        if path.parent and not path.parent.exists():
            raise FileNotFoundError(f"Output parent does not exist: {path.parent}")
        path.write_text(json.dumps({"summary": summary, "frames": raw_frames}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
