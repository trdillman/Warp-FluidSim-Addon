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


# Flag defaults this tool applies when the caller does not explicitly supply
# the flag (the parser defaults these flags to None so "explicit" can be
# tracked independently of value -- otherwise an explicit override equal to
# the ordinary default would be indistinguishable from an omitted flag). The
# dam_break scene preset only substitutes demo-scene values for flags the
# caller left unset, so an explicit CLI value always wins.
_DEFAULTS = {
    "voxel_size": 0.08,
    "max_particles": 400000,
    "min_substeps": 1,
    "max_substeps": 4,
}


def _phase_labels():
    from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS

    return {
        f"compute_{slot}": phase
        for phase, slot in TIMER_PHASE_SLOTS.items()
    }


# The classic dam-break benchmark: the domain, resolution and liquid column
# demos/dam_break.blend used before its 2026-09-29 rocky-cove redesign, with no
# colliders. It stays fixed so profiles remain comparable across releases; the
# shipped demo now adds mesh colliders a bpy-free profile cannot load.
DAM_BREAK_SCENE = {
    "domain_min": (0.0, -1.00146496, 0.0),
    "domain_max": (5.6, 1.00146496, 1.6),
    "voxel_size": 0.03,
    "max_particles": 600_000,
    "min_substeps": 1,
    "max_substeps": 16,
    "whitewater_enabled": True,
    # The source column lo=(0,0,0), hi=(0.6,0.6,1.2) -> centre (0.3,0.3,0.6),
    # half extents (0.3,0.3,0.6).
    "source_position": (0.3, 0.3, 0.6),
    "source_half_extents": (0.3, 0.3, 0.6),
    "source_emit_rate": 8.0,
    "source_velocity": (0.7, 0.0, 0.0),
}


def dam_break_sources():
    """Source column of the classic dam-break benchmark."""
    from fluidSim_warp_blender51.sim.enums import (
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
    from fluidSim_warp_blender51.sim.solver import SourceShapeDesc

    box = SourceShapeDesc()
    box.shape = SourceShapeShape.BOX
    box.stype = SourceShapeType.SOURCE
    box.mode = SourceShapeSourceMode.PARTICLES
    box.position = DAM_BREAK_SCENE["source_position"]
    box.half_extents = DAM_BREAK_SCENE["source_half_extents"]
    box.emit_rate = DAM_BREAK_SCENE["source_emit_rate"]
    box.velocity = DAM_BREAK_SCENE["source_velocity"]
    return [box]


def build_sources(args):
    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_sources

    if getattr(args, "scene_preset", "benchmark") == "dam_break":
        return dam_break_sources()
    return default_benchmark_sources()


def _sdf_splat_radius(settings):
    """Particle-splat radius (voxels) SurfacePipeline.build_sdf uses on the
    solver grid, including the enlargement surface tension forces."""
    import math

    from fluidSim_warp_blender51.sim.surface_tension_math import (
        required_sdf_splat_radius_voxels,
    )

    h = float(settings.voxel_size)
    radius = 0.5 * h * float(settings.particle_radius_scale)
    splat = max(1, int(settings.surface_splat_radius),
                int(math.ceil(radius / h + 1.0)))
    if (bool(getattr(settings, "surface_tension_enabled", False))
            and float(getattr(settings, "surface_tension_coefficient", 0.0)) > 0.0):
        filters = sum(max(0, int(value)) for value in (
            settings.surface_filter_iterations,
            settings.surface_dilate_iterations,
            settings.surface_erode_iterations,
            settings.surface_smooth_iterations,
        ))
        splat = max(splat, required_sdf_splat_radius_voxels(
            radius / h,
            float(getattr(settings, "surface_tension_band_voxels", 1.5)),
            int(getattr(settings, "surface_tension_smoothing_iterations", 1)),
            filters))
    return int(splat)


def _build_settings(args):
    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_settings

    settings = default_benchmark_settings()
    settings.device = args.device
    # None means the flag was left unset -- fall back to the ordinary default
    # -- while an explicit value (even one equal to that default) is tracked
    # as explicit so the dam_break preset below never overrides it.
    explicit = {
        flag: getattr(args, flag) is not None
        for flag in ("voxel_size", "max_particles", "min_substeps", "max_substeps")
    }
    for flag in explicit:
        setattr(settings, flag, getattr(args, flag) if explicit[flag] else _DEFAULTS[flag])
    settings.max_whitewater = args.max_whitewater
    settings.surface_enabled = not args.no_surface
    settings.whitewater_enabled = args.whitewater
    if getattr(args, "scene_preset", "benchmark") == "dam_break":
        settings.domain_min = DAM_BREAK_SCENE["domain_min"]
        settings.domain_max = DAM_BREAK_SCENE["domain_max"]
        for flag, is_explicit in explicit.items():
            if not is_explicit:
                setattr(settings, flag, DAM_BREAK_SCENE[flag])
        if not args.whitewater:
            settings.whitewater_enabled = DAM_BREAK_SCENE["whitewater_enabled"]
    if getattr(args, "no_whitewater", False):
        settings.whitewater_enabled = False
    settings.viscosity_enabled = bool(getattr(args, "viscosity", False))
    if settings.viscosity_enabled:
        settings.viscosity_constant = float(args.viscosity_constant)
        if args.viscosity_iterations is not None:
            settings.viscosity_iterations = int(args.viscosity_iterations)
        if args.viscosity_tolerance is not None:
            settings.viscosity_tolerance = float(args.viscosity_tolerance)
    settings.surface_tension_enabled = bool(getattr(args, "surface_tension", False))
    if settings.surface_tension_enabled:
        settings.surface_tension_coefficient = float(
            args.surface_tension_coefficient)
    return settings


def run_profile(args, on_frame=None):
    """Run one profile.

    ``on_frame(index, stats)`` is called after every stepped frame,
    warmup frames included, so callers running external probes can
    reset their counters at the warmup boundary.
    """
    import warp as wp

    from fluidSim_warp_blender51.diagnostics.profiler import Profiler
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    wp.init()
    if str(args.device).startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA is not available")

    settings = _build_settings(args)
    profiler = Profiler(device=args.device, sync=args.sync, enabled=not args.no_phase_timers)
    solver = FluidSolver(settings, profiler=profiler)
    solver.set_source_batch(build_sources(args))

    raw_frames = []
    total_frames = args.warmup + args.frames
    t_begin = time.perf_counter()
    try:
        for frame in range(total_frames):
            if frame == args.emit_frames:
                solver.set_source_batch()
            stats = solver.step_frame()
            if on_frame is not None:
                on_frame(frame, stats)
            if frame == args.warmup - 1:
                profiler.history.clear()
            if frame >= args.warmup:
                record = stats.as_dict()
                record["pressureIterations"] = stats.pressure_iterations
                record["pressureResidual"] = stats.pressure_residual
                record["substeps"] = stats.substeps
                record["surfaceTensionStatus"] = stats.surface_tension_status
                record["substepLimiter"] = stats.substep_limiter
                raw_frames.append(record)
        wall_total = time.perf_counter() - t_begin
    finally:
        # Repeated profile runs in one process (paired A/B sweeps) must not
        # accumulate device allocations from earlier solvers.
        solver.release()

    step_times = [record["stepTime"] for record in raw_frames]
    active_voxels = [record["activeVoxels"] for record in raw_frames]
    active_particles = [record["activeParticles"] for record in raw_frames]
    whitewater_particles = [record["whitewaterParticles"] for record in raw_frames]
    substeps = [record["substeps"] for record in raw_frames]
    pressure_iters = [record["pressureIterations"] for record in raw_frames]
    phase_labels = _phase_labels()
    phase_means = {}
    for record in raw_frames:
        for slot, value in record["timers"].items():
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
        "pressureResidualLast": raw_frames[-1]["pressureResidual"] if raw_frames else 0.0,
        "pressurePhaseMsMean": phase_means.get("compute_5", 0.0),
        "forcesPhaseMsMean": phase_means.get("compute_3", 0.0),
        "viscosityPhaseMsMean": phase_means.get("compute_4", 0.0),
        "surfacePhaseMsMean": phase_means.get("compute_11", 0.0),
        "whitewaterPhaseMsMean": phase_means.get("compute_12", 0.0),
        "viscosityEnabled": bool(settings.viscosity_enabled),
        "viscosityConstant": float(settings.viscosity_constant),
        "viscosityIterations": int(settings.viscosity_iterations),
        "viscosityTolerance": float(settings.viscosity_tolerance),
        "sdfSplatRadiusVoxels": _sdf_splat_radius(settings),
        "surfaceTensionEnabled": bool(settings.surface_tension_enabled),
        "whitewaterEnabled": bool(settings.whitewater_enabled),
        "voxelSize": float(settings.voxel_size),
        "maxSubsteps": int(settings.max_substeps),
        "scenePreset": str(getattr(args, "scene_preset", "benchmark")),
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
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--max-particles", type=int, default=None)
    parser.add_argument("--max-whitewater", type=int, default=200000)
    parser.add_argument("--min-substeps", type=int, default=None)
    parser.add_argument("--max-substeps", type=int, default=None)
    parser.add_argument("--whitewater", action="store_true")
    parser.add_argument("--no-surface", action="store_true")
    parser.add_argument("--sync", action="store_true", help="CUDA-sync every profiled phase for accurate phase timings.")
    parser.add_argument("--no-phase-timers", action="store_true")
    parser.add_argument(
        "--scene-preset", choices=("benchmark", "dam_break"), default="benchmark",
        help="'dam_break' is the classic dam-break benchmark: the pre-cove "
             "demos/dam_break.blend domain, resolution, substep range, "
             "whitewater state, and source column, without colliders.")
    parser.add_argument("--no-whitewater", action="store_true")
    parser.add_argument("--viscosity", action="store_true")
    parser.add_argument("--viscosity-constant", type=float, default=1.0)
    parser.add_argument("--viscosity-iterations", type=int, default=None)
    parser.add_argument(
        "--viscosity-tolerance", type=float, default=None,
        help="Override the solver default (C.DEFAULT_VISCOSITY_TOLERANCE).")
    parser.add_argument("--surface-tension", action="store_true")
    parser.add_argument("--surface-tension-coefficient", type=float, default=0.072)
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
