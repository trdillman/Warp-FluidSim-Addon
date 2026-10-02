"""Cross-version dam_break benchmark: the same workload on several checkouts.

Each version is a checkout (usually a detached ``git worktree``) of this
repository. Every trial is a fresh process running one version on one
workload; rounds interleave the versions in rotated order so slow drift on
the machine (thermal, background load) spreads evenly instead of landing on
whichever version ran last.

The child only touches API every benchmarked version shares (SolverSettings
fields, ``FluidSolver(settings, profiler)``, ``set_sources``, ``step_frame``,
``release``), and times each frame on the host around ``step_frame()`` plus
a device synchronize, so no version's own stats decide its score.

    python tools/benchmark_versions.py \\
        --version pre=C:/wt/bench_pre --version 2.0=C:/wt/rev015 \\
        --trials 5 --out-dir artifacts/bench

Run it with the Python that has the Warp you want to test on its path.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORKLOADS = ("plain", "visc", "st", "visc_st")
EMIT_FRAMES = 4


def _child(root, workload, frames, warmup, out):
    sys.path.insert(0, root)
    import warp as wp

    wp.config.quiet = True
    wp.init()

    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_settings
    from fluidSim_warp_blender51.sim.enums import (
        SourceShapeShape, SourceShapeSourceMode, SourceShapeType)
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SourceShapeDesc

    # demos/dam_break.blend, as mirrored by tools/profile_gpu_solver.py
    s = default_benchmark_settings()
    s.device = "cuda:0"
    s.domain_min = (0.0, -1.00146496, 0.0)
    s.domain_max = (5.6, 1.00146496, 1.6)
    s.voxel_size = 0.03
    s.max_particles = 600_000
    s.max_whitewater = 200_000
    s.min_substeps = 1
    s.max_substeps = 16
    s.surface_enabled = True
    s.whitewater_enabled = True
    # Versions before the Export Attributes switch always exported the enabled
    # attributes; turn it on where it exists so every version does that work.
    if hasattr(s, "export_attributes"):
        s.export_attributes = True
    s.viscosity_enabled = workload in ("visc", "visc_st")
    if s.viscosity_enabled:
        s.viscosity_constant = 1.0
    s.surface_tension_enabled = workload in ("st", "visc_st")
    if s.surface_tension_enabled:
        s.surface_tension_coefficient = 0.072

    box = SourceShapeDesc()
    box.shape = SourceShapeShape.BOX
    box.stype = SourceShapeType.SOURCE
    box.mode = SourceShapeSourceMode.PARTICLES
    box.position = (0.3, 0.3, 0.6)
    box.half_extents = (0.3, 0.3, 0.6)
    box.emit_rate = 8.0
    box.velocity = (0.7, 0.0, 0.0)

    solver = FluidSolver(s, profiler=None)
    solver.set_sources([box])
    records = []
    try:
        for frame in range(warmup + frames):
            if frame == EMIT_FRAMES:
                solver.set_sources([])
            wp.synchronize_device("cuda:0")
            t0 = time.perf_counter()
            stats = solver.step_frame()
            wp.synchronize_device("cuda:0")
            dt = time.perf_counter() - t0
            if frame >= warmup:
                d = stats.as_dict()
                records.append({
                    "frame": frame,
                    "hostMs": 1000.0 * dt,
                    "activeParticles": d.get("activeParticles"),
                    "activeVoxels": d.get("activeVoxels"),
                    "substeps": getattr(stats, "substeps", None),
                    "pressureIterations": getattr(stats, "pressure_iterations", None),
                })
    finally:
        solver.release()

    host = [r["hostMs"] for r in records]
    substeps = sum(r["substeps"] or 0 for r in records)
    result = {
        "root": root, "workload": workload, "warp": wp.config.version,
        "frames": frames, "warmup": warmup,
        "hostMsMedian": statistics.median(host),
        "hostMsMean": statistics.fmean(host),
        "substepsTotal": substeps,
        "msPerSubstep": sum(host) / substeps if substeps else None,
        "particlesLast": records[-1]["activeParticles"],
        "voxelsMean": statistics.fmean(r["activeVoxels"] for r in records),
        "records": records,
    }
    Path(out).write_text(json.dumps(result, indent=1))


def _summarize(runs, labels, workloads):
    """Per (workload, version): median of trial medians, trial range, and the
    median per-round ratio against the first version (paired by round)."""
    table = {}
    base = labels[0]
    for workload in workloads:
        for label in labels:
            mine = [r for r in runs if r["label"] == label and r["workload"] == workload]
            medians = [r["hostMsMedian"] for r in mine]
            ratios = []
            for r in mine:
                ref = [b for b in runs if b["label"] == base and b["workload"] == workload
                       and b["round"] == r["round"]]
                if ref:
                    ratios.append(r["hostMsMedian"] / ref[0]["hostMsMedian"])
            table[f"{workload}/{label}"] = {
                "trials": len(medians),
                "msMedian": statistics.median(medians),
                "msMin": min(medians), "msMax": max(medians),
                "msPerSubstep": statistics.median(r["msPerSubstep"] for r in mine),
                "substepsTotal": statistics.median(r["substepsTotal"] for r in mine),
                "particlesLast": statistics.median(r["particlesLast"] for r in mine),
                "ratioVsFirst": statistics.median(ratios) if ratios else None,
                "ratioVsFirstRange": [min(ratios), max(ratios)] if ratios else None,
            }
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="append", required=True,
                        help="LABEL=CHECKOUT_ROOT; the first one is the ratio baseline")
    parser.add_argument("--workload", action="append", choices=WORKLOADS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    versions = [v.split("=", 1) for v in args.version]
    labels = [label for label, _ in versions]
    workloads = args.workload or list(WORKLOADS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for rnd in range(args.trials):
        for w_index, workload in enumerate(workloads):
            shift = (rnd + w_index) % len(versions)
            for label, root in versions[shift:] + versions[:shift]:
                out = out_dir / f"r{rnd}_{workload}_{label}.json"
                proc = subprocess.run(
                    [sys.executable, "-u", os.path.abspath(__file__), "--child",
                     root, workload, str(args.frames), str(args.warmup), str(out)],
                    capture_output=True, text=True)
                if proc.returncode != 0:
                    (out_dir / f"r{rnd}_{workload}_{label}.err").write_text(
                        proc.stdout + proc.stderr)
                    raise SystemExit(f"{label}/{workload} round {rnd} failed "
                                     f"(exit {proc.returncode}); see {out_dir}")
                run = json.loads(out.read_text())
                run.update(label=label, round=rnd)
                runs.append(run)
                print(f"round {rnd} {workload:8s} {label:8s} "
                      f"median={run['hostMsMedian']:8.2f} ms "
                      f"ms/substep={run['msPerSubstep']:.2f}", flush=True)

    summary = {
        "versions": dict(versions), "workloads": workloads,
        "trials": args.trials, "frames": args.frames, "warmup": args.warmup,
        "warp": runs[0]["warp"],
        "table": _summarize(runs, labels, workloads),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["table"], indent=1))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        root, workload, frames, warmup, out = sys.argv[2:7]
        _child(root, workload, int(frames), int(warmup), out)
        sys.exit(0)
    sys.exit(main())
