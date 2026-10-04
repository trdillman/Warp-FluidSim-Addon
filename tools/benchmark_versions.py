"""Cross-version dam_break benchmark: the same workload on several checkouts.

Each version is a checkout (usually a detached ``git worktree``) of this
repository. Every trial is a fresh process running one version on one
workload; rounds interleave the versions in rotated order so slow drift on
the machine (thermal, background load) spreads evenly instead of landing on
whichever version ran last.

The child only touches API every benchmarked version shares (SolverSettings
fields, ``FluidSolver(settings, profiler)``, ``set_source_batch`` (or
``set_sources`` on older versions), ``step_frame``, ``release``), and times each frame on the host around ``step_frame()`` plus
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
import tempfile
import tarfile
import shutil
import hashlib
from pathlib import Path

WORKLOADS = ("plain", "visc", "st", "visc_st")
EMIT_FRAMES = 4
REFERENCE_COMMITS = {
    "1.0.255": "9c0d99e62adda14bb7b075a447c01fefe7187da3",
    "1.1.0": "c5919d668f469942200c58274eca13337d3aa624",
}


def _inventory(root):
    paths = list((root / "fluidSim_warp_blender51").rglob("*"))
    paths += [root / "demos" / f"{demo}.blend" for demo in
              ("dam_break", "crown_splash", "paddle_stir", "chromatic_vortex")]
    paths += [root / "tools" / name for name in
              ("repo_addon_loader.py", "benchmark_demo_trial.py", "benchmark_versions.py",
               "performance_budget.py", "run_production_gate.py")]
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _snapshot_candidate(source, destination):
    """Bind all trials to one private copy, even while other chats edit master."""
    before = _inventory(source)
    shutil.copytree(source / "fluidSim_warp_blender51", destination / "fluidSim_warp_blender51",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "demos").mkdir()
    for demo in ("dam_break", "crown_splash", "paddle_stir", "chromatic_vortex"):
        shutil.copy2(source / "demos" / f"{demo}.blend", destination / "demos" / f"{demo}.blend")
    (destination / "tools").mkdir()
    for name in ("repo_addon_loader.py", "benchmark_demo_trial.py", "benchmark_versions.py",
                 "performance_budget.py", "run_production_gate.py"):
        shutil.copy2(source / "tools" / name, destination / "tools" / name)
    if before != _inventory(source) or before != _inventory(destination):
        raise RuntimeError("candidate changed during snapshot; rerun with stable source")
    return before


def _run_trial(command, out, *, env=None, cwd=None, expect_marker=None):
    """One fresh child process, with retained failure logs for either harness."""
    out = Path(out)
    out.unlink(missing_ok=True)
    proc = subprocess.run(command, capture_output=True, text=True, env=env,
                          cwd=cwd, timeout=900)
    out.with_suffix(".log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"trial failed (exit {proc.returncode}); see {out.with_suffix('.log')}")
    if expect_marker and expect_marker not in proc.stdout:
        raise RuntimeError(f"trial missing marker {expect_marker}; see {out.with_suffix('.log')}")
    return json.loads(out.read_text(encoding="utf-8"))


def demo_budget_main(argv):
    """Release lane: exact saved workloads, pinned references, fresh ABBA trials."""
    from tools.performance_budget import REFERENCE_VERSIONS, abba_schedule, evaluate_budget
    parser = argparse.ArgumentParser(description=demo_budget_main.__doc__)
    parser.add_argument("--demo-budget", action="store_true")
    parser.add_argument("--blender", required=True)
    parser.add_argument("--candidate-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    schedule = list(abba_schedule(args.trials))
    if args.frames != 48:
        parser.error("release budget requires the unchanged frames 1–48 window")
    candidate = Path(args.candidate_root).resolve()
    source_root = candidate
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["BLENDER_USER_EXTENSIONS"] = str(out_dir / "blender_user_extensions")
    runs = []
    inventory = None
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=candidate, text=True)
    try:
        with tempfile.TemporaryDirectory(prefix="fluidsim-perf-references-") as temp:
            snapshot = Path(temp) / "candidate"
            inventory = _snapshot_candidate(candidate, snapshot)
            candidate = snapshot
            roots = {}
            for version, commit in REFERENCE_COMMITS.items():
                archive = Path(temp) / f"{version}.tar"
                subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", commit],
                               cwd=source_root, check=True)
                root = Path(temp) / version
                root.mkdir()
                with tarfile.open(archive) as bundle:
                    bundle.extractall(root, filter="data")
                roots[version] = root
            for demo, reference in REFERENCE_VERSIONS.items():
                blend = candidate / "demos" / f"{demo}.blend"
                payload = out_dir / f"{demo}_settings.json"
                extract_command = [args.blender, "--python-use-system-env", "--background",
                                   "--factory-startup", "--python-exit-code", "1", "--python",
                                   str(candidate / "tools" / "benchmark_demo_trial.py"), "--",
                                   "--extract", str(candidate), str(blend), str(payload)]
                _run_trial(extract_command, payload, env=env, cwd=candidate,
                           expect_marker="DEMO-SETTINGS-EXTRACT-PASS")
                for block, slot, label in schedule:
                    root = roots[reference] if label == "baseline" else candidate
                    out = out_dir / f"{demo}_b{block}_s{slot}_{label}.json"
                    # Never accept a stale trial after an abnormal child exit.
                    out.unlink(missing_ok=True)
                    command = [args.blender, "--python-use-system-env", "--background",
                               "--factory-startup", "--python-exit-code", "1", "--python",
                               str(candidate / "tools" / "benchmark_demo_trial.py"), "--",
                               str(root), str(blend), str(out), str(args.frames), reference, str(payload)]
                    run = _run_trial(command, out, env=env, cwd=root,
                                     expect_marker="DEMO-PERFORMANCE-TRIAL-PASS")
                    if run["version"] != (reference if label == "baseline" else _candidate_version(candidate)):
                        raise RuntimeError("runtime add-on version does not match selected checkout")
                    run.update(label=label, round=block, slot=slot, workload=demo,
                               gitHead=REFERENCE_COMMITS[reference] if label == "baseline" else head,
                               candidateInventorySha256=inventory)
                    out.write_text(json.dumps(run, indent=2), encoding="utf-8")
                    runs.append(run)
                    print(f"{demo} block={block} slot={slot} {label}: trial recorded", flush=True)
        verdict = evaluate_budget(runs)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        verdict = {"passed": False, "status": "invalid", "error": str(exc)}
    verdict.update(gitHead=head, gitStatus=status.splitlines(), referenceCommits=REFERENCE_COMMITS,
                   trials=args.trials, frames=args.frames, runs=runs,
                   candidateInventorySha256=inventory,
                   measurementKind="saved_demo_scene_bridge_with_stage_sync")
    summary_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print("PERFORMANCE-BUDGET-PASS" if verdict["passed"] else "PERFORMANCE-BUDGET-FAIL",
          verdict["status"], summary_path, flush=True)
    return 0 if verdict["passed"] else 1


def _candidate_version(root):
    namespace = {}
    exec((root / "fluidSim_warp_blender51" / "version.py").read_text(encoding="utf-8"), namespace)
    return namespace["VERSION_STRING"]


def _child(root, workload, frames, warmup, out):
    sys.path.insert(0, root)
    import warp as wp

    wp.config.quiet = True
    wp.init()

    from fluidSim_warp_blender51.diagnostics.benchmark import default_benchmark_settings
    from fluidSim_warp_blender51.sim.enums import (
        SourceShapeShape, SourceShapeSourceMode, SourceShapeType)
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SourceShapeDesc

    # demos/dam_break.blend, as mirrored by tools/profiling/profile_gpu_solver.py
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
    # Older benchmarked versions only have ``set_sources``; current ones only
    # have ``set_source_batch``. Both replace the whole installed source set.
    install_sources = getattr(solver, "set_source_batch", None) or solver.set_sources
    install_sources([box])
    records = []
    try:
        for frame in range(warmup + frames):
            if frame == EMIT_FRAMES:
                install_sources([])
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
    argv = sys.argv[1:] if argv is None else argv
    if "--demo-budget" in argv:
        return demo_budget_main(argv)
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
                run = _run_trial(
                    [sys.executable, "-u", os.path.abspath(__file__), "--child",
                     root, workload, str(args.frames), str(args.warmup), str(out)], out)
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if sys.argv[1:2] == ["--child"]:
        root, workload, frames, warmup, out = sys.argv[2:7]
        _child(root, workload, int(frames), int(warmup), out)
        sys.exit(0)
    sys.exit(main())
