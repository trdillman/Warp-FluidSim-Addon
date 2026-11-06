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
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

_qspec = importlib.util.spec_from_file_location("_qualification", Path(__file__).with_name("gate") / "qualification.py")
q = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(q)

WORKLOADS = ("plain", "visc", "st", "visc_st")
EMIT_FRAMES = 4
REFERENCE_COMMITS = {
    "4.0.1": "46cdf3ce67b16a92ea47ad372bcd6e4bba8fe877",
}


def _inventory(root):
    paths = list((root / "fluidSim_warp_blender51").rglob("*"))
    paths += [root / "demos" / f"{demo}.blend" for demo in
              ("dam_break", "crown_splash", "paddle_stir", "chromatic_vortex")]
    paths += list((root / "tools/gate").rglob("*.py"))
    paths += list((root / "tools/demos").rglob("*"))
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
    shutil.copytree(source / "tools/gate", destination / "tools/gate")
    if (source / "tools/demos").is_dir():
        shutil.copytree(source / "tools/demos", destination / "tools/demos",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if before != _inventory(source) or before != _inventory(destination):
        raise RuntimeError("candidate changed during snapshot; rerun with stable source")
    return before


def _run_trial(command, out, *, env=None, cwd=None, expect_marker=None):
    """One fresh child process, with retained failure logs for either harness."""
    out = Path(out)
    out.unlink(missing_ok=True)
    try:
        if expect_marker == "DEMO-PERFORMANCE-TRIAL-PASS":
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, env=env, cwd=cwd) as child:
                try:
                    stdout, stderr = child.communicate(timeout=900)
                except BaseException as exc:
                    child.kill()
                    stdout, stderr = child.communicate()
                    out.with_suffix('.log').write_text(stdout + stderr, encoding='utf-8')
                    if isinstance(exc, subprocess.TimeoutExpired):
                        exc.stdout, exc.stderr = stdout, stderr
                    raise
                proc = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
                observed_pid = child.pid
        else:
            proc = subprocess.run(command, capture_output=True, text=True, env=env,
                                  cwd=cwd, timeout=900)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
        out.with_suffix('.log').write_text(decoded(exc.stdout) + decoded(exc.stderr), encoding='utf-8')
        raise
    out.with_suffix(".log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"trial failed (exit {proc.returncode}); see {out.with_suffix('.log')}")
    if expect_marker and expect_marker not in proc.stdout:
        raise RuntimeError(f"trial missing marker {expect_marker}; see {out.with_suffix('.log')}")
    report = json.loads(out.read_text(encoding="utf-8"))
    if expect_marker == "DEMO-PERFORMANCE-TRIAL-PASS" and report.get("processId") != observed_pid:
        raise RuntimeError("reported child process identity differs from launched process")
    return report


def _checkpoint_summary(path, report):
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _validate_trial_identity(run, root, inventory):
    if (Path(run.get('root', '')).resolve() != root.resolve()
            or Path(run.get('addon', '')).resolve() != (root / 'fluidSim_warp_blender51/__init__.py').resolve()
            or run.get('rootInventorySha256') != inventory
            or _inventory(root) != inventory
            or type(run.get('processId')) is not int or run['processId'] <= 0):
        raise RuntimeError('observed child source/addon/process identity differs from selected snapshot')


def _measurement_context(candidate, blender, out_dir, env):
    context_path = env.get("FLUIDSIM_QUALIFICATION_CONTEXT")
    if context_path:
        context = json.loads(Path(context_path).read_text(encoding="utf-8"))
        q.check_source(candidate, context["frozen_source"], context["generated_semantics"],
                       context["generated_semantics"])
        q.validate_binding(context, context)
        return context
    frozen = q.freeze_source(candidate)
    runtime = q.observe_runtime(candidate, blender, env.get("BLENDER_PY", ""), env,
                                out_dir / "admission", frozen["verifier_fingerprint"])
    return dict(runtime, frozen_source=frozen, run_id=uuid.uuid4().hex, gitStatus=[],
                source_fingerprint=frozen["source_fingerprint"],
                verifier_fingerprint=frozen["verifier_fingerprint"], generated_semantics={},
                environment_request=str(out_dir / "admission/environment-request.json"))


def _snapshot_measurement_sources(candidate, demo, context, payload):
    semantics = {}
    name = f'demos/{demo}.blend'
    if name in context.get('generated_semantics', {}):
        try:
            observed = json.loads(payload.with_suffix('.semantics.json').read_text())['semantic_fingerprint']
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError('missing snapshot semantic observation') from exc
        if observed != context['generated_semantics'][name]:
            raise RuntimeError('measured snapshot semantic identity differs from admission')
        semantics[name] = observed
    return q.measurement_sources(candidate, demo, semantics)


def demo_budget_main(argv):
    """Release lane: exact saved workloads, pinned references, fresh ABBA trials."""
    from tools.performance_budget import (
        REFERENCE_VERSIONS,
        abba_schedule,
        evaluate_budget,
    )
    parser = argparse.ArgumentParser(description=demo_budget_main.__doc__)
    parser.add_argument("--demo-budget", action="store_true")
    parser.add_argument("--blender", required=True)
    parser.add_argument("--candidate-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--demos", nargs="+", choices=tuple(REFERENCE_VERSIONS),
                        help="measure only these demos; --carry-summary covers the rest")
    parser.add_argument("--carry-summary",
                        help="earlier summary whose runs cover the demos not measured")
    parser.add_argument("--full-schedule", action="store_true",
                        help="disable sequential stopping; run every block for every demo")
    args = parser.parse_args(argv)
    if bool(args.demos) != bool(args.carry_summary):
        parser.error("--demos and --carry-summary go together: carried runs cover the other demos")
    measure = tuple(demo for demo in REFERENCE_VERSIONS if not args.demos or demo in args.demos)
    carried_runs, measured_at, inventories = [], {}, {}
    if args.carry_summary:
        prior = json.loads(Path(args.carry_summary).read_text(encoding="utf-8"))
        for demo in REFERENCE_VERSIONS:
            if demo in measure:
                continue
            # Summaries written before per-demo reuse measured every demo together.
            measured_at[demo] = (prior.get("demoMeasuredAt") or {}).get(demo, prior.get("gitHead"))
            inventories[demo] = ((prior.get("demoInventory") or {}).get(demo)
                                 or prior.get("candidateInventorySha256"))
            carried_runs.extend(run for run in prior["runs"] if run.get("workload") == demo)
    schedule = list(abba_schedule(args.trials))
    if args.frames != 48:
        parser.error("release budget requires the unchanged frames 1–48 window")
    candidate = Path(args.candidate_root).resolve()
    source_root = candidate
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    evidence_dir = out_dir / "runs" / uuid.uuid4().hex
    evidence_dir.mkdir(parents=True, exist_ok=False)
    if summary_path.exists():
        shutil.copy2(summary_path, evidence_dir / "prior-summary.json")
    summary_path.unlink(missing_ok=True)
    from tools.release.build_extension import _blender_process_environment
    env = _blender_process_environment(os.environ, profile_dir=evidence_dir / 'blender-profile',
                                       allow_source_imports=True)
    for key in ('BLENDER_USER_RESOURCES', 'PYTHONHOME', 'PYTHONSTARTUP'):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    context = _measurement_context(candidate, args.blender, out_dir, env)
    env["FLUIDSIM_ENVIRONMENT_REQUEST"] = context["environment_request"]
    for run in carried_runs:
        expected = dict(context, source_fingerprint=q.measurement_sources(
            candidate, run['workload'], context.get('generated_semantics')))
        q.validate_binding(run, expected)
        q.retained_trial(run)
    runs = list(carried_runs)
    inventory = None
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate, text=True).strip()
    def checkpoint(verdict):
        report = dict(context, **verdict, gitHead=head, gitStatus=[], referenceCommits=REFERENCE_COMMITS,
                      retained_summary=str(evidence_dir / "summary.json"),
                      trials=args.trials, frames=args.frames, runs=runs,
                      candidateInventorySha256=inventory,
                      demoMeasuredAt={demo: measured_at.get(demo, head) for demo in REFERENCE_VERSIONS},
                      demoInventory={demo: inventories.get(demo, inventory)
                                     for demo in REFERENCE_VERSIONS},
                      measurementKind="saved_demo_scene_bridge_with_stage_sync")
        _checkpoint_summary(evidence_dir / "summary.json", report)
        _checkpoint_summary(summary_path, report)
    checkpoint({'passed': False, 'status': 'incomplete'})
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
                if demo not in measure:
                    continue
                blend = candidate / "demos" / f"{demo}.blend"
                payload = evidence_dir / f"{demo}_settings.json"
                extract_command = [args.blender, "--python-use-system-env", "--background",
                                   "--factory-startup", "--python-exit-code", "1", "--python",
                                   str(candidate / "tools" / "benchmark_demo_trial.py"), "--",
                                   "--extract", str(candidate), str(blend), str(payload)]
                _run_trial(extract_command, payload, env=env, cwd=candidate,
                           expect_marker="DEMO-SETTINGS-EXTRACT-PASS")
                measured_sources = _snapshot_measurement_sources(candidate, demo, context, payload)
                for block, slot, label in schedule:
                    root = roots[reference] if label == "baseline" else candidate
                    root_inventory = _inventory(root)
                    if _inventory(candidate) != inventory:
                        raise RuntimeError('candidate snapshot changed during measurement')
                    out = evidence_dir / f"{demo}_b{block}_s{slot}_{label}.json"
                    # Never accept a stale trial after an abnormal child exit.
                    out.unlink(missing_ok=True)
                    command = [args.blender, "--python-use-system-env", "--background",
                               "--factory-startup", "--python-exit-code", "1", "--python",
                               str(candidate / "tools" / "benchmark_demo_trial.py"), "--",
                               str(root), str(blend), str(out), str(args.frames), reference, str(payload)]
                    run = _run_trial(command, out, env=env, cwd=root,
                                     expect_marker="DEMO-PERFORMANCE-TRIAL-PASS")
                    _validate_trial_identity(run, root, root_inventory)
                    if _inventory(candidate) != inventory:
                        raise RuntimeError('candidate snapshot changed during measurement')
                    if run["version"] != (reference if label == "baseline" else _candidate_version(candidate)):
                        raise RuntimeError("runtime add-on version does not match selected checkout")
                    if run.get('environment_fingerprint') != context['environment_fingerprint']:
                        raise RuntimeError('observed child environment differs from qualification')
                    run.update(source_fingerprint=measured_sources,
                               verifier_fingerprint=context['verifier_fingerprint'], run_id=context['run_id'],
                               gitStatus=[], label=label, round=block, slot=slot, workload=demo,
                               gitHead=REFERENCE_COMMITS[reference] if label == "baseline" else head,
                               candidateInventorySha256=inventory)
                    q.atomic_json(out, run)
                    run.update(trial_artifact=str(out), trial_sha256=q.sha256(out))
                    runs.append(run)
                    checkpoint({'passed': False, 'status': 'incomplete'})
                    print(f"{demo} block={block} slot={slot} {label}: trial recorded", flush=True)
                    if block == 1 and slot == 3 and not args.full_schedule:
                        early = evaluate_budget([r for r in runs if r["workload"] == demo],
                                                demos=(demo,))
                        demo_status = early.get("demos", {}).get(demo, {}).get("status")
                        if demo_status in ("pass", "regression"):
                            print(f"{demo}: early-stop {demo_status} after the first ABBA and BAAB blocks",
                                  flush=True)
                            break
        q.check_source(source_root, context['frozen_source'], context['generated_semantics'],
                       context['generated_semantics'])
        for run in runs:
            q.validate_binding(run, dict(context, source_fingerprint=q.measurement_sources(
                source_root, run['workload'], context['generated_semantics'])))
            q.retained_trial(run)
        verdict = evaluate_budget(runs)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        verdict = {"passed": False, "status": "invalid", "error": str(exc)}
    checkpoint(verdict)
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
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
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
