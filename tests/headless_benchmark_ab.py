"""Headless before/after regression benchmark for the ST-FLIP work.

Runs the deterministic dam-break from ``diagnostics/benchmark.py`` in a single
configuration and writes the aggregated ``BENCHMARK_FIELDS`` payload as JSON so
two revisions of the tree can be compared with ``compare_benchmark``.

Two configurations matter, because every ST-FLIP change is gated on
``spacetime_enabled``:

  * ``--st off`` is the regression guard.  EXACT_FIELDS must stay bit-identical
    across the change; a mismatch means something leaked out of the ST gate.
  * ``--st on`` is the measurement.  EXACT_FIELDS legitimately move, so only
    PERFORMANCE_FIELDS are compared against the baseline; determinism is checked
    by running this configuration twice and requiring an exact match.

Usage (plain Python or Blender's Python both work; the solver does not need bpy):

    python tests/headless_benchmark_ab.py --st off --output before_st_off.json
    python tests/headless_benchmark_ab.py --st on  --output before_st_on.json
    python tests/headless_benchmark_ab.py --st on  --output after_st_on_2.json \\
        --baseline after_st_on.json --exact

    blender --background --python tests/headless_benchmark_ab.py -- --st on
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _ensure_package_on_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root / "fluidSim_warp_blender51"


def main(argv: list[str] | None = None) -> int:
    package_dir = _ensure_package_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--emit-frames", type=int, default=6)
    parser.add_argument("--st", choices=("on", "off"), default="off",
                        help="enable the spatiotemporal FLIP path")
    parser.add_argument("--cfl", type=float, default=None,
                        help="override SolverSettings.cfl (default: leave as-is)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--baseline", default=None,
                        help="compare the run against this saved benchmark JSON")
    parser.add_argument("--exact", action="store_true",
                        help="fail the comparison on EXACT_FIELDS mismatch too")
    parser.add_argument("--perf-tolerance", type=float, default=0.25)
    args, _ = parser.parse_known_args(argv)

    from fluidSim_warp_blender51.diagnostics.benchmark import (
        compare_benchmark, default_benchmark_settings, default_benchmark_sources,
        load_benchmark, run_benchmark, save_benchmark,
    )

    st_on = args.st == "on"
    settings = default_benchmark_settings()
    settings.device = args.device
    settings.spacetime_enabled = st_on
    if args.cfl is not None:
        settings.cfl = float(args.cfl)

    output = pathlib.Path(args.output) if args.output else (
        package_dir / "demo_outputs" / f"benchmark_ab_st_{args.st}.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(done, total, stats):
        print(f"  frame {done}/{total}: {stats.step_time * 1000.0:8.2f} ms  "
              f"active_voxels={stats.active_voxels}", flush=True)

    print(f"benchmark A/B: st={args.st} device={args.device} frames={args.frames} "
          f"cfl={settings.cfl}", flush=True)
    result = run_benchmark(frames=args.frames, emit_frames=args.emit_frames,
                           settings=settings, sources=default_benchmark_sources(),
                           progress=progress).as_dict()
    result["spacetimeEnabled"] = st_on
    result["device"] = args.device
    result["cfl"] = float(settings.cfl)
    save_benchmark(str(output), result)

    print(f"  timeStepAverage        = {result['timeStepAverage'] * 1000.0:.3f} ms")
    print(f"  voxelsPerSecondAverage = {result['voxelsPerSecondAverage']:.1f}")
    print(f"  activeVoxels avg/min/max = {result['activeVoxelsAverage']}/"
          f"{result['activeVoxelsMinimum']}/{result['activeVoxelsMaximum']}")
    print(f"  written to {output}")

    if not args.baseline:
        return 0

    report = compare_benchmark(result, load_benchmark(args.baseline),
                               perf_tolerance=args.perf_tolerance)
    print(json.dumps(report, indent=2))
    exact_ok = all(e["match"] for e in report["exact"].values())
    perf_ok = all(p["within_tolerance"] for p in report["performance"].values())
    ok = perf_ok and (exact_ok or not args.exact)
    print(f"comparison: {'PASS' if ok else 'FAIL'} "
          f"(exact={'match' if exact_ok else 'differ'}, "
          f"perf={'ok' if perf_ok else 'regression'}, "
          f"exact_enforced={args.exact})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None))
