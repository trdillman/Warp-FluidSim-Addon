"""Evaluate a learned-guess weights file against the 4.0 ship gate.

For every scene this runs the production solver twice -- learned guess off
(the zero guess) and on, with ``--weights`` -- and records every pressure
solve: its PCG iteration count, residual and status. Iterations are counted
two ways:

* ``exact``: the PCG checks convergence every iteration (``check_every=1``),
  so the count is the true number of iterations the guess saves;
* ``production``: the shipped convergence-check cadence, i.e. what a bake
  actually executes.

Wall clock is measured separately, without the recording hooks, as paired
interleaved runs in ABBA order (off, on, on, off, ...), frames 2..N only
(frame 1 carries kernel compilation and graph capture).

Ship gate (spec): median iterations over frames 2..48 at least 2x lower on
the four demos and 1.5x lower on the held-out scenes, a net wall-clock win,
and every guess-on solve converged to tolerance. Bitwise guess-off parity is
proven by the test suite, not here. The demos need Blender; this tool runs
the pure-Warp scenes from ``scenes.py`` (``--seeds``, held-out by default).

Usage::

    python tools/learned_guess/evaluate.py --weights W.npz --out report.json
        [--seeds 1000:1020] [--frames 48] [--pairs 2]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.learned_guess import scenes  # noqa: E402

FIRST_FRAME, LAST_FRAME = 2, 48
DEMO_RATIO, HELDOUT_RATIO = 2.0, 1.5


def _use_weights(path):
    from fluidSim_warp_blender51.sim import learned_guess as lg

    lg.DEFAULT_WEIGHTS_PATH = os.path.abspath(path) if path else lg.DEFAULT_WEIGHTS_PATH


def record_solves(spec, device, frames, *, guess, exact):
    """Run one scene; return per-solve rows and the guess message."""
    from fluidSim_warp_blender51.sim.pressure import PRESSURE_STATUS_CONVERGED

    solver = scenes.build_solver(spec, device, pressure_learned_guess=guess)
    rows = []
    try:
        ps = solver.pressure_solver
        original = ps.solve_pcg
        state = {"frame": 0}

        def solve(*args, **kwargs):
            if exact:
                kwargs["check_every"] = 1
            status = original(*args, **kwargs)
            rows.append({
                "frame": state["frame"],
                "iterations": int(ps.last_iterations),
                "residual": float(ps.last_residual),
                "converged": ps.last_pressure_status == PRESSURE_STATUS_CONVERGED,
            })
            return status

        ps.solve_pcg = solve
        for frame in range(1, frames + 1):
            state["frame"] = frame
            solver.step_frame()
        message = solver.stats.learned_guess_message
    finally:
        solver.release()
    return rows, message


def timed_run(spec, device, frames, *, guess):
    """Wall seconds for frames 2..N of one scene (frame 1 excluded)."""
    import warp as wp

    solver = scenes.build_solver(spec, device, pressure_learned_guess=guess)
    try:
        solver.step_frame()
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(frames - 1):
            solver.step_frame()
        wp.synchronize_device(device)
        return time.perf_counter() - start
    finally:
        solver.release()


def median_iterations(rows, first=FIRST_FRAME, last=LAST_FRAME):
    values = [r["iterations"] for r in rows if first <= r["frame"] <= last]
    return float(statistics.median(values)) if values else 0.0


def ratio(off, on):
    if on <= 0.0:
        return float("inf") if off > 0.0 else 1.0
    return off / on


def summarise_scene(off_rows, on_rows):
    off_med, on_med = median_iterations(off_rows), median_iterations(on_rows)
    return {
        "off_median": off_med, "on_median": on_med, "ratio": ratio(off_med, on_med),
        "off_total": sum(r["iterations"] for r in off_rows),
        "on_total": sum(r["iterations"] for r in on_rows),
        "all_on_converged": all(r["converged"] for r in on_rows),
        "solves": len(on_rows),
    }


def abba(times_off, times_on):
    """Paired wall-clock ratio off/on: median over pairs of (sum off / sum on)."""
    pairs = [a / b for a, b in zip(times_off, times_on) if b > 0.0]
    return float(statistics.median(pairs)) if pairs else 0.0


def gate_verdict(heldout_ratios, demo_ratios, wall_ratio, all_converged):
    """The spec's ship gate. ``demo_ratios`` may be empty (not measured here)."""
    checks = {
        "demos_2x": bool(demo_ratios) and min(demo_ratios) >= DEMO_RATIO,
        "heldout_1_5x": bool(heldout_ratios)
        and statistics.median(heldout_ratios) >= HELDOUT_RATIO,
        "wall_clock_win": wall_ratio > 1.0,
        "residual_within_tolerance": bool(all_converged),
    }
    return {"checks": checks, "met": all(checks.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default=f"{scenes.HELDOUT_SEEDS.start}:"
                                          f"{scenes.HELDOUT_SEEDS.start + 20}")
    parser.add_argument("--frames", type=int, default=LAST_FRAME)
    parser.add_argument("--pairs", type=int, default=2, help="ABBA blocks per scene")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    import warp as wp

    from tools.learned_guess.export_dataset import parse_seeds

    wp.init()
    _use_weights(args.weights)
    report = {"weights": os.path.abspath(args.weights), "frames": args.frames, "scenes": []}
    for seed in parse_seeds(args.seeds):
        spec = scenes.make_scene(seed)
        entry = {"seed": seed, "heldout": spec.heldout}
        for mode in ("exact", "production"):
            off, _ = record_solves(spec, args.device, args.frames, guess=False,
                                   exact=mode == "exact")
            on, message = record_solves(spec, args.device, args.frames, guess=True,
                                        exact=mode == "exact")
            entry[mode] = summarise_scene(off, on)
            entry["message"] = message
        times = {False: [], True: []}
        for _ in range(args.pairs):
            for guess in (False, True, True, False):
                times[guess].append(timed_run(spec, args.device, args.frames, guess=guess))
        entry["wall_off_s"], entry["wall_on_s"] = times[False], times[True]
        entry["wall_ratio"] = abba(times[False], times[True])
        report["scenes"].append(entry)
        print(json.dumps({k: entry[k] for k in ("seed", "message", "wall_ratio")}
                         | {"exact_ratio": entry["exact"]["ratio"],
                            "production_ratio": entry["production"]["ratio"]}), flush=True)
    rows = report["scenes"]
    heldout = [e["production"]["ratio"] for e in rows if e["heldout"]]
    report["gate"] = gate_verdict(
        heldout, [], statistics.median([e["wall_ratio"] for e in rows]) if rows else 0.0,
        all(e[m]["all_on_converged"] for e in rows for m in ("exact", "production")))
    report["gate"]["note"] = "demos_2x is measured in Blender (see RELEASE_4_0_STATUS.md)"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["gate"]))


if __name__ == "__main__":
    main()
