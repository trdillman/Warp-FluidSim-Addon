#!/usr/bin/env python3
"""Paired viscosity / surface-tension cost measurement on a real solver scene.

This is a measurement harness, not an optimization.  It answers one question
with paired before/after data instead of code reading: when a user turns on
viscosity, surface tension, or both, where does the extra frame time go?

Design notes
------------
* It reuses ``tools/profile_gpu_solver.py`` for the actual run (settings build,
  source construction, per-phase profiler wiring, per-frame record capture) and
  only adds the pairing, repetition, probe lanes, and raw-JSON persistence.
* Two lanes are run for every configuration:

  ``wall``  - ``Profiler(sync=False)``.  Phase slots are submission-time only
              and are NOT reported; the headline number is ``stepTime``, which
              ``FluidSolver.step_frame`` measures across an explicit
              ``wp.synchronize_device``, so it is a true wall-clock frame cost
              with no added profiling stalls.
  ``phase`` - ``Profiler(sync=True)``.  Every phase span brackets itself with
              ``wp.synchronize_device``, so ``compute_<slot>`` values describe
              real GPU work.  This lane's absolute frame time is inflated by
              the added synchronization and is used only for attribution.

* Phase-slot semantics that matter when reading the output:

  ``compute_3``  ("forces") encloses the surface-tension solve, and therefore
                 also encloses the SDF rebuild that solve triggers.
  ``compute_11`` ("surface") is NOT a subset of ``compute_3``.  Three separate
                 call sites accumulate into it: the surface-tension SDF rebuild
                 inside forces, ``SurfacePipeline.after_substep`` (which
                 rebuilds the solver SDF once per substep whenever whitewater
                 is enabled), and ``SurfacePipeline.after_frame`` (render mesh).
                 So ``compute_3 - compute_11`` is NOT forces-net-of-surface-
                 tension.  Isolation comes from the paired delta instead:
                 ``compute_11(ST on) - compute_11(ST off)`` is the cost the
                 surface-tension SDF rebuild adds, and
                 ``compute_3(ST on) - compute_3(ST off)`` is the whole
                 surface-tension cost including that rebuild.

* Probe lanes (``--probes``) monkeypatch measurement-only wrappers into the
  running solver; they never change production code.  They add synchronization
  and are reported separately from the timing lanes.

Run (from the repository root):

    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\5.1\\python\\bin\\python.exe" \\
        tools/profile_viscosity_surface_tension.py --device cuda:0 \\
        --frames 8 --warmup 3 --trials 3 --scene-preset dam_break \\
        --out-dir artifacts/visc_st_measurement
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profile_gpu_solver import _DEFAULTS, _phase_labels, run_profile  # noqa: E402


# Paired configurations.  ``baseline`` is the control for the whitewater-on
# lanes; ``baseline_no_whitewater`` is the control for the whitewater-off lane
# that tests whether whitewater forces a second solver-SDF rebuild per substep.
CONFIGURATIONS = (
    {"name": "baseline", "viscosity": False,
     "surface_tension": False, "no_whitewater": False},
    {"name": "viscosity_on", "viscosity": True,
     "surface_tension": False, "no_whitewater": False},
    {"name": "surface_tension_on", "viscosity": False,
     "surface_tension": True, "no_whitewater": False},
    {"name": "both_on", "viscosity": True,
     "surface_tension": True, "no_whitewater": False},
    {"name": "baseline_no_whitewater", "viscosity": False,
     "surface_tension": False, "no_whitewater": True},
    {"name": "surface_tension_on_no_whitewater", "viscosity": False,
     "surface_tension": True, "no_whitewater": True},
)


def _args_for(base: argparse.Namespace, config: dict, *, sync: bool) -> Namespace:
    """Build a profile_gpu_solver argument namespace for one run."""
    return Namespace(
        device=base.device,
        frames=base.frames,
        warmup=base.warmup,
        emit_frames=base.emit_frames,
        voxel_size=base.voxel_size,
        max_particles=base.max_particles,
        max_whitewater=base.max_whitewater,
        min_substeps=base.min_substeps,
        max_substeps=base.max_substeps,
        whitewater=False,
        no_surface=False,
        sync=sync,
        no_phase_timers=False,
        scene_preset=base.scene_preset,
        no_whitewater=bool(config["no_whitewater"]),
        viscosity=bool(config["viscosity"]),
        viscosity_constant=base.viscosity_constant,
        viscosity_iterations=base.viscosity_iterations,
        viscosity_tolerance=base.viscosity_tolerance,
        surface_tension=bool(config["surface_tension"]),
        surface_tension_coefficient=base.surface_tension_coefficient,
        json_out=None,
    )


# ---------------------------------------------------------------------------
# Probe lanes
# ---------------------------------------------------------------------------

def _install_viscosity_probe(device: str) -> tuple[dict, callable]:
    """Time each per-component PCG solve and each residual host readback.

    ``ViscositySolver.solve`` issues three fully independent ``_solve_component``
    calls (u, v, w).  Inside each, the PCG loop performs a synchronous
    ``residual.numpy()`` host readback on every tenth iteration and on the last
    iteration.  ``array.numpy`` is wrapped only while a ``_solve_component``
    call is on the stack, and the residual readback is the only ``numpy()`` call
    made there, so the accumulated time is exactly the host-sync stall cost.
    """
    import warp as wp

    from fluidSim_warp_blender51.sim.viscosity import ViscositySolver

    stats = {
        "component_calls": 0,
        "component_ms": [0.0, 0.0, 0.0],
        "component_iterations": [0, 0, 0],
        "solve_calls": 0,
        "readback_calls": 0,
        "readback_ms": 0.0,
    }
    state = {"inside": False}

    real_solve = ViscositySolver.solve
    real_solve_component = ViscositySolver._solve_component
    # warp moved its array class to warp._src.types in 1.16; resolve it from
    # a live instance so this probe does not hard-code an internal path.
    array_cls = type(wp.zeros(1, dtype=float, device=device))
    real_numpy = array_cls.numpy

    def probed_numpy(self, *args, **kwargs):
        if not state["inside"]:
            return real_numpy(self, *args, **kwargs)
        t0 = time.perf_counter()
        out = real_numpy(self, *args, **kwargs)
        stats["readback_ms"] += (time.perf_counter() - t0) * 1000.0
        stats["readback_calls"] += 1
        return out

    def probed_solve_component(self, *args, **kwargs):
        index = stats["component_calls"] % 3
        state["inside"] = True
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        try:
            return real_solve_component(self, *args, **kwargs)
        finally:
            wp.synchronize_device(device)
            stats["component_ms"][index] += (time.perf_counter() - t0) * 1000.0
            stats["component_iterations"][index] += int(
                getattr(self, "last_iterations", 0))
            stats["component_calls"] += 1
            state["inside"] = False

    def probed_solve(self, *args, **kwargs):
        stats["solve_calls"] += 1
        return real_solve(self, *args, **kwargs)

    ViscositySolver.solve = probed_solve
    ViscositySolver._solve_component = probed_solve_component
    array_cls.numpy = probed_numpy

    def restore():
        ViscositySolver.solve = real_solve
        ViscositySolver._solve_component = real_solve_component
        array_cls.numpy = real_numpy

    return stats, restore


def _install_legacy_viscosity_check() -> callable:
    """Restore the pre-optimization viscosity convergence check.

    Measurement-only: production code is untouched.  The shipped loop tested
    convergence on the host every tenth iteration; this forces both of those
    back (host-checked loop, cadence 10) so a before/after pair can be run
    back-to-back under the same machine load instead of being compared across
    runs minutes apart.
    """
    from fluidSim_warp_blender51.sim import viscosity as V

    real_init = V.ViscositySolver.__init__
    real_default = V.DEFAULT_VISCOSITY_CHECK_EVERY
    real_defaults = V.ViscositySolver.solve.__defaults__

    def legacy_init(self, grid, device):
        real_init(self, grid, device)
        self._conditional_available = False

    V.ViscositySolver.__init__ = legacy_init
    V.DEFAULT_VISCOSITY_CHECK_EVERY = 10
    V.ViscositySolver.solve.__defaults__ = real_defaults[:-1] + (10,)
    V.ViscositySolver._solve_component.__defaults__ = (10,)

    def restore():
        V.ViscositySolver.__init__ = real_init
        V.DEFAULT_VISCOSITY_CHECK_EVERY = real_default
        V.ViscositySolver.solve.__defaults__ = real_defaults
        V.ViscositySolver._solve_component.__defaults__ = (real_default,)

    return restore


def _install_surface_sdf_probe(device: str) -> tuple[dict, callable]:
    """Count and time solver-grid SDF rebuilds and their cache hits/misses."""
    import warp as wp

    from fluidSim_warp_blender51.sim.surface import SurfacePipeline

    stats = {
        "ensure_calls": 0,
        "ensure_cache_hits": 0,
        "build_sdf_calls": 0,
        "build_sdf_ms": 0.0,
    }

    real_ensure = SurfacePipeline.ensure_solver_sdf_current
    real_build_sdf = SurfacePipeline.build_sdf

    def probed_ensure(self, *args, **kwargs):
        stats["ensure_calls"] += 1
        before = stats["build_sdf_calls"]
        result = real_ensure(self, *args, **kwargs)
        if stats["build_sdf_calls"] == before:
            stats["ensure_cache_hits"] += 1
        return result

    def probed_build_sdf(self, *args, **kwargs):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        try:
            return real_build_sdf(self, *args, **kwargs)
        finally:
            wp.synchronize_device(device)
            stats["build_sdf_ms"] += (time.perf_counter() - t0) * 1000.0
            stats["build_sdf_calls"] += 1

    SurfacePipeline.ensure_solver_sdf_current = probed_ensure
    SurfacePipeline.build_sdf = probed_build_sdf

    def restore():
        SurfacePipeline.ensure_solver_sdf_current = real_ensure
        SurfacePipeline.build_sdf = real_build_sdf

    return stats, restore


def _zero_probe(stats: dict) -> None:
    for key, value in stats.items():
        if isinstance(value, list):
            stats[key] = [type(item)() for item in value]
        else:
            stats[key] = type(value)()


def run_probe(base: argparse.Namespace, config: dict) -> dict:
    """Run one configuration with the measurement-only probes installed.

    Counters are zeroed at the warmup boundary so every count and duration
    below covers exactly the measured frames returned by ``run_profile``.
    """
    visc_stats, visc_restore = _install_viscosity_probe(base.device)
    sdf_stats, sdf_restore = _install_surface_sdf_probe(base.device)

    def on_frame(index, _stats):
        if index == base.warmup - 1:
            _zero_probe(visc_stats)
            _zero_probe(sdf_stats)

    try:
        args = _args_for(base, config, sync=False)
        summary, frames = run_profile(args, on_frame=on_frame)
    finally:
        visc_restore()
        sdf_restore()
    measured = max(1, len(frames))
    substeps = sum(int(record["substeps"]) for record in frames) or 1
    return {
        "config": config["name"],
        "framesMeasured": measured,
        "substepsMeasured": substeps,
        "viscosity": visc_stats,
        "surfaceSdf": sdf_stats,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _median(values):
    return statistics.median(values) if values else 0.0


def _frame_phase_ms(record: dict, slot: str) -> float:
    return float(record.get("timers", {}).get(slot, 0.0))


def summarize(lane_runs: list[dict]) -> dict:
    """Per-trial medians for one configuration in one lane.

    The per-trial statistic is the median over that trial's measured frames;
    the reported statistic is the median over trials.  Medians are used at both
    levels so a single slow first frame or a background stall cannot move the
    headline.
    """
    per_trial_step_ms = []
    per_trial_phase_ms: dict[str, list[float]] = {}
    per_trial_particles = []
    per_trial_voxels = []
    per_trial_substeps = []
    for run in lane_runs:
        frames = run["frames"]
        if not frames:
            continue
        per_trial_step_ms.append(
            _median([float(f["stepTime"]) * 1000.0 for f in frames]))
        per_trial_particles.append(
            _median([float(f["activeParticles"]) for f in frames]))
        per_trial_voxels.append(
            _median([float(f["activeVoxels"]) for f in frames]))
        per_trial_substeps.append(
            _median([float(f["substeps"]) for f in frames]))
        for slot in _phase_labels():
            per_trial_phase_ms.setdefault(slot, []).append(
                _median([_frame_phase_ms(f, slot) for f in frames]))
    return {
        "trials": len(per_trial_step_ms),
        "stepMsMedian": _median(per_trial_step_ms),
        "stepMsPerTrial": per_trial_step_ms,
        "activeParticlesMedian": _median(per_trial_particles),
        "activeVoxelsMedian": _median(per_trial_voxels),
        "substepsMedian": _median(per_trial_substeps),
        "phaseMsMedian": {
            slot: _median(values) for slot, values in per_trial_phase_ms.items()
        },
    }


def _gpu_state():
    """Best-effort GPU clock/power/utilisation sample.

    This is a shared desktop GPU: other graphics clients contend for it and the
    achieved SM clock moves during a run.  Sampling it per run lets a reader
    tell a real regression from a contention artefact.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,utilization.gpu,power.draw,temperature.gpu,"
             "memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:  # noqa: BLE001 - diagnostics only, never fail the run
        return None
    fields = [part.strip() for part in out.strip().splitlines()[0].split(",")]
    keys = ("smClockMHz", "utilizationPct", "powerW", "temperatureC",
            "memoryUsedMiB")
    state = {}
    for key, value in zip(keys, fields):
        try:
            state[key] = float(value)
        except ValueError:
            state[key] = None
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--emit-frames", type=int, default=6)
    parser.add_argument("--scene-preset", choices=("benchmark", "dam_break"),
                        default="dam_break")
    parser.add_argument("--voxel-size", type=float,
                        default=_DEFAULTS["voxel_size"])
    parser.add_argument("--max-particles", type=int,
                        default=_DEFAULTS["max_particles"])
    parser.add_argument("--max-whitewater", type=int, default=200000)
    parser.add_argument("--min-substeps", type=int,
                        default=_DEFAULTS["min_substeps"])
    parser.add_argument("--max-substeps", type=int,
                        default=_DEFAULTS["max_substeps"])
    parser.add_argument("--viscosity-constant", type=float, default=1.0)
    parser.add_argument("--viscosity-iterations", type=int, default=None)
    parser.add_argument(
        "--viscosity-tolerance", type=float, default=None,
        help="Override the solver default (C.DEFAULT_VISCOSITY_TOLERANCE).")
    parser.add_argument("--surface-tension-coefficient", type=float,
                        default=0.072)
    parser.add_argument(
        "--viscosity-convergence-check", choices=("current", "legacy-host"),
        default="current",
        help="legacy-host reinstates the pre-optimization host readback check "
             "at cadence 10, for a contention-matched before/after pair.")
    parser.add_argument("--probes", action="store_true", default=True)
    parser.add_argument("--no-probes", dest="probes", action="store_false")
    parser.add_argument("--out-dir", default="artifacts/visc_st_measurement")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")

    import warp as wp

    wp.init()
    if args.viscosity_convergence_check == "legacy-host":
        _install_legacy_viscosity_check()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lanes = {"wall": {}, "phase": {}}
    # Interleave trials across configurations so thermal/clock drift affects
    # every configuration in the pair roughly equally instead of biasing the
    # configurations that happen to run last.
    for trial in range(args.trials):
        for lane, sync in (("wall", False), ("phase", True)):
            for config in CONFIGURATIONS:
                run_args = _args_for(args, config, sync=sync)
                gpu_before = _gpu_state()
                t0 = time.perf_counter()
                summary, frames = run_profile(run_args)
                lanes[lane].setdefault(config["name"], []).append({
                    "trial": trial,
                    "wallSeconds": time.perf_counter() - t0,
                    "gpuStateBefore": gpu_before,
                    "gpuStateAfter": _gpu_state(),
                    "summary": summary,
                    "frames": frames,
                })
                print(
                    f"[{lane}] trial {trial} {config['name']}: "
                    f"step_ms_median="
                    f"{_median([float(f['stepTime']) * 1000.0 for f in frames]):.3f}",
                    flush=True,
                )

    probes = []
    if args.probes:
        for config in CONFIGURATIONS:
            # Every configuration is probed, including the two baselines: the
            # SDF-rebuild counts and durations are only interpretable as a
            # paired delta against their own control.
            probes.append(run_probe(args, config))
            print(f"[probe] {config['name']} done", flush=True)

    payload = {
        "tool": "profile_viscosity_surface_tension",
        "timestampUtc": stamp,
        "device": str(wp.get_device(args.device).name),
        "deviceId": args.device,
        "warpVersion": getattr(wp, "__version__", "unknown"),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "configurations": list(CONFIGURATIONS),
        "phaseLabels": _phase_labels(),
        "lanes": lanes,
        "probes": probes,
    }
    raw_path = out_dir / f"visc_st_raw_{stamp}.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    aggregate = {
        lane: {name: summarize(runs) for name, runs in configs.items()}
        for lane, configs in lanes.items()
    }
    agg_path = out_dir / f"visc_st_aggregate_{stamp}.json"
    agg_path.write_text(
        json.dumps({"source": raw_path.name, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )

    print("\nVISC-ST-MEASUREMENT-COMPLETE")
    print(f"raw:       {raw_path}")
    print(f"aggregate: {agg_path}")
    for lane in ("wall", "phase"):
        print(f"\nlane={lane} (median of per-trial medians)")
        for name in (config["name"] for config in CONFIGURATIONS):
            entry = aggregate[lane].get(name)
            if not entry:
                continue
            print(
                f"  {name:34s} step_ms={entry['stepMsMedian']:9.3f} "
                f"particles={entry['activeParticlesMedian']:10.0f} "
                f"voxels={entry['activeVoxelsMedian']:9.0f} "
                f"substeps={entry['substepsMedian']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
