"""Equal-quality ST-FLIP benchmark: wall time to reach a known end state.

``headless_benchmark_ab.py`` compares ST-on against ST-off at *matched
settings*, which answers the paper's "negligible additional cost per time step"
claim and nothing else.  It cannot answer the claim that actually matters --
"several-fold speedups" -- because ST-FLIP does not make a step cheaper, it
makes a *bigger* step legal.  Comparing at matched settings therefore measures
ST at its worst by construction.

This harness compares at matched *quality* instead.  A closed box is filled
once on one side, released, and left to settle under gravity.  The end state is
known without simulating anything: still fluid, flat surface, on the floor, at
a depth fixed by the fill volume.  Every configuration runs the same simulated
duration, so a run that settles early is over-dissipative rather than fast --
the kinetic-energy history is recorded to catch exactly that.  What varies is
the CFL number, and what is measured is the wall clock needed to arrive at the
correct end state.

Why the end state needs no reference run: a dam break is chaotic, so particle
positions from two runs diverge exponentially and comparing them measures the
chaos, not the solver.  Equilibrium is not chaotic.  Conserved volume + flat
surface + zero residual motion + resting on the floor pins the hydrostatic
state completely, and all four are absolute checks.

Note on resolution: substeps per frame are ``ceil(frame_dt / (cfl*dx/vmax))``,
so at a coarse voxel size the CFL knob saturates almost immediately -- at
dx=0.05 with dam-break speeds it runs out of range by about CFL 4.  A sweep
worth reading has to run fine enough that CFL 1 costs many substeps.  Raise
``--max-substeps`` accordingly, and check ``cflSubstepsClamped`` in the output:
if it is true the cap, not the CFL number, set the timestep and that data point
says nothing about CFL.

Usage:

    python tests/headless_equilibrium_ab.py --st off --cfl 1 --output ref.json
    python tests/headless_equilibrium_ab.py --st on --cfl 8 --output st8.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time


def _ensure_package_on_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root / "fluidSim_warp_blender51"


def equilibrium_settings(voxel_size: float, max_substeps: int):
    """Closed box, solid on all sides, nothing but gravity acting."""
    from fluidSim_warp_blender51.sim.solver import SolverSettings

    s = SolverSettings()
    s.domain_min = (-1.0, -1.0, 0.0)
    s.domain_max = (1.0, 1.0, 2.0)
    s.voxel_size = voxel_size
    s.max_particles = 8_000_000
    # Both are post-processing that runs on top of the solve and would add
    # cost unrelated to the question being asked.
    s.whitewater_enabled = False
    s.surface_enabled = False
    s.max_substeps = max_substeps
    return s


def equilibrium_sources():
    """One column of liquid against the -x wall, emitted once.

    INITIAL_FILL rather than a continuous source: the fill volume has to be
    exactly the box volume for the analytic rest depth below to mean anything,
    and a rate-driven source makes the emitted volume depend on how many frames
    happened to elapse.
    """
    from fluidSim_warp_blender51.sim.enums import (
        SourceEmissionMode, SourceShapeShape, SourceShapeSourceMode,
        SourceShapeType,
    )
    from fluidSim_warp_blender51.sim.solver import SourceShapeDesc

    box = SourceShapeDesc()
    box.shape = SourceShapeShape.BOX
    box.stype = SourceShapeType.SOURCE
    box.mode = SourceShapeSourceMode.PARTICLES
    box.emission_mode = SourceEmissionMode.INITIAL_FILL
    # Required, not cosmetic: the one-shot disarm in Sourcing keys on
    # source_id and skips any source whose id is empty, so an INITIAL_FILL
    # source left unnamed re-fires every substep and the fill volume grows
    # without bound.
    box.source_id = "equilibrium_fill"
    box.position = (-0.55, 0.0, 0.9)
    box.half_extents = (0.45, 1.0, 0.9)
    return [box]


def fill_volume(sources) -> float:
    hx, hy, hz = sources[0].half_extents
    return 8.0 * hx * hy * hz


def measure(solver, settings, voxel_size):
    """Absolute equilibrium metrics, computed from live particles only."""
    import numpy as np

    p = solver.particles
    n = p.refresh_host_count()
    if n <= 0:
        return {"alive": 0}
    alive = p.alive.numpy()[:n].astype(bool)
    pos = p.position.numpy()[:n][alive]
    vel = p.velocity.numpy()[:n][alive]
    if pos.shape[0] == 0:
        return {"alive": 0}

    lo = np.asarray(settings.domain_min, dtype=np.float64)
    hi = np.asarray(settings.domain_max, dtype=np.float64)
    inside = np.all((pos >= lo) & (pos <= hi), axis=1)

    speed2 = np.sum(vel.astype(np.float64) ** 2, axis=1)

    # Occupied volume from an analysis grid independent of the solver's own
    # sparse grid, so clumping shows up as volume loss rather than hiding in
    # the solver's voxel bookkeeping.
    #
    # Flattened to a single int64 key first: np.unique(..., axis=0) on a
    # 2-D array of a few million rows costs seconds, enough that an earlier
    # revision of this harness spent more wall time measuring than simulating
    # and reported flat cost across a 7x range of substep counts.
    dims = np.maximum(np.ceil((hi - lo) / voxel_size).astype(np.int64), 1)
    cell = np.floor((pos - lo) / voxel_size).astype(np.int64)
    np.clip(cell, 0, dims - 1, out=cell)
    col = cell[:, 0] * dims[1] + cell[:, 1]
    lin = col * dims[2] + cell[:, 2]
    occupied_volume = np.unique(lin).shape[0] * voxel_size ** 3

    # Surface height per floor column: the top particle in each (x, y) column.
    order = np.lexsort((pos[:, 2], col))
    col_sorted = col[order]
    z_sorted = pos[order, 2]
    last = np.ones(col_sorted.shape[0], dtype=bool)
    last[:-1] = col_sorted[:-1] != col_sorted[1:]
    heights = z_sorted[last]

    return {
        "alive": int(pos.shape[0]),
        "outsideDomain": int((~inside).sum()),
        "kineticEnergyMean": float(speed2.mean()),
        "speedMax": float(math.sqrt(speed2.max())),
        "occupiedVolume": float(occupied_volume),
        "surfaceHeightMean": float(heights.mean()),
        "surfaceHeightStd": float(heights.std()),
        "columnsOccupied": int(heights.shape[0]),
        "zMin": float(pos[:, 2].min()),
        "zMax": float(pos[:, 2].max()),
    }


def main(argv: list[str] | None = None) -> int:
    _ensure_package_on_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--st", choices=("on", "off"), default="off")
    # Alias, because every other harness here spells it --spacetime.  Without
    # it, parse_known_args silently drops the flag and the run reports itself
    # as ST-on in the filename while solving ST-off -- which is exactly how an
    # entire 8-config sweep got measured against itself.
    ap.add_argument("--spacetime", action="store_true",
                    help="same as --st on")
    ap.add_argument("--cfl", type=float, default=1.0)
    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="simulated duration, identical for every config")
    ap.add_argument("--max-substeps", type=int, default=256)
    ap.add_argument("--pressure-iterations", type=int, default=None)
    ap.add_argument("--pressure-tolerance", type=float, default=None)
    ap.add_argument("--output", default=None)
    # Phase attribution.  CUDA-syncs every phase, so a --profile run costs more
    # than the same config without it -- read solveSeconds from a plain run and
    # the split from a profiled one, never both from the same run.
    ap.add_argument("--profile", action="store_true",
                    help="collect per-phase GPU timings (implies phase sync)")
    # Generic knob probe, rather than one flag per SolverSettings field: the ST
    # path has a dozen sub-settings and isolating which one moves a metric means
    # turning them off one at a time.  Coerced to the existing field's type and
    # echoed into the JSON so a result file always says what produced it.
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override a SolverSettings field (repeatable)")
    args, extra = ap.parse_known_args(argv)
    # parse_known_args is here for Blender's "--" passthrough, not to swallow
    # typos.  A dropped flag produces a plausible number for the wrong config,
    # which is worse than a crash.
    stray = [tok for tok in extra if tok.startswith("--")]
    if stray:
        ap.error(f"unrecognised options: {' '.join(stray)}")

    from fluidSim_warp_blender51.sim.solver import FluidSolver

    settings = equilibrium_settings(args.voxel_size, args.max_substeps)
    settings.device = args.device
    settings.spacetime_enabled = args.st == "on" or args.spacetime
    settings.cfl = float(args.cfl)
    if args.pressure_iterations is not None:
        settings.pressure_iterations = int(args.pressure_iterations)
    if args.pressure_tolerance is not None:
        settings.pressure_tolerance = float(args.pressure_tolerance)

    overrides = {}
    for item in args.set:
        field, _, raw = item.partition("=")
        field = field.strip()
        if not _ or not hasattr(settings, field):
            ap.error(f"--set: no SolverSettings field {field!r}")
        current = getattr(settings, field)
        if isinstance(current, bool):
            # bool("false") is True, so parse the word, not the truthiness.
            value = raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            value = type(current)(raw)
        setattr(settings, field, value)
        overrides[field] = value

    sources = equilibrium_sources()

    frames = int(round(args.seconds * settings.frame_rate / settings.time_scale))
    v_fill = fill_volume(sources)
    footprint = ((settings.domain_max[0] - settings.domain_min[0]) *
                 (settings.domain_max[1] - settings.domain_min[1]))
    h_expected = settings.domain_min[2] + v_fill / footprint

    print(f"equilibrium: spacetime_enabled={settings.spacetime_enabled} "
          f"cfl={args.cfl} dx={args.voxel_size} "
          f"frames={frames} ({args.seconds}s) fill_volume={v_fill:.4f} "
          f"expected_rest_depth={h_expected:.4f} overrides={overrides}",
          flush=True)

    profiler = None
    if args.profile:
        from fluidSim_warp_blender51.diagnostics.profiler import Profiler
        profiler = Profiler(device=args.device, sync=True, enabled=True)
    solver = FluidSolver(settings, profiler=profiler)
    solver.set_sources(sources)

    import warp as wp
    device = wp.get_device(args.device)

    history = []
    clamped_frames = 0
    # Solve time only: measure() reads back every particle and sorts them on
    # the CPU, which costs the same regardless of substep count and would
    # otherwise flatten the cost axis this whole harness exists to measure.
    # Synchronised so async kernel launches land inside the interval.
    solve = 0.0
    t_wall = time.perf_counter()
    for f in range(frames):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        stats = solver.step_frame()
        wp.synchronize_device(device)
        solve += time.perf_counter() - t0
        if getattr(stats, "cfl_substeps_clamped", False):
            clamped_frames += 1
        history.append({
            "frame": f + 1,
            "stepTime": float(stats.step_time),
            "activeVoxels": int(stats.active_voxels),
            "substeps": int(getattr(stats, "substeps", 0)),
            # Volume loss and PPE convergence are the same question: an
            # under-converged projection leaves divergence in the field and
            # the fluid compresses.  Recorded so a resolution scan can tell
            # the two apart.
            "pressureIterations": int(getattr(stats, "pressure_iterations", 0)),
            "pressureResidual": float(getattr(stats, "pressure_residual", 0.0)),
            # FrameStats carries the number but not why the solve stopped, and
            # "hit the iteration cap" and "broke down at iteration 100" call
            # for opposite fixes.  Read the reason off the solver directly.
            "pressureStatus": str(getattr(
                solver.pressure_solver, "last_pressure_status", "")),
            "pressureFailure": str(getattr(
                solver.pressure_solver, "last_pcg_failure", "")),
            "timers": {k: float(v) for k, v in
                       getattr(stats, "timers", {}).items()},
        })
        # Frame 1 is the baseline: the fill is still a clean box, so its
        # occupied-cell volume is the true volume.  Sampling the baseline
        # mid-collapse instead measures splash aeration, and the later
        # closing of those air pockets then reads as spurious volume loss.
        if f == 0 or (f + 1) % 24 == 0:
            m = measure(solver, settings, args.voxel_size)
            history[-1].update(m)
            print(f"  frame {f + 1}/{frames} {stats.step_time * 1000.0:7.1f} ms "
                  f"ke={m.get('kineticEnergyMean', 0.0):.4f} "
                  f"h={m.get('surfaceHeightMean', 0.0):.4f} "
                  f"alive={m.get('alive', 0)}", flush=True)
    wall = time.perf_counter() - t_wall

    final = measure(solver, settings, args.voxel_size)
    first = next((h for h in history if "alive" in h), {})
    alive0 = int(first.get("alive", 0)) or 1
    vol0 = float(first.get("occupiedVolume", 0.0)) or 1.0

    result = {
        "spacetimeEnabled": settings.spacetime_enabled,
        "device": args.device,
        "cfl": float(settings.cfl),
        "voxelSize": float(args.voxel_size),
        "maxSubsteps": int(args.max_substeps),
        "simulatedSeconds": float(args.seconds),
        "frames": frames,
        "wallSeconds": wall,
        # The cost number to compare.  wallSeconds includes measurement.
        "solveSeconds": solve,
        "cflSubstepsClampedFrames": clamped_frames,
        "expectedRestDepth": h_expected,
        "fillVolume": v_fill,
        "overrides": overrides,
        "final": final,
        # Equilibrium verdict inputs.  Kept as raw numbers rather than a
        # pass/fail so a sweep can plot them; thresholds belong to whoever
        # reads the sweep, not to the run.
        "restDepthError": (final.get("surfaceHeightMean", 0.0) - h_expected),
        "particleDrift": (final.get("alive", 0) - alive0) / alive0,
        "volumeDrift": (final.get("occupiedVolume", 0.0) - vol0) / vol0,
        # Absolute, needs no baseline frame: the fill volume is known exactly
        # from the source box.  volumeDrift is relative to frame 1 and so
        # still carries whatever discretisation error the fill itself had.
        "volumeErrorVsFill": (final.get("occupiedVolume", 0.0) - v_fill) / v_fill,
        "baselineVolume": vol0,
        "history": history,
    }

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  written to {out}")

    print(f"  solve={solve:.2f}s  wall={wall:.2f}s  "
          f"clampedFrames={clamped_frames}")
    print(f"  restDepth={final.get('surfaceHeightMean', 0.0):.4f} "
          f"(expected {h_expected:.4f}, error {result['restDepthError']:+.4f})")
    print(f"  flatness(std)={final.get('surfaceHeightStd', 0.0):.4f}  "
          f"ke={final.get('kineticEnergyMean', 0.0):.6f}  "
          f"speedMax={final.get('speedMax', 0.0):.4f}")
    print(f"  particleDrift={result['particleDrift']:+.4%}  "
          f"volumeDrift={result['volumeDrift']:+.4%}  "
          f"outsideDomain={final.get('outsideDomain', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(
        sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None))
