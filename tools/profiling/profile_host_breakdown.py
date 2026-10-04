#!/usr/bin/env python3
"""cProfile the host side of ``FluidSolver.step_frame`` (post-warmup frames).

Complements ``profile_kernel_breakdown.py``: that shows GPU-busy vs wall; this
shows which Python call sites own the host time (launches, syncs, readbacks).

    python tools/profiling/profile_host_breakdown.py --scene-preset dam_break
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profiling import profile_gpu_solver as pgs  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--emit-frames", type=int, default=4)
    parser.add_argument("--scene-preset", default="dam_break")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out")
    ns = parser.parse_args(argv)
    args = argparse.Namespace(
        device=ns.device, frames=ns.frames, warmup=ns.warmup,
        emit_frames=ns.emit_frames, voxel_size=None, max_particles=None,
        max_whitewater=200000, min_substeps=None, max_substeps=None,
        whitewater=False, no_surface=False, sync=False, no_phase_timers=True,
        scene_preset=ns.scene_preset, no_whitewater=False, viscosity=False,
        viscosity_constant=1.0, viscosity_iterations=None,
        viscosity_tolerance=None, surface_tension=False,
        surface_tension_coefficient=0.072)
    prof = cProfile.Profile()
    last = ns.warmup + ns.frames - 1

    def hook(index, _stats):
        if index == ns.warmup - 1:
            prof.enable()
        elif index == last:
            prof.disable()

    summary, _ = pgs.run_profile(args, on_frame=hook)
    print(f"stepMsMean={summary['stepMsMean']:.2f}")
    stats = pstats.Stats(prof)
    if ns.out:
        stats.dump_stats(ns.out)
    stats.sort_stats("tottime").print_stats(ns.top)
    stats.sort_stats("cumulative").print_stats(ns.top)


if __name__ == "__main__":
    main()
