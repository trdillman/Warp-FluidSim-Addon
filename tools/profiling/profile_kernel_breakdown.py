#!/usr/bin/env python3
"""Per-kernel GPU time vs wall time for the solver (launch-bound vs GPU-bound).

Wraps ``tools/profile_gpu_solver.run_profile`` with Warp's CUDA activity
timing for the measured (post-warmup) frames and aggregates by kernel/memcpy
name, so the ratio of summed GPU time to wall time shows how much of a frame
is launch/host overhead rather than device work.

    python tools/profiling/profile_kernel_breakdown.py --scene-preset dam_break \
        --frames 8 --warmup 3 --json-out artifacts/perf75/kernels_dam.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.profiling import profile_gpu_solver as pgs  # noqa: E402


def main(argv=None):
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--emit-frames", type=int, default=4)
    parser.add_argument("--scene-preset", default="dam_break")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-out")
    ns, rest = parser.parse_known_args(argv)

    args = argparse.Namespace(
        device=ns.device, frames=ns.frames, warmup=ns.warmup,
        emit_frames=ns.emit_frames, voxel_size=None, max_particles=None,
        max_whitewater=200000, min_substeps=None, max_substeps=None,
        whitewater=False, no_surface=False, sync=False, no_phase_timers=True,
        scene_preset=ns.scene_preset, no_whitewater=False, viscosity=False,
        viscosity_constant=1.0, viscosity_iterations=None,
        viscosity_tolerance=None, surface_tension=False,
        surface_tension_coefficient=0.072)

    state = {"on": False}

    def on_frame(index, _stats):
        if index == ns.warmup - 1 and not state["on"]:
            wp.synchronize_device(ns.device)
            wp.timing_begin(cuda_filter=wp.TIMING_ALL, synchronize=True)
            state["on"] = True

    # timing_end must run before solver.release(); run_profile releases in a
    # finally block, so collect inside a wrapper around the last frame.
    results = {}
    last = ns.warmup + ns.frames - 1

    def hook(index, stats):
        on_frame(index, stats)
        if index == last and state["on"]:
            wp.synchronize_device(ns.device)
            results["r"] = wp.timing_end(synchronize=True)
            state["on"] = False

    summary, _frames = pgs.run_profile(args, on_frame=hook)
    per = defaultdict(lambda: [0, 0.0])
    for r in results.get("r", []):
        entry = per[(r.name, getattr(r, "activity", getattr(r, "category", "")))]
        entry[0] += 1
        entry[1] += float(r.elapsed)
    total_gpu = sum(v[1] for v in per.values())
    wall_ms = summary["stepMsMean"] * ns.frames
    rows = sorted(per.items(), key=lambda kv: kv[1][1], reverse=True)
    print(f"frames={ns.frames} wall_ms_total={wall_ms:.1f} "
          f"gpu_busy_ms_total={total_gpu:.1f} "
          f"busy_fraction={total_gpu / max(wall_ms, 1e-9):.2f} "
          f"launches={sum(v[0] for v in per.values())}")
    print(f"per-frame: wall={wall_ms / ns.frames:.2f} ms "
          f"gpu={total_gpu / ns.frames:.2f} ms "
          f"ops/frame={sum(v[0] for v in per.values()) / ns.frames:.0f}")
    for (name, activity), (count, ms) in rows[: ns.top]:
        print(f"  {ms / ns.frames:8.3f} ms/frame {count / ns.frames:8.1f} x "
              f"{ms / max(count, 1) * 1000:8.1f} us  [{activity}] {name}")
    if ns.json_out:
        Path(ns.json_out).write_text(json.dumps({
            "summary": summary,
            "gpuBusyMsPerFrame": total_gpu / ns.frames,
            "wallMsPerFrame": wall_ms / ns.frames,
            "kernels": [
                {"name": n, "activity": a, "count": c, "totalMs": ms}
                for (n, a), (c, ms) in rows],
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
