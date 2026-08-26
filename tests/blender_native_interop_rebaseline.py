"""Record the native presentation benchmark without hiding failed thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy
import gpu
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import blender_native_interop_benchmark as benchmark
from tools.repo_addon_loader import enable_repo_addon


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = _args()
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    device = wp.get_device("cuda:0")
    workloads = [benchmark.benchmark_count(scene, count, device) for count in benchmark.COUNTS]
    report = {
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "device": device.name,
            "warp_version": wp.__version__,
            "blender_version": bpy.app.version_string,
        },
        "workloads": workloads,
        "gates": {
            "25000_native_within_10_percent": workloads[0]["native_end_to_end_ms"]["median"]
            <= workloads[0]["cpu_end_to_end_ms"]["median"] * 1.1,
            "250000_speedup_at_least_2x": workloads[1]["median_speedup"] >= 2.0,
            "1000000_speedup_at_least_2x": workloads[2]["median_speedup"] >= 2.0,
        },
        "measurement_scope": (
            "same deterministic Warp allocation; CPU path includes device-to-host copy, PointCloud "
            "replacement, and normal View3D redraw; native path includes shared-VBO publication and "
            "the same normal View3D redraw"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_REBASELINE", json.dumps(report, sort_keys=True), flush=True)

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
