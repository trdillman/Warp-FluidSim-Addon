"""Run one fresh-process 250k publication ceiling measurement."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback

import bpy
import gpu
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXECUTABLE = Path(r"F:\blender-gpu-interop-official-build\bin\blender.exe")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import blender_native_interop_benchmark as benchmark
from tools.repo_addon_loader import enable_repo_addon


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_limits():
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    handle = ctypes.c_void_p(-1)
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GetProcessAffinityMask(
        handle, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError()
    return {
        "priority_class": hex(kernel32.GetPriorityClass(handle)),
        "affinity_mask": hex(process_mask.value),
    }


def _schedule_quit():
    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-mode", choices=("packed", "no_repack"), required=True)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    device = wp.get_device("cuda:0")
    executable = Path(bpy.app.binary_path).resolve()
    limits_before = _process_limits()
    executable_hash_before = _sha256(executable)
    workload = benchmark.benchmark_stage_ceiling(
        bpy.context.scene,
        250_000,
        device,
        args.first_mode,
    )
    packed = workload["packed_batch_mean_ms"]
    no_repack = workload["no_repack_batch_mean_ms"]
    packed_spread = statistics.pstdev(packed["samples"]) / statistics.fmean(
        packed["samples"])
    no_repack_spread = statistics.pstdev(
        no_repack["samples"]
    ) / statistics.fmean(no_repack["samples"])
    stats = workload["native_stats"]
    pack_median = workload["cuda_stage_ms"]["position_pack_ms"]["median"]
    limits_after = _process_limits()
    executable_hash_after = _sha256(executable)
    gates = {
        "viewport_equivalent": workload["viewport_equivalent"],
        "visible_framebuffer_changed": workload["visible_framebuffer_changed"],
        "gpu_draw_count_matches": workload["gpu_point_count"] == 250_000,
        "redraws_completed": (
            workload["redraws"]["completed"] == workload["redraws"]["requested"]
        ),
        "zero_bulk_host_position_bytes": stats["bulk_host_position_bytes"] == 0,
        "fixed_imports_and_maps": stats["imports"] == 6 and stats["maps"] == 2,
        "packed_spread_within_10_percent": packed_spread <= 0.1,
        "no_repack_spread_within_10_percent": no_repack_spread <= 0.1,
        "expected_executable": executable == EXPECTED_EXECUTABLE,
        "executable_unchanged": executable_hash_before == executable_hash_after,
        "execution_limits_retained": (
            limits_before == limits_after
            == {"priority_class": "0x4000", "affinity_mask": "0xffff"}
        ),
        "idealized_reaches_2x": workload["idealized_speedup"] >= 2.0,
        "packing_below_1ms": pack_median < 1.0,
        "packing_below_10_percent": pack_median < packed["median"] * 0.1,
    }
    passed = all((
            gates["viewport_equivalent"],
            gates["visible_framebuffer_changed"],
            gates["gpu_draw_count_matches"],
            gates["redraws_completed"],
            gates["zero_bulk_host_position_bytes"],
            gates["fixed_imports_and_maps"],
            gates["packed_spread_within_10_percent"],
            gates["no_repack_spread_within_10_percent"],
            gates["expected_executable"],
            gates["executable_unchanged"],
            gates["execution_limits_retained"],
        ))
    result = {
        "passed": passed,
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "device": device.name,
            "warp_version": wp.__version__,
            "blender_version": bpy.app.version_string,
            "executable": str(executable),
            "executable_sha256_before": executable_hash_before,
            "executable_sha256_after": executable_hash_after,
            "addon_source": str(REPO_ROOT.resolve()),
            "process_limits_before": limits_before,
            "process_limits_after": limits_after,
        },
        "measurement_scope": (
            "unchanged shared VBO data; both modes retain lease, timeline, metadata, "
            "PointCloud binding, and normal View3D redraw"
        ),
        "sample_spread_fraction": {
            "packed": packed_spread,
            "no_repack": no_repack_spread,
        },
        "sample_spread_metric": (
            "population standard deviation divided by mean of "
            f"{benchmark.STAGE_BATCH_FRAMES}-frame batch means; all per-frame samples, "
            "p95, and maximum are retained"
        ),
        "gates": gates,
        "decision": "awaiting_cross_process_aggregate" if passed else "measurement_invalid",
        "workload": workload,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_STAGE_PROFILE=" + json.dumps(result, sort_keys=True), flush=True)

    _schedule_quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        args = _args()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"passed": False, "error": repr(exc), "traceback": traceback.format_exc()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _schedule_quit()
