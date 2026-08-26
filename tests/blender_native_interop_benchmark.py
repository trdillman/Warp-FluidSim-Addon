"""Compare equivalent CPU and native PointCloud presentation workloads."""

import json
from pathlib import Path
import statistics
import sys
import time

import bpy
import gpu
import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon

COUNTS = (25_000, 250_000, 1_000_000)
ITERATIONS = 20
RADIUS = 0.02


def percentile_95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def redraw_viewport():
    area = next(
        area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D"
    )
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(
        area=area, region=region, space_data=area.spaces.active
    ):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def sample(call):
    start = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def summarize(samples):
    return {
        "median": statistics.median(samples),
        "p95": percentile_95(samples),
        "samples": samples,
    }


def source_positions(count):
    indices = np.arange(count, dtype=np.float32)
    side = float(np.ceil(np.sqrt(count)))
    positions = np.empty((count, 3), dtype=np.float32)
    positions[:, 0] = (indices % side) * 0.002
    positions[:, 1] = np.floor(indices / side) * 0.002
    positions[:, 2] = np.sin(indices * 0.01) * 0.01
    return positions


def benchmark_count(scene, count, device):
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.native_interop import NativeParticlePresenter

    host_source = source_positions(count)
    device_source = wp.array(host_source, dtype=wp.vec3, device=device)
    active_count = wp.array([count], dtype=wp.int32, device=device)
    obj = bpy.data.objects.get(viewport.PARTICLES_NAME)
    if obj is None:
        pointcloud = bpy.data.pointclouds.new(viewport.PARTICLES_NAME)
        obj = bpy.data.objects.new(viewport.PARTICLES_NAME, pointcloud)
        scene.collection.objects.link(obj)

    def cpu_frame():
        viewport._swap_points(obj, device_source.numpy(), RADIUS)
        redraw_viewport()

    cpu_frame()
    cpu_samples = [sample(cpu_frame) for _ in range(ITERATIONS)]

    obj, pointcloud = viewport._ensure_native_pointcloud(scene, count)
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    presenter = NativeParticlePresenter(count, pointcloud=evaluated, radius=RADIUS)

    def native_frame():
        presenter.publish_device_count(device_source, active_count, count)
        redraw_viewport()

    native_frame()
    native_samples = [sample(native_frame) for _ in range(ITERATIONS)]
    stats = dict(presenter.bridge.stats)
    presenter.close()

    cpu = summarize(cpu_samples)
    native = summarize(native_samples)
    assert stats["bulk_host_position_bytes"] == 0
    assert stats["imports"] == 6
    assert stats["maps"] == 2
    return {
        "particle_count": count,
        "iterations": ITERATIONS,
        "bytes_per_frame": count * 12,
        "cpu_end_to_end_ms": {
            **cpu,
            "bulk_host_position_bytes": count * 12 * ITERATIONS,
        },
        "native_end_to_end_ms": {
            **native,
            "bulk_host_position_bytes": 0,
            "imports": stats["imports"],
            "maps": stats["maps"],
        },
        "median_speedup": cpu["median"] / native["median"],
    }


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    device = wp.get_device("cuda:0")
    workloads = [benchmark_count(scene, count, device) for count in COUNTS]
    assert workloads[0]["native_end_to_end_ms"]["median"] <= (
        workloads[0]["cpu_end_to_end_ms"]["median"] * 1.1
    )
    assert workloads[1]["median_speedup"] >= 2.0
    assert workloads[2]["median_speedup"] >= 2.0
    report = {
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "device": device.name,
            "warp_version": wp.__version__,
            "blender_version": bpy.app.version_string,
        },
        "workloads": workloads,
        "measurement_scope": (
            "same deterministic Warp allocation; CPU path includes device-to-host copy, PointCloud "
            "replacement, and normal View3D redraw; native path includes shared-VBO publication and "
            "the same normal View3D redraw"
        ),
    }
    evidence_dir = REPO_ROOT / "docs" / "evidence" / "native_interop"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_BENCHMARK", json.dumps(report, sort_keys=True))

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
