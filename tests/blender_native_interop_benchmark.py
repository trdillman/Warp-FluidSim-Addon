"""Compare equivalent CPU and native PointCloud presentation workloads."""

import json
import hashlib
from pathlib import Path
import statistics
import sys
import time

import bpy
import gpu
from mathutils import Quaternion
import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon

COUNTS = (25_000, 250_000, 1_000_000)
ITERATIONS = 20
STAGE_ITERATIONS = 4
STAGE_BATCH_FRAMES = 500
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
        result = bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    assert result == {"FINISHED"}
    return True


def sample(call):
    start = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def summarize(samples):
    return {
        "median": statistics.median(samples),
        "p95": percentile_95(samples),
        "max": max(samples),
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


def viewport_capture():
    area = next(
        area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D"
    )
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(
        area=area, region=region, space_data=area.spaces.active
    ):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
        framebuffer = gpu.state.active_framebuffer_get()
        pixels = framebuffer.read_color(
            0, 0, region.width, region.height, 4, 0, "UBYTE"
        )
        pixels.dimensions = region.width * region.height * 4
        return bytes(pixels)


def framebuffer_difference(left, right):
    assert len(left) == len(right)
    delta = np.abs(
        np.frombuffer(left, dtype=np.uint8).astype(np.int16)
        - np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    )
    return {
        "mean_absolute_8bit": float(delta.mean()),
        "max_absolute_8bit": int(delta.max()),
        "p99_absolute_8bit": float(np.percentile(delta, 99.0)),
        "p999_absolute_8bit": float(np.percentile(delta, 99.9)),
        "changed_bytes": int(np.count_nonzero(delta)),
        "changed_fraction": float(np.count_nonzero(delta) / len(delta)),
        "fraction_above_4": float(np.count_nonzero(delta > 4) / len(delta)),
        "fraction_above_16": float(np.count_nonzero(delta > 16) / len(delta)),
    }


def benchmark_stage_ceiling(scene, count, device, first_mode):
    """Measure the idealized ceiling from retaining already-packed VBO contents."""
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.native_interop import NativeParticlePresenter

    assert first_mode in {"packed", "no_repack"}
    host_source = source_positions(count)
    device_source = wp.array(host_source, dtype=wp.vec3, device=device)
    active_count = wp.array([count], dtype=wp.int32, device=device)
    obj, pointcloud = viewport._ensure_native_pointcloud(scene, count)
    for name in ("Cube", "Camera", "Light"):
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_viewport = True
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    presenter = NativeParticlePresenter(count, pointcloud=evaluated, radius=RADIUS)

    redraws = {"requested": 0, "completed": 0}

    def frame(mode):
        presenter.publish_device_count(
            device_source,
            active_count,
            count,
            repack=mode == "packed",
        )
        redraws["requested"] += 1
        if redraw_viewport():
            redraws["completed"] += 1

    try:
        # Populate both frame slots before the diagnostic is allowed to retain their contents.
        frame("packed")
        frame("packed")
        area = next(
            area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D"
        )
        view = area.spaces.active.region_3d
        view.view_location = (0.5, 0.5, 0.0)
        view.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
        view.view_distance = 1.2

        for _index in range(250):
            frame("packed")
            frame("no_repack")

        frame("packed")
        visible_framebuffer = viewport_capture()
        obj.hide_viewport = True
        bpy.context.view_layer.update()
        hidden_framebuffer = viewport_capture()
        obj.hide_viewport = False
        bpy.context.view_layer.update()
        presenter.bind_pointcloud(
            obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        )
        frame("packed")
        frame("packed")
        packed_framebuffer = viewport_capture()
        frame("no_repack")
        no_repack_framebuffer = viewport_capture()
        viewport_difference = framebuffer_difference(
            packed_framebuffer, no_repack_framebuffer
        )
        visibility_difference = framebuffer_difference(
            visible_framebuffer, hidden_framebuffer
        )

        active_bridge = presenter._slots[presenter._slot_index].bridge
        wp.synchronize_stream(active_bridge.stream)
        gpu_draw_vertex_count = int(active_bridge.draw_vertex_count.numpy()[0])
        gpu_point_count = gpu_draw_vertex_count // 8

        frame_samples = {"packed": [], "no_repack": []}
        batch_samples = {"packed": [], "no_repack": []}
        first_pair = (first_mode, "no_repack" if first_mode == "packed" else "packed")
        for index in range(STAGE_ITERATIONS):
            pair = first_pair if index % 2 == 0 else tuple(reversed(first_pair))
            for mode in pair:
                batch = []
                for _frame_index in range(STAGE_BATCH_FRAMES):
                    started = time.perf_counter_ns()
                    frame(mode)
                    batch.append(
                        (time.perf_counter_ns() - started) / 1_000_000.0
                    )
                frame_samples[mode].extend(batch)
                batch_samples[mode].append(statistics.fmean(batch))

        presenter.publication_profiling_enable()
        stage_samples = {key: [] for key in (
            "external_wait_ms",
            "position_pack_ms",
            "metadata_ms",
            "cuda_publication_ms",
        )}
        for _index in range(5):
            presenter.publish_device_count(device_source, active_count, count, repack=True)
            for key, value in presenter.publication_profile.items():
                stage_samples[key].append(value)

        packed = summarize(frame_samples["packed"])
        no_repack = summarize(frame_samples["no_repack"])
        packed_batches = summarize(batch_samples["packed"])
        no_repack_batches = summarize(batch_samples["no_repack"])
        stats = dict(presenter.bridge.stats)
        return {
            "particle_count": count,
            "iterations_per_mode": STAGE_ITERATIONS,
            "frames_per_sample": STAGE_BATCH_FRAMES,
            "warmup_frames_per_mode": 251,
            "first_mode": first_mode,
            "packed_end_to_end_ms": packed,
            "no_repack_end_to_end_ms": no_repack,
            "packed_batch_mean_ms": packed_batches,
            "no_repack_batch_mean_ms": no_repack_batches,
            "idealized_speedup": (
                packed_batches["median"] / no_repack_batches["median"]
            ),
            "viewport_digest": hashlib.sha256(packed_framebuffer).hexdigest(),
            "hidden_viewport_digest": hashlib.sha256(hidden_framebuffer).hexdigest(),
            "viewport_difference": viewport_difference,
            "visibility_difference": visibility_difference,
            "viewport_equivalent": (
                viewport_difference["mean_absolute_8bit"] <= 0.5
                and viewport_difference["p99_absolute_8bit"] <= 3.0
                and viewport_difference["fraction_above_4"] <= 0.005
                and viewport_difference["fraction_above_16"] <= 0.001
            ),
            "visible_framebuffer_changed": visibility_difference["changed_bytes"] > 1000,
            "gpu_draw_vertex_count": gpu_draw_vertex_count,
            "gpu_point_count": gpu_point_count,
            "redraws": redraws,
            "cuda_stage_ms": {
                key: summarize(values) for key, values in stage_samples.items()
            },
            "native_stats": stats,
        }
    finally:
        presenter.close()


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    device = wp.get_device("cuda:0")
    workloads = [benchmark_count(scene, count, device) for count in COUNTS]
    gates = {
        "25000_native_within_10_percent": (
            workloads[0]["native_end_to_end_ms"]["median"]
            <= workloads[0]["cpu_end_to_end_ms"]["median"] * 1.1
        ),
        "250000_speedup_at_least_2x": workloads[1]["median_speedup"] >= 2.0,
        "1000000_speedup_at_least_2x": workloads[2]["median_speedup"] >= 2.0,
    }
    report = {
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "device": device.name,
            "warp_version": wp.__version__,
            "blender_version": bpy.app.version_string,
        },
        "workloads": workloads,
        "gates": gates,
        "measurement_scope": (
            "same deterministic Warp allocation; CPU path includes device-to-host copy, PointCloud "
            "replacement, and normal View3D redraw; native path includes shared-VBO publication "
            "and the same normal View3D redraw"
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
    assert all(gates.values()), gates


if __name__ == "__main__":
    main()
