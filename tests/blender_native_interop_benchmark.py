"""Compare steady-state CPU datablock presentation with native GPU interop."""

from tools.repo_addon_loader import enable_repo_addon
import json
from pathlib import Path
import statistics
import sys
import time

import bpy
import gpu
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


ITERATIONS = 20


def percentile_95(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def draw_pending(presenter, offscreen):
    with offscreen.bind():
        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            presenter.draw()


def configure_scene():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    settings = bpy.context.scene.fluidsim
    settings.voxel_size = 0.14
    settings.max_particles = 120_000
    settings.max_whitewater = 20_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status
    return settings


def main():
    settings = configure_scene()
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    runtime.solver.step_frame()
    count = runtime.solver.particles.host_count
    assert count > 0

    settings.native_gpu_interop = False
    viewport.update_outputs(bpy.context.scene, runtime)
    cpu_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        viewport.update_outputs(bpy.context.scene, runtime)
        cpu_times.append((time.perf_counter_ns() - start) / 1_000_000.0)

    settings.native_gpu_interop = True
    offscreen = gpu.types.GPUOffScreen(64, 64)
    try:
        viewport.update_outputs(bpy.context.scene, runtime)
        assert runtime.native_presenter is not None, runtime.native_interop_error
        draw_pending(runtime.native_presenter, offscreen)
        native_times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter_ns()
            viewport.update_outputs(bpy.context.scene, runtime)
            draw_pending(runtime.native_presenter, offscreen)
            native_times.append((time.perf_counter_ns() - start) / 1_000_000.0)
        with offscreen.bind():
            gpu.state.active_framebuffer_get().read_color(0, 0, 1, 1, 4, 0, "UBYTE")
    finally:
        offscreen.free()

    stats = runtime.native_presenter.bridge.stats
    cpu_median = statistics.median(cpu_times)
    native_median = statistics.median(native_times)
    report = {
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "device": runtime.native_presenter.bridge.device.name,
            "device_uuid": str(runtime.native_presenter.bridge.device.uuid),
            "warp_version": runtime.native_presenter.bridge.wp.__version__,
            "blender_version": bpy.app.version_string,
        },
        "workload": {
            "particle_count": count,
            "iterations": ITERATIONS,
            "bytes_per_frame": count * 12,
            "surface_enabled": False,
        },
        "cpu_datablock_ms": {
            "median": cpu_median,
            "p95": percentile_95(cpu_times),
            "samples": cpu_times,
            "bulk_host_position_bytes": count * 12 * ITERATIONS,
        },
        "native_interop_ms": {
            "median": native_median,
            "p95": percentile_95(native_times),
            "samples": native_times,
            "bulk_host_position_bytes": stats["bulk_host_position_bytes"],
            "imports": stats["imports"],
            "maps": stats["maps"],
        },
        "median_speedup": cpu_median / native_median,
        "measurement_scope": (
            "steady-state presentation only; native samples enqueue CUDA copy, timeline handoff, "
            "and Vulkan draw, followed by one final framebuffer drain"
        ),
    }
    assert report["native_interop_ms"]["bulk_host_position_bytes"] == 0
    assert report["native_interop_ms"]["imports"] == 3
    assert report["native_interop_ms"]["maps"] == 1
    evidence_dir = REPO_ROOT / "docs" / "evidence" / "native_interop"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output = evidence_dir / "benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_BENCHMARK", json.dumps(report, sort_keys=True))
    bpy.ops.fluidsim.release()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
