"""Attribute canonical Dam Break time to solver, interop publication, and PointCloud draw."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from time import perf_counter_ns, process_time_ns
import traceback

import addon_utils
import bpy
import gpu


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLEND = (REPO_ROOT / "demos" / "dam_break.blend").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import blender_native_interop_benchmark as benchmark
from blender_native_interop_stage_profile import (
    EXPECTED_EXECUTABLE,
    _process_limits,
    _schedule_quit,
    _sha256,
)
from tools.repo_addon_loader import enable_repo_addon


EXPECTED_LIMITS = {"priority_class": "0x4000", "affinity_mask": "0xffff"}
EXPECTED_PARTICLES = 600_000
_NVTX_AVAILABLE = None


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-mode", choices=("visible", "hidden"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--cuda-profiler-range", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.frames < 1 or args.batch_size < 1:
        parser.error("--warmup, --frames, and --batch-size must be positive")
    if args.frames % args.batch_size:
        parser.error("--frames must be divisible by --batch-size")
    return args


def _enforce_process_limits():
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not kernel32.SetPriorityClass(process, 0x4000):
        raise ctypes.WinError()
    if not kernel32.SetProcessAffinityMask(process, 0xFFFF):
        raise ctypes.WinError()
    if _process_limits() != EXPECTED_LIMITS:
        raise RuntimeError(f"Failed to retain process limits: {_process_limits()}")


def _nvtx_range(wp, name, enabled):
    global _NVTX_AVAILABLE
    if not enabled:
        return nullcontext()
    if _NVTX_AVAILABLE is None:
        try:
            __import__("nvtx")
        except ModuleNotFoundError:
            _NVTX_AVAILABLE = False
        else:
            _NVTX_AVAILABLE = True
    if not _NVTX_AVAILABLE:
        return nullcontext()
    return wp.ScopedTimer(
        name,
        print=False,
        use_nvtx=True,
        synchronize=False,
        skip_tape=True,
    )


def _cuda_profiler(enabled, start):
    if not enabled:
        return
    cuda = ctypes.WinDLL("nvcuda.dll")
    function = cuda.cuProfilerStart if start else cuda.cuProfilerStop
    result = function()
    if result != 0:
        raise RuntimeError(f"{'cuProfilerStart' if start else 'cuProfilerStop'} failed: {result}")


def _batch_means(values, batch_size):
    return [fmean(values[index:index + batch_size]) for index in range(0, len(values), batch_size)]


def _spread(values):
    return pstdev(values) / fmean(values)


def _set_visible(obj, presenter, visible):
    obj.hide_viewport = not visible
    bpy.context.view_layer.update()
    if visible:
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        presenter.bind_pointcloud(evaluated)
        presenter._restore_active_binding()


def _frame_pointcloud(obj):
    for other in bpy.context.scene.objects:
        if other != obj:
            other.hide_viewport = True
    bpy.context.view_layer.update()
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(
        area=area,
        region=region,
        space_data=area.spaces.active,
        active_object=obj,
        object=obj,
        selected_objects=[obj],
        selected_editable_objects=[obj],
    ):
        bpy.ops.view3d.view_axis(type="FRONT", align_active=False)
        bpy.ops.view3d.view_selected(use_all_regions=False)
        bounds = tuple(obj.data["native_interop_bounds"])
        lo = bounds[:3]
        hi = bounds[3:]
        center = tuple((low + high) * 0.5 for low, high in zip(lo, hi))
        extent = max(high - low for low, high in zip(lo, hi))
        view = area.spaces.active.region_3d
        view.view_location = center
        view.view_distance = extent * 0.75
        area.spaces.active.clip_start = max(extent * 0.0001, 0.0001)
        area.spaces.active.clip_end = max(extent * 100.0, 1000.0)
    return {
        "bounds": bounds,
        "center": center,
        "extent": extent,
        "view_distance": view.view_distance,
        "region_size": (region.width, region.height),
    }


def _redraw_timed(wp, label, nvtx):
    started = perf_counter_ns()
    with _nvtx_range(wp, label, nvtx):
        benchmark.redraw_viewport()
    return (perf_counter_ns() - started) / 1_000_000.0


def _configure_and_initialize():
    scene = bpy.context.scene
    settings = scene.fluidsim
    bpy.ops.fluidsim.release()
    settings.native_gpu_interop = True
    settings.show_particles = True
    settings.liquid_view_mode = "PARTICLES"
    settings.surface_enabled = False
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    settings.live = False
    scene.frame_set(scene.frame_start)
    if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
        raise RuntimeError(settings.status)
    return scene


def _run_profile(args):
    import warp as wp
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = _configure_and_initialize()
    runtime = session()
    for warmup_index in range(args.warmup):
        runtime.solver.step_frame()
        wp.synchronize_device(runtime.solver.device)
        viewport.update_outputs(scene, runtime)
        benchmark.redraw_viewport()
        if warmup_index == 0:
            if runtime.native_presenter is None:
                raise RuntimeError(
                    "Native presenter unavailable after warmup: "
                    f"backend={gpu.platform.backend_type_get()}, "
                    f"reason={runtime.native_interop_error}"
                )
            runtime.native_presenter.publication_profiling_enable()

    presenter = runtime.native_presenter
    if presenter is None:
        raise RuntimeError(f"Native presenter unavailable: {runtime.native_interop_error}")
    obj = bpy.data.objects[viewport.PARTICLES_NAME]
    viewport_state = _frame_pointcloud(obj)
    _set_visible(obj, presenter, True)
    benchmark.redraw_viewport()
    stats_before = dict(presenter.bridge.stats)
    samples = []
    _cuda_profiler(args.cuda_profiler_range, True)
    try:
        first_pair = (
            (args.first_mode, "hidden" if args.first_mode == "visible" else "visible")
        )
        for index in range(args.frames):
            order = first_pair if index % 2 == 0 else tuple(reversed(first_pair))
            frame_started = perf_counter_ns()
            cpu_started = process_time_ns()

            phase_started = perf_counter_ns()
            with _nvtx_range(wp, "dam_break_solver_gpu_wall", args.cuda_profiler_range):
                runtime.solver.step_frame()
                wp.synchronize_device(runtime.solver.device)
            solver_ms = (perf_counter_ns() - phase_started) / 1_000_000.0

            phase_started = perf_counter_ns()
            with _nvtx_range(wp, "interop_publication_host", args.cuda_profiler_range):
                viewport.update_outputs(scene, runtime)
            presentation_ms = (perf_counter_ns() - phase_started) / 1_000_000.0
            profile = dict(presenter.publication_profile or {})

            redraw_ms = {}
            for mode in order:
                _set_visible(obj, presenter, mode == "visible")
                redraw_ms[mode] = _redraw_timed(
                    wp,
                    f"pointcloud_{mode}_redraw",
                    args.cuda_profiler_range,
                )
            samples.append(
                {
                    "index": index,
                    "order": list(order),
                    "solver_gpu_wall_ms": solver_ms,
                    "presentation_host_ms": presentation_ms,
                    "cuda_publication_ms": profile.get("cuda_publication_ms"),
                    "cuda_stage_ms": profile,
                    "visible_redraw_ms": redraw_ms["visible"],
                    "hidden_redraw_ms": redraw_ms["hidden"],
                    "whole_frame_ms": (perf_counter_ns() - frame_started) / 1_000_000.0,
                    "process_cpu_ms": (process_time_ns() - cpu_started) / 1_000_000.0,
                }
            )
    finally:
        _cuda_profiler(args.cuda_profiler_range, False)

    _set_visible(obj, presenter, True)
    visible_framebuffer = benchmark.viewport_capture()
    _set_visible(obj, presenter, False)
    hidden_framebuffer = benchmark.viewport_capture()
    _set_visible(obj, presenter, True)
    visibility_difference = benchmark.framebuffer_difference(
        visible_framebuffer, hidden_framebuffer
    )

    active_bridge = presenter._slots[presenter._slot_index].bridge
    wp.synchronize_stream(active_bridge.stream)
    draw_vertex_count = int(active_bridge.draw_vertex_count.numpy()[0])
    stats_after = dict(presenter.bridge.stats)
    values = {
        key: [sample[key] for sample in samples]
        for key in (
            "solver_gpu_wall_ms",
            "presentation_host_ms",
            "cuda_publication_ms",
            "visible_redraw_ms",
            "hidden_redraw_ms",
            "whole_frame_ms",
            "process_cpu_ms",
        )
    }
    batch_values = {
        key: _batch_means(values[key], args.batch_size)
        for key in ("visible_redraw_ms", "hidden_redraw_ms")
    }
    return {
        "particle_count": runtime.solver.particles.host_count,
        "first_mode": args.first_mode,
        "warmup_frames": args.warmup,
        "measured_frames": args.frames,
        "batch_size": args.batch_size,
        "samples": samples,
        "solver_gpu_wall_ms": benchmark.summarize(values["solver_gpu_wall_ms"]),
        "presentation_host_ms": benchmark.summarize(values["presentation_host_ms"]),
        "cuda_publication_ms": benchmark.summarize(values["cuda_publication_ms"]),
        "visible_redraw_ms": benchmark.summarize(values["visible_redraw_ms"]),
        "hidden_redraw_ms": benchmark.summarize(values["hidden_redraw_ms"]),
        "visible_redraw_batch_mean_ms": benchmark.summarize(
            batch_values["visible_redraw_ms"]
        ),
        "hidden_redraw_batch_mean_ms": benchmark.summarize(
            batch_values["hidden_redraw_ms"]
        ),
        "whole_frame_ms": benchmark.summarize(values["whole_frame_ms"]),
        "process_cpu_ms": benchmark.summarize(values["process_cpu_ms"]),
        "batch_spread_fraction": {
            "visible": _spread(batch_values["visible_redraw_ms"]),
            "hidden": _spread(batch_values["hidden_redraw_ms"]),
        },
        "visibility_difference": visibility_difference,
        "viewport_state": viewport_state,
        "gpu_draw_vertex_count": draw_vertex_count,
        "gpu_point_count": draw_vertex_count // 8,
        "native_stats_before": stats_before,
        "native_stats": stats_after,
    }


def main():
    args = _args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _enforce_process_limits()
    executable = Path(bpy.app.binary_path).resolve()
    limits_before = _process_limits()
    executable_hash_before = _sha256(executable)
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    if Path(bpy.data.filepath).resolve() != EXPECTED_BLEND:
        raise RuntimeError(f"Unexpected blend file: {bpy.data.filepath}")
    if addon_utils.check("fluidSim_warp_blender51") != (True, True):
        raise RuntimeError("Repository FluidSim add-on is not enabled")
    module = sys.modules["fluidSim_warp_blender51"]
    addon_module = Path(module.__file__).resolve()
    if not addon_module.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"Unexpected add-on source: {addon_module}")

    workload = _run_profile(args)
    limits_after = _process_limits()
    executable_hash_after = _sha256(executable)
    stats_before = workload["native_stats_before"]
    stats = workload["native_stats"]
    timing_keys = (
        "solver_gpu_wall_ms",
        "presentation_host_ms",
        "cuda_publication_ms",
        "visible_redraw_ms",
        "hidden_redraw_ms",
        "whole_frame_ms",
        "process_cpu_ms",
    )
    finite = all(
        math.isfinite(sample[key])
        for sample in workload["samples"]
        for key in timing_keys
    )
    gates = {
        "expected_executable": executable == EXPECTED_EXECUTABLE,
        "executable_unchanged": executable_hash_before == executable_hash_after,
        "execution_limits_retained": limits_before == limits_after == EXPECTED_LIMITS,
        "vulkan_backend": gpu.platform.backend_type_get() == "VULKAN",
        "canonical_blend": Path(bpy.data.filepath).resolve() == EXPECTED_BLEND,
        "repository_addon": addon_module.is_relative_to(REPO_ROOT),
        "particle_count": workload["particle_count"] == EXPECTED_PARTICLES,
        "gpu_draw_count_matches": workload["gpu_point_count"] == EXPECTED_PARTICLES,
        "visible_framebuffer_changed": (
            workload["visibility_difference"]["changed_bytes"] > 1000
        ),
        "zero_bulk_host_position_bytes": stats["bulk_host_position_bytes"] == 0,
        "fixed_imports_and_maps": stats["imports"] == 6 and stats["maps"] == 2,
        "fixed_imports_and_maps_during_measurement": (
            stats["imports"] == stats_before["imports"]
            and stats["maps"] == stats_before["maps"]
        ),
        "exact_publication_count": (
            stats["published_frames"] - stats_before["published_frames"] == args.frames
        ),
        "finite_timings": finite,
        "visible_batch_spread_within_10_percent": (
            workload["batch_spread_fraction"]["visible"] <= 0.1
        ),
        "hidden_batch_spread_within_10_percent": (
            workload["batch_spread_fraction"]["hidden"] <= 0.1
        ),
    }
    result = {
        "passed": all(gates.values()),
        "blend": str(EXPECTED_BLEND),
        "addon_module": str(addon_module),
        "environment": {
            "backend": gpu.platform.backend_type_get(),
            "blender_version": bpy.app.version_string,
            "executable": str(executable),
            "executable_sha256_before": executable_hash_before,
            "executable_sha256_after": executable_hash_after,
            "addon_source": str(REPO_ROOT.resolve()),
            "process_limits_before": limits_before,
            "process_limits_after": limits_after,
        },
        "measurement_scope": (
            "canonical 600k Dam Break; solver is synchronized for GPU wall attribution; "
            "one native publication followed by paired visible and hidden redraws of the same frame"
        ),
        "gates": gates,
        "decision": (
            "awaiting_cross_process_aggregate"
            if all(gates.values())
            else "measurement_invalid"
        ),
        "workload": workload,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_DRAW_PROFILE=" + json.dumps(result, sort_keys=True), flush=True)
    bpy.ops.fluidsim.release()
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
