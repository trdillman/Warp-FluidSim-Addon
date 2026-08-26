"""Compare real solver-step time with CPU and native particle presentation."""

import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon

WARMUP_FRAMES = 5
MEASURED_FRAMES = 20


def run_case(native):
    scene = bpy.context.scene
    settings = scene.fluidsim
    bpy.ops.fluidsim.release()
    settings.voxel_size = 0.18
    settings.max_particles = 50_000
    settings.max_whitewater = 1_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.particle_attribute_mode = "NONE"
    settings.native_gpu_interop = native
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status

    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    samples = []
    for frame in range(WARMUP_FRAMES + MEASURED_FRAMES):
        started = perf_counter()
        runtime.solver.step_frame()
        viewport.update_outputs(scene, runtime)
        elapsed_ms = (perf_counter() - started) * 1000.0
        if frame >= WARMUP_FRAMES:
            samples.append(elapsed_ms)
    if native:
        assert runtime.native_presenter is not None, runtime.native_interop_error
        assert runtime.native_presenter.bridge.stats["bulk_host_position_bytes"] == 0
    else:
        assert runtime.native_presenter is None
        assert runtime.presentation_mode == "CPU"
    bpy.ops.fluidsim.release()
    return samples


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    cpu_samples = run_case(False)
    native_samples = run_case(True)
    native_samples += run_case(True)
    cpu_samples += run_case(False)
    cpu_median = median(cpu_samples)
    native_median = median(native_samples)
    ratio = native_median / cpu_median
    limit = 1.05

    result = {
        "passed": ratio <= limit,
        "workload": {
            "capacity": 50000,
            "warmup_frames_per_run": WARMUP_FRAMES,
            "measured_frames_per_run": MEASURED_FRAMES,
            "order": ["cpu", "native", "native", "cpu"],
        },
        "measurement_scope": "solver step plus particle presentation; View3D redraw excluded",
        "cpu_whole_frame_median_ms": cpu_median,
        "native_whole_frame_median_ms": native_median,
        "native_to_cpu_ratio": ratio,
        "limit": limit,
    }
    evidence = REPO_ROOT / "docs" / "evidence" / "native_interop" / "benchmark.json"
    report = json.loads(evidence.read_text(encoding="utf-8"))
    history = list(report.get("solver_regression_runs", ()))
    for entry in (report.get("solver_regression"), result):
        if entry is not None and (not history or history[-1] != entry):
            history.append(entry)
    report["solver_regression_runs"] = history[-4:]
    report["solver_regression"] = result
    evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_SOLVER_REGRESSION", json.dumps(result, sort_keys=True))
    assert result["passed"], (cpu_median, native_median, ratio)


if __name__ == "__main__":
    try:
        main()
    finally:
        bpy.ops.wm.quit_blender()
