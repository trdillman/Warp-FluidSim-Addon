"""Profile canonical Dam Break native versus CPU particle presentation in a real View3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter, process_time
import traceback

import addon_utils
import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLEND = (REPO_ROOT / "demos" / "dam_break.blend").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from blender_native_interop_stage_profile import EXPECTED_EXECUTABLE, _process_limits, _sha256
from tools.repo_addon_loader import enable_repo_addon


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=12)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.frames < 1:
        parser.error("--warmup must be non-negative and --frames must be positive")
    return args


def _view3d_override():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(region for region in area.regions if region.type == "WINDOW")
                return {
                    "window": window,
                    "screen": window.screen,
                    "area": area,
                    "region": region,
                    "space_data": area.spaces.active,
                }
    raise RuntimeError("No VIEW_3D area is available")


def _redraw():
    with bpy.context.temp_override(**_view3d_override()):
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)


def _run_case(native, warmup, frames):
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = bpy.context.scene
    settings = scene.fluidsim
    bpy.ops.fluidsim.release()
    settings.native_gpu_interop = native
    settings.liquid_view_mode = "PARTICLES"
    settings.surface_enabled = False
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    settings.live = False
    scene.frame_set(scene.frame_start)
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status

    runtime = session()
    samples = []
    for index in range(warmup + frames):
        process_started = process_time()
        frame_started = perf_counter()

        phase_started = perf_counter()
        runtime.solver.step_frame()
        solver_ms = (perf_counter() - phase_started) * 1000.0

        phase_started = perf_counter()
        viewport.update_outputs(scene, runtime)
        presentation_ms = (perf_counter() - phase_started) * 1000.0

        phase_started = perf_counter()
        _redraw()
        redraw_ms = (perf_counter() - phase_started) * 1000.0

        if index >= warmup:
            samples.append(
                {
                    "solver_submission_ms": solver_ms,
                    "presentation_ms": presentation_ms,
                    "redraw_ms": redraw_ms,
                    "whole_frame_ms": (perf_counter() - frame_started) * 1000.0,
                    "process_cpu_ms": (process_time() - process_started) * 1000.0,
                }
            )

    presenter = runtime.native_presenter
    result = {
        "mode": "native" if native else "cpu",
        "particle_count": runtime.solver.particles.host_count,
        "presentation_mode": runtime.presentation_mode,
        "presentation_bytes_last_frame": runtime.presentation_bytes,
        "native_stats": dict(presenter.bridge.stats) if presenter else None,
        "medians_ms": {
            key: median(sample[key] for sample in samples) for key in samples[0]
        },
        "samples": samples,
    }
    bpy.ops.fluidsim.release()
    return result


def _median_for(runs, mode, key):
    return median(run["medians_ms"][key] for run in runs if run["mode"] == mode)


def main():
    args = _args()
    executable = Path(bpy.app.binary_path).resolve()
    limits_before = _process_limits()
    executable_hash_before = _sha256(executable)
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    assert Path(bpy.data.filepath).resolve() == EXPECTED_BLEND
    assert addon_utils.check("fluidSim_warp_blender51") == (True, True)
    module = sys.modules["fluidSim_warp_blender51"]
    assert Path(module.__file__).resolve().is_relative_to(REPO_ROOT)

    runs = [_run_case(native, args.warmup, args.frames) for native in (True, False, False, True)]
    native_whole = _median_for(runs, "native", "whole_frame_ms")
    cpu_whole = _median_for(runs, "cpu", "whole_frame_ms")
    native_cpu = _median_for(runs, "native", "process_cpu_ms")
    cpu_cpu = _median_for(runs, "cpu", "process_cpu_ms")
    limits_after = _process_limits()
    executable_hash_after = _sha256(executable)
    native_runs = [run for run in runs if run["mode"] == "native"]
    cpu_runs = [run for run in runs if run["mode"] == "cpu"]
    provenance_passed = (
        executable == EXPECTED_EXECUTABLE
        and executable_hash_before == executable_hash_after
        and limits_before == limits_after
        == {"priority_class": "0x4000", "affinity_mask": "0xffff"}
    )
    functional_passed = (
        all(run["particle_count"] == 600_000 for run in runs)
        and all(run["presentation_mode"] == "Native GPU" for run in native_runs)
        and all(run["presentation_mode"] == "CPU" for run in cpu_runs)
        and all(run["presentation_bytes_last_frame"] == 0 for run in native_runs)
        and all(
            run["native_stats"]["imports"] == 6
            and run["native_stats"]["maps"] == 2
            and run["native_stats"]["bulk_host_position_bytes"] == 0
            for run in native_runs
        )
    )
    result = {
        "passed": provenance_passed and functional_passed,
        "blend": str(EXPECTED_BLEND),
        "addon_module": str(Path(module.__file__).resolve()),
        "environment": {
            "executable": str(executable),
            "executable_sha256_before": executable_hash_before,
            "executable_sha256_after": executable_hash_after,
            "addon_source": str(REPO_ROOT.resolve()),
            "process_limits_before": limits_before,
            "process_limits_after": limits_after,
        },
        "order": [run["mode"] for run in runs],
        "warmup_frames_per_run": args.warmup,
        "measured_frames_per_run": args.frames,
        "profile_scope": {
            "authoritative": "whole_frame_ms: solver + presentation + View3D redraw",
            "phase_caveat": "phase timings are host boundaries; GPU work may overlap phases",
        },
        "gates": {
            "provenance": provenance_passed,
            "particle_count": all(run["particle_count"] == 600_000 for run in runs),
            "presentation_modes": (
                all(run["presentation_mode"] == "Native GPU" for run in native_runs)
                and all(run["presentation_mode"] == "CPU" for run in cpu_runs)
            ),
            "zero_native_host_position_bytes": all(
                run["presentation_bytes_last_frame"] == 0
                and run["native_stats"]["bulk_host_position_bytes"] == 0
                for run in native_runs
            ),
            "fixed_native_imports_and_maps": all(
                run["native_stats"]["imports"] == 6 and run["native_stats"]["maps"] == 2
                for run in native_runs
            ),
        },
        "runs": runs,
        "summary": {
            "native_whole_frame_median_ms": native_whole,
            "cpu_whole_frame_median_ms": cpu_whole,
            "whole_frame_speedup": cpu_whole / native_whole,
            "whole_frame_reduction_fraction": (cpu_whole - native_whole) / cpu_whole,
            "native_process_cpu_median_ms": native_cpu,
            "cpu_process_cpu_median_ms": cpu_cpu,
            "process_cpu_reduction_fraction": (cpu_cpu - native_cpu) / cpu_cpu,
            "native_solver_submission_median_ms": _median_for(
                runs, "native", "solver_submission_ms"
            ),
            "cpu_solver_submission_median_ms": _median_for(
                runs, "cpu", "solver_submission_ms"
            ),
            "native_presentation_median_ms": _median_for(runs, "native", "presentation_ms"),
            "cpu_presentation_median_ms": _median_for(runs, "cpu", "presentation_ms"),
            "native_redraw_median_ms": _median_for(runs, "native", "redraw_ms"),
            "cpu_redraw_median_ms": _median_for(runs, "cpu", "redraw_ms"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DAM_BREAK_PROFILE=" + json.dumps(result, sort_keys=True), flush=True)

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


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
        bpy.ops.wm.quit_blender()
