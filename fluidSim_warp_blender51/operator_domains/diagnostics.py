"""Capability-owned Blender operators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty

from . import common as _runtime

_FLUIDSIM_OT_session_base = _runtime._FLUIDSIM_OT_session_base


class FLUIDSIM_OT_run_validation(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Run the GPU validation suite (may take ~30s)"""

    bl_idname = "fluidsim.run_validation"
    bl_label = "Run Validation Suite"

    def execute(self, context):
        bridge = _runtime._bridge()
        sess = bridge.session()
        try:
            from ..diagnostics import validation

            results = validation.run_all(device=_runtime._configured_device())
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Validation failed to run: {exc}")
            return {"CANCELLED"}
        sess.validation_results = [r.as_dict() for r in results]
        passed = sum(1 for r in results if r.passed)
        msg = f"Validation: {passed}/{len(results)} passed"
        if passed == len(results):
            self.report({"INFO"}, msg)
        else:
            failed = [r.name for r in results if not r.passed]
            self.report({"WARNING"}, f"{msg} - failed: {', '.join(failed)}")
        context.scene.fluidsim.status = msg
        return {"FINISHED"}


class FLUIDSIM_OT_run_benchmark(_FLUIDSIM_OT_session_base, bpy.types.Operator):
    """Run the deterministic benchmark scenario and store results"""

    bl_idname = "fluidsim.run_benchmark"
    bl_label = "Run Benchmark"

    save_baseline: BoolProperty(
        name="Save As Baseline", default=False,
        description="Enable to store this benchmark run as the comparison baseline; disable to compare without replacing the existing baseline")

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _runtime._bridge()
        sess = bridge.session()
        try:
            from ..diagnostics import benchmark as bench

            result = bench.run_benchmark(
                frames=fs.benchmark_frames,
                device=_runtime._configured_device(),
            )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Benchmark failed: {exc}")
            return {"CANCELLED"}
        data = result.as_dict()
        sess.benchmark_result = data
        baseline_path = bpy.path.abspath(fs.benchmark_baseline)
        if self.save_baseline:
            try:
                bench.save_benchmark(baseline_path, data)
            except (OSError, TypeError, ValueError) as exc:
                message = f"Benchmark baseline invalid: {baseline_path}: {exc}"
                fs.status = message
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
            self.report({"INFO"}, f"Benchmark baseline saved: {baseline_path}")
        else:
            try:
                baseline = bench.load_benchmark(baseline_path)
                comparison = bench.compare_benchmark(data, baseline)
                if comparison.get("error"):
                    message = (
                        f"Benchmark baseline invalid: {baseline_path}: "
                        f"{comparison['error']}"
                    )
                    comparison["error"] = message
                    sess.benchmark_compare = comparison
                    fs.status = message
                    self.report({"ERROR"}, message)
                    return {"CANCELLED"}
                sess.benchmark_compare = comparison
                self.report(
                    {"INFO"},
                    f"Benchmark {sess.benchmark_compare['result']}: "
                    f"{data['timeStepAverage'] * 1000:.1f} ms/frame avg, "
                    f"{data['voxelsPerSecondAverage'] / 1e6:.1f}M voxels/s")
            except FileNotFoundError:
                sess.benchmark_compare = None
                self.report(
                    {"INFO"},
                    f"Benchmark complete (no baseline): "
                    f"{data['timeStepAverage'] * 1000:.1f} ms/frame avg, "
                    f"{data['voxelsPerSecondAverage'] / 1e6:.1f}M voxels/s")
            except (OSError, TypeError, ValueError) as exc:
                message = f"Benchmark baseline invalid: {baseline_path}: {exc}"
                sess.benchmark_compare = {
                    "contract": {},
                    "exact": {},
                    "performance": {},
                    "result": "fail",
                    "error": message,
                }
                fs.status = message
                self.report({"ERROR"}, message)
                return {"CANCELLED"}
        fs.status = f"Benchmark {data['result']}"
        return {"FINISHED"}


class FLUIDSIM_OT_write_report(bpy.types.Operator):
    """Write profiling / validation / benchmark data to a JSON report"""

    bl_idname = "fluidsim.write_report"
    bl_label = "Write Diagnostics Report"

    def execute(self, context):
        scene = context.scene
        fs = scene.fluidsim
        bridge = _runtime._bridge()
        sess = bridge.session()
        from ..diagnostics import report as report_mod

        path = bpy.path.abspath(fs.report_path)
        try:
            from ..blender import effects as effects_mod
            from ..io import cache as cache_mod

            settings_signature = ""
            if sess.active and getattr(sess, "solver", None) is not None:
                settings_signature = cache_mod.settings_record(
                    sess.solver.settings)["signature"]
            rep = report_mod.build_report(
                profiler=sess.profiler,
                frame_stats=sess.frame_stats,
                benchmark=sess.benchmark_result,
                validation=sess.validation_results,
                extra={"benchmark_comparison": sess.benchmark_compare}
                if sess.benchmark_compare else None,
                package_name=__package__.rsplit(".", 1)[0],
                blender_version=".".join(str(value) for value in bpy.app.version[:3]),
                selected_device=_runtime._configured_device(),
                cache_path=bpy.path.abspath(fs.cache_path),
                scene_signature=str(getattr(sess, "scene_key", "")),
                settings_signature=settings_signature,
                effect_state=effects_mod.effect_report_state(scene),
            )
            report_mod.write_report(path, rep)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Report write failed: {exc}")
            bridge.set_error(scene, f"Report write failed: {exc}")
            fs.status = f"Report write failed: {exc}"
            return {"CANCELLED"}
        self.report({"INFO"}, f"Report written: {path}")
        return {"FINISHED"}


# deprecated compatibility adapter entry points forward to Custom Effects v2.
