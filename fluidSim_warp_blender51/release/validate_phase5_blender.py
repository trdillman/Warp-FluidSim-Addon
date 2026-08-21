"""Blender integration proof for Phase-5 surface-tension controls.

Runs in the supplied Blender package.  This validator deliberately does not
claim CUDA solver certification: it proves registration, lifecycle policy,
strict presets, settings projection, capillary preview math, and .blend
save/reopen persistence while the separate Warp validator executes the kernels.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
import warp as wp

EXPECTED_BLENDER_VERSION = (5, 1, 0)
EXPECTED_WARP_VERSION = "1.16.0"
EXPECTED_FLUIDSIM_VERSION = "1.0.255"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _assert_close(actual, expected, label, tolerance=1.0e-8):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def validate(module_name: str, work_dir: Path, expected_warp_root: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    if tuple(bpy.app.version[:3]) != EXPECTED_BLENDER_VERSION:
        raise AssertionError(
            f"expected Blender {EXPECTED_BLENDER_VERSION}, got {bpy.app.version[:3]}")
    if str(wp.__version__) != EXPECTED_WARP_VERSION:
        raise AssertionError(f"expected Warp {EXPECTED_WARP_VERSION}, got {wp.__version__}")
    warp_module = Path(wp.__file__).resolve()
    if expected_warp_root.resolve() not in warp_module.parents:
        raise AssertionError(
            f"Warp loaded from {warp_module}, expected under {expected_warp_root}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon = importlib.import_module(module_name)
    addon.register()
    try:
        version = importlib.import_module(f"{module_name}.version")
        if version.VERSION_STRING != EXPECTED_FLUIDSIM_VERSION:
            raise AssertionError(f"unexpected FluidSim version {version.VERSION_STRING}")
        library = importlib.import_module(f"{module_name}.presets.library")
        solver_settings = importlib.import_module(
            f"{module_name}.blender.solver_settings")
        pending_settings = importlib.import_module(
            f"{module_name}.blender.pending_settings")
        tension = importlib.import_module(f"{module_name}.sim.surface_tension")

        pending_settings.validate_transition_registry()
        scene = bpy.context.scene
        fs = scene.fluidsim
        physical = library.load_preset("BUILTIN:water_surface_tension")
        result = library.apply_preset(scene, physical)
        if result.preset_id != "water_surface_tension" or not result.requires_reset:
            raise AssertionError(f"physical preset result is invalid: {result}")
        if not fs.surface_tension_enabled:
            raise AssertionError("physical preset did not enable surface tension")
        _assert_close(fs.surface_tension_coefficient, 0.072, "physical coefficient")
        _assert_close(fs.surface_tension_density, 1000.0, "physical density")
        _assert_close(fs.surface_tension_band_voxels, 1.5, "physical band")
        _assert_close(fs.surface_tension_safety, 0.5, "physical safety")
        if fs.surface_tension_smoothing_iterations != 1:
            raise AssertionError("physical preset smoothing is not one pass")

        settings = solver_settings.build_settings(scene)
        for name in (
            "surface_tension_enabled",
            "surface_tension_coefficient",
            "surface_tension_density",
            "surface_tension_band_voxels",
            "surface_tension_safety",
            "surface_tension_smoothing_iterations",
        ):
            if getattr(settings, name) != getattr(fs, name):
                raise AssertionError(f"scene-to-solver mapping drifted for {name}")

        frame_dt = float(fs.time_scale) / (
            float(scene.render.fps) / float(scene.render.fps_base))
        stable_dt = tension.capillary_dt(
            fs.surface_tension_density,
            fs.voxel_size,
            fs.surface_tension_coefficient,
            fs.surface_tension_safety,
        )
        required = tension.required_capillary_substeps(frame_dt, stable_dt)
        if not math.isfinite(stable_dt) or stable_dt <= 0.0 or required < 1:
            raise AssertionError("capillary preview returned invalid values")
        panel = getattr(bpy.types, "FLUIDSIM_PT_sim_surface_tension", None)
        if panel is None:
            raise AssertionError("native surface-tension panel is not registered")
        if panel.bl_space_type != "PROPERTIES" or panel.bl_context != "scene":
            raise AssertionError(
                "surface-tension panel is not registered in Scene Properties"
            )
        if panel.bl_parent_id != "FLUIDSIM_PT_scene_simulation":
            raise AssertionError(
                "surface-tension panel is not parented under Simulation Physics"
            )

        blend_path = work_dir / "phase5_surface_tension_settings.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
        reopened = bpy.context.scene.fluidsim
        if reopened.active_preset_id != "water_surface_tension":
            raise AssertionError("active Phase-5 preset did not survive save/reopen")
        if not reopened.surface_tension_enabled:
            raise AssertionError("surface-tension enable state did not survive save/reopen")
        _assert_close(
            reopened.surface_tension_coefficient, 0.072,
            "reopened physical coefficient")

        stylized = library.load_preset("BUILTIN:stylized_high_tension")
        stylized_result = library.apply_preset(bpy.context.scene, stylized)
        reopened = bpy.context.scene.fluidsim
        if stylized_result.preset_id != "stylized_high_tension":
            raise AssertionError("stylized preset did not apply")
        _assert_close(reopened.surface_tension_coefficient, 0.25, "stylized coefficient")
        _assert_close(reopened.surface_tension_safety, 0.4, "stylized safety")
        if reopened.max_substeps != 16:
            raise AssertionError("stylized preset did not increase max substeps")

        library.apply_preset(
            bpy.context.scene, library.load_preset("BUILTIN:preview_water"))
        if bpy.context.scene.fluidsim.surface_tension_enabled:
            raise AssertionError(
                "pre-Phase-5 preset retained hidden surface-tension state")

        inventory = {record.preset.id for record in library.built_in_presets()}
        required_presets = {"water_surface_tension", "stylized_high_tension"}
        if not required_presets.issubset(inventory):
            raise AssertionError(f"Phase-5 presets missing from inventory: {inventory}")

        return {
            "status": "passed",
            "blender_version": bpy.app.version_string,
            "blender_python": sys.version.split()[0],
            "module": module_name,
            "fluidsim_version": version.VERSION_STRING,
            "warp_version": str(wp.__version__),
            "warp_module": str(warp_module),
            "device_executed": "none (Blender settings integration only)",
            "cuda_devices": [
                str(device.alias) for device in wp.get_devices() if device.is_cuda],
            "checks": {
                "registration": "passed",
                "transition_registry": "passed",
                "strict_physical_preset": "passed",
                "strict_stylized_preset": "passed",
                "legacy_preset_disables_surface_tension": "passed",
                "scene_to_solver_projection": "passed",
                "capillary_ui_preview": "passed",
                "surface_tension_panel": "passed",
                "blend_save_reopen": "passed",
                "preset_inventory": "passed",
            },
            "physical_preview": {
                "frame_dt": frame_dt,
                "capillary_dt": stable_dt,
                "required_substeps": required,
            },
            "artifacts": {"blend": str(blend_path)},
        }
    finally:
        try:
            addon.unregister()
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    report = validate(
        options.module, options.work_dir, options.expected_warp_root)
    _write_json(options.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(arguments))
