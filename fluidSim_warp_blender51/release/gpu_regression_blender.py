"""Full NVIDIA GPU validation plus a deterministic ten-frame preview run."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import bpy

SUCCESS_MARKER = "FLUIDSIM_GPU_REGRESSION_OK"
EXPECTED_BLENDER_VERSION = (5, 1, 0)
EXPECTED_WARP_VERSION = "1.16.0"


def _installed_module() -> str:
    matches = [
        item.module
        for item in bpy.context.preferences.addons
        if item.module.endswith(".fluidsim_warp") or item.module == "fluidsim_warp"
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one enabled FluidSim module, found {matches}")
    return matches[0]


def _require_finished(label: str, result) -> None:
    if "FINISHED" not in result:
        raise AssertionError(f"{label} failed: {result}")


def _assert_packaged_runtime(module: str, probe) -> None:
    extension = importlib.import_module(module)
    extension_root = Path(extension.__file__).resolve().parent
    repository_value = os.environ.get("FLUIDSIM_CI_REPOSITORY", "")
    user_extensions_value = os.environ.get("BLENDER_USER_EXTENSIONS", "")
    if not repository_value or not user_extensions_value:
        raise AssertionError("GPU release environment is not isolated")
    repository_root = Path(repository_value).resolve()
    user_extensions_root = Path(user_extensions_value).resolve()
    if not probe.module_file:
        raise AssertionError("Warp runtime reported no module source path")
    warp_path = Path(probe.module_file).resolve()
    try:
        extension_root.relative_to(repository_root)
    except ValueError as exc:
        raise AssertionError(
            f"FluidSim loaded outside the isolated installed repository: {extension_root}"
        ) from exc
    try:
        warp_path.relative_to(user_extensions_root)
    except ValueError as exc:
        raise AssertionError(
            f"Warp loaded outside Blender's isolated extension wheel directory: {warp_path}"
        ) from exc


def main() -> None:
    if bpy.app.version[:3] != EXPECTED_BLENDER_VERSION:
        raise AssertionError(
            f"expected Blender 5.1.x, running {bpy.app.version_string}"
        )
    started = time.time()
    module = _installed_module()
    runtime = importlib.import_module(f"{module}.runtime.warp_runtime")
    validation = importlib.import_module(f"{module}.sim.validation")
    probe = runtime.probe_runtime(force=True)
    if probe.warp_version != EXPECTED_WARP_VERSION:
        raise AssertionError(
            f"expected Warp {EXPECTED_WARP_VERSION}, loaded {probe.warp_version}"
        )
    if not probe.available:
        raise AssertionError(probe.message)
    if probe.warp_version != EXPECTED_WARP_VERSION:
        raise AssertionError(
            f"expected Warp {EXPECTED_WARP_VERSION}, loaded {probe.warp_version}"
        )
    _assert_packaged_runtime(module, probe)
    device = probe.cuda_devices[0].alias
    bpy.context.preferences.addons[module].preferences.device = device

    expected_validation_count = len(validation.VALIDATION_TESTS)
    results = validation.run_all(device=device)
    if len(results) != expected_validation_count:
        raise AssertionError(
            f"expected {expected_validation_count} GPU validations, got {len(results)}"
        )
    failed = [result.as_dict() for result in results if not result.passed]
    if failed:
        raise AssertionError(f"GPU validations failed: {failed}")

    _require_finished("fluidsim.setup_scene", bpy.ops.fluidsim.setup_scene())
    _require_finished("fluidsim.initialize", bpy.ops.fluidsim.initialize())
    _require_finished("fluidsim.step(frames=10)", bpy.ops.fluidsim.step(frames=10))

    report = {
        "status": "passed",
        "module": module,
        "blender_version": bpy.app.version_string,
        "warp_version": probe.warp_version,
        "warp_module": probe.module_file,
        "device": device,
        "device_name": probe.cuda_devices[0].name,
        "cuda_driver_version": probe.cuda_driver_version,
        "cuda_toolkit_version": probe.cuda_toolkit_version,
        "validation_count": len(results),
        "validations": [result.as_dict() for result in results],
        "preview_frames": 10,
        "elapsed_seconds": time.time() - started,
    }
    report_path = Path(os.environ.get("FLUIDSIM_GPU_REPORT", "gpu-regression-report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _require_finished("fluidsim.release", bpy.ops.fluidsim.release())
    print(SUCCESS_MARKER)


if __name__ == "__main__":
    main()
