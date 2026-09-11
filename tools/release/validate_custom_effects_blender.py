"""Validate Custom Effects demos, workflow, F12 renders, and trust in supplied Blender."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .tool_loading import load_addon_relative_tool
except ImportError:  # Blender executes this file directly via --python.
    from tool_loading import load_addon_relative_tool

import bpy
import numpy as np

DEMO_NAMES = (
    "05_custom_effects_vortex.blend",
    "06_custom_effects_stack.blend",
    "07_warp_code_authoring.blend",
)


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-warp-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _assert_warp_root(expected: Path):
    import warp as wp

    actual = Path(wp.__file__).resolve()
    expected = expected.resolve()
    if expected != actual and expected not in actual.parents:
        raise AssertionError(f"Warp imported from {actual}, expected beneath {expected}")
    if str(getattr(wp, "__version__", "")) != "1.16.0":
        raise AssertionError(f"unexpected Warp version {getattr(wp, '__version__', None)!r}")
    return wp, actual


def _render_stats(path: Path) -> dict[str, int]:
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    try:
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))[:, :3]
    finally:
        bpy.data.images.remove(image)
    luminance = pixels.mean(axis=1)
    stats = {
        "non_dark_pixels": int(np.count_nonzero(luminance > 0.08)),
        "orange_pixels": int(np.count_nonzero((pixels[:, 0] > pixels[:, 2] * 1.35) & (pixels[:, 0] > 0.2))),
        "cyan_pixels": int(np.count_nonzero((pixels[:, 2] > pixels[:, 0] * 1.2) & (pixels[:, 1] > pixels[:, 0] * 1.15))),
        "green_pixels": int(np.count_nonzero((pixels[:, 1] > pixels[:, 0] * 1.2) & (pixels[:, 1] > pixels[:, 2] * 1.1))),
    }
    if stats["non_dark_pixels"] < 500 or stats["orange_pixels"] < 20 or stats["cyan_pixels"] < 20:
        raise AssertionError(f"F12 evidence render lacks expected data colors: {path.name}: {stats}")
    return stats


def _numeric_close(left, right, *, atol=1.0e-6, rtol=1.0e-6):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))


def _validate_scene(path: Path, module_name: str, rerun_record: dict, examples: Path) -> dict[str, object]:
    result = bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    if "FINISHED" not in result:
        raise AssertionError(f"could not open {path}: {result}")
    scene = bpy.context.scene
    effects = importlib.import_module(f"{module_name}.blender.effects")
    package_module = importlib.import_module(f"{module_name}.sim.effects.package")
    metrics = json.loads(scene["fluidsim_custom_effect_metrics_json"])
    expected_metrics = rerun_record["metrics"]
    for key in (
        "tangential_speed_mean",
        "kinetic_energy_ratio",
        "stack_order_sensitivity",
        "x_velocity_rms",
    ):
        if key in metrics and not _numeric_close(metrics[key], expected_metrics[key]):
            raise AssertionError(f"{path.name}: metric {key} changed: {metrics[key]} vs {expected_metrics[key]}")
    for key in ("deterministic_rerun_equal", "canaries_intact", "finite"):
        if key in metrics and metrics[key] is not True:
            raise AssertionError(f"{path.name}: {key} is not true")
    signature, inventory = effects.effect_identity(scene)
    if signature != scene["fluidsim_custom_effect_applied_signature"]:
        raise AssertionError(f"{path.name}: applied signature mismatch")
    if len(signature) != 64 or not inventory:
        raise AssertionError(f"{path.name}: invalid effect identity")
    render_payload = json.loads(scene["fluidsim_custom_effect_render_json"])
    render_path = Path(render_payload["path"])
    if not render_path.is_file():
        fallback = examples / "custom_effects" / "renders" / render_path.name
        render_path = fallback
    render_stats = _render_stats(render_path)
    effect_ids = [item.effect_id for item in scene.fluidsim_effects.items]
    controller_names = [
        obj.name for obj in scene.objects if obj.get("fluidsim_effect_controller")
    ]
    if path.name == DEMO_NAMES[0]:
        if effect_ids != ["fluidsim.vortex_force"] or "Vortex Controller" not in controller_names:
            raise AssertionError(f"vortex demo inventory is incorrect: {effect_ids}, {controller_names}")
        if metrics["tangential_speed_mean"] <= 0.25:
            raise AssertionError("vortex demo lacks tangential motion")
    elif path.name == DEMO_NAMES[1]:
        if effect_ids != ["fluidsim.radial_pulse", "fluidsim.velocity_damping"]:
            raise AssertionError(f"stack demo order is incorrect: {effect_ids}")
        if metrics["stack_order_sensitivity"] <= 0.05:
            raise AssertionError("stack order demo is not sensitive to order")
        if metrics["kinetic_energy_ratio_error"] > 2.0e-5:
            raise AssertionError("damping result does not match analytic energy ratio")
    elif path.name == DEMO_NAMES[2]:
        custom = next(item for item in scene.fluidsim_effects.items if item.effect_id == "demo.lateral_wave")
        if custom.enabled or custom.status != "DISABLED":
            raise AssertionError("embedded Warp Code must be disabled")
        if custom.trust_state == "BUILT_IN_TRUSTED":
            raise AssertionError("embedded custom code spoofed built-in trust")
        text_name = scene.get("fluidsim_authoring_text", "")
        text = bpy.data.texts.get(text_name)
        if text is None or "wp.sin" not in text.as_string():
            raise AssertionError("Warp Code Text datablock is missing")
        definition = package_module.read_effect_package(
            examples / "custom_effects" / "lateral_wave-1.0.0.fseffect"
        )
        if definition.built_in or definition.effect_id != "demo.lateral_wave":
            raise AssertionError("authoring package identity/trust boundary is incorrect")
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "scene": scene.name,
        "signature": signature,
        "effect_ids": effect_ids,
        "trust": [item.trust_state for item in scene.fluidsim_effects.items],
        "statuses": [item.status for item in scene.fluidsim_effects.items],
        "controllers": controller_names,
        "metrics": metrics,
        "render": render_stats,
    }


def _package_library_validation(module_name: str, runtime_module, work_dir: Path) -> dict[str, object]:
    effect_library = importlib.import_module(f"{module_name}.blender.effect_library")
    package_module = importlib.import_module(f"{module_name}.sim.effects.package")
    library_root = work_dir / "effect-library"
    if library_root.exists():
        shutil.rmtree(library_root)
    library_root.mkdir(parents=True)
    original_user_library_path = effect_library.user_library_path
    effect_library.user_library_path = lambda: library_root
    try:
        definition = replace(
            runtime_module.authoring_definition(),
            effect_id="demo.library-wave",
            name="Library Wave",
            package_origin="validation://custom-effects-library",
        )
        incoming = work_dir / "library-wave-1.0.0.fseffect"
        package_module.write_effect_package(incoming, definition)
        import_result = bpy.ops.fluidsim.effect_import_package(
            filepath=str(incoming.resolve()), conflict="cancel"
        )
        if "FINISHED" not in import_result:
            raise AssertionError(f"Import Package operator failed: {import_result}")
        imported_item = bpy.context.scene.fluidsim_effects.items[
            bpy.context.scene.fluidsim_effects.active_index
        ]
        if imported_item.effect_id != definition.effect_id or imported_item.enabled:
            raise AssertionError("Import Package did not add the expected disabled scene effect")

        candidate = replace(definition, version="1.1.0")
        candidate_path = work_dir / "library-wave-1.1.0.fseffect"
        package_module.write_effect_package(candidate_path, candidate)
        comparison = effect_library.compare_package_update(candidate_path)
        if comparison.status != "NEWER":
            raise AssertionError(f"expected NEWER update comparison, got {comparison}")
        operator_result = bpy.ops.fluidsim.effect_compare_update(
            filepath=str(candidate_path.resolve())
        )
        if "FINISHED" not in operator_result:
            raise AssertionError(f"Compare Update operator failed: {operator_result}")

        installed = effect_library.installed_package_for_effect(definition.effect_id)
        if not callable(effect_library.rename_installed_package):
            raise TypeError("Rename Package service is unavailable")
        rename_result = bpy.ops.fluidsim.effect_rename_installed(
            new_effect_id="demo.library-wave-renamed",
            new_name="Library Wave Renamed",
        )
        if "FINISHED" not in rename_result:
            raise AssertionError(f"Rename Package operator failed: {rename_result}")
        renamed = effect_library.installed_package_for_effect("demo.library-wave-renamed")
        restored = package_module.read_effect_package(renamed.path)
        if installed.path.exists():
            raise AssertionError("old installed package survived successful rename")
        if restored.effect_id != "demo.library-wave-renamed":
            raise AssertionError("renamed package did not persist its new identity")
        if restored.kernels != definition.kernels:
            raise AssertionError("package rename changed executable kernel source")
        active = bpy.context.scene.fluidsim_effects.items[
            bpy.context.scene.fluidsim_effects.active_index
        ]
        if active.effect_id != definition.effect_id:
            raise AssertionError("Rename Package mutated the embedded scene instance")
        return {
            "package_import": True,
            "update_comparison": "NEWER",
            "package_rename": True,
            "renamed_effect_id": restored.effect_id,
            "renamed_package": str(renamed.path),
            "scene_identity_preserved": True,
        }
    finally:
        effect_library.user_library_path = original_user_library_path


def _workflow_validation(module_name: str, runtime_module, work_dir: Path, device: str) -> dict[str, object]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    effects = importlib.import_module(f"{module_name}.blender.effects")

    add_result = bpy.ops.fluidsim.effect_add(effect_id="fluidsim.vortex_force")
    if "FINISHED" not in add_result:
        raise AssertionError(f"Add Vortex failed: {add_result}")
    controller_result = bpy.ops.fluidsim.effect_create_controller(parameter_key="controller")
    if "FINISHED" not in controller_result:
        raise AssertionError(f"Create Controller failed: {controller_result}")
    test_result = bpy.ops.fluidsim.effect_test_draft(device=device)
    apply_result = bpy.ops.fluidsim.effect_apply_stack(device=device)
    if "FINISHED" not in test_result or "FINISHED" not in apply_result:
        raise AssertionError(f"Test/Apply failed: {test_result}, {apply_result}")
    stable_signature = scene.fluidsim_effects.applied_signature

    duplicate_result = bpy.ops.fluidsim.effect_duplicate()
    move_result = bpy.ops.fluidsim.effect_move(direction="UP")
    mute_result = bpy.ops.fluidsim.effect_toggle_mute()
    solo_result = bpy.ops.fluidsim.effect_toggle_solo()
    if not all("FINISHED" in value for value in (duplicate_result, move_result, mute_result, solo_result)):
        raise AssertionError("stack edit operators failed")

    # Remove the duplicate and restore a clean applied stack before testing the
    # failure transaction. This proves stack editing without changing the gate.
    bpy.ops.fluidsim.effect_remove()
    effects.revert_draft(scene)
    if scene.fluidsim_effects.applied_signature != stable_signature:
        raise AssertionError("Revert Draft did not restore the applied signature")

    definition = runtime_module.authoring_definition()
    bad_kernel = replace(
        definition.kernels[0],
        source="particle_velocity[particle_index] = 1.0",
    )
    bad_definition = replace(definition, kernels=(bad_kernel,))
    bad_instance = runtime_module.EffectInstanceV2(
        instance_id="workflow-bad-code",
        definition=bad_definition,
        enabled=True,
        order=len(scene.fluidsim_effects.items),
        display_name="Bad Type Fixture",
    )
    bad_pg = effects.add_instance_to_scene(scene, bad_instance)
    effects.trust_effect(scene, len(scene.fluidsim_effects.items) - 1)
    before_failed_apply = scene.fluidsim_effects.applied_signature
    try:
        failed = bpy.ops.fluidsim.effect_apply_stack(device=device)
        expected_rejection = "CANCELLED" in failed
    except RuntimeError as exc:
        # Blender raises reported operator errors to scripted callers even though
        # the operator correctly rejected the invalid candidate transaction.
        expected_rejection = True
        if "Error while parsing" not in str(exc):
            raise AssertionError(f"unexpected Apply failure: {exc}") from exc
    if not expected_rejection:
        raise AssertionError("invalid Warp code was accepted")
    if scene.fluidsim_effects.applied_signature != before_failed_apply:
        raise AssertionError("failed Apply replaced the last working stack")
    if not bad_pg.diagnostics or bad_pg.kernels[0].error_line != 1:
        raise AssertionError("compile error was not mapped to the user source line")
    mapped_error_line = int(bad_pg.kernels[0].error_line)
    bad_pg.enabled = False
    effects.untrust_effect(scene, len(scene.fluidsim_effects.items) - 1)
    effects.refresh_stack_state(scene)

    # A built-in flag on modified source must never bypass exact-digest trust.
    shipped = runtime_module._definition("fluidsim.velocity_damping")
    modified = replace(shipped, kernels=(replace(
        shipped.kernels[0], source=shipped.kernels[0].source + "\npass"
    ),))
    spoof = runtime_module.EffectInstanceV2(
        instance_id="workflow-spoof",
        definition=modified,
        enabled=False,
        order=len(scene.fluidsim_effects.items),
        display_name="Modified Built-in",
    )
    spoof_pg = effects.add_instance_to_scene(scene, spoof)
    if spoof_pg.trust_state == "BUILT_IN_TRUSTED":
        raise AssertionError("modified built-in content inherited built-in trust")

    save_path = work_dir / "custom_effects_workflow_roundtrip.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path.resolve()), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(save_path.resolve()))
    reopened = bpy.context.scene
    reopened_bad = next(item for item in reopened.fluidsim_effects.items if item.effect_id == "demo.lateral_wave")
    if reopened_bad.enabled or reopened_bad.trust_state == "BUILT_IN_TRUSTED":
        raise AssertionError("untrusted code changed state across save/reopen")

    return {
        "add_vortex": True,
        "controller_created": True,
        "test_and_apply": True,
        "stack_actions": True,
        "failed_apply_preserved_signature": True,
        "source_mapped_error_line": mapped_error_line,
        "modified_builtin_untrusted": True,
        "roundtrip_path": str(save_path),
    }


def main() -> None:
    args = _args()
    args.examples = args.examples.resolve()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    wp, warp_path = _assert_warp_root(args.expected_warp_root)
    addon = importlib.import_module(args.module)
    addon.register()
    try:
        runtime_module = load_addon_relative_tool(
            args.module, "custom_effect_demo_runtime"
        )
        rerun = runtime_module.run_all_demos(args.work_dir / "rerun", device=args.device)
        rerun_by_name = {item["name"]: item for item in rerun["demos"]}
        records = [
            _validate_scene(args.examples / DEMO_NAMES[0], args.module, rerun_by_name["vortex"], args.examples),
            _validate_scene(args.examples / DEMO_NAMES[1], args.module, rerun_by_name["stack"], args.examples),
            _validate_scene(args.examples / DEMO_NAMES[2], args.module, rerun_by_name["authoring"], args.examples),
        ]
        workflow = _workflow_validation(args.module, runtime_module, args.work_dir, args.device)
        package_library = _package_library_validation(args.module, runtime_module, args.work_dir)
    finally:
        addon.unregister()
    payload = {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "blender_python": sys.version.split()[0],
        "warp_version": str(getattr(wp, "__version__", "unknown")),
        "warp_module": str(warp_path),
        "device": args.device,
        "demo_count": len(records),
        "demos": records,
        "workflow": workflow,
        "package_library": package_library,
        "cuda_certified": False,
    }
    _atomic_json(args.report_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("FLUIDSIM_CUSTOM_EFFECTS_BLENDER_VALIDATED")


if __name__ == "__main__":
    main()
