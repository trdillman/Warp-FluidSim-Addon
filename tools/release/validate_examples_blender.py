"""Open and validate every generated FluidSim example in Blender."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import bpy

EXPECTED = {
    "01_live_pour.blend": ("preview_water", {"SOURCE", "COLLIDER"}),
    "02_moving_collider_splash.blend": ("production_water", {"SOURCE", "COLLIDER"}),
    "03_viscous_honey.blend": ("honey", {"SOURCE", "COLLIDER"}),
    "04_whitewater_bake_resume.blend": ("splash_whitewater", {"SOURCE", "COLLIDER"}),
}

CUSTOM_EFFECT_EXPECTED = {
    "05_custom_effects_vortex.blend": ("fluidsim.vortex_force",),
    "06_custom_effects_stack.blend": (
        "fluidsim.radial_pulse",
        "fluidsim.velocity_damping",
    ),
    "07_warp_code_authoring.blend": (
        "fluidsim.particle_speed_limit",
        "demo.lateral_wave",
    ),
}


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args(argv)


def _owned_child(parent, kind: str):
    matches = [
        item
        for item in parent.children
        if item.get("fluidsim_owned") and item.get("fluidsim_kind") == kind
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {kind}, found {[item.name for item in matches]}")
    return matches[0]


def _open(path: Path, module_name: str):
    result = bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
    if "FINISHED" not in result:
        raise AssertionError(f"could not open {path.name}: {result}")
    scene = bpy.context.scene
    # A shipped example must not trip the 2.0 "saved before 1.1" warning.
    file_compat = importlib.import_module(f"{module_name}.blender.file_compat")
    if file_compat.is_pre_bridge_scene(scene):
        raise AssertionError(f"{path.name}: stores pre-1.1 settings; regenerate it")
    return scene


def _validate(path: Path, module_name: str) -> dict[str, object]:
    scene = _open(path, module_name)
    fs = scene.fluidsim
    preset, expected_roles = EXPECTED[path.name]
    if fs.active_preset_id != preset:
        raise AssertionError(
            f"{path.name}: expected preset {preset}, got {fs.active_preset_id!r}"
        )
    if fs.domain_object is None or fs.domain_object.name not in scene.objects:
        raise AssertionError(f"{path.name}: domain assignment is missing")
    roles = {
        obj.fluidsim.role
        for obj in scene.objects
        if hasattr(obj, "fluidsim") and obj.fluidsim.role != "NONE"
    }
    if not expected_roles.issubset(roles):
        raise AssertionError(f"{path.name}: expected roles {expected_roles}, got {roles}")
    root = _owned_child(scene.collection, "ROOT_COLLECTION")
    inputs = _owned_child(root, "INPUT_COLLECTION")
    outputs = _owned_child(root, "OUTPUT_COLLECTION")
    helpers = _owned_child(root, "HELPER_COLLECTION")
    publication = importlib.import_module(f"{module_name}.blender.output_publication")
    output_names = (
        publication.SURFACE_NAME,
        publication.PARTICLES_NAME,
        *publication.WHITEWATER_OUTPUT_NAMES,
    )
    output_objects = [publication.output_object(scene, name) for name in output_names]
    if any(obj is None for obj in output_objects):
        raise AssertionError(f"{path.name}: FluidSim output shells are missing")
    if any(obj.name not in outputs.objects for obj in output_objects):
        raise AssertionError(f"{path.name}: output shells are outside the owned collection")
    if any(len(getattr(obj.data, "materials", ())) != 1 for obj in output_objects):
        raise AssertionError(f"{path.name}: output shell material assignment is incomplete")
    geometry_nodes = importlib.import_module(f"{module_name}.blender.geometry_nodes")
    node_state = geometry_nodes.installed_state(scene)
    if set(node_state) != set(geometry_nodes.GROUP_BY_OUTPUT):
        raise AssertionError(f"{path.name}: render-node output inventory is incomplete")
    if any(value != geometry_nodes.ASSET_REVISION for value in node_state.values()):
        raise AssertionError(f"{path.name}: stale render-node revision: {node_state}")
    if path.name == "02_moving_collider_splash.blend":
        paddle = scene.objects.get("Animated Paddle")
        action = getattr(getattr(paddle, "animation_data", None), "action", None)
        if paddle is None or action is None:
            raise AssertionError("moving-collider example lost its animation")
    if path.name == "04_whitewater_bake_resume.blend":
        ds = fs.domain_object.fluidsim_domain

        def setting(name):
            # Domain physics lives on the domain object, display/cache on the scene.
            return getattr(ds if name in ds.bl_rna.properties else fs, name)

        if not setting("whitewater_enabled") or setting("cache_format") != "V2":
            raise AssertionError("whitewater resume example lost whitewater/cache-v2 settings")
        if (not setting("whitewater_attribute_type")
                or setting("whitewater_output_mode") != "BOTH"):
            raise AssertionError("whitewater example lost typed combined/separate output settings")
        if tuple(setting(name) for name in ("bake_start", "bake_end", "checkpoint_interval")) != (1, 48, 1):
            raise AssertionError("whitewater resume example lost its bake range")
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "scene": scene.name,
        "preset": fs.active_preset_id,
        "domain": fs.domain_object.name,
        "roles": sorted(roles),
        "collections": [root.name, inputs.name, outputs.name, helpers.name],
        "outputs": [obj.name for obj in output_objects],
        "whitewater_output_mode": fs.whitewater_output_mode,
        "render_nodes": node_state,
    }



def _validate_custom_effect(path: Path, module_name: str) -> dict[str, object]:
    scene = _open(path, module_name)
    effects = importlib.import_module(f"{module_name}.blender.effects")
    expected_ids = CUSTOM_EFFECT_EXPECTED[path.name]
    actual_ids = tuple(item.effect_id for item in scene.fluidsim_effects.items)
    if actual_ids != expected_ids:
        raise AssertionError(f"{path.name}: expected effects {expected_ids}, got {actual_ids}")
    signature, inventory = effects.effect_identity(scene)
    stored = str(scene.get("fluidsim_custom_effect_applied_signature", ""))
    if signature != stored or len(signature) != 64 or not inventory:
        raise AssertionError(f"{path.name}: invalid applied Custom Effects identity")
    if path.name == "07_warp_code_authoring.blend":
        custom = next(item for item in scene.fluidsim_effects.items if item.effect_id == "demo.lateral_wave")
        if custom.enabled or custom.status != "DISABLED":
            raise AssertionError("authoring example custom code must reopen disabled")
        if custom.trust_state == "BUILT_IN_TRUSTED":
            raise AssertionError("authoring example custom code spoofed built-in trust")
        text_name = str(scene.get("fluidsim_authoring_text", ""))
        if not text_name or bpy.data.texts.get(text_name) is None:
            raise AssertionError("authoring example lost its Warp Code Text datablock")
    return {
        "file": path.name,
        "size": path.stat().st_size,
        "scene": scene.name,
        "kind": "custom_effects",
        "effect_ids": list(actual_ids),
        "signature": signature,
        "statuses": [item.status for item in scene.fluidsim_effects.items],
        "trust": [item.trust_state for item in scene.fluidsim_effects.items],
    }

def main() -> None:
    args = _args()
    module = importlib.import_module(args.module)
    module.register()
    try:
        records = [_validate(args.examples / name, args.module) for name in EXPECTED]
        records.extend(
            _validate_custom_effect(args.examples / name, args.module)
            for name in CUSTOM_EFFECT_EXPECTED
        )
    finally:
        module.unregister()
    payload = {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "module": args.module,
        "example_count": len(records),
        "examples": records,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("FLUIDSIM_EXAMPLES_VALIDATED")


if __name__ == "__main__":
    main()
