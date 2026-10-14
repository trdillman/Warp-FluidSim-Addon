"""Generate the current FluidSim example .blend files from Blender primitives.

Usage:
    blender --background --factory-startup --python tools/release/generate_examples.py -- \\
      --module fluidsim_dev --output resources/examples

The selected module must resolve to this source tree or an installed FluidSim
extension. No solver step is performed, so example generation does not require
CUDA; registration and authored data still run inside the supplied Blender.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import bpy

EXAMPLE_NAMES = (
    "01_live_pour.blend",
    "02_moving_collider_splash.blend",
    "03_viscous_honey.blend",
    "04_whitewater_bake_resume.blend",
)

CUSTOM_EFFECT_EXAMPLE_NAMES = (
    "05_custom_effects_vortex.blend",
    "06_custom_effects_stack.blend",
    "07_warp_code_authoring.blend",
)

ALL_EXAMPLE_NAMES = EXAMPLE_NAMES + CUSTOM_EFFECT_EXAMPLE_NAMES


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


def _new_scene(name: str):
    scene = bpy.data.scenes.new(name)
    bpy.context.window.scene = scene
    for candidate in tuple(bpy.data.scenes):
        if candidate is not scene:
            bpy.data.scenes.remove(candidate)
    # Remove data orphaned by the previous generated scene so every file has
    # clean, stable names and contains only its own primitive assets.
    for collection in (
        bpy.data.collections,
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.pointclouds,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in tuple(collection):
            if datablock.users == 0:
                collection.remove(datablock)
    scene.frame_start = 1
    scene.frame_end = 48
    scene.render.engine = "BLENDER_EEVEE"
    return scene


def _select_only(obj) -> None:
    for candidate in bpy.context.selected_objects:
        candidate.select_set(False)
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _cube(name: str, location, scale):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.context.view_layer.update()
    return obj


def _sphere(name: str, location, radius: float):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def _guided_setup(mode: str, preset: str) -> None:
    result = bpy.ops.fluidsim.setup_scene(mode=mode, preset_id=f"BUILTIN:{preset}")
    if "FINISHED" not in result:
        raise RuntimeError(f"guided setup failed for {mode}/{preset}: {result}")


def _add_collider(obj) -> None:
    _select_only(obj)
    result = bpy.ops.fluidsim.setup_scene(mode="ADD_COLLIDER")
    if "FINISHED" not in result:
        raise RuntimeError(f"collider setup failed for {obj.name}: {result}")


def _save(scene, output: Path, filename: str, description: str) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    path = (output / filename).resolve()
    scene["fluidsim_example"] = filename.removesuffix(".blend")
    scene["fluidsim_example_description"] = description
    scene["fluidsim_example_generated"] = True
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"could not save {path}: {result}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "scene": scene.name,
        "preset": scene.fluidsim.active_preset_id,
        "domain": scene.fluidsim.domain_object.name,
        "objects": sorted(obj.name for obj in scene.objects),
    }


def _live_pour(output: Path) -> dict[str, object]:
    scene = _new_scene("FluidSim Example — Live Pour")
    source = _cube("Pour Inflow", (0.0, 0.0, 2.0), (0.45, 0.45, 0.35))
    _select_only(source)
    _guided_setup("QUICK_LIQUID", "preview_water")
    source.fluidsim.emission_mode = "CONTINUOUS_INFLOW"
    source.fluidsim.mesh_evaluation_mode = "EVERY_SIM_FRAME"
    source.fluidsim.initial_velocity = (1.5, 0.0, 0.0)
    floor = _cube("Pour Basin", (0.0, 0.0, -0.65), (2.5, 2.0, 0.15))
    _add_collider(floor)
    scene.frame_end = 72
    return _save(
        scene,
        output,
        EXAMPLE_NAMES[0],
        "Continuous inflow using Preview Water with a simple basin collider.",
    )


def _moving_collider(output: Path) -> dict[str, object]:
    scene = _new_scene("FluidSim Example — Moving Collider Splash")
    source = _cube("Splash Fill", (0.0, 0.0, 0.3), (1.4, 1.2, 0.45))
    _select_only(source)
    _guided_setup("QUICK_LIQUID", "production_water")
    paddle = _cube("Animated Paddle", (-1.6, 0.0, 0.5), (0.18, 1.0, 0.8))
    _add_collider(paddle)
    paddle.location.x = -1.6
    paddle.keyframe_insert(data_path="location", frame=1)
    paddle.location.x = 1.6
    paddle.keyframe_insert(data_path="location", frame=28)
    paddle.location.x = -1.6
    paddle.keyframe_insert(data_path="location", frame=56)
    paddle.fluidsim.transfer_velocity = True
    scene.frame_end = 56
    return _save(
        scene,
        output,
        EXAMPLE_NAMES[1],
        "Initial-fill liquid with an animated velocity-transferring paddle collider.",
    )


def _viscous_honey(output: Path) -> dict[str, object]:
    scene = _new_scene("FluidSim Example — Viscous Honey")
    source = _sphere("Honey Source", (0.0, 0.0, 1.2), 0.8)
    _select_only(source)
    _guided_setup("QUICK_VISCOUS", "honey")
    floor = _cube("Honey Plate", (0.0, 0.0, -0.45), (2.2, 2.2, 0.12))
    _add_collider(floor)
    scene.frame_end = 64
    return _save(
        scene,
        output,
        EXAMPLE_NAMES[2],
        "Honey preset with implicit viscosity and a broad plate collider.",
    )


def _whitewater_resume(output: Path) -> dict[str, object]:
    scene = _new_scene("FluidSim Example — Whitewater Bake Resume")
    source = _cube("Whitewater Inflow", (-0.8, 0.0, 1.4), (0.45, 0.55, 0.45))
    _select_only(source)
    _guided_setup("QUICK_LIQUID", "splash_whitewater")
    source.fluidsim.emission_mode = "CONTINUOUS_INFLOW"
    source.fluidsim.mesh_evaluation_mode = "EVERY_SIM_FRAME"
    source.fluidsim.initial_velocity = (2.5, 0.0, 0.0)
    obstacle = _sphere("Splash Obstacle", (0.5, 0.0, 0.2), 0.65)
    _add_collider(obstacle)
    fs = scene.fluidsim
    fs.bake_start = 1
    fs.bake_end = 48
    fs.checkpoint_interval = 1
    fs.cache_format = "V2"
    fs.whitewater_output_mode = "BOTH"
    fs.domain_object.fluidsim_domain.whitewater_attribute_type = True
    scene.frame_end = 48
    return _save(
        scene,
        output,
        EXAMPLE_NAMES[3],
        "Typed foam, bubble, and spray scene authored for resumable-cache bake/resume.",
    )


def main() -> None:
    args = _args()
    module = importlib.import_module(args.module)
    module.register()
    try:
        records = [
            _live_pour(args.output),
            _moving_collider(args.output),
            _viscous_honey(args.output),
            _whitewater_resume(args.output),
        ]
    finally:
        # The saved examples retain RNA values; unregister only the generator process.
        module.unregister()
    payload = {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "module": args.module,
        "example_count": len(records),
        "examples": records,
        "custom_effect_examples": [
            {
                "file": name,
                "path": str((args.output / name).resolve()),
                "generated_by": "tools/release/generate_custom_effect_demos.py",
                "exists": (args.output / name).is_file(),
            }
            for name in CUSTOM_EFFECT_EXAMPLE_NAMES
        ],
    }
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("FLUIDSIM_EXAMPLES_OK")


if __name__ == "__main__":
    main()
