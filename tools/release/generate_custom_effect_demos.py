"""Generate real Custom Effects demo scenes from supplied-Warp CPU results.

Run with the supplied Blender package. The script compiles, sandboxes, and
executes Custom Effects v2 kernels through the real runtime, validates analytic
metrics, authors three explanatory Blender scenes, renders Eevee F12 evidence,
and saves machine-readable records.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
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


def _new_scene(name: str):
    scene = bpy.data.scenes.new(name)
    bpy.context.window.scene = scene
    for candidate in tuple(bpy.data.scenes):
        if candidate is not scene:
            bpy.data.scenes.remove(candidate)
    for collection in (
        bpy.data.collections,
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
    ):
        for datablock in tuple(collection):
            if datablock.users == 0:
                collection.remove(datablock)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new(name + " World")
    scene.world.color = (0.008, 0.012, 0.02)
    scene.frame_start = 1
    scene.frame_end = 48
    return scene


def _material(name: str, color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.32
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = color
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.5
    material.diffuse_color = color
    return material


def _cube_cloud(name: str, points: np.ndarray, offset, size: float, material):
    vertices = []
    faces = []
    ox, oy, oz = offset
    corners = (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    )
    local_faces = (
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7),
    )
    for point in np.asarray(points, dtype=float):
        base = len(vertices)
        center = (point[0] + ox, point[1] + oy, point[2] * 0.2 + oz)
        vertices.extend(
            (center[0] + x * size, center[1] + y * size, center[2] + z * size)
            for x, y, z in corners
        )
        faces.extend(tuple(base + index for index in face) for face in local_faces)
    mesh = bpy.data.meshes.new(name + " Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    obj["fluidsim_demo_points"] = len(points)
    return obj


def _segments(name: str, starts: np.ndarray, ends: np.ndarray, offset, material):
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    vertices = []
    edges = []
    ox, oy, oz = offset
    for start, end in zip(starts, ends, strict=True):
        index = len(vertices)
        vertices.extend((
            (start[0] + ox, start[1] + oy, start[2] * 0.2 + oz),
            (end[0] + ox, end[1] + oy, end[2] * 0.2 + oz),
        ))
        edges.append((index, index + 1))
    mesh = bpy.data.meshes.new(name + " Mesh")
    mesh.from_pydata(vertices, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    obj.modifiers.new("FluidSim Demo Wire", "SKIN")
    obj["fluidsim_owned"] = True
    return obj


def _trajectory_segments(name: str, trajectory: np.ndarray, offset, material, stride: int = 4):
    trajectory = np.asarray(trajectory)
    starts = []
    ends = []
    for particle in range(0, trajectory.shape[1], max(1, stride)):
        for step in range(trajectory.shape[0] - 1):
            starts.append(trajectory[step, particle])
            ends.append(trajectory[step + 1, particle])
    return _segments(name, np.asarray(starts), np.asarray(ends), offset, material)


def _label(text: str, location, size: float = 0.42, color=None):
    curve = bpy.data.curves.new("Label", type="FONT")
    curve.body = text
    curve.align_x = "CENTER"
    curve.size = size
    obj = bpy.data.objects.new(text, curve)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    if color is not None:
        obj.data.materials.append(color)
    return obj


def _controller(name: str, location=(0.0, 0.0, 0.0), display="CIRCLE"):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = 1.25
    obj.location = location
    obj["fluidsim_owned"] = True
    obj["fluidsim_effect_controller"] = True
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _camera_and_light(scene, ortho_scale: float = 14.0):
    camera_data = bpy.data.cameras.new("Demo Camera")
    camera = bpy.data.objects.new("Demo Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 16.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    scene.camera = camera
    light_data = bpy.data.lights.new("Demo Area", type="AREA")
    light_data.energy = 1000.0
    light_data.shape = "DISK"
    light_data.size = 12.0
    light = bpy.data.objects.new("Demo Area", light_data)
    light.location = (0.0, 0.0, 9.0)
    scene.collection.objects.link(light)


def _render(scene, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path.resolve())
    result = bpy.ops.render.render(write_still=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"F12 render failed: {result}")
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    try:
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))
    finally:
        bpy.data.images.remove(image)
    rgb = pixels[:, :3]
    luminance = rgb.mean(axis=1)
    stats = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "non_dark_pixels": int(np.count_nonzero(luminance > 0.08)),
        "orange_pixels": int(np.count_nonzero((rgb[:, 0] > rgb[:, 2] * 1.35) & (rgb[:, 0] > 0.2))),
        "cyan_pixels": int(np.count_nonzero((rgb[:, 2] > rgb[:, 0] * 1.2) & (rgb[:, 1] > rgb[:, 0] * 1.15))),
        "green_pixels": int(np.count_nonzero((rgb[:, 1] > rgb[:, 0] * 1.2) & (rgb[:, 1] > rgb[:, 2] * 1.1))),
    }
    if stats["non_dark_pixels"] < 500:
        raise AssertionError(f"demo render is unexpectedly blank: {stats}")
    return stats


def _set_parameter(item, key: str, value, *, controller=None):
    parameter = next(entry for entry in item.parameters if entry.key == key)
    if parameter.kind == "FLOAT":
        parameter.float_value = float(value)
    elif parameter.kind == "INT":
        parameter.int_value = int(value)
    elif parameter.kind == "BOOL":
        parameter.bool_value = bool(value)
    elif parameter.kind == "VECTOR":
        parameter.vector_value = tuple(value)
    elif parameter.kind == "COLOR":
        parameter.color_value = tuple(value)
    elif parameter.kind == "ENUM":
        parameter.enum_value = str(value)
    elif parameter.kind == "OBJECT":
        parameter.object_value = controller
    else:
        raise ValueError(parameter.kind)


def _apply(scene, effects, device: str):
    candidate = effects.apply_stack(scene, device=device)
    if not candidate.success:
        raise AssertionError([issue.to_dict() for issue in candidate.issues])
    effects.refresh_stack_state(scene)
    if not scene.fluidsim_effects.applied_signature:
        raise AssertionError("applied effect signature is empty")


def _save(scene, output: Path, filename: str, metrics: dict, render: dict, simulation_record: dict):
    path = (output / filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    scene["fluidsim_custom_effect_demo"] = filename.removesuffix(".blend")
    scene["fluidsim_custom_effect_metrics_json"] = json.dumps(metrics, sort_keys=True, allow_nan=False)
    scene["fluidsim_custom_effect_applied_signature"] = scene.fluidsim_effects.applied_signature
    scene["fluidsim_custom_effect_inventory_json"] = scene.fluidsim_effects.applied_inventory_json
    scene["fluidsim_custom_effect_render_json"] = json.dumps(render, sort_keys=True, allow_nan=False)
    scene["fluidsim_custom_effect_simulation_sha256"] = simulation_record["npz_sha256"]
    scene["fluidsim_custom_effect_validation_device"] = "cpu"
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    if "FINISHED" not in result:
        raise RuntimeError(f"could not save {path}: {result}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "scene": scene.name,
        "signature": scene.fluidsim_effects.applied_signature,
        "effects": [item.effect_id for item in scene.fluidsim_effects.items],
        "trust": [item.trust_state for item in scene.fluidsim_effects.items],
        "statuses": [item.status for item in scene.fluidsim_effects.items],
        "render": render,
    }


def _load_result(record: dict[str, object]):
    return np.load(record["npz"], allow_pickle=False)


def _vortex_scene(args, effects, record):
    data = _load_result(record)
    metrics = record["metrics"]
    scene = _new_scene("Custom Effects Demo — Vortex Force")
    initial_mat = _material("Initial Particles", (0.03, 0.55, 1.0, 1.0))
    final_mat = _material("Vortex Result", (1.0, 0.22, 0.02, 1.0))
    line_mat = _material("Vortex Trajectories", (0.2, 1.0, 0.25, 1.0))
    text_mat = _material("Demo Text", (0.85, 0.9, 1.0, 1.0))
    _cube_cloud("Initial Particle State", data["initial_position"], (-3.2, 0.0, 0.0), 0.075, initial_mat)
    _cube_cloud("Final Vortex State", data["final_position"], (3.2, 0.0, 0.0), 0.085, final_mat)
    _trajectory_segments("Vortex Trajectories", data["trajectory"], (3.2, 0.0, -0.05), line_mat, stride=5)
    controller = _controller("Vortex Controller", (0.0, 0.0, 0.0), "CIRCLE")
    item = effects.add_builtin_effect(scene, "fluidsim.vortex_force", instance_id="demo-vortex")
    for key, value in {
        "controller": "Vortex Controller",
        "axis": (0.0, 0.0, 1.0),
        "strength": 6.0,
        "radius": 2.5,
        "suction": 0.12,
        "lift": 0.08,
    }.items():
        _set_parameter(item, key, value, controller=controller if key == "controller" else None)
    _apply(scene, effects, args.device)
    _label("BEFORE", (-3.2, -2.0, 0.2), color=text_mat)
    _label("VORTEX FORCE — AFTER", (3.2, -2.0, 0.2), color=text_mat)
    _label(
        f"Mean tangential speed: {metrics['tangential_speed_mean']:.3f} m/s  |  deterministic: yes",
        (0.0, 2.5, 0.2), size=0.32, color=text_mat,
    )
    _camera_and_light(scene, 13.5)
    render = _render(scene, args.output / "custom_effects" / "renders" / "05_custom_effects_vortex.png")
    return _save(scene, args.output, DEMO_NAMES[0], metrics, render, record)


def _stack_scene(args, effects, record):
    data = _load_result(record)
    metrics = record["metrics"]
    scene = _new_scene("Custom Effects Demo — Stack Order")
    initial_mat = _material("Initial Stack State", (0.03, 0.55, 1.0, 1.0))
    primary_mat = _material("Pulse Then Damping", (1.0, 0.22, 0.02, 1.0))
    compare_mat = _material("Damping Then Pulse", (0.85, 0.05, 0.9, 1.0))
    text_mat = _material("Stack Text", (0.85, 0.9, 1.0, 1.0))
    _cube_cloud("Initial Stack State", data["initial_position"], (-4.2, 0.0, 0.0), 0.08, initial_mat)
    _cube_cloud("Pulse Then Damping Result", data["final_position"], (0.0, 0.0, 0.0), 0.085, primary_mat)
    _cube_cloud("Damping Then Pulse Result", data["comparison_final_position"], (4.2, 0.0, 0.0), 0.085, compare_mat)
    controller = _controller("Pulse Controller", (0.0, 0.0, 0.0), "SPHERE")
    pulse = effects.add_builtin_effect(scene, "fluidsim.radial_pulse", instance_id="demo-pulse")
    damping = effects.add_builtin_effect(scene, "fluidsim.velocity_damping", instance_id="demo-damping")
    _set_parameter(pulse, "controller", "Pulse Controller", controller=controller)
    _set_parameter(pulse, "strength", 8.0)
    _set_parameter(pulse, "radius", 3.0)
    _set_parameter(damping, "damping", 3.0)
    _apply(scene, effects, args.device)
    _label("INITIAL", (-4.2, -2.0, 0.2), color=text_mat)
    _label("PULSE → DAMPING", (0.0, -2.0, 0.2), color=text_mat)
    _label("DAMPING → PULSE", (4.2, -2.0, 0.2), color=text_mat)
    _label(
        f"Order sensitivity L2: {metrics['stack_order_sensitivity']:.3f}  |  damping energy ratio: {metrics['kinetic_energy_ratio']:.4f}",
        (0.0, 2.5, 0.2), size=0.30, color=text_mat,
    )
    _camera_and_light(scene, 15.5)
    render = _render(scene, args.output / "custom_effects" / "renders" / "06_custom_effects_stack.png")
    return _save(scene, args.output, DEMO_NAMES[1], metrics, render, record)


def _authoring_scene(args, effects, runtime_module, package_module, record):
    data = _load_result(record)
    metrics = record["metrics"]
    scene = _new_scene("Custom Effects Demo — Warp Code Authoring")
    initial_mat = _material("Warp Code Initial", (0.03, 0.55, 1.0, 1.0))
    final_mat = _material("Warp Code Result", (1.0, 0.22, 0.02, 1.0))
    text_mat = _material("Warp Code Text", (0.85, 0.9, 1.0, 1.0))
    _cube_cloud("Warp Code Initial", data["initial_position"], (-3.0, 0.0, 0.0), 0.075, initial_mat)
    _cube_cloud("Warp Code Result", data["final_position"], (3.0, 0.0, 0.0), 0.08, final_mat)
    speed_limit = effects.add_builtin_effect(scene, "fluidsim.particle_speed_limit", instance_id="demo-safe-guard")
    _set_parameter(speed_limit, "maximum_speed", 25.0)
    definition = runtime_module.authoring_definition()
    custom = runtime_module.EffectInstanceV2(
        instance_id="demo-lateral-wave",
        definition=definition,
        enabled=False,
        order=1,
        display_name=definition.name,
    )
    custom_pg = effects.add_instance_to_scene(scene, custom)
    _apply(scene, effects, args.device)
    effects.refresh_stack_state(scene)
    if custom_pg.trust_state != "UNTRUSTED_IMPORTED" or custom_pg.enabled:
        raise AssertionError("embedded Warp Code did not remain disabled and untrusted")
    source_text = custom_pg.kernels[0].source_text
    scene["fluidsim_authoring_text"] = source_text.name
    scene["fluidsim_authoring_content_digest"] = metrics["content_digest"]
    package_path = args.output / "custom_effects" / "lateral_wave-1.0.0.fseffect"
    package_module.write_effect_package(package_path, definition)
    _label("WARP CODE — INPUT", (-3.0, -2.15, 0.2), color=text_mat)
    _label("VALIDATED RESULT", (3.0, -2.15, 0.2), color=text_mat)
    _label(
        "Embedded custom code is intentionally DISABLED + UNTRUSTED until reviewed",
        (0.0, 2.45, 0.2), size=0.31, color=text_mat,
    )
    _label(
        f"Temporary validation run x-velocity RMS: {metrics['x_velocity_rms']:.3f} m/s",
        (0.0, 1.85, 0.2), size=0.27, color=text_mat,
    )
    _camera_and_light(scene, 13.5)
    render = _render(scene, args.output / "custom_effects" / "renders" / "07_warp_code_authoring.png")
    return _save(scene, args.output, DEMO_NAMES[2], metrics, render, record)


def _write_readme(output: Path, warp_version: str) -> None:
    path = output / "custom_effects" / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# FluidSim Custom Effects demos

These scenes are generated from real Custom Effects v2 kernel executions using the
supplied Warp 1.16.0 package on the CPU validation backend.

- **05 — Vortex Force:** compares the initial particle fixture with the validated
  Vortex Force result and trajectory lines.
- **06 — Stack Order:** demonstrates that Radial Pulse → Velocity Damping differs
  from Velocity Damping → Radial Pulse, while independently checking the analytic
  kinetic-energy damping ratio.
- **07 — Warp Code:** includes a developer-authored Lateral Wave Text datablock and
  package. The code was compiled and executed for evidence, but remains disabled and
  untrusted in the distributed `.blend` until the user reviews its exact SHA-256.

CPU validation proves compilation, bounded execution, deterministic fixture results,
and Blender publication. It does not certify full FluidSolver CUDA numerics or GPU
performance; the NVIDIA CUDA release gate remains required on compatible hardware.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = _args()
    args.output = args.output.resolve()
    args.work_dir = args.work_dir.resolve()
    addon = importlib.import_module(args.module)
    addon.register()
    try:
        effects = importlib.import_module(f"{args.module}.blender.effects")
        runtime_module = load_addon_relative_tool(
            args.module, "custom_effect_demo_runtime"
        )
        package_module = importlib.import_module(f"{args.module}.sim.effects.package")
        simulation = runtime_module.run_all_demos(args.work_dir, device=args.device)
        records = {item["name"]: item for item in simulation["demos"]}
        generated = [
            _vortex_scene(args, effects, records["vortex"]),
            _stack_scene(args, effects, records["stack"]),
            _authoring_scene(args, effects, runtime_module, package_module, records["authoring"]),
        ]
        _write_readme(args.output, simulation["warp_version"])
    finally:
        addon.unregister()
    payload = {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "blender_python": sys.version.split()[0],
        "module": args.module,
        "device": args.device,
        "simulation": simulation,
        "demo_count": len(generated),
        "demos": generated,
    }
    _atomic_json(args.report_json.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    print("FLUIDSIM_CUSTOM_EFFECT_DEMOS_GENERATED")


if __name__ == "__main__":
    main()
