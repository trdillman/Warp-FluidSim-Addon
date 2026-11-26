"""Blender background acceptance. Pass gate-specific script arguments after --."""
import argparse
import hashlib
import importlib
import json
import platform
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ENGINE = "FLUIDSIM_FSPT"
ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 128, 96


def arguments(default_report):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=default_report)
    parser.add_argument("--addon-root", type=Path, default=ROOT)
    parser.add_argument("--addon-module", default="fluidSim_warp_blender51")
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    pending.replace(path)


def checkpoint(path, report, stage):
    report["stage"] = stage
    write_report(path, report)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_addon(args, report):
    root = args.addon_root.resolve()
    name = args.addon_module.split(".")[-1]
    package = root if (root / "__init__.py").is_file() else root / name
    assert (package / "__init__.py").is_file(), f"Missing add-on package at {package}"
    sys.path.insert(0, str(package.parent))
    addon = importlib.import_module(args.addon_module)
    loaded = Path(addon.__file__).resolve()
    assert loaded.parent == package, f"Loaded {loaded}, expected {package}"
    digest = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        digest.update(source.relative_to(package).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha(source)))
    report.update(addon=str(loaded), python_tree_sha256=digest.hexdigest(),
                  blender=bpy.app.version_string, blender_build=bpy.app.build_hash.decode(),
                  blender_executable=bpy.app.binary_path, python=sys.version, os=platform.platform())
    print("FSPT loaded", loaded, flush=True)
    addon.register()
    return addon


def runtime_info(addon):
    runtime = importlib.import_module(addon.__name__ + ".runtime.warp_runtime")
    wp = runtime._import_warp()
    gpu = wp.get_device("cuda:0")
    assert gpu.is_cuda, gpu
    state = runtime.probe_runtime(force=True)
    assert state.available, state.message
    return {"warp": wp.__version__, "warp_path": wp.__file__, "device": str(gpu), "gpu": gpu.name,
                "cuda_driver": state.cuda_driver_version, "cuda_toolkit": state.cuda_toolkit_version}


def info(addon, attr):
    for name, module in tuple(sys.modules.items()):
        if name.startswith(addon.__name__ + ".") and module and attr in vars(module):
            return dict(getattr(module, attr))
    return {}


def material(name, shader, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    node = tree.nodes.new(shader)
    node.inputs["Color"].default_value = (*color, 1)
    if "Roughness" in node.inputs:
        node.inputs["Roughness"].default_value = 0
    if "Strength" in node.inputs:
        node.inputs["Strength"].default_value = 5
    tree.links.new(node.outputs[0], out.inputs["Surface"])
    return mat, node


def fixture():
    scene = bpy.context.scene
    for obj in tuple(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.name = "FSPT P1 validation fixture"
    world = bpy.data.worlds.new("Validation constant sky")
    scene.world = world
    world.use_nodes = True
    sky = (0.12, 0.22, 0.34)
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (*sky, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1
    diffuse, node = material("Validation diffuse", "ShaderNodeBsdfDiffuse", (0.8, 0.06, 0.03))
    ground, _ = material("Validation ground", "ShaderNodeBsdfDiffuse", (0.45, 0.45, 0.45))
    glass, _ = material("Validation smooth glass", "ShaderNodeBsdfGlass", (0.97, 0.99, 1))
    emitter, _ = material("Validation emissive mesh", "ShaderNodeEmission", (1, 0.75, 0.4))
    bpy.ops.mesh.primitive_plane_add(size=14)
    bpy.context.object.data.materials.append(ground)
    for x, z, radius, mat in [(-0.95, 1, 1, diffuse), (1.1, 0.8, 0.8, glass),
                               (0, 3.1, 0.55, emitter)]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius,
                                              location=(x, 0, z))
        bpy.context.object.data.materials.append(mat)
    bpy.ops.object.camera_add(location=(0, -8.5, 3.0))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 1.4)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 42
    scene.camera = camera
    scene.render.engine = ENGINE
    scene.render.resolution_x, scene.render.resolution_y = WIDTH, HEIGHT
    scene.render.resolution_percentage = 100
    settings = scene.fspt
    settings.device = "cuda:0"
    settings.samples, settings.viewport_samples, settings.max_bounces = 32, 4, 6
    settings.seed = 31
    settings.resolution_percentage = 100
    bpy.context.view_layer.update()
    return scene, node, sky


def validate_info(value):
    assert value.get("status") != "ERROR", value
    assert value.get("device") == "cuda:0", value
    assert value.get("backend") and value.get("transport") and value.get("key"), value


def render(addon, scene, folder, label, report):
    filepath = folder / (label + ".exr")
    if filepath.exists():
        filepath.unlink()
    scene.render.filepath = str(filepath)
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    result = bpy.ops.render.render(write_still=True)
    assert result == {"FINISHED"} and filepath.is_file(), (result, filepath)
    evidence = info(addon, "LAST_RENDER_INFO")
    validate_info(evidence)
    assert evidence["samples"] == scene.fspt.samples, evidence
    img = bpy.data.images.load(str(filepath), check_existing=False)
    try:
        assert tuple(img.size) == (WIDTH, HEIGHT), tuple(img.size)
        values = np.empty(WIDTH * HEIGHT * 4, dtype=np.float32)
        img.pixels.foreach_get(values)
        pixels = values.reshape(HEIGHT, WIDTH, 4)
        assert np.isfinite(pixels).all(), label
        assert np.allclose(pixels[..., 3], 1, atol=1e-5), label
    finally:
        bpy.data.images.remove(img)
    png = filepath.with_suffix(".png")
    scene.render.image_settings.file_format = "PNG"
    bpy.data.images["Render Result"].save_render(str(png), scene=scene)
    report.setdefault("renders", {})[label] = dict(
        **evidence, exr=str(filepath), exr_sha256=sha(filepath), png=str(png), png_sha256=sha(png),
        mean_rgb=pixels[..., :3].mean(axis=(0, 1)).tolist())
    return pixels[..., :3], evidence["key"]


def main():
    args = arguments("artifacts/fspt-p1/render.json")
    path = Path(args.json_out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "verifier_sha256": sha(Path(__file__))}
    addon = None
    try:
        checkpoint(path, report, "register")
        addon = load_addon(args, report)
        checkpoint(path, report, "runtime_probe")
        report["runtime"] = runtime_info(addon)
        checkpoint(path, report, "fixture_setup")
        scene, node, sky = fixture()
        checkpoint(path, report, "fixture_render")
        base, base_key = render(addon, scene, path.parent, "fixture", report)
        assert np.ptp(base.mean(axis=2)) > 0.25 and base.std(axis=(0, 1)).max() > 0.05
        checkpoint(path, report, "camera_edit")
        scene.camera.location.x += 0.8
        shifted, shifted_key = render(addon, scene, path.parent, "camera_edit", report)
        camera_delta = float(np.abs(shifted - base).mean())
        assert shifted_key != base_key and camera_delta > 0.01, camera_delta
        report["camera_mean_abs_delta"] = camera_delta
        checkpoint(path, report, "material_edit")
        node.inputs["Color"].default_value = (0.02, 0.9, 0.07, 1)
        edited, edited_key = render(addon, scene, path.parent, "material_edit", report)
        material_delta = float(np.abs(edited - shifted).mean())
        assert edited_key != shifted_key and material_delta > 0.001, material_delta
        report["material_mean_abs_delta"] = material_delta
        checkpoint(path, report, "cpu_rejection")
        backend = importlib.import_module(addon.__name__ + ".fspt.backend")
        try:
            instance = backend.CudaBackend(device="cpu")
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            assert "cuda" in msg.lower() or "cpu" in msg.lower(), msg
            report["cpu_rejection"] = msg
        else:
            instance.close()
            raise AssertionError("CPU backend accepted")
        checkpoint(path, report, "unsupported_export")
        adapter = importlib.import_module(addon.__name__ + ".blender.fspt_scene")
        noise = node.id_data.nodes.new("ShaderNodeTexNoise")
        node.id_data.links.new(noise.outputs["Color"], node.inputs["Color"])
        bpy.context.view_layer.update()
        try:
            adapter.extract_scene(bpy.context.evaluated_depsgraph_get())
        except adapter.UnsupportedSceneError as exc:
            msg = str(exc)
            assert any(s in msg.lower() for s in ("link", "texture", "noise", "graph")), msg
            report["unsupported_material_rejection"] = msg
        else:
            raise AssertionError("Active noise texture was accepted")
        refused = path.parent / "unsupported.exr"
        if refused.exists():
            refused.unlink()
        scene.render.filepath = str(refused)
        scene.render.image_settings.file_format = "OPEN_EXR"
        checkpoint(path, report, "unsupported_render")
        try:
            result = bpy.ops.render.render(write_still=True)
            report["unsupported_operator"] = sorted(result)
        except RuntimeError as exc:
            report["unsupported_operator"] = str(exc)
        rejection = info(addon, "LAST_RENDER_INFO")
        assert rejection.get("status") == "ERROR", rejection
        assert any(s in str(rejection.get("error", "")).lower()
                   for s in ("link", "texture", "noise", "graph")), rejection
        report["unsupported_render"] = rejection
        assert not refused.exists(), f"Unsupported scene published {refused}"
        checkpoint(path, report, "world_only")
        for obj in tuple(scene.objects):
            if obj.type == "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)
        empty, _ = render(addon, scene, path.parent, "world_only", report)
        assert np.allclose(empty, sky, atol=1e-4), (empty.min(), empty.max(), sky)
        checkpoint(path, report, "unregister")
        scene.render.engine = "BLENDER_WORKBENCH"
        addon.unregister()
        addon = None
        assert not hasattr(bpy.types.Scene, "fspt"), "Scene.fspt survived unregistration"
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        write_report(path, report)
        raise
    finally:
        if addon is not None:
            try:
                addon.unregister()
            except Exception:  # noqa: BLE001 - retain teardown diagnostics
                report["cleanup_error"] = traceback.format_exc()
        write_report(path, report)
    print("FSPT-ACCEPTANCE-PASS", path, flush=True)


if __name__ == "__main__":
    main()
