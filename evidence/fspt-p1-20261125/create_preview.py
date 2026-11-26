"""Render the static P1 validation fixture in factory-startup, background Blender."""
import importlib.util
import traceback
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tests" / "acceptance" / "blender_fspt_acceptance.py"
spec = importlib.util.spec_from_file_location("fspt_preview_fixture", HELPER)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def checker_floor(scene):
    floor, = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.polygons) == 1]
    old_mesh = floor.data
    light = old_mesh.materials[0]
    dark, _ = fixture.material("Validation dark checker", "ShaderNodeBsdfDiffuse", (0.08, 0.1, 0.12))
    count = 20
    stride = count + 1
    vertices = [(14 * x / count - 7, 14 * y / count - 7, 0)
                for y in range(stride) for x in range(stride)]
    faces = []
    for y in range(count):
        for x in range(count):
            index = x + y * stride
            faces.append((index, index + 1, index + stride + 1, index + stride))
    mesh = bpy.data.meshes.new("Validation checker tiles")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(light)
    mesh.materials.append(dark)
    for face in mesh.polygons:
        y, x = divmod(face.index, count)
        face.material_index = (x + y) % 2
    mesh.update()
    floor.data = mesh
    bpy.data.meshes.remove(old_mesh)
    for sphere in scene.objects:
        if sphere.type == "MESH" and sphere != floor:
            modifier = sphere.modifiers.new("Validation sphere subdivision", "SUBSURF")
            modifier.levels = modifier.render_levels = 2


def main():
    args = fixture.arguments("artifacts/fspt-p1/preview/preview.json")
    path = Path(args.json_out).resolve()
    report = {"status": "RUNNING", "subject": "Static P1 validation fixture",
              "script_sha256": fixture.sha(Path(__file__)), "helper_sha256": fixture.sha(HELPER)}
    addon = None
    try:
        fixture.checkpoint(path, report, "register")
        assert bpy.app.background, "Use a factory-startup background Blender lease"
        addon = fixture.load_addon(args, report)
        report["runtime"] = fixture.runtime_info(addon)
        scene, _, _ = fixture.fixture()
        scene.name = "Static P1 validation fixture"
        checker_floor(scene)
        width, height = 512, 384
        scene.render.resolution_x, scene.render.resolution_y = width, height
        scene.render.resolution_percentage = 100
        scene.fspt.samples = 128
        scene.fspt.max_bounces = 6
        bpy.context.view_layer.update()
        report["settings"] = {"width": width, "height": height, "samples": scene.fspt.samples,
                              "max_bounces": scene.fspt.max_bounces, "seed": scene.fspt.seed,
                              "engine": scene.render.engine, "device": scene.fspt.device,
                              "display": scene.display_settings.display_device,
                              "view_transform": scene.view_settings.view_transform,
                              "look": scene.view_settings.look, "exposure": scene.view_settings.exposure,
                              "gamma": scene.view_settings.gamma}
        name = "static-p1-validation"
        exr, png, blend = [path.with_name(name + ext) for ext in (".exr", ".png", ".blend")]
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = "//" + exr.name
        fixture.checkpoint(path, report, "save_blend")
        result = bpy.ops.wm.save_as_mainfile(filepath=str(blend), relative_remap=False)
        assert result == {"FINISHED"} and blend.is_file(), result
        assert Path(bpy.path.abspath(scene.render.filepath)).resolve() == exr
        if exr.exists():
            exr.unlink()
        fixture.checkpoint(path, report, "render")
        result = bpy.ops.render.render(write_still=True)
        assert result == {"FINISHED"} and exr.is_file(), result
        completed = fixture.info(addon, "LAST_RENDER_INFO")
        fixture.validate_info(completed)
        assert completed["status"] == "COMPLETE" and completed["samples"] == 128, completed
        assert (completed["width"], completed["height"]) == (width, height), completed
        report["completed_render"] = completed
        fixture.checkpoint(path, report, "verify_pixels")
        image = bpy.data.images.load(str(exr), check_existing=False)
        try:
            assert tuple(image.size) == (width, height)
            rgba = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(rgba)
            pixels = rgba.reshape(height, width, 4)
            assert np.isfinite(pixels).all() and np.allclose(pixels[..., 3], 1.0, atol=1e-5)
            assert np.ptp(pixels[..., :3].mean(axis=2)) > 0.25
            report["mean_linear_rgb"] = pixels[..., :3].mean(axis=(0, 1)).tolist()
        finally:
            bpy.data.images.remove(image)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        bpy.data.images["Render Result"].save_render(str(png), scene=scene)
        report["artifacts"] = {item.suffix[1:]: {"path": str(item), "sha256": fixture.sha(item),
                               "bytes": item.stat().st_size} for item in (png, exr, blend)}
        fixture.checkpoint(path, report, "unregister")
        scene.render.engine = "BLENDER_WORKBENCH"
        addon.unregister()
        addon = None
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        fixture.write_report(path, report)
        raise
    finally:
        if addon is not None:
            try:
                addon.unregister()
            except Exception:  # noqa: BLE001 - retain teardown diagnostics
                report["cleanup_error"] = traceback.format_exc()
        fixture.write_report(path, report)
    print("FSPT-PREVIEW-PASS", path, flush=True)


if __name__ == "__main__":
    main()
