"""Independent Blender/CUDA cases for the P1 extractor, final guards and ray bounds."""
import importlib
import importlib.util
import math
import traceback
from pathlib import Path
from unittest import SkipTest

import bpy
import numpy as np
from mathutils import Vector

sibling = Path(__file__).with_name("blender_fspt_acceptance.py")
spec = importlib.util.spec_from_file_location("fspt_fixture", sibling)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
SKY = (0.125, 0.25, 0.5)
COLOR = (0.25, 0.5, 0.75)


def reset_scene():
    scene = bpy.context.scene
    for layer in tuple(scene.view_layers)[1:]:
        scene.view_layers.remove(layer)
    scene.view_layers[0].material_override = None
    scene.view_layers[0].use = True
    scene.render.film_transparent = False
    scene.render.use_multiview = False
    scene.render.use_border = False
    for obj in tuple(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    world = bpy.data.worlds.new("Edge constant sky")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*SKY, 1)
    bg.inputs["Strength"].default_value = 1
    bpy.ops.object.camera_add(location=(0, 0, 0), rotation=(0, 0, 0))
    scene.camera = bpy.context.object
    scene.camera.data.clip_start = 0.1
    scene.camera.data.clip_end = 10
    scene.render.engine = fixture.ENGINE
    scene.render.resolution_x = scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100
    scene.fspt.samples = 2
    return scene


def plane(shader="ShaderNodeBsdfDiffuse"):
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, -5))
    obj = bpy.context.object
    mat, _ = fixture.material("Edge " + shader, shader, COLOR)
    obj.data.materials.append(mat)
    return obj, mat


def depsgraph():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def expect_rejection(adapter, terms):
    try:
        adapter.extract_scene(depsgraph())
    except adapter.UnsupportedSceneError as exc:
        assert any(term in str(exc).lower() for term in terms), str(exc)
        return {"rejection": str(exc)}
    raise AssertionError(f"Extractor accepted unsupported scene; expected one of {terms}")


def slot_override(adapter, supported):
    obj, original = plane()
    slot = obj.material_slots[0]
    if supported:
        override, _ = fixture.material("Object emission override", "ShaderNodeEmission", COLOR)
    else:
        override = bpy.data.materials.new("Object Principled override")
        override.use_nodes = True
    slot.link = "OBJECT"
    slot.material = override
    assert obj.data.materials[0] == original and slot.material == override
    if not supported:
        return expect_rejection(adapter, ("principled", "unsupported shader"))
    snapshot = adapter.extract_scene(depsgraph())
    assert len(snapshot.triangles) == 2
    assigned = [snapshot.materials[i] for i in snapshot.material_indices]
    assert all(m.kind == "emission" and np.allclose(m.emission, np.asarray(COLOR) * 5)
               for m in assigned), assigned
    return {"assigned_emission": list(assigned[0].emission)}


def layer_override(adapter):
    _, mat = plane()
    bpy.context.view_layer.material_override = mat
    assert bpy.context.view_layer.material_override == mat
    return expect_rejection(adapter, ("override",))


def muted_link(adapter):
    _, mat = plane()
    link = mat.node_tree.links[0]
    prop = link.bl_rna.properties.get("is_muted")
    if prop is None or prop.is_readonly:
        raise SkipTest("NodeLink.is_muted is unavailable or read-only on this Blender build")
    link.is_muted = True
    assert link.is_muted
    return expect_rejection(adapter, ("mut", "surface"))


def points_output(adapter):
    obj, _ = plane()
    mod = obj.modifiers.new("Mesh to points regression", "NODES")
    tree = bpy.data.node_groups.new("Edge point cloud", "GeometryNodeTree")
    for direction in ("INPUT", "OUTPUT"):
        tree.interface.new_socket(name="Geometry", in_out=direction, socket_type="NodeSocketGeometry")
    inp = tree.nodes.new("NodeGroupInput")
    convert = tree.nodes.new("GeometryNodeMeshToPoints")
    out = tree.nodes.new("NodeGroupOutput")
    tree.links.new(inp.outputs["Geometry"], convert.inputs["Mesh"])
    join = tree.nodes.new("GeometryNodeJoinGeometry")
    tree.links.new(inp.outputs["Geometry"], join.inputs["Geometry"])
    tree.links.new(convert.outputs["Points"], join.inputs["Geometry"])
    tree.links.new(join.outputs["Geometry"], out.inputs["Geometry"])
    mod.node_group = tree
    evaluated = obj.evaluated_get(depsgraph())
    geometry = evaluated.evaluated_geometry()
    assert geometry.mesh is not None and len(geometry.mesh.polygons) == 1, str(geometry)
    assert geometry.pointcloud is not None and len(geometry.pointcloud.points) == 4, str(geometry)
    evidence = expect_rejection(adapter, ("point", "non-mesh", "nonmesh"))
    evidence["evaluated_points"] = 4
    return evidence


def render_guard(addon, scene, dest, case):
    plane()
    if case == "film_transparent":
        scene.render.film_transparent = True
        terms = ("transparent", "transparen")
    elif case == "multiview":
        scene.render.use_multiview = True
        terms = ("multiview", "multi-view", "stereo", "multiple view")
    else:
        scene.view_layers.new("Second enabled validation layer").use = True
        assert sum(layer.use for layer in scene.view_layers) == 2
        terms = ("layer",)
    dest.mkdir(parents=True, exist_ok=True)
    assert not list(dest.iterdir()), f"Refusal output directory must be empty: {dest}"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.filepath = str(dest / "refused.exr")
    try:
        op = sorted(bpy.ops.render.render(write_still=True))
    except RuntimeError as exc:
        op = str(exc)
    info = fixture.info(addon, "LAST_RENDER_INFO")
    emitted = sorted(p.name for p in dest.iterdir())
    assert info.get("status") == "ERROR" and not emitted, (op, info, emitted)
    assert any(t in str(info.get("error", "")).lower() for t in terms), info
    return {"operator": op, "engine": info, "published": emitted}


def numerical(adapter, backend_cls, scene, case):
    backend = backend_cls(device="cuda:0")
    readings = {}
    try:
        clip_test = case in {"PERSP", "ORTHO"}
        plane("ShaderNodeEmission" if clip_test else "ShaderNodeBsdfDiffuse")
        if clip_test:
            scene.camera.data.type = case
            scene.camera.data.ortho_scale = 4
            pairs = [(2, 1, SKY), (10, 1, np.asarray(COLOR) * 5)]
        else:
            pairs = [(10, 1, np.zeros(3)), (10, 2, np.asarray(SKY) * COLOR)]
        for clip_end, bounces, expected in pairs:
            scene.camera.data.clip_end = clip_end
            dg = depsgraph()
            snapshot = adapter.extract_scene(dg)
            camera = adapter.camera_snapshot(dg, 8, 8)
            frame = backend.render(snapshot, camera, 8, 8, samples=2, max_bounces=bounces, seed=23)
            assert frame.device == "cuda:0" and frame.samples == 2, (frame.device, frame.samples)
            values = frame.pixels[..., :3]
            error = float(np.max(np.abs(values - expected)))
            label = f"clip_{clip_end}_bounces_{bounces}"
            readings[label] = {"max_abs_error": error, "expected": np.asarray(expected).tolist(),
                                   "mean": values.mean(axis=(0, 1)).tolist(), "device": frame.device}
            assert np.isfinite(values).all() and error < 2e-5, readings
    finally:
        backend.close()
    return readings


def glass_transport(adapter, backend_cls, scene, total_reflection):
    glass, mat = plane("ShaderNodeBsdfGlass")
    shader = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfGlass")
    emission, _ = plane("ShaderNodeEmission")
    if total_reflection:
        # The ray exits through +Z. A bright plane above the interface distinguishes
        # accidental transmission from the reflected constant sky below it.
        shader.inputs["IOR"].default_value = 1.5
        glass.location.z = 0
        emission.location.z = 2
        scene.camera.data.type = "ORTHO"
        scene.camera.data.ortho_scale = 1
        scene.camera.data.clip_end = 50
        scene.camera.location.z = -2
        angle = math.radians(70)
        direction = Vector((math.sin(angle), 0, math.cos(angle)))
        scene.camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        expected = np.asarray(SKY) * COLOR
    else:
        shader.inputs["IOR"].default_value = 1.0
        shader.inputs["Color"].default_value = (1, 1, 1, 1)
        emission.location.z = -8
        expected = np.asarray(COLOR) * 5
    dg = depsgraph()
    snapshot = adapter.extract_scene(dg)
    camera = adapter.camera_snapshot(dg, 8, 8)
    backend = backend_cls(device="cuda:0")
    try:
        frame = backend.render(snapshot, camera, 8, 8, samples=2, max_bounces=2, seed=23)
        assert frame.device == "cuda:0" and frame.samples == 2
        rgb = frame.pixels[..., :3]
        evidence = {"expected": expected.tolist(), "mean": rgb.mean(axis=(0, 1)).tolist(),
                    "max_abs_error": float(np.max(np.abs(rgb - expected))), "device": frame.device}
        assert evidence["max_abs_error"] < 2e-5, evidence
        return evidence
    finally:
        backend.close()


def main():
    import tempfile

    args = fixture.arguments("artifacts/fspt-p1/edges.json")
    path = Path(args.json_out).resolve()
    report = {"status": "RUNNING", "verifier_sha256": fixture.sha(Path(__file__)),
                  "fixture_sha256": fixture.sha(sibling), "cases": {}}
    addon = None
    try:
        fixture.checkpoint(path, report, "register")
        addon = fixture.load_addon(args, report)
        report["runtime"] = fixture.runtime_info(addon)
        adapter = importlib.import_module(addon.__name__ + ".blender.fspt_scene")
        backend = importlib.import_module(addon.__name__ + ".fspt.backend")
        # Each attempt has a fresh output directory, including stereo suffixes.
        scratch = Path(tempfile.mkdtemp(prefix=path.stem + "-", dir=path.parent))
        report["output_directory"] = str(scratch)
        cases = {
            "object_emission_override": lambda scene: slot_override(adapter, True),
            "object_principled_override": lambda scene: slot_override(adapter, False),
            "view_layer_material_override": lambda scene: layer_override(adapter),
            "muted_surface_link": lambda scene: muted_link(adapter),
            "mixed_mesh_points": lambda scene: points_output(adapter),
            "glass_ior_one": lambda scene: glass_transport(adapter, backend.CudaBackend, scene, False),
            "glass_tir_70_degrees": lambda scene: glass_transport(adapter, backend.CudaBackend, scene, True),
            "clip_perspective": lambda scene: numerical(adapter, backend.CudaBackend, scene, "PERSP"),
            "clip_orthographic": lambda scene: numerical(adapter, backend.CudaBackend, scene, "ORTHO"),
            "diffuse_bounce_depth": lambda scene: numerical(adapter, backend.CudaBackend, scene, "diffuse"),
        }
        for name in ("film_transparent", "multiview", "multiple_layers"):
            cases[name] = lambda scene, name=name: render_guard(addon, scene, scratch / name, name)
        for name, run in cases.items():
            fixture.checkpoint(path, report, name)
            try:
                evidence = run(reset_scene())
                result = {"status": "VERIFIED", "evidence": evidence}
            except SkipTest as exc:
                result = {"status": "NOT_RUN", "reason": str(exc)}
            except Exception:  # noqa: BLE001 - record independent acceptance and cleanup failures
                result = {"status": "FAILED", "error": traceback.format_exc()}
            report["cases"][name] = result
            fixture.write_report(path, report)
            print("FSPT-EDGE", name, result["status"], flush=True)
        fixture.checkpoint(path, report, "unregister")
        reset_scene().render.engine = "BLENDER_WORKBENCH"
        addon.unregister()
        addon = None
        failures = [name for name, result in report["cases"].items() if result["status"] != "VERIFIED"]
        assert not failures, f"Cases requiring attention: {failures}"
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        fixture.write_report(path, report)
        raise
    finally:
        if addon is not None:
            try:
                addon.unregister()
            except Exception:  # noqa: BLE001 - record independent acceptance and cleanup failures
                report["cleanup_error"] = traceback.format_exc()
        fixture.write_report(path, report)
    print("FSPT-EDGE-ACCEPTANCE-PASS", path, flush=True)


if __name__ == "__main__":
    main()
