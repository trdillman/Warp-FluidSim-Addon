"""Blender integration validation for Phase-4 whitewater outputs.

Run with the supplied Blender package. The script validates six persistent
output shells, user-safe Geometry Nodes installation, typed cache publication,
legacy-unclassified playback, save/reopen persistence, and one Eevee F12 render
with independently materialized foam, bubble, and spray objects.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _look_at(obj, target=(0.0, 0.0, 0.0)) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.35
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = color
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = 0.2
    material.diffuse_color = color
    return material


def _live_count(publication, obj) -> int:
    return int(obj.get(publication.POINT_COUNT_TAG, publication.live_point_count(obj)))


def _read_int_attribute(obj, name: str, count: int) -> np.ndarray:
    attribute = obj.data.attributes.get(name)
    if attribute is None:
        raise AssertionError(f"{obj.name} is missing {name}")
    if str(attribute.data_type) != "INT" or str(attribute.domain) != "POINT":
        raise AssertionError(
            f"{obj.name}/{name} has {attribute.data_type}/{attribute.domain}")
    values = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", values)
    return values[:count]


def _typed_frame(attributes, positions: np.ndarray, types: np.ndarray) -> dict:
    return {
        "attribute_schema_version": np.array(attributes.ATTRIBUTE_SCHEMA_VERSION, dtype=np.int32),
        "attribute_inventory_json": np.array(json.dumps({
            "surface": [], "particles": [], "whitewater": ["type"],
        }, sort_keys=True, separators=(",", ":"))),
        "surface_attribute_transfer_method": np.array(attributes.SURFACE_TRANSFER_METHOD),
        "voxel_size": np.array(0.5, dtype=np.float32),
        "positions": np.zeros((0, 3), dtype=np.float32),
        "velocities": np.zeros((0, 3), dtype=np.float32),
        "surface_verts": np.zeros((0, 3), dtype=np.float32),
        "surface_tris": np.zeros((0, 3), dtype=np.int32),
        "whitewater": np.ascontiguousarray(positions, dtype=np.float32),
        "whitewater_velocities": np.zeros_like(positions, dtype=np.float32),
        "whitewater_age": np.zeros(len(positions), dtype=np.float32),
        "whitewater_life": np.full(len(positions), 4.0, dtype=np.float32),
        "whitewater_type": np.ascontiguousarray(types, dtype=np.int32),
    }


def _legacy_frame(positions: np.ndarray) -> dict:
    return {
        "voxel_size": np.array(0.5, dtype=np.float32),
        "positions": np.zeros((0, 3), dtype=np.float32),
        "velocities": np.zeros((0, 3), dtype=np.float32),
        "surface_verts": np.zeros((0, 3), dtype=np.float32),
        "surface_tris": np.zeros((0, 3), dtype=np.int32),
        "whitewater": np.ascontiguousarray(positions, dtype=np.float32),
        "whitewater_velocities": np.zeros_like(positions, dtype=np.float32),
        "whitewater_age": np.zeros(len(positions), dtype=np.float32),
        "whitewater_life": np.full(len(positions), 4.0, dtype=np.float32),
    }


def _assert_groups(geometry_nodes) -> dict:
    expected = set(geometry_nodes.GROUP_BY_OUTPUT.values())
    found = {}
    for group in bpy.data.node_groups:
        if group.get("fluidsim_source_revision") != geometry_nodes.ASSET_REVISION:
            continue
        found[str(group.get("fluidsim_asset_key", group.name))] = {
            "name": group.name,
            "kind": str(group.get("fluidsim_output_kind", "")),
            "nodes": sorted({node.bl_idname for node in group.nodes}),
        }
    if set(found) != expected:
        raise AssertionError(f"unexpected bundled groups: {sorted(found)}")
    for key, record in found.items():
        if "GeometryNodeSetMaterial" not in record["nodes"]:
            raise AssertionError(f"{key} does not assign a material")
        if record["kind"] != "SURFACE":
            for node_type in (
                "GeometryNodeSetPointRadius", "GeometryNodeInstanceOnPoints",
                "GeometryNodeRealizeInstances",
            ):
                if node_type not in record["nodes"]:
                    raise AssertionError(f"{key} is missing {node_type}")
    return found


def _render_typed(scene, publication, path: Path) -> dict:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 160
    scene.render.resolution_y = 112
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Phase 4 Validation World")
    scene.world.color = (0.015, 0.015, 0.015)

    bpy.ops.object.camera_add(location=(0.0, -8.0, 1.2))
    camera = bpy.context.object
    _look_at(camera, (0.0, 0.0, 0.1))
    scene.camera = camera
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 5.0))
    light = bpy.context.object
    light.data.energy = 900.0
    light.data.shape = "DISK"
    light.data.size = 5.0
    _look_at(light, (0.0, 0.0, 0.0))

    for name in publication.WHITEWATER_OUTPUT_NAMES:
        obj = publication.output_object(scene, name)
        if obj is not None:
            obj.hide_render = name == publication.WHITEWATER_NAME
    scene.render.filepath = str(path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape((-1, 4))[:, :3]
    finally:
        bpy.data.images.remove(image)
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    stats = {
        "path": str(path),
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "max": float(np.max(pixels)),
        "red_dominant_pixels": int(np.count_nonzero(
            (red > 0.12) & (red > green * 1.2) & (red > blue * 1.2))),
        "green_dominant_pixels": int(np.count_nonzero(
            (green > 0.12) & (green > red * 1.2) & (green > blue * 1.2))),
        "blue_dominant_pixels": int(np.count_nonzero(
            (blue > 0.12) & (blue > red * 1.2) & (blue > green * 1.2))),
    }
    if min(
        stats["red_dominant_pixels"],
        stats["green_dominant_pixels"],
        stats["blue_dominant_pixels"],
    ) <= 10:
        raise AssertionError(f"typed F12 render did not preserve three materials: {stats}")
    return stats


def validate(module_name: str, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon = importlib.import_module(module_name)
    addon.register()
    try:
        publication = importlib.import_module(f"{module_name}.blender.output_publication")
        geometry_nodes = importlib.import_module(f"{module_name}.blender.geometry_nodes")
        attributes = importlib.import_module(f"{module_name}.sim.attributes")

        scene = bpy.context.scene
        fs = scene.fluidsim
        # Domain settings live only on the designated domain object.
        domain = bpy.data.objects.new("P4 Domain", bpy.data.meshes.new("P4 Domain"))
        scene.collection.objects.link(domain)
        fs.domain_object = domain
        ds = domain.fluidsim_domain
        fs.show_surface = False
        ds.surface_enabled = False
        fs.show_particles = False
        ds.whitewater_enabled = True
        fs.show_whitewater = True
        fs.whitewater_output_mode = "BOTH"
        fs.whitewater_attribute_mode = "TYPE"
        fs.whitewater_display_scale = 1.0
        fs.whitewater_material = _material("P4 Combined", (0.7, 0.7, 0.7, 1.0))
        fs.foam_material = _material("P4 Foam", (1.0, 0.04, 0.02, 1.0))
        fs.bubble_material = _material("P4 Bubbles", (0.02, 1.0, 0.05, 1.0))
        fs.spray_material = _material("P4 Spray", (0.02, 0.08, 1.0, 1.0))

        outputs = publication.ensure_output_shells(scene)
        output_map = {
            str(obj.get(publication.OUTPUT_TAG, obj.name)): obj for obj in outputs}
        if set(output_map) != {
            publication.SURFACE_NAME, publication.PARTICLES_NAME,
            *publication.WHITEWATER_OUTPUT_NAMES,
        }:
            raise AssertionError(f"unexpected output shells: {sorted(output_map)}")

        # A user-created modifier with the public name must be skipped, not
        # overwritten. Removing it then permits the owned modifier transaction.
        foam = output_map[publication.FOAM_NAME]
        user_modifier = foam.modifiers.new(
            name=geometry_nodes.MODIFIER_BY_OUTPUT[publication.FOAM_NAME],
            type="NODES")
        protected = geometry_nodes.install_render_nodes(
            scene, output_objects=output_map)
        if foam.name not in protected.skipped_objects or not protected.warnings:
            raise AssertionError("user-owned modifier was not protected")
        foam.modifiers.remove(user_modifier)
        installed = geometry_nodes.install_render_nodes(
            scene, output_objects=output_map)
        installed_again = geometry_nodes.install_render_nodes(
            scene, output_objects=output_map)
        if installed_again.installed_groups:
            raise AssertionError("Geometry Nodes installation is not idempotent")
        if any(len(obj.modifiers) != 1 for obj in outputs):
            raise AssertionError("output shells do not each have one owned modifier")
        group_report = _assert_groups(geometry_nodes)

        positions = np.array([
            [-1.7, 0.0, -0.25], [-1.35, 0.0, 0.35],
            [-0.2, 0.0, -0.25], [0.2, 0.0, 0.35],
            [1.35, 0.0, -0.25], [1.7, 0.0, 0.35],
        ], dtype=np.float32)
        types = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        typed_payload = publication.prepare_cache_payload(
            scene, _typed_frame(attributes, positions, types))
        publication.publish_payload(scene, typed_payload)
        expected_counts = {
            publication.WHITEWATER_NAME: 6,
            publication.FOAM_NAME: 2,
            publication.BUBBLES_NAME: 2,
            publication.SPRAY_NAME: 2,
        }
        type_values = {}
        material_values = {}
        for name, expected_count in expected_counts.items():
            obj = publication.output_object(scene, name)
            count = _live_count(publication, obj)
            if count != expected_count:
                raise AssertionError(f"{name} has {count} points, expected {expected_count}")
            type_values[name] = _read_int_attribute(
                obj, "fluidsim_whitewater_type", count).tolist()
            material_values[name] = obj.data.materials[0].name
        if type_values[publication.FOAM_NAME] != [0, 0]:
            raise AssertionError("foam output is not a stable type-0 partition")
        if type_values[publication.BUBBLES_NAME] != [1, 1]:
            raise AssertionError("bubble output is not a stable type-1 partition")
        if type_values[publication.SPRAY_NAME] != [2, 2]:
            raise AssertionError("spray output is not a stable type-2 partition")

        # Cache-v1 / Phase-3 data is unclassified. Separate mode must retain it
        # on the combined object while all physically typed outputs stay empty.
        fs.whitewater_output_mode = "SEPARATE"
        legacy_positions = np.array(
            [[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32)
        publication.publish_payload(
            scene, publication.prepare_cache_payload(
                scene, _legacy_frame(legacy_positions)))
        combined = publication.output_object(scene, publication.WHITEWATER_NAME)
        if _live_count(publication, combined) != 2:
            raise AssertionError("legacy unclassified whitewater disappeared")
        if int(combined.get(publication.LEGACY_UNCLASSIFIED_COUNT_TAG, 0)) != 2:
            raise AssertionError("legacy unclassified count was not recorded")
        for name in (publication.FOAM_NAME, publication.BUBBLES_NAME, publication.SPRAY_NAME):
            if _live_count(publication, publication.output_object(scene, name)) != 0:
                raise AssertionError("legacy whitewater was falsely classified")

        fs.whitewater_output_mode = "SEPARATE"
        publication.publish_payload(scene, typed_payload)
        render_path = work_dir / "phase4_typed_whitewater.png"
        render_stats = _render_typed(scene, publication, render_path)
        blend_path = work_dir / "phase4_typed_whitewater.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
        reopened_scene = bpy.context.scene
        reopened = {}
        for name in expected_counts:
            obj = publication.output_object(reopened_scene, name)
            if obj is None:
                raise AssertionError(f"{name} did not survive save/reopen")
            reopened[name] = {
                "count": _live_count(publication, obj),
                "material": obj.data.materials[0].name if obj.data.materials else "",
                "modifier_count": len(obj.modifiers),
            }
        if reopened[publication.FOAM_NAME]["count"] != 2:
            raise AssertionError("typed output count did not survive save/reopen")
        if any(record["modifier_count"] != 1 for record in reopened.values()):
            raise AssertionError("owned render modifiers did not survive save/reopen")

        return {
            "status": "passed",
            "blender_version": bpy.app.version_string,
            "module": module_name,
            "asset_revision": geometry_nodes.ASSET_REVISION,
            "output_shells": sorted(output_map),
            "installed": {
                "groups_appended": list(installed.installed_groups),
                "objects_attached": list(installed.attached_objects),
                "user_modifier_protected": True,
                "idempotent": True,
                "group_structure": group_report,
            },
            "typed_counts": expected_counts,
            "typed_values": type_values,
            "materials": material_values,
            "legacy_unclassified": {
                "combined_count": 2,
                "separate_counts": {"foam": 0, "bubbles": 0, "spray": 0},
            },
            "save_reopen": reopened,
            "render": render_stats,
            "artifacts": {"blend": str(blend_path), "image": str(render_path)},
        }
    finally:
        try:
            addon.unregister()
        except Exception:  # noqa: BLE001, S110 - preserve the primary gate failure.
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    report = validate(options.module, options.work_dir)
    _write_json(options.report_json, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(arguments))
