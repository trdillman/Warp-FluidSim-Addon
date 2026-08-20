"""Blender integration validation for Phase-3 attributes and render nodes."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_attribute(data, spec, count: int) -> np.ndarray:
    attr = data.attributes.get(spec.blender_name)
    if attr is None:
        raise AssertionError(f"missing Blender attribute {spec.blender_name}")
    if attr.data_type != spec.data_type or attr.domain != spec.domain:
        raise AssertionError(
            f"{spec.blender_name} has {attr.data_type}/{attr.domain}; "
            f"expected {spec.data_type}/{spec.domain}"
        )
    total = len(attr.data)
    shape = (total,) if spec.components == 1 else (total, spec.components)
    values = np.empty(shape, dtype=spec.numpy_dtype)
    prop = {
        "FLOAT": "value",
        "INT": "value",
        "FLOAT_VECTOR": "vector",
        "FLOAT_COLOR": "color",
    }[spec.data_type]
    attr.data.foreach_get(prop, values.ravel())
    values = values[:count]
    if spec.data_type != "INT" and not np.all(np.isfinite(values)):
        raise AssertionError(f"{spec.blender_name} contains non-finite values")
    return values


def _look_at(obj, target=(0.0, 0.0, 0.0)) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _simple_material(name: str, color) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.35
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 0.0
    return material


def _render(
    scene, path: Path, surface_obj, particles_obj, whitewater_obj, *,
    show_surface: bool, show_particles: bool, show_whitewater: bool,
) -> dict:
    fs = scene.fluidsim
    fs.show_surface = bool(show_surface)
    fs.show_particles = bool(show_particles)
    fs.show_whitewater = bool(show_whitewater)
    surface_obj.hide_render = not show_surface
    particles_obj.hide_render = not show_particles
    whitewater_obj.hide_render = not show_whitewater
    bpy.context.view_layer.update()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        raw = np.asarray(image.pixels[:], dtype=np.float32)
        if raw.size == 0:
            raise AssertionError(f"render image could not be read: {path}")
        pixels = raw.reshape((-1, 4))[:, :3]
    finally:
        bpy.data.images.remove(image)
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    stats = {
        "path": str(path),
        "mean": float(np.mean(pixels)),
        "std": float(np.std(pixels)),
        "max": float(np.max(pixels)),
        "red_dominant_pixels": int(np.count_nonzero(
            (red > 0.12) & (red > green * 1.25) & (red > blue * 1.25)
        )),
        "green_dominant_pixels": int(np.count_nonzero(
            (green > 0.12) & (green > red * 1.25) & (green > blue * 1.25)
        )),
        "blue_dominant_pixels": int(np.count_nonzero(
            (blue > 0.12) & (blue > red * 1.25) & (blue > green * 1.25)
        )),
    }
    if stats["std"] <= 0.005 or stats["max"] <= 0.08:
        raise AssertionError(f"render appears empty or uniform: {stats}")
    return stats


def _assert_group_structure(group, kind: str, expected_revision: str) -> None:
    if group.get("fluidsim_source_revision") != expected_revision:
        raise AssertionError(f"{group.name} has the wrong source revision")
    node_types = {node.bl_idname for node in group.nodes}
    if "GeometryNodeSetMaterial" not in node_types:
        raise AssertionError(f"{group.name} does not assign a material")
    if kind != "surface":
        if "GeometryNodeSetPointRadius" not in node_types:
            raise AssertionError(f"{group.name} does not preserve/scale point radius")
        if "GeometryNodeRealizeInstances" not in node_types:
            raise AssertionError(
                f"{group.name} does not realize instances for shader attributes"
            )


def validate(module_name: str, work_dir: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon = importlib.import_module(module_name)
    addon.register()
    try:
        geometry_nodes = importlib.import_module(f"{module_name}.blender.geometry_nodes")
        materials_module = importlib.import_module(f"{module_name}.blender.materials")
        output_publication = importlib.import_module(
            f"{module_name}.blender.output_publication"
        )
        cache = importlib.import_module(f"{module_name}.io.cache")
        attributes = importlib.import_module(f"{module_name}.sim.attributes")
        ATTRIBUTE_SCHEMA_VERSION = attributes.ATTRIBUTE_SCHEMA_VERSION
        ATTRIBUTE_SPECS = attributes.ATTRIBUTE_SPECS
        OUTPUT_ATTRIBUTE_KEYS = attributes.OUTPUT_ATTRIBUTE_KEYS
        OUTPUT_TOGGLE_NAMES = attributes.OUTPUT_TOGGLE_NAMES
        SURFACE_TRANSFER_METHOD = attributes.SURFACE_TRANSFER_METHOD
        cache_attribute_name = attributes.cache_attribute_name

        scene = bpy.context.scene
        fs = scene.fluidsim
        fs.show_surface = True
        fs.surface_enabled = True
        fs.show_particles = True
        fs.whitewater_enabled = True
        fs.show_whitewater = True
        fs.particle_display_fraction = 1.0
        fs.surface_attribute_mode = "NONE"
        fs.particle_attribute_mode = "NONE"
        fs.whitewater_attribute_mode = "NONE"
        for output_kind, keys in OUTPUT_ATTRIBUTE_KEYS.items():
            for key in keys:
                setattr(fs, OUTPUT_TOGGLE_NAMES[output_kind][key], True)

        surface_material = _simple_material("P3 Surface Test", (0.8, 0.1, 0.05, 1.0))
        particle_material = _simple_material("P3 Particle Test", (0.05, 0.25, 1.0, 1.0))
        whitewater_material = _simple_material("P3 Whitewater Test", (1.0, 0.9, 0.2, 1.0))
        fs.surface_material = surface_material
        fs.particle_material = particle_material
        fs.whitewater_material = whitewater_material

        outputs = output_publication.ensure_output_shells(scene)
        output_map = {
            str(obj.get(output_publication.OUTPUT_TAG, obj.name)): obj for obj in outputs
        }
        installed = geometry_nodes.install_render_nodes(
            scene, output_objects=output_map
        )
        installed_again = geometry_nodes.install_render_nodes(
            scene, output_objects=output_map
        )
        if installed_again.installed_groups:
            raise AssertionError("render-node installation is not idempotent")
        if any(len(obj.modifiers) != 1 for obj in outputs):
            raise AssertionError("render-node installation duplicated modifiers")

        surface = output_publication.output_object(
            scene, output_publication.SURFACE_NAME)
        particles = output_publication.output_object(
            scene, output_publication.PARTICLES_NAME)
        whitewater = output_publication.output_object(
            scene, output_publication.WHITEWATER_NAME)
        if any(obj is None for obj in (surface, particles, whitewater)):
            raise AssertionError("Phase-3 compatibility outputs are missing")
        surface_verts = np.array(
            [[-2.0, 0.0, -1.2], [2.0, 0.0, -1.2], [2.0, 0.0, 1.2], [-2.0, 0.0, 1.2]],
            dtype=np.float32,
        )
        surface_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        particle_positions = np.array(
            [[-1.1, -0.45, -0.3], [0.0, -0.45, 0.35], [1.1, -0.45, -0.1]],
            dtype=np.float32,
        )
        whitewater_positions = np.array(
            [[-0.55, -0.65, 0.85], [0.55, -0.65, 0.95]], dtype=np.float32
        )
        particle_velocity = np.array(
            [[2.4, 0.0, 0.0], [0.0, 1.2, 0.0], [-1.2, 0.0, 0.6]], dtype=np.float32
        )
        surface_velocity = np.array(
            [[2.4, 0.0, 0.0], [2.4, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 1.2, 0.0]],
            dtype=np.float32,
        )
        whitewater_velocity = np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        surface_attributes = {
            "velocity": surface_velocity,
            "speed": np.linalg.norm(surface_velocity, axis=1).astype(np.float32),
            "vorticity": np.tile(np.array([[0.0, 0.0, 0.5]], dtype=np.float32), (4, 1)),
            "age": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "lifetime": np.full(4, 4.0, dtype=np.float32),
            "color": np.array(
                [[1.0, 0.1, 0.1, 1.0], [1.0, 0.2, 0.1, 1.0],
                 [0.1, 0.4, 1.0, 1.0], [0.1, 0.6, 1.0, 1.0]], dtype=np.float32
            ),
            "source_id": np.array([11, 11, 22, 22], dtype=np.int32),
            "viscosity": np.array([0.1, 0.1, 0.5, 0.5], dtype=np.float32),
            "density": np.array([1.0, 1.0, 0.9, 0.9], dtype=np.float32),
            "motion": np.ascontiguousarray(surface_velocity / 24.0),
        }
        particle_attributes = {
            "velocity": particle_velocity,
            "speed": np.linalg.norm(particle_velocity, axis=1).astype(np.float32),
            "vorticity": np.tile(np.array([[0.0, 0.5, 0.0]], dtype=np.float32), (3, 1)),
            "age": np.array([0.2, 0.4, 0.6], dtype=np.float32),
            "lifetime": np.full(3, 4.0, dtype=np.float32),
            "color": np.array(
                [[1.0, 0.15, 0.05, 1.0], [0.05, 1.0, 0.2, 1.0],
                 [0.05, 0.2, 1.0, 1.0]], dtype=np.float32
            ),
            "source_id": np.array([11, 22, 33], dtype=np.int32),
            "uid": np.array([101, 102, 103], dtype=np.int32),
            "viscosity": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "density": np.array([1.0, 0.95, 0.9], dtype=np.float32),
        }
        whitewater_attributes = {
            "velocity": whitewater_velocity,
            "speed": np.linalg.norm(whitewater_velocity, axis=1).astype(np.float32),
            "age": np.array([0.1, 0.2], dtype=np.float32),
            "lifetime": np.array([2.0, 2.0], dtype=np.float32),
        }

        payload = output_publication.PublicationPayload(
            voxel_size=0.5,
            publish_surface=True,
            surface_visible=True,
            surface_attribute_mode="NONE",
            surface_verts=surface_verts,
            surface_tris=surface_tris,
            surface_attributes=surface_attributes,
            particles_visible=True,
            particle_attribute_mode="NONE",
            particle_radius=0.38,
            particles=particle_positions,
            particle_attributes=particle_attributes,
            whitewater_visible=True,
            whitewater_attribute_mode="NONE",
            whitewater_output_mode="COMBINED",
            whitewater_radius=0.28,
            whitewater=whitewater_positions,
            whitewater_attributes=whitewater_attributes,
        )
        output_publication.publish_payload(scene, payload)

        expected_by_kind = {
            "surface": (surface, len(surface_verts), surface_attributes),
            "particles": (particles, len(particle_positions), particle_attributes),
            "whitewater": (whitewater, len(whitewater_positions), whitewater_attributes),
        }
        for kind, (obj, count, expected) in expected_by_kind.items():
            for key, values in expected.items():
                actual = _read_attribute(obj.data, ATTRIBUTE_SPECS[key], count)
                np.testing.assert_allclose(actual, values, rtol=0.0, atol=1.0e-7)
            modifier = obj.modifiers[0]
            group = modifier.node_group
            _assert_group_structure(
                group, kind, geometry_nodes.ASSET_REVISION)
            assigned_material = geometry_nodes._modifier_input_value(
                modifier, group, geometry_nodes.MATERIAL_SOCKET
            )
            if assigned_material is not obj.data.materials[0]:
                raise AssertionError(f"{obj.name} render material input is stale")
            if kind != "surface":
                assigned_radius = geometry_nodes._modifier_input_value(
                    modifier, group, geometry_nodes.RADIUS_SOCKET
                )
                if abs(float(assigned_radius) - 1.0) > 1.0e-8:
                    raise AssertionError(f"{obj.name} radius scale input is stale")

        motion = _read_attribute(surface.data, ATTRIBUTE_SPECS["motion"], 4)
        np.testing.assert_allclose(motion, surface_velocity / 24.0, rtol=0.0, atol=1.0e-7)

        # Cache-v1 remains readable while Phase-3 inventory/arrays round-trip.
        cache_dir = work_dir / "cache_roundtrip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame_payload = {
            "format_version": np.array(cache.CACHE_FORMAT_VERSION, dtype=np.int32),
            "frame": np.array(1, dtype=np.int32),
            "voxel_size": np.array(0.5, dtype=np.float32),
            "positions": particle_positions,
            "velocities": particle_velocity,
            "whitewater": whitewater_positions,
            "whitewater_velocities": whitewater_velocity,
            "surface_verts": surface_verts,
            "surface_tris": surface_tris,
            # This validator deliberately writes a Phase-3 schema-v1 frame
            # to prove backward-compatible playback under the current release.
            "attribute_schema_version": np.array(1, dtype=np.int32),
            "attribute_inventory_json": np.array(json.dumps(
                {
                    kind: [key for key in values if key != "type"]
                    for kind, values in OUTPUT_ATTRIBUTE_KEYS.items()
                },
                sort_keys=True,
                separators=(",", ":"),
            )),
            "surface_attribute_transfer_method": np.array(SURFACE_TRANSFER_METHOD),
        }
        for kind, attributes in (
            ("surface", surface_attributes),
            ("particles", particle_attributes),
            ("whitewater", whitewater_attributes),
        ):
            for key, values in attributes.items():
                frame_payload[cache_attribute_name(kind, key)] = values
        np.savez(cache_dir / "fluidsim_00001.npz", **frame_payload)
        restored = cache.read_frame(str(cache_dir), 1)
        if restored is None:
            raise AssertionError("Phase-3 cache frame did not round-trip")
        output_publication.update_outputs_from_cache_frame(scene, restored)
        for kind, (obj, count, expected) in expected_by_kind.items():
            for key, values in expected.items():
                actual = _read_attribute(obj.data, ATTRIBUTE_SPECS[key], count)
                np.testing.assert_allclose(actual, values, rtol=0.0, atol=1.0e-7)

        blend_path = work_dir / "phase3_attributes_roundtrip.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
        scene = bpy.context.scene
        scene.fluidsim.particle_display_scale = 3.0
        scene.fluidsim.whitewater_display_scale = 3.0
        surface = output_publication.output_object(scene, output_publication.SURFACE_NAME)
        particles = output_publication.output_object(scene, output_publication.PARTICLES_NAME)
        whitewater = output_publication.output_object(scene, output_publication.WHITEWATER_NAME)
        geometry_nodes.sync_render_inputs(scene)
        for obj in (surface, particles, whitewater):
            if obj is None or not obj.modifiers or obj.modifiers[0].node_group is None:
                raise AssertionError("render modifiers did not survive .blend reopen")
        for obj in (particles, whitewater):
            radius_scale = geometry_nodes._modifier_input_value(
                obj.modifiers[0], obj.modifiers[0].node_group,
                geometry_nodes.RADIUS_SOCKET,
            )
            if abs(float(radius_scale) - 3.0) > 1.0e-8:
                raise AssertionError(f"{obj.name} radius input did not update")
        np.testing.assert_allclose(
            _read_attribute(surface.data, ATTRIBUTE_SPECS["motion"], 4),
            surface_velocity / 24.0,
            rtol=0.0,
            atol=1.0e-7,
        )

        # Evaluate every bundled modifier before rendering.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        evaluated = {}
        for obj in (surface, particles, whitewater):
            evaluated_obj = obj.evaluated_get(depsgraph)
            evaluated[obj.name] = {
                "type": evaluated_obj.type,
                "modifier_count": len(evaluated_obj.modifiers),
            }

        camera_data = bpy.data.cameras.new("P3 Camera")
        camera = bpy.data.objects.new("P3 Camera", camera_data)
        scene.collection.objects.link(camera)
        camera.location = (0.0, -8.0, 0.0)
        _look_at(camera)
        camera_data.lens = 48.0
        scene.camera = camera
        light_data = bpy.data.lights.new("P3 Key", "AREA")
        light_data.energy = 1200.0
        light_data.shape = "DISK"
        light_data.size = 5.0
        light = bpy.data.objects.new("P3 Key", light_data)
        scene.collection.objects.link(light)
        light.location = (0.0, -4.0, 4.0)
        _look_at(light)
        if scene.world is None:
            scene.world = bpy.data.worlds.new("P3 World")
        scene.world.color = (0.005, 0.005, 0.005)
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = 96
        scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False
        render_surface = _render(
            scene,
            work_dir / "phase3_surface_f12.png",
            surface, particles, whitewater,
            show_surface=True,
            show_particles=False,
            show_whitewater=False,
        )
        render_points = _render(
            scene,
            work_dir / "phase3_points_f12.png",
            surface, particles, whitewater,
            show_surface=False,
            show_particles=True,
            show_whitewater=True,
        )
        if render_surface == render_points:
            raise AssertionError("surface and point render validation results are identical")

        # Validate the actual production color-attribute material through the
        # bundled point-instancing Geometry Nodes group, not only on the source
        # point cloud. Each source color must remain distinguishable in F12.
        scene.fluidsim.particle_material = None
        default_particle_material = materials_module.ensure_particle_material(
            scene, particles
        )
        attribute_nodes = {
            node.attribute_name for node in default_particle_material.node_tree.nodes
            if node.bl_idname == "ShaderNodeAttribute"
        }
        if {"fluidsim_color"} - attribute_nodes:
            raise AssertionError("default particle material does not read fluidsim_color")
        geometry_nodes.sync_render_inputs(scene)
        render_attribute_color = _render(
            scene,
            work_dir / "phase3_attribute_color_f12.png",
            surface, particles, whitewater,
            show_surface=False,
            show_particles=True,
            show_whitewater=False,
        )
        for channel in ("red", "green", "blue"):
            if render_attribute_color[f"{channel}_dominant_pixels"] < 2:
                raise AssertionError(
                    "point color attributes did not survive Geometry Nodes/F12: "
                    f"{render_attribute_color}"
                )

        scene.fluidsim.surface_material = None
        default_surface_material = materials_module.ensure_surface_material(
            scene, surface
        )
        surface_attribute_nodes = {
            node.attribute_name for node in default_surface_material.node_tree.nodes
            if node.bl_idname == "ShaderNodeAttribute"
        }
        required_surface_nodes = {"fluidsim_color", "fluidsim_viscosity"}
        if required_surface_nodes - surface_attribute_nodes:
            raise AssertionError(
                "default surface material is missing simulation attributes: "
                f"{sorted(required_surface_nodes - surface_attribute_nodes)}"
            )
        geometry_nodes.sync_render_inputs(scene)
        render_surface_attributes = _render(
            scene,
            work_dir / "phase3_surface_attributes_f12.png",
            surface, particles, whitewater,
            show_surface=True,
            show_particles=False,
            show_whitewater=False,
        )
        if (render_surface_attributes["red_dominant_pixels"] < 2
                or render_surface_attributes["blue_dominant_pixels"] < 2):
            raise AssertionError(
                "surface color attributes did not survive Geometry Nodes/F12: "
                f"{render_surface_attributes}"
            )

        # Prove an identically named user modifier is never claimed even under
        # the explicit update operation.
        modifier = surface.modifiers[0]
        original_group = modifier.node_group
        owner_key = geometry_nodes._MODIFIER_OWNER_KEYS[output_publication.SURFACE_NAME]
        del surface[owner_key]
        user_group = bpy.data.node_groups.new("User Surface Group", "GeometryNodeTree")
        modifier.node_group = user_group
        protected = geometry_nodes.install_render_nodes(
            scene,
            force=True,
            output_objects={
                output_publication.SURFACE_NAME: surface,
                output_publication.PARTICLES_NAME: particles,
                output_publication.WHITEWATER_NAME: whitewater,
            },
        )
        if modifier.node_group is not user_group or surface.name not in protected.skipped_objects:
            raise AssertionError("explicit install overwrote a user-owned modifier")
        modifier.node_group = original_group
        surface[owner_key] = modifier.name

        return {
            "status": "passed",
            "blender_version": bpy.app.version_string,
            "blender_binary": bpy.app.binary_path,
            "attribute_schema_version": ATTRIBUTE_SCHEMA_VERSION,
            "asset_revision": geometry_nodes.ASSET_REVISION,
            "installed_groups": list(installed.installed_groups),
            "evaluated_outputs": evaluated,
            "cache_roundtrip": "passed",
            "blend_reopen": str(blend_path),
            "motion_vector_attribute": "velocity",
            "render_engine": scene.render.engine,
            "renders": {
                "surface": render_surface,
                "points": render_points,
                "attribute_color": render_attribute_color,
                "surface_attributes": render_surface_attributes,
            },
            "default_material_attribute_nodes": {
                "particles": sorted(attribute_nodes),
                "surface": sorted(surface_attribute_nodes),
            },
            "user_modifier_protection": "passed",
        }
    finally:
        try:
            addon.unregister()
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="fluidsim_phase3_work")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    options = parser.parse_args(argv)
    options.work_dir.mkdir(parents=True, exist_ok=True)
    report = validate(options.module, options.work_dir.resolve())
    _write_json(options.report_json.resolve(), report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    raise SystemExit(main(args))
