"""Blender 5.1 acceptance for the Phase 3 role-first source UI translators."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.repo_addon_loader import enable_repo_addon

FAILURES = []
CHECKS = []


def check(name, condition, detail=""):
    passed = bool(condition)
    CHECKS.append({"name": name, "passed": passed, "detail": str(detail)})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")
    if not passed:
        FAILURES.append(name)


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def add_cube(name, location=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(ROOT / "artifacts" / "phase3_source_ui_acceptance.json"))
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = parse_args()
    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51.blender.scene_inputs import (
        SceneInputs,
        persistent_source_key,
    )
    from fluidSim_warp_blender51.sim.enums import MeshSourceRegion, SourceRegionKind
    from fluidSim_warp_blender51.sim.source_regions import CompileContext, SourceCompiler
    from fluidSim_warp_blender51.version import VERSION

    def plan_for(obj, candidates):
        source_key = persistent_source_key(obj)
        return next((plan for plan in candidates
                     if str(plan.stable_id) == source_key), None)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene

    emitter = add_cube("P3_MeshSolid", (0, 0, 0))
    select_only(emitter)
    check("role-specific emitter operator", bpy.ops.fluidsim.tag_liquid_emitter() == {"FINISHED"})
    check("new emitter schema/default", emitter.fluidsim.source_ui_schema_version == 4
          and emitter.fluidsim.emission_mode == "INITIAL_FILL"
          and emitter.fluidsim.source_sampling_mode == "INHERIT")
    emitter.fluidsim.source_region_kind = "MESH_SOLID"
    emitter.fluidsim.mesh_object = emitter

    point = add_cube("P3_PointEmitter", (1.25, 0, 0))
    select_only(point)
    check("point emitter operator", bpy.ops.fluidsim.tag_liquid_emitter() == {"FINISHED"})
    point.fluidsim.source_region_kind = "POINTS"
    point.fluidsim.point_input_object = point
    point.fluidsim.point_source_representation = "RASTERIZED_REGION"
    point.fluidsim.point_rasterize_radius = 0.2

    volume_path = ROOT / "artifacts" / "phase3_fixtures" / "ui_volume.vdb"
    volume_path.parent.mkdir(parents=True, exist_ok=True)
    import openvdb
    density = openvdb.FloatGrid(); density.name = "density"
    accessor = density.getAccessor()
    for x in range(3):
        for y in range(3):
            for z in range(3):
                accessor.setValueOn((x, y, z), 1.0)
    velocity = openvdb.Vec3SGrid(); velocity.name = "velocity"
    velocity_accessor = velocity.getAccessor()
    for x in range(3):
        for y in range(3):
            for z in range(3):
                velocity_accessor.setValueOn((x, y, z), (float(x), float(y), float(z)))
    openvdb.write(str(volume_path), grids=[density, velocity])
    volume_data = bpy.data.volumes.new("P3_UI_VolumeData")
    volume_data.filepath = str(volume_path)
    volume = bpy.data.objects.new("P3_VolumeEmitter", volume_data)
    scene.collection.objects.link(volume)
    select_only(volume)
    check("volume emitter operator", bpy.ops.fluidsim.tag_liquid_emitter() == {"FINISHED"})
    volume.fluidsim.source_region_kind = "VOLUME_GRID"
    volume.fluidsim.volume_input_object = volume
    volume.fluidsim.volume_mask_grid = "density"

    field = add_cube("P3_FieldPressure", (0, 1.25, 0))
    select_only(field)
    check("field operator", bpy.ops.fluidsim.tag_field_source() == {"FINISHED"})
    field.fluidsim.field_operation = "PRESSURE"
    field.fluidsim.field_region_kind = "ANALYTIC"
    field.fluidsim.field_value_source = "CONSTANT"
    field.fluidsim.field_scalar_value = 2.0

    volume_sink = bpy.data.objects.new("P3_VolumeSink", volume_data)
    scene.collection.objects.link(volume_sink)
    select_only(volume_sink)
    check("volume sink operator", bpy.ops.fluidsim.tag_liquid_sink() == {"FINISHED"})
    volume_sink.fluidsim.sink_region_kind = "VOLUME_GRID"
    volume_sink.fluidsim.volume_input_object = volume_sink
    volume_sink.fluidsim.volume_mask_grid = "density"

    sink = add_cube("P3_Sink", (-1.25, 0, 0))
    select_only(sink)
    check("sink operator", bpy.ops.fluidsim.tag_liquid_sink() == {"FINISHED"})
    sink.fluidsim.sink_region_kind = "MESH_SHELL"
    sink.fluidsim.mesh_object = sink

    collider = add_cube("P3_Collider", (0, -1.25, 0))
    select_only(collider)
    check("collider operator", bpy.ops.fluidsim.tag_collider() == {"FINISHED"})

    force = add_cube("P3_Force", (0, 0, 1.25))
    select_only(force)
    check("force operator", bpy.ops.fluidsim.tag_force_region() == {"FINISHED"})
    force.fluidsim.force_type = "DIRECTIONAL"

    legacy = add_cube("P3_LegacyShell", (1.25, 1.25, 0))
    legacy.fluidsim.role = "SOURCE"
    legacy.fluidsim.source_ui_schema_version = 0
    legacy.fluidsim.geometry_mode = "MESH"
    legacy.fluidsim.source_mode = "PARTICLES"
    legacy.fluidsim.emit_rate = 5.0

    image = bpy.data.images.new("P3_HeightProfile", width=4, height=4)
    pixels = []
    for y in range(4):
        for x in range(4):
            value = (x + y) / 6.0
            pixels.extend((value, value, value, 1.0))
    image.pixels[:] = pixels
    scene.fluidsim.height_enabled = True
    scene.fluidsim.height_back_mode = "PROFILE"
    scene.fluidsim.height_profile_image = image
    scene.fluidsim.height_profile_scale = 0.5

    inputs = SceneInputs()
    descriptors = inputs.gather_sources(scene)
    plans = inputs.gather_source_plans(scene)
    point.fluidsim.point_source_representation = "DIRECT_APPEND"
    direct_point_plan = plan_for(point, inputs.gather_source_plans(scene))
    check("direct point representation translates once", direct_point_plan is not None
          and direct_point_plan.lifecycle.name == "DIRECT_APPEND"
          and direct_point_plan.region.kind == SourceRegionKind.POINTS)
    point.fluidsim.point_source_representation = "RASTERIZED_REGION"
    height = inputs.gather_height_source(scene)
    solid = next((item for item in descriptors
                  if getattr(item, "runtime_source_key", "").endswith(str(emitter.as_pointer()))), None)
    legacy_desc = next((item for item in descriptors
                        if getattr(item, "runtime_source_key", "").endswith(str(legacy.as_pointer()))), None)
    check("mesh solid translator", solid is not None
          and int(solid.source_region) == int(MeshSourceRegion.SOLID_VOLUME))
    emitter.fluidsim.source_region_kind = "MESH_SHELL"
    shell_desc = next((item for item in inputs.gather_sources(scene)
                       if getattr(item, "runtime_source_key", "").endswith(
                           str(emitter.as_pointer()))), None)
    check("mesh shell translator", shell_desc is not None
          and int(shell_desc.source_region) == int(MeshSourceRegion.SURFACE_SHELL))
    emitter.fluidsim.source_region_kind = "MESH_SOLID"
    check("legacy mesh remains shell", legacy_desc is not None
          and int(legacy_desc.source_region) == int(MeshSourceRegion.SURFACE_SHELL))
    kinds = {plan.region.kind for plan in plans}
    check("point and volume plans translated", SourceRegionKind.POINTS in kinds
          and SourceRegionKind.VOLUME_GRID in kinds, str(kinds))
    check("Volume sink plan translated", any(
        plan.operation.name == "SINK" and plan.region.kind == SourceRegionKind.VOLUME_GRID
        for plan in plans))
    check("advanced pressure plan translated", any(plan.operation.name == "PRESSURE" for plan in plans))
    translated_field_ops = set()
    for operation in ("PRESSURE", "SCALAR", "VECTOR", "STAGGERED_VELOCITY"):
        field.fluidsim.field_operation = operation
        field.fluidsim.field_region_kind = "ANALYTIC"
        field.fluidsim.field_value_source = "CONSTANT"
        field_plan = plan_for(field, inputs.gather_source_plans(scene))
        if field_plan is not None:
            translated_field_ops.add(field_plan.operation.name)
    check("all accepted field operations translate", translated_field_ops == {
        "PRESSURE", "SCALAR_FIELD", "VECTOR_FIELD", "STAGGERED_VELOCITY"},
        translated_field_ops)
    field.fluidsim.field_operation = "SCALAR"
    field.fluidsim.field_region_kind = "VOLUME_GRID"
    field.fluidsim.field_value_source = "VOLUME_GRID"
    field.fluidsim.volume_input_object = volume
    field.fluidsim.volume_mask_grid = "density"
    field.fluidsim.volume_scalar_grid = "density"
    imported_field = plan_for(field, inputs.gather_source_plans(scene))
    check("Volume field payload translates", imported_field is not None
          and imported_field.operation.name == "SCALAR_FIELD"
          and imported_field.region.kind == SourceRegionKind.VOLUME_GRID)
    field.fluidsim.field_operation = "STAGGERED_VELOCITY"
    field.fluidsim.volume_scalar_grid = ""
    field.fluidsim.volume_vector_grid = "velocity"
    imported_vector = plan_for(field, inputs.gather_source_plans(scene))
    check("Volume vector field payload translates", imported_vector is not None
          and imported_vector.operation.name == "STAGGERED_VELOCITY"
          and imported_vector.region.vector_grid == "velocity")
    field.fluidsim.field_operation = "PRESSURE"
    field.fluidsim.field_region_kind = "ANALYTIC"
    field.fluidsim.field_value_source = "CONSTANT"
    check("height image profile translated", height is not None
          and height.profile is not None and height.profile.shape == (4, 4))
    check("roles remain separated", sink.fluidsim.role == "SINK"
          and collider.fluidsim.role == "COLLIDER"
          and force.fluidsim.role == "FORCE" and field.fluidsim.role == "FIELD")

    compiler = SourceCompiler()
    batch = compiler.compile(plans, CompileContext(
        1, 0, 1, "cpu", (-2, -2, -2), 0.25,
        (-2, -2, -2), (3, 3, 3), upload=False))
    check("translated plans compile", not any(d.severity == "ERROR" for d in batch.diagnostics),
          [(d.code, d.message) for d in batch.diagnostics])

    invalid = add_cube("P3_InvalidVolume")
    select_only(invalid)
    bpy.ops.fluidsim.tag_liquid_emitter()
    invalid.fluidsim.source_region_kind = "VOLUME_GRID"
    invalid_plans = inputs.gather_source_plans(scene)
    check("invalid input owns actionable diagnostic", invalid.fluidsim.source_status_code == "SOURCE_INPUT_INVALID"
          and "Volume" in invalid.fluidsim.source_status_message)

    fixture = ROOT / "artifacts" / "phase3_fixtures" / "phase3_roles.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(fixture), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(fixture), load_ui=False)
    check("saved roles reopen", bpy.data.objects["P3_MeshSolid"].fluidsim.role == "SOURCE"
          and bpy.data.objects["P3_FieldPressure"].fluidsim.role == "FIELD")
    reopened_legacy = bpy.data.objects["P3_LegacyShell"].fluidsim
    check("legacy values migrate without occupancy drift",
          reopened_legacy.geometry_mode == "MESH"
          and reopened_legacy.source_ui_schema_version == 4
          and reopened_legacy.source_sampling_mode == "LEGACY_ABSOLUTE"
          and reopened_legacy.emit_rate == 5.0)

    required_panels = (
        "FLUIDSIM_PT_liquid_emitters", "FLUIDSIM_PT_liquid_emitter_attributes",
        "FLUIDSIM_PT_liquid_sinks", "FLUIDSIM_PT_colliders", "FLUIDSIM_PT_forces",
        "FLUIDSIM_PT_domain_height_fill", "FLUIDSIM_PT_point_particle_injection",
        "FLUIDSIM_PT_advanced_field_sources",
    )
    check("role panels registered", all(hasattr(bpy.types, name) for name in required_panels))
    check("legacy mixed source panel absent", not hasattr(bpy.types, "FLUIDSIM_PT_sources"))

    report = {
        "schema": "fluidsim-phase3-source-ui-v1",
        "ok": not FAILURES,
        "identity": {
            "blender_version": bpy.app.version_string,
            "blender_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
            "addon_path": str(Path(addon.__file__).resolve()),
            "addon_version": list(VERSION),
        },
        "fixture": str(fixture.resolve()),
        "checks": CHECKS,
        "errors": FAILURES,
    }
    out = Path(args.json_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    if FAILURES:
        raise SystemExit(f"PHASE3-SOURCE-UI-ACCEPTANCE-FAIL ({len(FAILURES)}): {FAILURES}")
    print("PHASE3-SOURCE-UI-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
