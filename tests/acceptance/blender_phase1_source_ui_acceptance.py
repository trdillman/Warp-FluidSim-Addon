"""Profile-safe interactive acceptance for Phase 1 source UI workflows.

This script is executed by ``launch_profile_safe_addon_review.py`` only after
that launcher has enabled a verified temporary add-on copy.  It deliberately
does not load the checkout, enable an add-on, or persist Blender preferences.
"""

from __future__ import annotations

import importlib
import json
import os
import traceback
from pathlib import Path

import bpy


RESULT_ENV = "FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"
SCHEMA = "fluidsim-phase1-source-ui-review-v1"
RESULT_PATH = Path(os.environ[RESULT_ENV]).resolve()
RESULT = {
    "schema": SCHEMA,
    "ok": False,
    "checks": [],
    "controls": {},
    "disabled_states": {},
    "lifecycle": {},
    "translation": {},
    "sink_validation": {},
    "injection": {},
    "skipped": [],
    "errors": [],
}


def check(name, condition, detail=""):
    passed = bool(condition)
    RESULT["checks"].append({
        "name": name, "passed": passed, "detail": str(detail)})
    if not passed:
        RESULT["errors"].append(f"{name}: {detail or 'check failed'}")
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")
    return passed


def write_result():
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_name(RESULT_PATH.name + ".tmp")
    temporary.write_text(
        json.dumps(RESULT, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULT_PATH)


def quit_owned_process():
    bpy.ops.wm.quit_blender()
    return None


def select_only(obj):
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def cube(name, location, scale):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.context.view_layer.update()
    return obj


def tag(obj, role):
    select_only(obj)
    result = bpy.ops.fluidsim.tag_object(role=role)
    check(
        f"tag {role}",
        result == {"FINISHED"} and obj.fluidsim.role == role,
        f"result={result}, role={obj.fluidsim.role}",
    )
    return result


class CaptureLayout:
    """Small Blender-layout recorder for deterministic panel-state evidence."""

    def __init__(self, records=None):
        self.records = [] if records is None else records
        self.enabled = True
        self.alert = False

    def _child(self, kind):
        self.records.append({
            "kind": kind, "enabled": bool(self.enabled), "alert": bool(self.alert)})
        return CaptureLayout(self.records)

    def column(self, align=False):
        return self._child("column")

    def row(self, align=False):
        return self._child("row")

    def box(self):
        return self._child("box")

    def separator(self):
        self.records.append({"kind": "separator"})

    def prop(self, data, property_name, **kwargs):
        self.records.append({
            "kind": "property",
            "property": property_name,
            "text": kwargs.get("text"),
            "enabled": bool(self.enabled),
            "value": str(getattr(data, property_name, "<missing>")),
        })

    def prop_enum(self, data, property_name, value, **kwargs):
        self.records.append({
            "kind": "property_enum",
            "property": property_name,
            "enum_value": value,
            "text": kwargs.get("text"),
            "enabled": bool(self.enabled),
        })

    def label(self, text="", icon="NONE"):
        self.records.append({
            "kind": "label", "text": text, "icon": icon,
            "enabled": bool(self.enabled), "alert": bool(self.alert),
        })

    def operator(self, operator_id, **kwargs):
        entry = {"kind": "operator", "id": operator_id, **kwargs}
        self.records.append(entry)

        class CapturedOperator:
            def __setattr__(self, name, value):
                entry[name] = value

        return CapturedOperator()


def capture_panel(panel_class, active_object):
    select_only(active_object)
    layout = CaptureLayout()
    panel = type("Phase1PanelProbe", (), {"layout": layout})()
    panel_class.draw(panel, bpy.context)
    return layout.records


def control_names(records):
    return [item["property"] for item in records
            if item.get("kind") in {"property", "property_enum"}]


def label_text(records):
    return [item.get("text", "") for item in records
            if item.get("kind") in {"label", "property_enum", "operator"}]


def count_in_source_box(solver):
    count = solver.particles.refresh_host_count()
    if count <= 0:
        return 0
    positions = solver.particles.position.numpy()[:count]
    alive = solver.particles.alive.numpy()[:count].astype(bool)
    points = positions[alive]
    if not len(points):
        return 0
    inside = (
        (points[:, 0] >= -0.3) & (points[:, 0] <= 0.3)
        & (points[:, 1] >= -0.3) & (points[:, 1] <= 0.3)
        & (points[:, 2] >= 0.60) & (points[:, 2] <= 1.10)
    )
    return int(inside.sum())


def step_frames(frames):
    result = bpy.ops.fluidsim.step(frames=frames)
    if result != {"FINISHED"}:
        raise RuntimeError(
            f"step({frames}) failed: {result}; {bpy.context.scene.fluidsim.status}")


def lifecycle_run(scene, emitter, mode):
    from fluidSim_warp_blender51.blender import scene_bridge

    scene_bridge.release_session(scene)
    emitter.fluidsim.enabled = True
    emitter.fluidsim.geometry_mode = "ANALYTIC"
    emitter.fluidsim.emission_mode = mode
    select_only(emitter)
    result = bpy.ops.fluidsim.initialize()
    if result != {"FINISHED"}:
        raise RuntimeError(f"initialize {mode} failed: {scene.fluidsim.status}")
    session = scene_bridge.session()
    expected_source_id = f"blender-object:{int(emitter.as_pointer())}"
    translated = [d for d in session.scene_inputs.gather_sources(scene)
                  if getattr(d, "source_id", "") == expected_source_id]
    if len(translated) != 1:
        raise RuntimeError(f"{mode} translated descriptors={len(translated)}")
    translated_mode = int(translated[0].emission_mode)
    step_frames(1)
    first = count_in_source_box(session.solver)
    fired_after_first = sorted(session.solver.sourcing._initial_fired)
    rename_stable = None
    original_name = emitter.name
    if mode == "INITIAL_FILL":
        emitter.name = "Phase1 Review Emitter Renamed"
        renamed = [d for d in session.scene_inputs.gather_sources(scene)
                   if getattr(d, "source_id", "") == expected_source_id]
        rename_stable = len(renamed) == 1
        check("Initial Fill identity survives Blender object rename",
              rename_stable, [getattr(d, "source_id", "") for d in renamed])
    step_frames(8)
    later = count_in_source_box(session.solver)
    fired_after_later = sorted(session.solver.sourcing._initial_fired)
    emitter.name = original_name
    return {
        "enum": mode,
        "translated_mode": translated_mode,
        "source_id": expected_source_id,
        "rename_stable": rename_stable,
        "first_box_count": first,
        "later_box_count": later,
        "fired_after_first": fired_after_first,
        "fired_after_later": fired_after_later,
    }


def run():
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51 import panels
    from fluidSim_warp_blender51.blender import scene_bridge
    from fluidSim_warp_blender51.sim.enums import (
        SourceEmissionMode, SourceShapeSourceMode)
    from fluidSim_warp_blender51.sim.sourcing import MeshSourceDesc

    scene = bpy.context.scene
    fs = scene.fluidsim
    RESULT["module"] = {
        "name": addon.__name__,
        "path": str(Path(addon.__file__).resolve()),
        "version": list(importlib.import_module(addon.__name__ + ".version").VERSION),
    }
    check("temporary add-on already loaded", addon.__name__ == "fluidSim_warp_blender51",
          RESULT["module"])

    scene_bridge.release_session(scene)
    fs.live = False
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is not None and pg.role in {"SOURCE", "SINK", "FORCE"}:
            pg.enabled = False

    ds = scene_bridge.active_domain_settings(scene)
    check("designated domain present", ds is not None)
    ds.voxel_size = 0.1
    ds.max_particles = 120_000
    ds.min_substeps = 1
    ds.max_substeps = 2
    ds.use_scene_gravity = False
    ds.gravity = (0.0, 0.0, -9.81)
    ds.height_enabled = False
    fs.cache_enabled = False
    ds.whitewater_enabled = False
    fs.show_whitewater = False
    fs.show_surface = False
    fs.show_particles = False

    domain = cube("Phase1 Review Domain", (0.0, 0.0, 0.6), (0.6, 0.6, 0.6))
    domain.display_type = "WIRE"
    domain.hide_render = True
    fs.domain_object = domain

    emitter = cube("Phase1 Review Emitter", (0.0, 0.0, 0.85), (0.3, 0.3, 0.25))
    tag(emitter, "SOURCE")
    emitter.fluidsim.geometry_mode = "ANALYTIC"
    emitter.fluidsim.shape = "BOX"
    emitter.fluidsim.emit_rate = 6.0
    emitter.fluidsim.initial_velocity = (0.0, 0.0, 0.0)
    emitter.fluidsim.source_mode = "PARTICLES"

    scene_bridge.release_session(scene)
    select_only(emitter)
    RESULT["disabled_states"]["inject_before_initialize"] = bool(
        bpy.ops.fluidsim.inject_particles.poll())
    check("point injection disabled before initialize",
          not RESULT["disabled_states"]["inject_before_initialize"])
    bpy.ops.object.select_all(action="DESELECT")
    RESULT["disabled_states"]["tag_without_selection"] = bool(
        bpy.ops.fluidsim.tag_object.poll())
    check("tag action disabled without selection",
          not RESULT["disabled_states"]["tag_without_selection"])

    initial = lifecycle_run(scene, emitter, "INITIAL_FILL")
    continuous = lifecycle_run(scene, emitter, "CONTINUOUS_INFLOW")
    RESULT["lifecycle"] = {"initial_fill": initial, "continuous_inflow": continuous}
    check("Initial Fill reaches scene translation",
          initial["translated_mode"] == int(SourceEmissionMode.INITIAL_FILL), initial)
    check("Continuous Inflow reaches scene translation",
          continuous["translated_mode"] == int(SourceEmissionMode.CONTINUOUS_INFLOW), continuous)
    check("Initial Fill emits once then drains",
          initial["first_box_count"] > 0
          and initial["later_box_count"] < initial["first_box_count"] * 0.5,
          initial)
    check("Continuous Inflow replenishes over frames",
          continuous["first_box_count"] > 0
          and continuous["later_box_count"] >= continuous["first_box_count"] * 0.5,
          continuous)
    check("lifecycle armed-state behavior differs",
          initial["source_id"] in initial["fired_after_first"]
          and initial["source_id"] in initial["fired_after_later"]
          and continuous["fired_after_first"] == []
          and continuous["fired_after_later"] == [],
          {"initial": initial["fired_after_later"],
           "continuous": continuous["fired_after_later"]})

    emitter.fluidsim.enabled = False
    mesh_emitter = cube(
        "Phase1 Review Mesh Shell", (0.35, 0.0, 0.8), (0.15, 0.15, 0.15))
    tag(mesh_emitter, "SOURCE")
    mesh_emitter.fluidsim.geometry_mode = "MESH"
    mesh_emitter.fluidsim.mesh_object = mesh_emitter
    mesh_emitter.fluidsim.mesh_surface_thickness = 0.08
    mesh_descs = scene_bridge.session().scene_inputs.gather_sources(scene)
    mesh_source_id = f"blender-object:{int(mesh_emitter.as_pointer())}"
    mesh_desc = next((d for d in mesh_descs
                      if getattr(d, "source_id", "") == mesh_source_id), None)
    RESULT["translation"]["mesh_surface_shell"] = {
        "descriptor": type(mesh_desc).__name__ if mesh_desc is not None else None,
        "mesh_error": mesh_emitter.fluidsim.mesh_error,
    }
    check("Mesh Surface Shell reaches mesh translation",
          isinstance(mesh_desc, MeshSourceDesc)
          and not mesh_emitter.fluidsim.mesh_error,
          RESULT["translation"]["mesh_surface_shell"])

    force = cube("Phase1 Review Force Region", (0.0, 0.0, 0.8), (0.5, 0.5, 0.5))
    tag(force, "FORCE")
    force.fluidsim.shape = "BOX"
    force.fluidsim.force_type = "DIRECTIONAL"
    force.fluidsim.force_strength = 4.0
    force.fluidsim.force_direction = (1.0, 0.0, 0.0)
    force_descs = scene_bridge.session().scene_inputs.gather_sources(scene)
    force_source_id = f"blender-object:{int(force.as_pointer())}"
    force_desc = next((d for d in force_descs
                       if getattr(d, "source_id", "") == force_source_id), None)
    RESULT["translation"]["force_region"] = {
        "role": force.fluidsim.role,
        "mode": int(force_desc.mode) if force_desc is not None else None,
        "emit_rate": float(force_desc.emit_rate) if force_desc is not None else None,
    }
    check("FORCE role translates as non-liquid FIELD representation",
          force_desc is not None
          and int(force_desc.mode) == int(SourceShapeSourceMode.FIELD)
          and float(force_desc.emit_rate) == 0.0,
          RESULT["translation"]["force_region"])

    bpy.ops.mesh.primitive_plane_add(size=0.6, location=(0.0, 0.0, 0.5))
    sink = bpy.context.object
    sink.name = "Phase1 Review Open Mesh Sink"
    tag(sink, "SINK")
    sink.fluidsim.geometry_mode = "MESH"
    sink.fluidsim.mesh_object = sink
    sink.fluidsim.sink_region_mode = "SOLID"
    scene_bridge.session().scene_inputs.gather_sources(scene)
    solid_error = sink.fluidsim.mesh_error
    sink.fluidsim.sink_region_mode = "SURFACE"
    scene_bridge.session().scene_inputs.gather_sources(scene)
    surface_error = sink.fluidsim.mesh_error
    RESULT["sink_validation"] = {
        "solid_error": solid_error,
        "surface_error": surface_error,
    }
    check("Solid open-mesh sink shows adjacent validation",
          bool(solid_error) and "watertight" in solid_error.lower(), solid_error)
    check("Surface mesh sink accepts open mesh", not surface_error, surface_error)

    source_records = capture_panel(panels.FLUIDSIM_PT_sources, emitter)
    attribute_records = capture_panel(
        panels.FLUIDSIM_PT_source_particle_attributes, emitter)
    sink.fluidsim.sink_region_mode = "SOLID"
    sink.fluidsim.mesh_error = solid_error
    sink_records = capture_panel(panels.FLUIDSIM_PT_colliders, sink)
    force_records = capture_panel(panels.FLUIDSIM_PT_forces, force)
    RESULT["controls"] = {
        "source": source_records,
        "particle_attributes": attribute_records,
        "sink": sink_records,
        "force": force_records,
        "source_order": control_names(source_records),
        "sink_order": control_names(sink_records),
        "force_order": control_names(force_records),
    }
    visible_properties = set(
        control_names(source_records + attribute_records + sink_records + force_records))
    visible_text = "\n".join(label_text(
        source_records + attribute_records + sink_records + force_records))
    check("truthful emitter controls visible",
          all(name in visible_properties for name in (
              "emission_mode", "geometry_mode", "emit_rate", "density")),
          sorted(visible_properties))
    check("Mesh Surface Shell label visible",
          "Mesh Surface Shell" in visible_text, visible_text)
    check("sink validation text is adjacent in sink panel",
          solid_error in visible_text, solid_error)
    check("Tag Force Region action visible",
          any(item.get("id") == "fluidsim.tag_object"
              and item.get("text") == "Tag Force Region"
              and item.get("role") == "FORCE" for item in force_records),
          force_records)
    forbidden_properties = {
        "source_mode", "blend_mode", "feather", "height_back_mode"}
    check("compatibility and ineffective controls hidden",
          not (visible_properties & forbidden_properties),
          sorted(visible_properties & forbidden_properties))
    check("Phase 2 source controls absent",
          "Mesh Solid Volume" not in visible_text
          and "Volume Grid" not in visible_text
          and "OpenVDB" not in visible_text,
          visible_text)
    check("FIELD representation is not presented as liquid emission",
          "FIELD" not in visible_text and "Field" not in visible_text,
          visible_text)

    mesh_emitter.fluidsim.enabled = False
    sink.fluidsim.enabled = False
    force.fluidsim.enabled = False
    emitter.fluidsim.enabled = False
    select_only(emitter)
    before_mesh = scene_bridge.session().solver.particles.refresh_host_count()
    mesh_result = bpy.ops.fluidsim.inject_particles()
    after_mesh = scene_bridge.session().solver.particles.refresh_host_count()
    RESULT["injection"]["mesh"] = {
        "result": sorted(mesh_result), "before": before_mesh, "after": after_mesh,
        "expected_points": len(emitter.data.vertices),
    }
    check("Inject Active Object Points Once supports mesh vertices",
          mesh_result == {"FINISHED"}
          and after_mesh - before_mesh == len(emitter.data.vertices),
          RESULT["injection"]["mesh"])

    point_data = bpy.data.pointclouds.new("Phase1 Review Points")
    point_data.resize(3)
    point_data.attributes["position"].data.foreach_set(
        "vector", (-0.1, 0.0, 0.9, 0.0, 0.0, 0.9, 0.1, 0.0, 0.9))
    point_object = bpy.data.objects.new("Phase1 Review PointCloud", point_data)
    scene.collection.objects.link(point_object)
    select_only(point_object)
    before_points = scene_bridge.session().solver.particles.refresh_host_count()
    point_result = bpy.ops.fluidsim.inject_particles()
    after_points = scene_bridge.session().solver.particles.refresh_host_count()
    RESULT["injection"]["point_cloud"] = {
        "result": sorted(point_result), "before": before_points,
        "after": after_points, "expected_points": 3,
    }
    check("Inject Active Object Points Once supports PointCloud points",
          point_result == {"FINISHED"} and after_points - before_points == 3,
          RESULT["injection"]["point_cloud"])

    particle_objects = [obj for obj in scene.objects
                        if len(getattr(obj, "particle_systems", ())) > 0]
    if particle_objects:
        particle_object = particle_objects[0]
        select_only(particle_object)
        before_particles = scene_bridge.session().solver.particles.refresh_host_count()
        particle_result = bpy.ops.fluidsim.inject_particles()
        after_particles = scene_bridge.session().solver.particles.refresh_host_count()
        RESULT["injection"]["particle_system"] = {
            "object": particle_object.name,
            "result": sorted(particle_result),
            "before": before_particles,
            "after": after_particles,
        }
        check("first particle-system input is accepted",
              particle_result == {"FINISHED"} and after_particles > before_particles,
              RESULT["injection"]["particle_system"])
    else:
        RESULT["skipped"].append(
            "particle-system injection: dam_break fixture has no particle system")
        RESULT["injection"]["particle_system"] = {
            "status": "not_available_in_fixture"}

    RESULT["ok"] = not RESULT["errors"]


try:
    run()
except Exception as exc:  # noqa: BLE001 - acceptance must fail closed
    RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
    RESULT["traceback"] = traceback.format_exc()
finally:
    RESULT["ok"] = bool(RESULT["ok"] and not RESULT["errors"])
    try:
        from fluidSim_warp_blender51.blender import scene_bridge
        scene_bridge.release_session(bpy.context.scene)
    except Exception as exc:  # noqa: BLE001
        RESULT["errors"].append(f"release failed: {exc}")
        RESULT["ok"] = False
    write_result()
    marker = "PHASE1-SOURCE-UI-REVIEW-PASS" if RESULT["ok"] else "PHASE1-SOURCE-UI-REVIEW-FAIL"
    print(marker + " " + json.dumps({
        "checks": len(RESULT["checks"]),
        "errors": RESULT["errors"],
        "skipped": RESULT["skipped"],
    }))
    bpy.app.timers.register(quit_owned_process, first_interval=0.1)
