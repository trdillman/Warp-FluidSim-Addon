"""Profile-safe visible Phase 3 role/UI and multi-frame playback review."""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

import bpy

RESULT_PATH = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"]).resolve()
RESULT = {
    "schema": "fluidsim-phase3-source-ui-review-v1",
    "ok": False,
    "checks": [],
    "roles": {},
    "playback": {},
    "errors": [],
}


def check(name, condition, detail=""):
    passed = bool(condition)
    RESULT["checks"].append({"name": name, "passed": passed, "detail": str(detail)})
    if not passed:
        RESULT["errors"].append(f"{name}: {detail or 'check failed'}")
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def select_only(obj):
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def cube(name, location):
    bpy.ops.mesh.primitive_cube_add(size=0.4, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def run():
    import addon_utils
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51.blender import scene_bridge
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    check("temporary addon active", addon_utils.check("fluidSim_warp_blender51")[1])
    check("active blend is dam_break", Path(bpy.data.filepath).name == "dam_break.blend",
          bpy.data.filepath)
    panel_ids = (
        "FLUIDSIM_PT_liquid_emitters", "FLUIDSIM_PT_liquid_emitter_attributes",
        "FLUIDSIM_PT_liquid_sinks", "FLUIDSIM_PT_colliders", "FLUIDSIM_PT_forces",
        "FLUIDSIM_PT_domain_height_fill", "FLUIDSIM_PT_point_particle_injection",
        "FLUIDSIM_PT_advanced_field_sources",
    )
    check("all role panels registered", all(hasattr(bpy.types, item) for item in panel_ids))
    check("mixed legacy source panel absent", not hasattr(bpy.types, "FLUIDSIM_PT_sources"))

    scene = bpy.context.scene
    specs = [
        ("Emitter", (0.0, 0.0, 1.5), "fluidsim.tag_liquid_emitter", "SOURCE"),
        ("Sink", (0.6, 0.0, 1.5), "fluidsim.tag_liquid_sink", "SINK"),
        ("Collider", (1.2, 0.0, 1.5), "fluidsim.tag_collider", "COLLIDER"),
        ("Force", (1.8, 0.0, 1.5), "fluidsim.tag_force_region", "FORCE"),
        ("Field", (2.4, 0.0, 1.5), "fluidsim.tag_field_source", "FIELD"),
    ]
    created = {}
    for label, location, operator_id, role in specs:
        obj = cube(f"P3 Review {label}", location)
        select_only(obj)
        result = getattr(bpy.ops.fluidsim, operator_id.split(".")[-1])()
        created[label] = obj
        RESULT["roles"][label] = {"result": sorted(result), "role": obj.fluidsim.role}
        check(f"create {label} role", result == {"FINISHED"} and obj.fluidsim.role == role,
              RESULT["roles"][label])

    emitter = created["Emitter"]
    check("new emitter defaults Initial Fill", emitter.fluidsim.emission_mode == "INITIAL_FILL")
    emitter.fluidsim.source_region_kind = "MESH_SOLID"
    emitter.fluidsim.mesh_object = emitter
    created["Sink"].fluidsim.sink_region_kind = "MESH_SHELL"
    created["Sink"].fluidsim.mesh_object = created["Sink"]
    created["Force"].fluidsim.force_type = "DIRECTIONAL"
    created["Field"].fluidsim.field_operation = "PRESSURE"
    created["Field"].fluidsim.field_scalar_value = 1.0

    image = bpy.data.images.new("P3 Review Height Profile", width=2, height=2)
    image.pixels[:] = [0,0,0,1, 1,1,1,1, 0.5,0.5,0.5,1, 0.25,0.25,0.25,1]
    scene.fluidsim.height_enabled = True
    scene.fluidsim.height_back_mode = "PROFILE"
    scene.fluidsim.height_profile_image = image
    height = SceneInputs().gather_height_source(scene)
    check("Height Profile uses real Image input", height is not None and height.profile.shape == (2, 2))
    # Role creation proof must not replace the saved dam-break source as the
    # playback workload. Keep the temporary collider visible but disable every
    # temporary source/field/force and the temporary height fill.
    for label in ("Emitter", "Sink", "Force", "Field"):
        created[label].fluidsim.enabled = False
    scene.fluidsim.height_enabled = False

    if scene_bridge.session().active:
        scene_bridge.release_session(scene)
    initialize = bpy.ops.fluidsim.initialize()
    check("initialize from role UI state", initialize == {"FINISHED"}, scene.fluidsim.status)
    session = scene_bridge.session()
    start_frame = session.solver.frame if session.active else -1
    step_result = bpy.ops.fluidsim.step(frames=3)
    end_frame = session.solver.frame if session.active else -1
    RESULT["playback"] = {
        "step_result": sorted(step_result), "start_frame": start_frame,
        "end_frame": end_frame, "status": scene.fluidsim.status,
        "particles": session.solver.particles.host_count if session.active else 0,
    }
    check("multi-frame playback advances", step_result == {"FINISHED"}
          and end_frame >= start_frame + 3, RESULT["playback"])
    check("Blender remains responsive after playback", session.active)
    RESULT["ok"] = not RESULT["errors"]


def finish():
    try:
        from fluidSim_warp_blender51.blender import scene_bridge
        scene_bridge.release_session(bpy.context.scene)
    except Exception as exc:
        RESULT["errors"].append(f"release failed: {exc}")
    RESULT["ok"] = bool(not RESULT["errors"])
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(RESULT, indent=2, default=str), encoding="utf8")
    print("PHASE3-PROFILE-SAFE-REVIEW-PASS" if RESULT["ok"] else "PHASE3-PROFILE-SAFE-REVIEW-FAIL")
    bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender(), first_interval=0.1)


try:
    run()
except Exception as exc:
    RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
    RESULT["traceback"] = traceback.format_exc()
finally:
    finish()
