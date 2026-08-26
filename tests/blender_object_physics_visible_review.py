"""Open the Chromatic Vortex with an emitter's Physics Properties visible."""

from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

import bpy


RESULT_PATH = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"]).resolve()
RESULT = {
    "schema": "fluidsim-object-physics-visible-review-v1",
    "ok": False,
    "checks": [],
    "errors": [],
}
DEFERRED = False


def check(name, condition, detail=""):
    passed = bool(condition)
    RESULT["checks"].append({
        "name": name, "passed": passed, "detail": str(detail)})
    if not passed:
        RESULT["errors"].append(f"{name}: {detail or 'check failed'}")
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)


try:
    import addon_utils
    import fluidSim_warp_blender51 as addon
    from fluidSim_warp_blender51 import panels

    check("temporary addon active", addon_utils.check(addon.__name__)[1])
    check("chromatic demo loaded", Path(bpy.data.filepath).name == "chromatic_vortex.blend")
    panel = panels.FLUIDSIM_PT_object_physics
    check("Physics Properties panel registered",
          hasattr(bpy.types, panel.__name__)
          and panel.bl_space_type == "PROPERTIES"
          and panel.bl_region_type == "WINDOW"
          and panel.bl_context == "physics")

    emitter = bpy.data.objects.get("Red Dye Initial Fill")
    check("review emitter exists", emitter is not None)
    if emitter is not None:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        emitter.select_set(True)
        bpy.context.view_layer.objects.active = emitter
        check("emitter panel poll succeeds", panel.poll(bpy.context))

    screen = bpy.context.window.screen
    viewports = [area for area in screen.areas if area.type == "VIEW_3D"]
    check("review has a 3D viewport", bool(viewports))
    if viewports:
        viewport = max(viewports, key=lambda area: area.width * area.height)
        before = {area.as_pointer() for area in screen.areas}
        with bpy.context.temp_override(
                window=bpy.context.window, screen=screen, area=viewport):
            split = bpy.ops.screen.area_split(direction="VERTICAL", factor=0.68)
        check("Physics editor area created", split == {"FINISHED"}, split)
        created = [area for area in screen.areas if area.as_pointer() not in before]
        properties_area = created[0] if created else min(
            screen.areas, key=lambda area: area.width * area.height)
        properties_area.type = "PROPERTIES"

        def finish_properties_context():
            try:
                properties_area.spaces.active.context = "PHYSICS"
                check("Physics context visible",
                      properties_area.type == "PROPERTIES"
                      and properties_area.spaces.active.context == "PHYSICS")
            except Exception as exc:
                RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
                RESULT["traceback"] = traceback.format_exc()
            RESULT["ok"] = not RESULT["errors"]
            RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULT_PATH.write_text(
                json.dumps(RESULT, indent=2), encoding="utf-8")
            print(
                "OBJECT-PHYSICS-VISIBLE-REVIEW-PASS"
                if RESULT["ok"] else "OBJECT-PHYSICS-VISIBLE-REVIEW-FAIL",
                flush=True,
            )
            return None

        bpy.app.timers.register(finish_properties_context, first_interval=0.5)
        DEFERRED = True

    bpy.context.scene["fluidsim_object_physics_panel_verified"] = True
except Exception as exc:
    RESULT["errors"].append(f"{type(exc).__name__}: {exc}")
    RESULT["traceback"] = traceback.format_exc()

if not DEFERRED:
    RESULT["ok"] = not RESULT["errors"]
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(RESULT, indent=2), encoding="utf-8")
    print(
        "OBJECT-PHYSICS-VISIBLE-REVIEW-PASS"
        if RESULT["ok"] else "OBJECT-PHYSICS-VISIBLE-REVIEW-FAIL",
        flush=True,
    )
    if not RESULT["ok"]:
        raise RuntimeError("object Physics Properties visible review failed")
