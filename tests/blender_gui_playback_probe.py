"""Drive real Blender timeline playback and report solver advancement.

Run without --background. The process closes itself after writing the report.
"""

from __future__ import annotations

import json
import os
import sys
import time

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

blend_path = os.environ["FLUIDSIM_PLAYBACK_BLEND"]
report_path = os.environ["FLUIDSIM_PLAYBACK_REPORT"]
enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
bpy.ops.wm.open_mainfile(filepath=blend_path)

from fluidSim_warp_blender51.blender.scene_bridge import session

scene = bpy.context.scene
start_frame = scene.frame_current
started_at = time.monotonic()
report = {
    "blend": bpy.data.filepath,
    "start_frame": start_frame,
    "handler_count": len([
        fn for fn in bpy.app.handlers.frame_change_post
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51")
    ]),
}


def finish():
    sess = session()
    report.update({
        "end_frame": scene.frame_current,
        "animation_playing": bool(bpy.context.screen and bpy.context.screen.is_animation_playing),
        "session_active": sess.active,
        "solver_frame": sess.solver.frame if sess.active else None,
        "status": scene.fluidsim.status,
        "error": sess.error,
        "elapsed_seconds": time.monotonic() - started_at,
    })
    report["passed"] = bool(
        report["handler_count"] == 1
        and report["end_frame"] > report["start_frame"]
        and report["session_active"]
        and report["solver_frame"] is not None
        and report["solver_frame"] > 0
        and not report["error"]
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if bpy.context.screen and bpy.context.screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    bpy.ops.wm.quit_blender()
    return None


def poll():
    sess = session()
    advanced = scene.frame_current > start_frame and sess.active and sess.solver.frame > 0
    if advanced or time.monotonic() - started_at > 45.0:
        return finish()
    return 0.25


def start():
    window = bpy.context.window_manager.windows[0]
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        report["start_error"] = "No VIEW_3D area"
        return finish()
    region = next(item for item in area.regions if item.type == "WINDOW")
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.animation_play()
    bpy.app.timers.register(poll, first_interval=0.25)
    return None


bpy.app.timers.register(start, first_interval=1.0)
