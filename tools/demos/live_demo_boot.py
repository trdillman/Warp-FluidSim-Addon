"""Runs inside Blender for tools/demos/live_demo.py: set the demo up and press Play.

It checks the add-on that loaded is the one installed in the temporary profile,
switches Live Simulation on, cache playback off, the Liquid Deck on, frames the
viewport and starts the timeline. It then writes a small status file every few
seconds so a run can be checked without watching the window.
"""

import json
import os
import time
import traceback
from pathlib import Path

import addon_utils
import bpy

CONFIG = json.loads(Path(os.environ["FLUIDSIM_LIVE_DEMO_CONFIG"]).read_text(encoding="utf-8"))
REPORT = Path(CONFIG["report"])
PROFILE = Path(CONFIG["profile"]).resolve()
HEARTBEAT_SECONDS = 5.0
MAX_WAIT_FOR_WINDOW = 60.0

status = {"stage": "start", "ok": False, "demo": CONFIG["demo"], "heartbeats": 0}
state = {"started": time.time(), "playing": False}


def write():
    status["updated"] = time.strftime("%H:%M:%S")
    try:
        REPORT.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass


def find_addon_module():
    suffix = "." + CONFIG["extension_id"]
    for addon in bpy.context.preferences.addons:
        if addon.module.endswith(suffix) or addon.module == CONFIG["extension_id"]:
            return addon.module
    return None


def view_area():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is not None:
                    return window, area, region
    return None


def prepare(window, area, region):
    module_name = find_addon_module()
    status["addon_module"] = module_name
    module = __import__(module_name, fromlist=["x"]) if module_name else None
    path = Path(getattr(module, "__file__", "") or "").resolve() if module else None
    status["addon_path"] = str(path) if path else None
    # The add-on that loaded must be the fresh copy in this temporary profile.
    status["addon_in_temp_profile"] = bool(path) and PROFILE in path.parents
    if not status["addon_in_temp_profile"]:
        raise RuntimeError(f"add-on loaded from {path}, not from the temporary profile")
    try:
        version = __import__(module_name + ".version", fromlist=["x"])
        status["fluidsim_version"] = version.VERSION_STRING
    except Exception as exc:  # noqa: BLE001
        status["fluidsim_version"] = f"unknown ({exc!r})"
    for key, value in CONFIG["expected_env"].items():
        status.setdefault("env", {})[key] = os.environ.get(key)
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} must be {value!r} for this demo")

    prefs = bpy.context.preferences.addons[module_name].preferences
    if hasattr(prefs, "deck_mode"):
        prefs.deck_mode = "ON"
    scene = bpy.context.scene
    fs = scene.fluidsim
    # Real-time solver only: no cache writing, no cache replay.
    fs.cache_enabled = False
    fs.cache_playback_enabled = False
    fs.deck_shown = True
    status["scene"] = scene.name
    status["frame_start"] = scene.frame_start
    status["frame_end"] = scene.frame_end

    space = area.spaces.active
    space.shading.type = CONFIG["shading"]
    space.show_region_ui = True
    if CONFIG["camera_view"] and scene.camera is not None:
        space.region_3d.view_perspective = "CAMERA"
    scene.frame_set(scene.frame_start)
    fs.live = True
    status["live"] = bool(fs.live)
    status["cache_playback_enabled"] = bool(fs.cache_playback_enabled)
    status["deck_shown"] = bool(fs.deck_shown)
    status["deck_mode"] = getattr(prefs, "deck_mode", None)
    with bpy.context.temp_override(window=window, area=area, region=region):
        bpy.ops.screen.animation_play()
    state["playing"] = True
    status["stage"] = "playing"
    status["ok"] = bool(fs.live) and not fs.cache_playback_enabled and fs.deck_shown


def heartbeat():
    scene = bpy.context.scene
    fs = scene.fluidsim
    status["heartbeats"] += 1
    status["frame_current"] = scene.frame_current
    status["is_playing"] = bool(bpy.context.screen.is_animation_playing)
    status["particles"] = int(fs.stat_particles)
    status["step_ms"] = float(fs.stat_step_ms)
    status["fluidsim_status"] = fs.status
    status["live"] = bool(fs.live)
    status["elapsed_s"] = round(time.time() - state["started"], 1)
    frames = status.setdefault("frames_seen", [])
    if not frames or frames[-1] != scene.frame_current:
        frames.append(scene.frame_current)
        del frames[:-200]
    write()


def tick():
    try:
        if not state["playing"]:
            found = view_area()
            if found is None:
                if time.time() - state["started"] > MAX_WAIT_FOR_WINDOW:
                    raise RuntimeError("no 3D Viewport appeared")
                return 1.0
            prepare(*found)
            write()
            return 2.0
        heartbeat()
    except Exception as exc:  # noqa: BLE001
        status["stage"] = "error"
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write()
        print("FLUIDSIM-LIVE-DEMO ERROR", status["error"], flush=True)
        return None
    return HEARTBEAT_SECONDS


addon_utils.modules_refresh()
bpy.app.timers.register(tick, first_interval=2.0, persistent=True)
print("FLUIDSIM-LIVE-DEMO armed for", CONFIG["demo"], flush=True)
