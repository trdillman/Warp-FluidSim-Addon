"""GUI acceptance (B29): open another .blend while a modal Production Bake runs.

The review suspected that load_pre releases the Solver Session but not the bake,
so the modal cancel could rebuild a solver for the outgoing scene and leave the
new file unable to Play/Step. On Blender 5.2 the load ends the bake cleanly. This
pins it: after the load no bake is active, no session survives for the old
scene, and Initialize + Step work in the new file.

Needs a GUI Blender (a modal bake does not run in --background); it drives
itself from a persistent timer, writes a JSON report and quits:
  blender --factory-startup --python-use-system-env --python tests/acceptance/blender_load_during_bake_acceptance.py
Prints LOAD-DURING-BAKE-ACCEPTANCE-PASS when every check passes.
"""
import json, os, sys, tempfile, time, traceback
from pathlib import Path
import bpy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("FLUIDSIM_LOAD_DURING_BAKE_REPORT", str(REPO / "artifacts" / "load_during_bake_acceptance.json")))
ROOT = Path(tempfile.mkdtemp(prefix="b29-"))
STATE = {"phase": "setup", "report": {"checks": {}, "events": []}, "t0": time.monotonic()}

def ev(msg):
    STATE["report"]["events"].append(f"{time.monotonic()-STATE['t0']:.2f} {msg}")
    print("B29", msg, flush=True)

def session_info():
    from fluidSim_warp_blender51.blender import scene_bridge, production_bake
    sess = scene_bridge.session()
    scene = bpy.context.scene
    owner = getattr(sess, "scene_key", None)
    return {
        "active": bool(sess.active),
        "owns_current": bool(sess.owns_scene(scene)) if sess.active else False,
        "owner": repr(owner),
        "active_bakes": len(production_bake.active_bakes()),
        "any_bake_active": production_bake.any_bake_active(),
        "current_scene": scene.name,
        "file": bpy.data.filepath,
    }

def make_file(path, name):
    bpy.ops.wm.read_homefile(use_empty=False)
    bpy.context.scene.name = name
    cube = bpy.data.objects.get("Cube")
    cube.select_set(True); bpy.context.view_layer.objects.active = cube
    window = bpy.context.window_manager.windows[0]
    area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    with bpy.context.temp_override(window=window, area=area, active_object=cube, selected_objects=[cube]):
        print("SETUP", bpy.ops.fluidsim.setup_scene(), flush=True)
    from fluidSim_warp_blender51.blender import scene_bridge
    dom = scene_bridge.active_domain_settings(bpy.context.scene)
    dom.voxel_size = 0.12; dom.whitewater_enabled = False; dom.surface_enabled = False
    fs = bpy.context.scene.fluidsim
    fs.cache_path = str(ROOT / f"cache_{name}")
    fs.bake_start = 1; fs.bake_end = 400; fs.checkpoint_interval = 1; fs.live = False
    bpy.ops.wm.save_as_mainfile(filepath=str(path))

def finish():
    rep = STATE["report"]
    after, init = rep.get("after_load", {}), rep.get("after_init", {})
    checks = {
        "bake was running before the load": rep.get("before_load", {}).get("active_bakes") == 1,
        "no bake is active after the load": after.get("active_bakes") == 0 and not after.get("any_bake_active", True),
        "no solver survives for the outgoing scene": after.get("active") is False,
        "Initialize works in the new file": rep.get("initialize_in_new_file") == "{'FINISHED'}",
        "Step works in the new file": rep.get("step_in_new_file") == "{'FINISHED'}",
        "the new scene owns the session": init.get("owns_current") is True and init.get("current_scene") == "SceneB",
    }
    rep["checks"] = checks
    passed = all(checks.values()) and "error" not in rep
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    print("LOAD-DURING-BAKE-ACCEPTANCE-PASS" if passed else "LOAD-DURING-BAKE-ACCEPTANCE-FAIL", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(STATE["report"], indent=2, default=str), encoding="utf-8")
    print("report:", OUT, flush=True)
    bpy.ops.wm.quit_blender()
    return None

def tick():
    from fluidSim_warp_blender51.blender import production_bake, scene_bridge
    try:
        if time.monotonic() - STATE["t0"] > 300:
            ev("timeout"); return finish()
        phase = STATE["phase"]
        if phase == "start_bake":
            window = bpy.context.window_manager.windows[0]
            area = next(a for a in window.screen.areas if a.type == "VIEW_3D")
            region = next(r for r in area.regions if r.type == "WINDOW")
            with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
                result = bpy.ops.fluidsim.bake_start("EXEC_DEFAULT", restart_existing=False)
            ev(f"bake_start -> {result}")
            STATE["phase"] = "wait_frames"; return 0.1
        if phase == "wait_frames":
            bakes = production_bake.active_bakes()
            if not bakes:
                ev("bake vanished before load"); STATE["report"]["error"] = "no bake"; return finish()
            ctrl = bakes[0][1]
            last = getattr(getattr(ctrl, "backend", None), "manifest", None)
            last = None if last is None else last.last_committed_frame
            if last is not None and last >= 3:
                STATE["report"]["before_load"] = session_info()
                STATE["report"]["before_load"]["last_committed"] = last
                ev(f"loading B while baking frame {last}")
                STATE["phase"] = "after_load"
                bpy.ops.wm.open_mainfile(filepath=str(STATE["file_b"]))
                return 0.5
            return 0.1
        if phase == "after_load":
            STATE.setdefault("after_ticks", 0)
            STATE["after_ticks"] += 1
            info = session_info()
            STATE["report"].setdefault("after_load_samples", []).append(info)
            if STATE["after_ticks"] < 8:
                return 0.5
            STATE["report"]["after_load"] = info
            scene = bpy.context.scene
            try:
                init = bpy.ops.fluidsim.initialize()
            except Exception as exc:
                init = f"{type(exc).__name__}: {exc}"
            STATE["report"]["initialize_in_new_file"] = str(init)
            try:
                step = bpy.ops.fluidsim.step(frames=1)
            except Exception as exc:
                step = f"{type(exc).__name__}: {exc}"
            STATE["report"]["step_in_new_file"] = str(step)
            STATE["report"]["after_init"] = session_info()
            STATE["report"]["status_text"] = str(scene.fluidsim.status)
            return finish()
    except Exception:
        STATE["report"]["error"] = traceback.format_exc()
        ev("exception"); return finish()

def boot():
    try:
        from tools.repo_addon_loader import enable_repo_addon
        enable_repo_addon(str(REPO), default_set=True, persistent=False)
        file_b = ROOT / "b.blend"; file_a = ROOT / "a.blend"
        make_file(file_b, "SceneB")
        make_file(file_a, "SceneA")
        STATE["file_b"] = file_b
        STATE["phase"] = "start_bake"
        bpy.app.timers.register(tick, first_interval=1.0, persistent=True)
    except Exception:
        STATE["report"]["error"] = traceback.format_exc(); finish()
    return None

bpy.app.timers.register(boot, first_interval=2.0, persistent=True)
