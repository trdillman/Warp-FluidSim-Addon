"""Real GUI modal-bake cleanup when its disposable owner scene is deleted."""
from __future__ import annotations
import json, os, shutil, subprocess, tempfile, time, traceback
from pathlib import Path
import bpy
STATE = {}
def _active_controller():
    """The modal operator's registered running bake, or None."""
    from fluidSim_warp_blender51.blender import production_bake
    bakes = production_bake.active_bakes()
    return bakes[0][1] if bakes else None
def _snapshot(scene):
    fs = scene.fluidsim
    objects = tuple(sorted(
        (obj.name, obj.type, len(getattr(obj.data, "vertices", ())),
         len(getattr(obj.data, "polygons", ())))
        for obj in scene.objects
    ))
    return {
        "frame": int(scene.frame_current),
        "settings": (
            str(fs.cache_path), int(fs.bake_start), int(fs.bake_end),
            str(fs.bake_status), float(fs.bake_progress), str(fs.bake_message),
            str(fs.status), bool(fs.live), bool(fs.cache_playback_enabled),
        ),
        "objects": objects,
    }
def _check(name, condition, detail=None):
    if not condition:
        raise AssertionError(f"{name}: {detail}" if detail is not None else name)
    STATE["report"]["checks"].append(name)
def _write():
    path = STATE["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(STATE["report"], indent=2, default=str), encoding="utf-8")
def _finish(passed):
    try:
        window, source = STATE.get("window"), STATE.get("source")
        if window is not None and source is not None:
            window.scene = source
        owned = STATE.get("owned")
        if owned is not None and owned.name in bpy.data.scenes:
            bpy.data.scenes.remove(owned)
        root = STATE["temp_root"]
        if root.exists():
            shutil.rmtree(root)
        if root.exists():
            raise RuntimeError(f"temporary cache still exists: {root}")
        STATE["report"]["temporary_cache_removed"] = True
    except Exception:  # cleanup failure invalidates a nominal pass
        passed = False
        STATE["report"]["temporary_cache_removed"] = False
        STATE["report"]["cleanup_error"] = traceback.format_exc()
    STATE["report"]["status"] = "PASS" if passed else "FAIL"
    _write()
    print("ARTIST-BAKE-DELETED-SCENE-ACCEPTANCE-PASS" if passed else
          STATE["report"].get("error", STATE["report"].get("cleanup_error")))
    STATE["phase"] = "done"
    return None
def _fail(exc):
    STATE["report"]["error"] = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__))
    controller = _active_controller()
    if controller is not None:
        controller.request_cancel()
        STATE["phase"] = "fail_wait"
        STATE["fail_deadline"] = time.monotonic() + 30.0
        return 0.1
    return _finish(False)
def start(report_path):
    from fluidSim_warp_blender51.blender import effects, scene_bridge
    if STATE.get("phase") not in {None, "done"}:
        raise RuntimeError("Deleted-scene acceptance is already running")
    if _active_controller() is not None:
        raise RuntimeError("A Production Bake is already active")
    source = bpy.context.scene
    if scene_bridge.active_domain_settings(source) is None:
        raise RuntimeError("The active scene is not a runnable FluidSim scene")
    if scene_bridge.session().active:
        raise RuntimeError("The source scene must not have an active solver session")
    window = next((w for w in bpy.context.window_manager.windows if w.scene == source), None)
    if window is None:
        raise RuntimeError("No Blender window is showing the source scene")
    area = next((a for a in window.screen.areas if a.type == "VIEW_3D"), None)
    region = next((r for r in area.regions if r.type == "WINDOW"), None) if area else None
    if area is None or region is None:
        raise RuntimeError("The source window needs a 3D View")
    root = Path(tempfile.mkdtemp(prefix="fluidsim-bake-deleted-scene-"))
    cache_path = (root / "cache").resolve()
    repo = Path(__file__).resolve().parents[1]
    source_before = _snapshot(source)
    owned = source.copy()
    owned.name = "FluidSim Disposable Bake Owner"
    owned.fluidsim.cache_path = str(cache_path)
    owned.fluidsim.cache_compress = False
    owned.fluidsim.checkpoint_interval = 1
    owned.fluidsim.bake_start = int(owned.frame_start)
    owned.fluidsim.bake_end = int(owned.frame_start) + 1000
    STATE.clear()
    STATE.update(
        phase="launch", path=Path(report_path).resolve(), source=source,
        source_before=source_before, owned=owned, window=window, area=area,
        region=region, temp_root=root, cache_path=cache_path,
        deadline=time.monotonic() + 240.0,
        report={"status": "RUNNING", "source_scene": source.name,
                "owned_scene": owned.name, "cache_path": str(cache_path),
                "addon_source": str(Path(effects.__file__).resolve()),
                "pid": os.getpid(), "blender": bpy.app.version_string,
                "head": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                "checks": []},
    )
    bpy.app.timers.register(_tick, first_interval=0.1)
    return {"scheduled": True, "report": str(STATE["path"])}
def _tick():
    from fluidSim_warp_blender51.blender import production_bake
    try:
        phase = STATE["phase"]
        if phase == "fail_wait":
            if _active_controller() is None:
                return _finish(False)
            if time.monotonic() >= STATE["fail_deadline"]:
                count = int(STATE["report"].get("cleanup_wait_reports", 0)) + 1
                STATE["report"].update(
                    status="FAIL", cleanup_pending=True, cleanup_wait_reports=count)
                _write()
                print(STATE["report"]["error"])
                STATE["fail_deadline"] = time.monotonic() + 30.0
                return 0.5
            return 0.1
        _check("acceptance stayed within deadline", time.monotonic() < STATE["deadline"])
        if phase == "launch":
            window = STATE["window"]
            window.scene = STATE["owned"]
            with bpy.context.temp_override(window=window, screen=window.screen,
                                           area=STATE["area"], region=STATE["region"]):
                result = bpy.ops.fluidsim.bake_start("EXEC_DEFAULT", restart_existing=False)
            _check("real modal bake started", result == {"RUNNING_MODAL"}, result)
            _check("temporary cache reserved", production_bake.cache_path_is_active(STATE["cache_path"]))
            STATE["phase"] = "wait_running"
            return 0.05
        controller = _active_controller()
        if phase == "wait_running":
            _check("modal controller remains active", controller is not None)
            if controller.status.value != "RUNNING":
                return 0.05
            STATE["backend_session"] = controller.backend.sess
            STATE["window"].scene = STATE["source"]
            bpy.data.scenes.remove(STATE["owned"])
            STATE["owned"] = None
            STATE["phase"] = "wait_cleanup"
            return 0.05
        if phase == "wait_cleanup":
            if controller is not None:
                return 0.05
            _check("active bake released", not production_bake.active_bakes())
            _check("cache reservation released", not production_bake.cache_path_is_active(STATE["cache_path"]))
            _check("backend session released", not STATE["backend_session"].active)
            _check("original scene unchanged", _snapshot(STATE["source"]) == STATE["source_before"])
            STATE["report"]["owner_deleted"] = True
            return _finish(True)
        raise AssertionError(f"unknown phase: {phase}")
    except Exception as exc:  # failure becomes an actionable persisted receipt
        return _fail(exc)
