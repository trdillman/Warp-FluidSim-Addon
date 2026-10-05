"""Timer-driven GUI regression for modal Production Bake scene ownership.

Call ``start(report_path)`` from an isolated, source-loaded Blender GUI session
whose active scene is a runnable FluidSim demo.  The probe never saves the
blend, reloads the add-on, changes the user profile, or quits Blender.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import bpy


STATE: dict[str, object] = {}
_SCENE_FIELDS = (
    "cache_path",
    "cache_compress",
    "checkpoint_interval",
    "bake_start",
    "bake_end",
    "bake_status",
    "bake_progress",
    "bake_message",
    "status",
    "live",
    "cache_playback_enabled",
)
_DOMAIN_FIELDS = (
    "voxel_size",
    "max_particles",
    "max_whitewater",
    "whitewater_enabled",
    "surface_enabled",
)


def _simple_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _field_snapshot(owner, names):
    return {
        name: _simple_value(getattr(owner, name))
        for name in names
        if hasattr(owner, name)
    }


def _restore_fields(owner, values):
    for name, value in values.items():
        setattr(owner, name, value)


def _scene_b_snapshot(scene):
    """Capture the settings and object payload that modal bake must not touch."""
    objects = []
    for obj in scene.objects:
        data = getattr(obj, "data", None)
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "vertices": len(getattr(data, "vertices", ())),
                "polygons": len(getattr(data, "polygons", ())),
                "probe": _simple_value(obj.get("artist_bake_probe")),
            }
        )
    return {
        "frame": int(scene.frame_current),
        "settings": _field_snapshot(scene.fluidsim, _SCENE_FIELDS),
        "objects": sorted(objects, key=lambda item: item["name"]),
    }


def _check(name, condition, detail=None):
    if not condition:
        suffix = "" if detail is None else f": {detail}"
        raise AssertionError(f"{name}{suffix}")
    STATE["report"]["checks"].append(name)


def _write_report():
    path = STATE["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(STATE["report"], indent=2, default=str), encoding="utf-8"
    )


def _remove_probe_scene():
    scene_a = STATE.get("scene_a")
    scene_b = STATE.get("scene_b")
    window = STATE.get("window")
    marker = STATE.get("marker")
    marker_mesh = STATE.get("marker_mesh")
    if window is not None and scene_a is not None:
        window.scene = scene_a
    if marker is not None and marker.name in bpy.data.objects:
        bpy.data.objects.remove(marker, do_unlink=True)
    if scene_b is not None and scene_b.name in bpy.data.scenes:
        bpy.data.scenes.remove(scene_b)
    if marker_mesh is not None and marker_mesh.name in bpy.data.meshes:
        bpy.data.meshes.remove(marker_mesh)


def _cleanup():
    scene_a = STATE.get("scene_a")
    domain = STATE.get("domain")
    if scene_a is not None:
        _restore_fields(scene_a.fluidsim, STATE.get("scene_a_original", {}))
        scene_a.frame_set(int(STATE.get("scene_a_frame", scene_a.frame_start)))
    if domain is not None:
        _restore_fields(domain, STATE.get("domain_original", {}))
    _remove_probe_scene()
    temp_root = STATE.get("temp_root")
    if temp_root is not None and temp_root.exists():
        shutil.rmtree(temp_root)
    if temp_root is not None and temp_root.exists():
        raise RuntimeError(f"Temporary cache cleanup did not remove {temp_root}")


def _finish(passed):
    try:
        _cleanup()
        STATE["report"]["temporary_cache_removed"] = True
    except Exception:  # noqa: BLE001 - cleanup failure invalidates the receipt
        passed = False
        STATE["report"]["temporary_cache_removed"] = False
        STATE["report"]["cleanup_error"] = traceback.format_exc()
    STATE["report"]["status"] = "PASS" if passed else "FAIL"
    _write_report()
    if passed:
        print("ARTIST-BAKE-GUI-ACCEPTANCE-PASS")
    else:
        print(STATE["report"].get("error", "ARTIST-BAKE-GUI-ACCEPTANCE-FAIL"))
    return None


def _begin_failure(exc):
    from fluidSim_warp_blender51.operator_domains import cache as cache_operators

    STATE["report"]["error"] = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    STATE["report"]["scene_b_before"] = STATE.get("scene_b_before")
    if STATE.get("scene_b") is not None:
        STATE["report"]["scene_b_at_failure"] = _scene_b_snapshot(STATE["scene_b"])
    active = cache_operators._ACTIVE_BAKE
    if active is not None:
        active.request_cancel()
        STATE["phase"] = "failure_wait"
        STATE["failure_deadline"] = time.monotonic() + 30.0
        return 0.1
    return _finish(False)


def start(report_path):
    """Schedule the real modal bake ownership probe in the current GUI."""
    from fluidSim_warp_blender51.blender import effects, production_bake, scene_bridge
    from fluidSim_warp_blender51.operator_domains import cache as cache_operators

    if STATE.get("phase") not in {None, "done"}:
        raise RuntimeError("Artist bake GUI acceptance is already scheduled")
    if cache_operators._ACTIVE_BAKE is not None:
        raise RuntimeError("A Production Bake is already active")

    scene_a = bpy.context.scene
    domain = scene_bridge.active_domain_settings(scene_a)
    if domain is None or scene_a.fluidsim.domain_object is None:
        raise RuntimeError("The active scene is not a runnable FluidSim demo")
    window = next(
        (item for item in bpy.context.window_manager.windows if item.scene == scene_a),
        None,
    )
    if window is None:
        raise RuntimeError("No Blender window is showing the active FluidSim scene")
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("The FluidSim window needs a 3D View")
    region = next((item for item in area.regions if item.type == "WINDOW"), None)
    if region is None:
        raise RuntimeError("The 3D View has no window region")

    temp_root = Path(tempfile.mkdtemp(prefix="fluidsim-artist-bake-gui-"))
    cache_path = (temp_root / "cache").resolve()
    scene_b = bpy.data.scenes.new("FluidSim Bake Ownership Control")
    marker_mesh = bpy.data.meshes.new("FluidSim Bake Control Mesh")
    marker_mesh.from_pydata(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    marker = bpy.data.objects.new("FluidSim Bake Control Output", marker_mesh)
    marker["artist_bake_probe"] = "must remain unchanged"
    scene_b.collection.objects.link(marker)

    repo = Path(__file__).resolve().parents[2]
    report = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "blend": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(Path(effects.__file__).resolve()),
        "scene_a": scene_a.name,
        "scene_b": scene_b.name,
        "cache_path": str(cache_path),
        "checks": [],
        "scene_deletion_gui_case": (
            "not run: preserve the source demo; removed-scene cleanup has a unit sibling"
        ),
    }
    STATE.clear()
    STATE.update(
        phase="launch",
        path=Path(report_path).resolve(),
        report=report,
        deadline=time.monotonic() + 240.0,
        scene_a=scene_a,
        scene_b=scene_b,
        window=window,
        area=area,
        region=region,
        domain=domain,
        marker=marker,
        marker_mesh=marker_mesh,
        temp_root=temp_root,
        cache_path=cache_path,
        scene_a_original=_field_snapshot(scene_a.fluidsim, _SCENE_FIELDS),
        domain_original=_field_snapshot(domain, _DOMAIN_FIELDS),
        scene_a_frame=int(scene_a.frame_current),
        scene_b_before=_scene_b_snapshot(scene_b),
    )

    with bpy.context.temp_override(
        window=window, screen=window.screen, area=area, region=region
    ):
        if window.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
    if scene_bridge.session().owns_scene(scene_a):
        scene_bridge.release_session(scene_a)
    fs = scene_a.fluidsim
    fs.live = False
    fs.cache_playback_enabled = False
    fs.cache_path = str(cache_path)
    fs.cache_compress = False
    fs.checkpoint_interval = 1
    fs.bake_start = int(scene_a.frame_start)
    fs.bake_end = int(scene_a.frame_start) + 1000
    domain.voxel_size = max(float(domain.voxel_size), 0.14)
    domain.max_particles = min(int(domain.max_particles), 50_000)
    domain.max_whitewater = 1
    domain.whitewater_enabled = False
    domain.surface_enabled = False

    _check("temporary cache is outside the blend", cache_path.is_absolute())
    _check("scene A has no active bake before launch", not production_bake.bake_is_active(scene_a))
    bpy.app.timers.register(_tick, first_interval=0.1)
    return {"scheduled": True, "report": str(STATE["path"])}


def _tick():
    from fluidSim_warp_blender51.blender import production_bake
    from fluidSim_warp_blender51.operator_domains import cache as cache_operators

    scene_a = STATE["scene_a"]
    scene_b = STATE["scene_b"]
    window = STATE["window"]
    try:
        phase = STATE["phase"]
        if phase == "failure_wait":
            if cache_operators._ACTIVE_BAKE is None:
                return _finish(False)
            if time.monotonic() >= STATE["failure_deadline"]:
                STATE["report"]["error"] += "\nTimed out waiting for failed bake cleanup"
                STATE["report"]["status"] = "FAIL"
                STATE["report"]["cleanup_pending"] = True
                STATE["report"]["temporary_cache_removed"] = False
                _write_report()
                print(STATE["report"]["error"])
                # Keep observing the owner's cancellation rather than abandon
                # an active controller and its cache reservation in this GUI.
                STATE["failure_deadline"] = time.monotonic() + 30.0
                return 0.5
            return 0.1
        _check("acceptance stayed within its deadline", time.monotonic() < STATE["deadline"])

        if phase == "launch":
            with bpy.context.temp_override(
                window=window,
                screen=window.screen,
                area=STATE["area"],
                region=STATE["region"],
            ):
                result = bpy.ops.fluidsim.bake_start(
                    "EXEC_DEFAULT", restart_existing=False
                )
            _check("real modal Production Bake started", result == {"RUNNING_MODAL"}, result)
            _check("scene A owns active bake", production_bake.bake_is_active(scene_a))
            _check("temporary cache path is reserved", production_bake.cache_path_is_active(STATE["cache_path"]))
            STATE["phase"] = "wait_running"
            return 0.05

        controller = cache_operators._ACTIVE_BAKE
        if phase == "wait_running":
            _check("modal bake remains active", controller is not None)
            if controller.status.value != "RUNNING":
                return 0.05
            STATE["committed_before_switch"] = controller.last_committed_frame
            STATE["progress_before_switch"] = float(controller.progress)
            STATE["bake_status_before_switch"] = scene_a.fluidsim.bake_status
            window.scene = scene_b
            _check("window switched to scene B", window.scene == scene_b)
            _check("scene B unchanged at switch", _scene_b_snapshot(scene_b) == STATE["scene_b_before"])
            STATE["phase"] = "wait_post_switch_commit"
            return 0.05

        if phase == "wait_post_switch_commit":
            _check("window remains on scene B", window.scene == scene_b)
            _check("scene B settings and output remain unchanged", _scene_b_snapshot(scene_b) == STATE["scene_b_before"])
            _check("scene A remains the registered bake owner", production_bake.active_bake(scene_a) is controller)
            before = STATE["committed_before_switch"]
            committed = controller.last_committed_frame
            advanced = committed is not None and (before is None or committed > before)
            if not advanced:
                return 0.05
            _check("scene A committed a frame after the context switch", advanced)
            _check("scene A received bake RNA after the switch", scene_a.fluidsim.bake_status == "RUNNING")
            _check("scene A progress advanced after the switch", float(controller.progress) > STATE["progress_before_switch"])
            STATE["report"]["scene_a_frame_after_switch"] = int(scene_a.frame_current)
            STATE["report"]["last_committed_frame"] = int(committed)
            STATE["report"]["cache_files_before_cancel"] = sorted(
                str(path.relative_to(STATE["cache_path"]))
                for path in STATE["cache_path"].rglob("*")
                if path.is_file()
            )
            result = bpy.ops.fluidsim.bake_cancel("EXEC_DEFAULT")
            _check("cancel requested from scene B", result == {"FINISHED"}, result)
            _check("cancel RNA was written to scene A", scene_a.fluidsim.bake_status == "CANCELLING")
            _check("cancel did not mutate scene B", _scene_b_snapshot(scene_b) == STATE["scene_b_before"])
            STATE["phase"] = "wait_cancelled"
            return 0.05

        if phase == "wait_cancelled":
            if cache_operators._ACTIVE_BAKE is not None:
                return 0.05
            _check("scene A active bake registration released", production_bake.active_bake(scene_a) is None)
            _check("temporary cache path reservation released", not production_bake.cache_path_is_active(STATE["cache_path"]))
            _check("operator active-scene reference released", cache_operators._ACTIVE_BAKE_SCENE is None)
            _check("scene B stayed unchanged through cancellation", _scene_b_snapshot(scene_b) == STATE["scene_b_before"])
            STATE["cancelled_snapshot"] = {
                "frame": int(scene_a.frame_current),
                "status": scene_a.fluidsim.bake_status,
                "progress": float(scene_a.fluidsim.bake_progress),
                "message": scene_a.fluidsim.bake_message,
            }
            STATE["settle_after"] = time.monotonic() + 0.25
            STATE["phase"] = "settle"
            return 0.05

        if phase == "settle":
            if time.monotonic() < STATE["settle_after"]:
                return 0.05
            snapshot = {
                "frame": int(scene_a.frame_current),
                "status": scene_a.fluidsim.bake_status,
                "progress": float(scene_a.fluidsim.bake_progress),
                "message": scene_a.fluidsim.bake_message,
            }
            _check("modal bake is quiescent after timer cleanup", snapshot == STATE["cancelled_snapshot"])
            _check("scene B final settings and output are unchanged", _scene_b_snapshot(scene_b) == STATE["scene_b_before"])
            STATE["report"]["scene_a_cancelled"] = snapshot
            STATE["report"]["scene_b_unchanged"] = True
            STATE["phase"] = "done"
            return _finish(True)
        raise AssertionError(f"Unknown acceptance phase: {phase}")
    except Exception as exc:  # noqa: BLE001 - failure must become a report
        return _begin_failure(exc)
