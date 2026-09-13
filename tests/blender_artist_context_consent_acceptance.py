"""Real-GUI acceptance for artist context routing and Apply consent.

Call ``start(report_path, expected_module_root)`` in an isolated source-loaded
Blender GUI launched with event simulation enabled. The active scene must be a
playable FluidSim demo. This test never saves the open file.

Call ``check_background(report_path)`` after the normal repository loader has
enabled the add-on in a factory-startup background process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import addon_utils
import bpy


STATE = {}


def _check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    STATE["report"]["checks"].append(name)


def _override(area):
    window = STATE["window"]
    region = next(region for region in area.regions if region.type == "WINDOW")
    return dict(window=window, screen=window.screen, area=area, region=region)


def _operator(area, operator, invoke=False, **kwargs):
    with bpy.context.temp_override(**_override(area)):
        call = getattr(bpy.ops.fluidsim, operator)
        return call("INVOKE_DEFAULT", **kwargs) if invoke else call(**kwargs)


def _simulate(event_type):
    simulate = getattr(STATE["window"], "event_simulate", None)
    if not callable(simulate):
        raise RuntimeError(
            "Window.event_simulate is unavailable; launch with --enable-event-simulate"
        )
    simulate(type=event_type, value="PRESS")
    simulate(type=event_type, value="RELEASE")


def _snapshot():
    from fluidSim_warp_blender51.blender import pending_settings, scene_bridge

    scene = STATE["scene"]
    sess = scene_bridge.session()
    return {
        "solver": sess.solver,
        "frame": scene.frame_current,
        "solver_frame": sess.solver.frame,
        "token": pending_settings.change_confirmation_token(scene),
    }


def _preserved(before):
    from fluidSim_warp_blender51.blender import pending_settings, scene_bridge

    scene = STATE["scene"]
    sess = scene_bridge.session()
    return (
        sess.solver is before["solver"]
        and scene.frame_current == before["frame"]
        and sess.solver.frame == before["solver_frame"]
        and pending_settings.change_confirmation_token(scene) == before["token"]
    )


def _write(status, error=""):
    report = STATE["report"]
    report.update(status=status, error=error, finished_at_unix=time.time())
    STATE["path"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ARTIST-CONTEXT-CONSENT-ACCEPTANCE-{status}", flush=True)


def _fail(exc):
    scene = STATE.get("scene")
    window = STATE.get("window")
    original_scene = STATE.get("original_window_scene") or scene
    try:
        if window is not None and original_scene is not None:
            window.scene = original_scene
            with bpy.context.temp_override(window=window, screen=window.screen):
                if window.screen.is_animation_playing:
                    bpy.ops.screen.animation_cancel(restore_frame=False)
        if scene is not None:
            scene.fluidsim.live = False
    except Exception:  # noqa: BLE001 - preserve the primary failure
        pass
    error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _write("FAIL", error)
    return None


def check_background(report_path):
    """Prove Preview rejects a real background context without allocating."""
    from fluidSim_warp_blender51.blender import scene_bridge

    path = Path(report_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    module = sys.modules.get("fluidSim_warp_blender51")
    loaded = Path(getattr(module, "__file__", "")).resolve()
    repo = Path(__file__).resolve().parents[1]
    expected_root = (repo / "fluidSim_warp_blender51").resolve()
    scene = bpy.context.scene
    sess = scene_bridge.session()
    report = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "background": bool(bpy.app.background),
        "blender": bpy.app.version_string,
        "source": str(loaded),
        "expected_source_root": str(expected_root),
        "head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "blend": bpy.data.filepath,
        "scene": scene.name,
        "checks": [],
    }
    try:
        if not bpy.app.background:
            raise AssertionError("check_background requires real background Blender")
        if not addon_utils.check("fluidSim_warp_blender51")[1]:
            raise AssertionError("FluidSim add-on is not runtime-enabled")
        if loaded != expected_root and expected_root not in loaded.parents:
            raise AssertionError(
                f"Wrong add-on source: expected {expected_root}, loaded {loaded}"
            )
        if sess.active:
            raise AssertionError("factory-startup background session unexpectedly owns a solver")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
        setup = bpy.ops.fluidsim.setup_scene(start_preview=False)
        if setup != {"FINISHED"}:
            raise AssertionError(f"guided setup failed: {setup}")
        before = (sess.solver, scene.frame_current)
        result = bpy.ops.fluidsim.preview_start()
        preserved = sess.solver is before[0] and scene.frame_current == before[1]
        passed = bool(
            result == {"CANCELLED"}
            and preserved
            and not scene.fluidsim.live
            and "needs a Blender window" in scene.fluidsim.status
        )
        report.update(
            status="PASS" if passed else "FAIL",
            result=sorted(result),
            message=scene.fluidsim.status,
            solver_preserved=preserved,
            source_enabled=True,
            checks=[
                "real background context",
                "expected source package enabled",
                "Preview returned CANCELLED",
                "no solver or frame mutation",
                "actionable no-window message",
            ] if passed else [],
        )
        if not passed:
            report["error"] = "Preview background rejection contract was not satisfied"
    except Exception as exc:  # noqa: BLE001 - receipt must survive failures
        report.update(status="FAIL", error="".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)))
    report["finished_at_unix"] = time.time()
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ARTIST-CONTEXT-CONSENT-BACKGROUND-{report['status']}", flush=True)
    return report


def start(report_path, expected_module_root):
    from fluidSim_warp_blender51.blender import effects, playback, scene_bridge

    path = Path(report_path).resolve()
    expected_root = Path(expected_module_root).resolve()
    module = sys.modules.get("fluidSim_warp_blender51")
    loaded = Path(getattr(module, "__file__", "")).resolve()
    if loaded != expected_root and expected_root not in loaded.parents:
        raise RuntimeError(f"Wrong add-on source: expected {expected_root}, loaded {loaded}")
    if not addon_utils.check("fluidSim_warp_blender51")[1]:
        raise RuntimeError("FluidSim add-on is not runtime-enabled")
    if "fluidSim_warp_blender51" not in bpy.context.preferences.addons:
        raise RuntimeError("FluidSim add-on is absent from Preferences")

    scene = bpy.context.scene
    window = next(w for w in bpy.context.window_manager.windows if w.scene == scene)
    properties = next(a for a in window.screen.areas if a.type == "PROPERTIES")
    view3d = next(a for a in window.screen.areas if a.type == "VIEW_3D")
    domain = scene_bridge.active_domain_settings(scene)
    if domain is None:
        raise RuntimeError("Active scene has no authoritative FluidSim Domain")

    path.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    report = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "blender": bpy.app.version_string,
        "blend": bpy.data.filepath,
        "scene": scene.name,
        "source": str(loaded),
        "expected_source_root": str(expected_root),
        "head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "checks": [],
    }
    STATE.clear()
    STATE.update(
        path=path,
        report=report,
        scene=scene,
        window=window,
        original_window_scene=scene,
        properties=properties,
        view3d=view3d,
        domain=domain,
        phase="start",
        deadline=time.monotonic() + 240.0,
    )
    try:
        with bpy.context.temp_override(**_override(view3d)):
            if window.screen.is_animation_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            scene.fluidsim.live = False
            if scene_bridge.session().owns_scene(scene):
                scene_bridge.session().release(scene)
            with playback.suppress_frame_handlers():
                scene.frame_set(scene.frame_start)
            domain.voxel_size = max(float(domain.voxel_size), 0.14)
            domain.max_particles = min(int(domain.max_particles), 50_000)
            domain.whitewater_enabled = False
            domain.surface_enabled = False
            effects.clear_effects(scene)
        bpy.app.timers.register(_tick, first_interval=0.1)
        return {"scheduled": True, "report": str(path)}
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)


def _tick():
    from fluidSim_warp_blender51.blender import effects, pending_settings, scene_bridge

    scene = STATE["scene"]
    window = STATE["window"]
    sess = scene_bridge.session()
    try:
        if time.monotonic() >= STATE["deadline"]:
            raise TimeoutError(f"Timed out in phase {STATE['phase']}")
        phase = STATE["phase"]

        if phase == "start":
            _check(
                "Properties Preview starts real playback",
                _operator(STATE["properties"], "preview_start") == {"FINISHED"}
                and window.screen.is_animation_playing,
                scene.fluidsim.status,
            )
            STATE.update(phase="idempotent", solver=sess.solver)

        elif phase == "idempotent" and sess.solver.frame >= 1:
            _check(
                "View3D Preview is already-playing idempotent",
                _operator(STATE["view3d"], "preview_start") == {"FINISHED"}
                and sess.solver is STATE["solver"]
                and window.screen.is_animation_playing,
                scene.fluidsim.status,
            )
            STATE["phase"] = "progress"

        elif phase == "progress" and sess.solver.frame >= 3:
            _check(
                "Pause preserves progressed solver",
                _operator(STATE["properties"], "preview_pause") == {"FINISHED"}
                and not window.screen.is_animation_playing
                and sess.solver is STATE["solver"],
                scene.fluidsim.status,
            )
            foreign = bpy.data.scenes.new("FluidSim Consent Foreign Scene")
            STATE["foreign"] = foreign
            window.scene = foreign
            try:
                try:
                    result = _operator(STATE["view3d"], "preview_start")
                except RuntimeError:
                    result = {"CANCELLED"}
            finally:
                window.scene = scene
            _check(
                "foreign scene/window Preview rejected",
                result == {"CANCELLED"}
                and sess.solver is STATE["solver"]
                and "Another scene owns" in foreign.fluidsim.status,
                getattr(foreign.fluidsim, "status", ""),
            )
            _check(
                "effect draft created",
                _operator(
                    STATE["properties"], "effect_add",
                    effect_id="fluidsim.planar_split",
                ) == {"FINISHED"},
            )
            before = _snapshot()
            STATE["cancel_before"] = before
            STATE["status_before_cancel"] = scene.fluidsim.status
            _check(
                "Apply opens real progress-loss dialog",
                _operator(STATE["properties"], "apply_changes", invoke=True)
                == {"RUNNING_MODAL"},
            )
            STATE["phase"] = "cancel_event"

        elif phase == "cancel_event":
            _simulate("ESC")
            STATE.update(phase="cancelled", not_before=time.monotonic() + 0.35)

        elif phase == "cancelled" and time.monotonic() >= STATE["not_before"]:
            _check(
                "dialog cancellation preserves solver frame and draft",
                _preserved(STATE["cancel_before"])
                and scene.fluidsim.status == STATE["status_before_cancel"],
                scene.fluidsim.status,
            )
            before = _snapshot()
            _check(
                "stale-settings dialog opens",
                _operator(STATE["properties"], "apply_changes", invoke=True)
                == {"RUNNING_MODAL"},
            )
            STATE["stale_settings_before"] = before
            STATE["phase"] = "stale_settings_event"

        elif phase == "stale_settings_event":
            STATE["domain"].voxel_size *= 0.97
            STATE["stale_settings_token"] = pending_settings.change_confirmation_token(scene)
            _simulate("RET")
            STATE.update(phase="stale_settings", not_before=time.monotonic() + 0.2)

        elif phase == "stale_settings" and time.monotonic() >= STATE["not_before"]:
            changed_token = pending_settings.change_confirmation_token(scene)
            before = STATE["stale_settings_before"]
            preserved = (
                sess.solver is before["solver"]
                and scene.frame_current == before["frame"]
                and sess.solver.frame == before["solver_frame"]
            )
            if "Scene or settings changed" not in scene.fluidsim.status:
                return 0.05
            _check(
                "stale settings token rejected before progress loss",
                changed_token == STATE["stale_settings_token"]
                and changed_token != before["token"] and preserved,
                scene.fluidsim.status,
            )
            scene_before = _snapshot()
            _check(
                "stale-scene dialog opens",
                _operator(STATE["properties"], "apply_changes", invoke=True)
                == {"RUNNING_MODAL"},
            )
            STATE["scene_before"] = scene_before
            STATE["phase"] = "stale_scene_event"

        elif phase == "stale_scene_event":
            window.scene = STATE["foreign"]
            _simulate("RET")
            STATE.update(phase="stale_scene", not_before=time.monotonic() + 0.2)

        elif phase == "stale_scene" and time.monotonic() >= STATE["not_before"]:
            foreign = STATE["foreign"]
            if "Scene or settings changed" not in foreign.fluidsim.status:
                return 0.05
            window.scene = scene
            before = STATE["scene_before"]
            _check(
                "stale scene token rejected before progress loss",
                _preserved(before),
                foreign.fluidsim.status,
            )
            bpy.data.scenes.remove(foreign)
            STATE.pop("foreign", None)
            STATE["confirmed_before"] = _snapshot()
            STATE["confirmed_draft"] = pending_settings.change_confirmation_token(scene)
            _check(
                "confirmation dialog reopens for current draft",
                _operator(STATE["properties"], "apply_changes", invoke=True)
                == {"RUNNING_MODAL"},
            )
            STATE["phase"] = "confirm_event"

        elif phase == "confirm_event":
            _simulate("RET")
            STATE.update(phase="confirmed", not_before=time.monotonic() + 0.2)

        elif phase == "confirmed" and time.monotonic() >= STATE["not_before"]:
            if pending_settings.pending_presentation(scene)["action"] != "NONE":
                return 0.05
            before = STATE["confirmed_before"]
            _check(
                "confirmed Apply performs progress-loss transition",
                scene.frame_current == scene.frame_start
                and sess.solver is not before["solver"]
                and not window.screen.is_animation_playing,
                scene.fluidsim.status,
            )
            _check(
                "confirmed Apply installs current effect draft",
                not effects.production_blocking_reason(scene)
                and bool(scene.fluidsim_effects.applied_signature)
                and pending_settings.change_confirmation_token(scene)
                == STATE["confirmed_draft"],
            )
            STATE["report"].update(
                progressed_frame=before["frame"],
                progressed_solver_frame=before["solver_frame"],
                confirmed_frame=scene.frame_current,
                confirmed_solver_frame=sess.solver.frame,
            )
            _write("PASS")
            return None

        return 0.05
    except Exception as exc:  # noqa: BLE001
        return _fail(exc)
