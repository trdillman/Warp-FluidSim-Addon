"""Drive real Blender timeline playback and report solver advancement.

Run without --background. A profile-safe launcher session stays open after the
probe; standalone production-gate runs close after writing the report.
Set FLUIDSIM_EXPECTED_ADDON_ROOT, FLUIDSIM_PLAYBACK_BLEND, and
FLUIDSIM_PLAYBACK_REPORT to reuse a source-loaded isolated GUI without
enabling, reopening, or quitting it. The already-open file must match.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import addon_utils
import bpy

expected_preloaded_addon_root = os.environ.get("FLUIDSIM_EXPECTED_ADDON_ROOT")
profile_safe_result_path = (
    os.environ.get("FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT")
    if os.environ.get("FLUIDSIM_PROFILE_SAFE_REVIEW_CONFIG")
    else None
)
if expected_preloaded_addon_root:
    blend_path = os.environ["FLUIDSIM_PLAYBACK_BLEND"]
    report_path = os.environ["FLUIDSIM_PLAYBACK_REPORT"]
    loaded_blend_path = bpy.data.filepath
    if Path(loaded_blend_path).resolve() != Path(blend_path).resolve():
        raise RuntimeError(
            f"Wrong preloaded blend: expected {Path(blend_path).resolve()}, "
            f"loaded {Path(loaded_blend_path).resolve()}"
        )
else:
    if not profile_safe_result_path:
        REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from tools.repo_addon_loader import enable_repo_addon

        blend_path = os.environ["FLUIDSIM_PLAYBACK_BLEND"]
        report_path = os.environ["FLUIDSIM_PLAYBACK_REPORT"]
        enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
        bpy.ops.wm.open_mainfile(filepath=blend_path)
    else:
        blend_path = bpy.data.filepath
        report_path = profile_safe_result_path

from fluidSim_warp_blender51.blender.scene_bridge import (
    active_domain_settings,
    session,
)


def _verify_preloaded_addon():
    expected_root = Path(expected_preloaded_addon_root).resolve()
    module = sys.modules.get("fluidSim_warp_blender51")
    loaded_path = Path(getattr(module, "__file__", "")).resolve()
    if loaded_path != expected_root and expected_root not in loaded_path.parents:
        raise RuntimeError(
            f"Wrong preloaded add-on: expected {expected_root}, loaded {loaded_path}"
        )
    if not addon_utils.check("fluidSim_warp_blender51")[1]:
        raise RuntimeError("Preloaded FluidSim add-on is not runtime-enabled")
    if "fluidSim_warp_blender51" not in bpy.context.preferences.addons:
        raise RuntimeError("Preloaded FluidSim add-on is absent from preferences")
    return loaded_path


loaded_addon_path = (
    _verify_preloaded_addon() if expected_preloaded_addon_root else
    Path(sys.modules["fluidSim_warp_blender51"].__file__).resolve()
)

scene = bpy.context.scene
domain = active_domain_settings(scene)
if domain is None:
    raise RuntimeError("Active scene has no authoritative FluidSim Domain settings")
start_frame = scene.frame_current
started_at = time.monotonic()
canonical_wall_expectation = {
    "dam_break.blend": True,
    "crown_splash.blend": True,
}
expected_domain_sdf_walls = canonical_wall_expectation.get(
    os.path.basename(blend_path), bool(domain.domain_sdf_walls)
)


def viewport_sample():
    from fluidSim_warp_blender51.blender import output_publication

    obj = output_publication.output_object(scene, "FluidSim Particles")
    if obj is None or obj.type != "POINTCLOUD":
        return ()
    position = obj.data.attributes.get("position")
    if position is None:
        return ()
    return tuple(
        round(float(value), 6)
        for item in position.data[:min(32, len(position.data))]
        for value in item.vector
    )


live_viewport_baseline = {"sample": None, "solver_frame": None}
report = {
    "blend": bpy.data.filepath,
    "start_frame": start_frame,
    "live": bool(scene.fluidsim.live),
    "saved_domain_sdf_walls": bool(domain.domain_sdf_walls),
    "expected_domain_sdf_walls": expected_domain_sdf_walls,
    "addon_module": str(loaded_addon_path),
    "handler_count": len([
        fn for fn in bpy.app.handlers.frame_change_post
        if getattr(fn, "__module__", "").startswith("fluidSim_warp_blender51")
    ]),
}


def finish():
    sess = session()
    particle_count = (
        sess.solver.particles.refresh_host_count() if sess.active else 0
    )
    from fluidSim_warp_blender51.blender import output_publication

    # Output clouds carry spare capacity, so len(data.points) is not the count.
    viewport_particle_count = output_publication.live_point_count(
        output_publication.output_object(scene, "FluidSim Particles"))
    report.update({
        "generated_at_unix": time.time(),
        "timed_out": bool(report.get("timed_out", False)),
        "end_frame": scene.frame_current,
        "animation_playing": bool(
            playback_window and playback_window.screen.is_animation_playing),
        "session_active": sess.active,
        "solver_frame": sess.solver.frame if sess.active else None,
        "device": str(sess.solver.device) if sess.active else None,
        "domain_sdf_walls": (
            bool(sess.solver.settings.domain_sdf_walls) if sess.active else False
        ),
        "particle_count": particle_count,
        "viewport_particle_count": viewport_particle_count,
        "viewport_baseline_solver_frame": live_viewport_baseline["solver_frame"],
        "viewport_changed": (
            live_viewport_baseline["sample"] is not None
            and viewport_sample() != live_viewport_baseline["sample"]
        ),
        "status": scene.fluidsim.status,
        "error": sess.error,
        "elapsed_seconds": time.monotonic() - started_at,
    })
    report["passed"] = bool(
        not report["timed_out"]
        and report["live"]
        and report["handler_count"] == 1
        and report["end_frame"] >= report["start_frame"] + 3
        and report["session_active"]
        and report["solver_frame"] is not None
        and report["solver_frame"] >= 3
        and report["device"].startswith("cuda")
        and report["domain_sdf_walls"] == report["expected_domain_sdf_walls"]
        and report["saved_domain_sdf_walls"] == report["expected_domain_sdf_walls"]
        and report["particle_count"] > 0
        and report["viewport_particle_count"] > 0
        and report["viewport_changed"]
        and not report["error"]
    )
    report["ok"] = report["passed"]
    report_parent = os.path.dirname(report_path)
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    marker = "GUI-PLAYBACK-PROBE-PASS" if report["passed"] else "GUI-PLAYBACK-PROBE-FAIL"
    print(marker, flush=True)
    if playback_window and playback_window.screen.is_animation_playing:
        with bpy.context.temp_override(window=playback_window,
                                       screen=playback_window.screen):
            bpy.ops.screen.animation_cancel(restore_frame=False)
    if not profile_safe_result_path and not expected_preloaded_addon_root:
        bpy.ops.wm.quit_blender()
    return None


def poll():
    sess = session()
    if (
        sess.active
        and sess.solver.frame >= 1
        and live_viewport_baseline["sample"] is None
    ):
        live_viewport_baseline["sample"] = viewport_sample()
        live_viewport_baseline["solver_frame"] = sess.solver.frame
        return 0.25
    advanced = (
        scene.frame_current >= start_frame + 3
        and sess.active
        and sess.solver.frame >= 3
        and live_viewport_baseline["solver_frame"] is not None
        and sess.solver.frame > live_viewport_baseline["solver_frame"]
    )
    timed_out = time.monotonic() - started_at > 45.0
    if advanced or timed_out:
        report["timed_out"] = timed_out
        return finish()
    return 0.25


def start():
    global playback_window
    window = next(
        (item for item in bpy.context.window_manager.windows if item.scene == scene),
        None,
    )
    playback_window = window
    if window is None:
        report["start_error"] = f"No Blender window owns scene {scene.name!r}"
        return finish()
    area = next((item for item in window.screen.areas if item.type == "VIEW_3D"), None)
    if area is None:
        report["start_error"] = "No VIEW_3D area"
        return finish()
    region = next(item for item in area.regions if item.type == "WINDOW")
    with bpy.context.temp_override(window=window, screen=window.screen,
                                   area=area, region=region):
        bpy.ops.screen.animation_play()
    bpy.app.timers.register(poll, first_interval=0.25)
    return None


playback_window = None
bpy.app.timers.register(start, first_interval=1.0)
