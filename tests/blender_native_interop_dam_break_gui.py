"""Prove the canonical Dam Break scene in a live native-interop viewport."""

import json
from pathlib import Path
import sys
import time
import traceback

import addon_utils
import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLEND = (REPO_ROOT / "demos" / "dam_break.blend").resolve()
EVIDENCE_DIR = REPO_ROOT / "docs" / "evidence" / "native_interop"
EVIDENCE_PATH = EVIDENCE_DIR / "dam-break-gui.json"
SCREENSHOT_PATH = EVIDENCE_DIR / "dam-break-viewport.png"
STARTED = time.perf_counter()
START_FRAME = 0
START_SOLVER_FRAME = 0
START_PUBLISHED = 0


def view3d_override():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(
                    region for region in area.regions if region.type == "WINDOW"
                )
                return {"window": window, "screen": window.screen,
                        "area": area, "region": region,
                        "space_data": area.spaces.active}
    raise RuntimeError("No VIEW_3D area is available")


def write_result(passed, **details):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": passed,
        "blend": str(Path(bpy.data.filepath).resolve()),
        "blender": bpy.app.version_string,
        "elapsed_seconds": round(time.perf_counter() - STARTED, 3),
        **details,
    }
    EVIDENCE_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("DAM_BREAK_NATIVE_GUI=" + json.dumps(report, sort_keys=True))


def finish():
    details = {}
    try:
        from fluidSim_warp_blender51.blender.scene_bridge import session

        scene = bpy.context.scene
        runtime = session()
        playing = bool(
            bpy.context.screen and bpy.context.screen.is_animation_playing
        )
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            with bpy.context.temp_override(**view3d_override()):
                bpy.ops.screen.animation_cancel(restore_frame=False)
        with bpy.context.temp_override(**view3d_override()):
            bpy.ops.view3d.view_camera()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
            bpy.ops.screen.screenshot(filepath=str(SCREENSHOT_PATH))

        presenter = runtime.native_presenter
        particles = bpy.data.objects.get("FluidSim Particles")
        stats = dict(presenter.bridge.stats) if presenter else {}
        details = {
            "addon_module": str(
                Path(sys.modules["fluidSim_warp_blender51"].__file__).resolve()
            ),
            "backend": "Vulkan",
            "device": str(runtime.solver.device),
            "timeline_start": START_FRAME,
            "timeline_end": scene.frame_current,
            "solver_frame_start": START_SOLVER_FRAME,
            "solver_frame": runtime.solver.frame,
            "animation_was_playing": playing,
            "runtime_active": runtime.active,
            "native_interop_error": runtime.native_interop_error,
            "particle_count": runtime.solver.particles.host_count,
            "pointcloud_type": particles.type if particles else None,
            "pointcloud_capacity": len(particles.data.points) if particles else 0,
            "pointcloud_visible": particles.visible_get() if particles else False,
            "pointcloud_hide_get": particles.hide_get() if particles else None,
            "pointcloud_hide_viewport": (
                particles.hide_viewport if particles else None
            ),
            "pointcloud_hide_render": particles.hide_render if particles else None,
            "pointcloud_collections": [
                {
                    "name": collection.name,
                    "hide_viewport": collection.hide_viewport,
                    "hide_render": collection.hide_render,
                }
                for collection in particles.users_collection
            ] if particles else [],
            "show_particles": scene.fluidsim.show_particles,
            "presentation_mode": runtime.presentation_mode,
            "presentation_reason": runtime.presentation_reason,
            "presentation_bytes": runtime.presentation_bytes,
            "native_stats": stats,
            "screenshot": str(SCREENSHOT_PATH),
            "screenshot_bytes": SCREENSHOT_PATH.stat().st_size,
        }
        assert scene.frame_current - START_FRAME >= 3, details
        assert runtime.solver.frame - START_SOLVER_FRAME >= 3, details
        assert str(runtime.solver.device).startswith("cuda"), details
        assert presenter is not None, details
        assert particles is not None and particles.type == "POINTCLOUD", details
        assert details["pointcloud_visible"], details
        assert not details["pointcloud_hide_viewport"], details
        assert not details["pointcloud_hide_render"], details
        assert runtime.presentation_mode == "Native GPU", details
        assert stats["imports"] == 6 and stats["maps"] == 2, details
        assert stats["published_frames"] - START_PUBLISHED >= 3, details
        assert stats["bulk_host_position_bytes"] == 0, details
        assert details["particle_count"] > 0, details
        write_result(True, **details)
    except Exception as exc:  # noqa: BLE001
        write_result(
            False,
            **details,
            error=repr(exc),
            traceback=traceback.format_exc(),
            scene_frame=bpy.context.scene.frame_current,
        )
    return None


def poll():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    scene = bpy.context.scene
    runtime = session()
    presenter = runtime.native_presenter
    if presenter:
        published = presenter.bridge.stats["published_frames"]
        if (scene.frame_current - START_FRAME >= 3
                and published - START_PUBLISHED >= 3):
            return finish()
    if time.perf_counter() - STARTED > 45.0:
        return finish()
    return 0.25


def main():
    global START_FRAME, START_PUBLISHED, START_SOLVER_FRAME

    assert Path(bpy.data.filepath).resolve() == EXPECTED_BLEND
    assert addon_utils.check("fluidSim_warp_blender51") == (True, True)
    scene = bpy.context.scene
    settings = scene.fluidsim
    settings.native_gpu_interop = True
    settings.liquid_view_mode = "PARTICLES"
    settings.surface_enabled = False
    settings.particle_display_fraction = 1.0
    settings.live = False
    scene.frame_set(scene.frame_start)
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status
    assert bpy.ops.fluidsim.step(frames=1) == {"FINISHED"}, settings.status
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    START_FRAME = scene.frame_current
    START_SOLVER_FRAME = runtime.solver.frame
    START_PUBLISHED = runtime.native_presenter.bridge.stats["published_frames"]
    assert bpy.ops.fluidsim.toggle_live() == {"FINISHED"}, settings.status
    with bpy.context.temp_override(**view3d_override()):
        bpy.ops.screen.animation_play()
    bpy.app.timers.register(poll, first_interval=0.5)


main()
