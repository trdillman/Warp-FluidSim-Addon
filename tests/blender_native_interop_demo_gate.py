"""Open the saved demo and prove it drives the native presentation path."""

from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    settings = scene.fluidsim
    assert scene.get("native_gpu_interop_demo") is True
    assert settings.native_gpu_interop is True
    assert settings.liquid_view_mode == "PARTICLES"
    assert settings.surface_enabled is False
    assert bpy.data.texts.get("NATIVE_GPU_INTEROP_README") is not None
    assert bpy.data.objects.get("FluidSim Domain") is not None
    assert bpy.data.objects.get("FluidSim Source") is not None

    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    runtime.solver.step_frame()
    viewport.update_outputs(scene, runtime)
    assert runtime.native_presenter is not None, runtime.native_interop_error

    area = next(
        area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D"
    )
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(
        area=area, region=region, space_data=area.spaces.active
    ):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)

    stats = runtime.native_presenter.bridge.stats
    assert stats["published_frames"] >= 1
    assert stats["bulk_host_position_bytes"] == 0
    print(
        "NATIVE_INTEROP_DEMO_GATE",
        {"particles": runtime.solver.particles.host_count, "stats": stats},
    )
    bpy.ops.fluidsim.release()

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
