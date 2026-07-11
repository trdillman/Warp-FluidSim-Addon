"""Open the saved demo and prove it drives the native presentation path."""

from tools.repo_addon_loader import enable_repo_addon
from pathlib import Path
import sys

import bpy
import gpu
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


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

    offscreen = gpu.types.GPUOffScreen(64, 64)
    try:
        with offscreen.bind():
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                runtime.native_presenter.draw()
            gpu.state.active_framebuffer_get().read_color(0, 0, 1, 1, 4, 0, "UBYTE")
    finally:
        offscreen.free()

    stats = runtime.native_presenter.bridge.stats
    assert stats["published_frames"] >= 1
    assert stats["bulk_host_position_bytes"] == 0
    print(
        "NATIVE_INTEROP_DEMO_GATE",
        {"particles": runtime.solver.particles.host_count, "stats": stats},
    )
    bpy.ops.fluidsim.release()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
