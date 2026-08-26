"""Prove native interop requests fall back correctly on Blender OpenGL."""

from pathlib import Path
import sys

import bpy
import gpu


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    settings = scene.fluidsim
    settings.voxel_size = 0.18
    settings.max_particles = 50_000
    settings.max_whitewater = 1_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.native_gpu_interop = True

    assert gpu.platform.backend_type_get() == "OPENGL"
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status

    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    runtime.solver.step_frame()
    viewport.update_outputs(scene, runtime)

    pointcloud = bpy.data.objects.get("FluidSim Particles")
    assert runtime.native_presenter is None
    assert runtime.presentation_mode == "CPU Fallback"
    assert "Vulkan" in runtime.presentation_reason
    assert runtime.presentation_bytes >= runtime.solver.particles.host_count * 12
    assert pointcloud is not None
    assert pointcloud.type == "POINTCLOUD"
    assert len(pointcloud.data.points) == runtime.solver.particles.host_count
    print(
        "NATIVE_INTEROP_OPENGL_FALLBACK",
        {
            "particles": runtime.solver.particles.host_count,
            "reason": runtime.presentation_reason,
            "bytes": runtime.presentation_bytes,
        },
    )
    bpy.ops.fluidsim.release()

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
