"""Run the liquid solver and prove its particle output takes the native interop path."""

from tools.repo_addon_loader import enable_repo_addon
from pathlib import Path
import sys

import bpy
import gpu
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def draw_pending(presenter):
    offscreen = gpu.types.GPUOffScreen(64, 64)
    try:
        with offscreen.bind():
            gpu.state.active_framebuffer_get().clear(color=(0.0, 0.0, 0.0, 1.0))
            with gpu.matrix.push_pop():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                presenter.draw()
            gpu.state.active_framebuffer_get().read_color(0, 0, 1, 1, 4, 0, "UBYTE")
    finally:
        offscreen.free()


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    settings = scene.fluidsim
    settings.voxel_size = 0.14
    settings.max_particles = 120_000
    settings.max_whitewater = 20_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "NONE"
    settings.native_gpu_interop = True

    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status

    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    assert runtime.active
    assert str(runtime.solver.device).startswith("cuda")

    for _ in range(2):
        runtime.solver.step_frame()
        viewport.update_outputs(scene, runtime)
        assert runtime.native_presenter is not None, runtime.native_interop_error
        draw_pending(runtime.native_presenter)

    stats = dict(runtime.native_presenter.bridge.stats)
    assert stats["imports"] == 3
    assert stats["maps"] == 1
    assert stats["published_frames"] >= 2
    assert stats["published_bytes"] > 0
    assert stats["bulk_host_position_bytes"] == 0
    particle_object = bpy.data.objects.get("FluidSim Particles")
    if particle_object is not None:
        assert len(particle_object.data.points) == 0

    from fluidSim_warp_blender51.blender import native_interop

    runtime.native_presenter.close()
    runtime.native_presenter = None
    presenter_type = native_interop.NativeParticlePresenter

    def forced_unavailable(*_args, **_kwargs):
        raise native_interop.NativeInteropUnavailable("forced unavailable")

    native_interop.NativeParticlePresenter = forced_unavailable
    try:
        viewport.update_outputs(scene, runtime)
    finally:
        native_interop.NativeParticlePresenter = presenter_type
    assert runtime.native_presenter is None
    assert runtime.native_interop_error == "forced unavailable"
    particle_object = bpy.data.objects.get("FluidSim Particles")
    assert particle_object is not None
    assert len(particle_object.data.points) == runtime.solver.particles.host_count

    print(
        "NATIVE_INTEROP_SOLVER_GATE",
        {
            "frame": runtime.solver.frame,
            "particles": runtime.solver.particles.host_count,
            "stats": stats,
            "forced_fallback_error": runtime.native_interop_error,
        },
    )
    bpy.ops.fluidsim.release()
    bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
