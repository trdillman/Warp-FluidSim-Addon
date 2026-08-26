"""Run the liquid solver and prove its particle output takes the native interop path."""

from array import array
from time import perf_counter
from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon


def redraw_viewport():
    area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region, space_data=area.spaces.active):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)


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

    for _ in range(4):
        runtime.solver.step_frame()
        viewport.update_outputs(scene, runtime)
        assert runtime.native_presenter is not None, runtime.native_interop_error
        redraw_viewport()
        particle_object = bpy.data.objects.get("FluidSim Particles")
        assert particle_object is not None
        assert particle_object.type == "POINTCLOUD"
        assert particle_object.visible_get()
        assert not particle_object.hide_viewport
        assert len(particle_object.data.points) == runtime.solver.particles.capacity

    settings.particle_display_fraction = 0.25
    viewport.update_outputs(scene, runtime)
    expected_visible = (runtime.solver.particles.host_count + 3) // 4
    expected_output = (runtime.solver.particles.capacity + 3) // 4
    assert runtime.native_presenter.active_count == expected_output
    assert runtime.native_presenter is not None, runtime.native_interop_error
    settings.particle_display_fraction = 1.0
    settings.particle_attribute_mode = "HEIGHT"
    viewport.update_outputs(scene, runtime)
    assert runtime.native_presenter is not None, runtime.native_interop_error
    materialization = dict(runtime.native_presenter.materialization_stats)
    assert materialization["reads"] >= 1
    assert materialization["reason"] == "particle_attribute:HEIGHT"
    assert materialization["bytes"] >= runtime.solver.particles.host_count * 4
    assert runtime.presentation_mode == "Native GPU"
    assert runtime.presentation_reason == "particle_attribute:HEIGHT"
    assert runtime.presentation_bytes == materialization["bytes"]
    color_attribute = bpy.data.objects["FluidSim Particles"].data.attributes[
        "display_color"]
    colors = array("f", [0.0]) * (len(color_attribute.data) * 4)
    color_attribute.data.foreach_get("color", colors)
    visible_colors = {
        tuple(round(value, 3) for value in colors[index:index + 3])
        for index in range(0, expected_visible * 4, 64)
    }
    assert len(visible_colors) > 1
    for mode in (
            "SPEED",
            "TEMPERATURE",
            "VISCOSITY",
            "DENSITY",
            "AGE",
            "LIFE",
            "TIME_OFFSET"):
        settings.particle_attribute_mode = mode
        viewport.update_outputs(scene, runtime)
        assert runtime.native_presenter is not None, runtime.native_interop_error
        assert runtime.native_presenter.materialization_stats[
            "reason"] == f"particle_attribute:{mode}"
    settings.particle_attribute_mode = "NONE"
    stats = dict(runtime.native_presenter.bridge.stats)
    assert stats["imports"] == 6
    assert stats["maps"] == 2
    assert stats["published_frames"] >= 5
    assert stats["published_bytes"] > 0
    assert stats["bulk_host_position_bytes"] == 0
    assert sum(slot.value for slot in runtime.native_presenter._slots) == stats[
        "published_frames"]
    from fluidSim_warp_blender51.blender import native_interop

    failed_presenter = runtime.native_presenter

    def forced_publish_failure(*_args, **_kwargs):
        raise RuntimeError("forced publish failure")

    failed_presenter.publish_device_count = forced_publish_failure
    fault_started = perf_counter()
    viewport.update_outputs(scene, runtime)
    fault_elapsed = perf_counter() - fault_started
    assert fault_elapsed < 5.0
    assert runtime.native_presenter is None
    assert runtime.native_interop_error == "forced publish failure"
    assert not failed_presenter._slots
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
    assert runtime.presentation_mode == "CPU Fallback"
    assert runtime.presentation_reason == "forced unavailable"
    assert runtime.presentation_bytes >= runtime.solver.particles.host_count * 12
    particle_object = bpy.data.objects.get("FluidSim Particles")
    assert particle_object is not None
    assert len(particle_object.data.points) == runtime.solver.particles.host_count

    print(
        "NATIVE_INTEROP_SOLVER_GATE",
        {
            "frame": runtime.solver.frame,
            "particles": runtime.solver.particles.host_count,
            "stats": stats,
            "publish_fault_seconds": fault_elapsed,
            "forced_fallback_error": runtime.native_interop_error,
            "fallback_bytes": runtime.presentation_bytes,
        },
    )
    bpy.ops.fluidsim.release()

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
