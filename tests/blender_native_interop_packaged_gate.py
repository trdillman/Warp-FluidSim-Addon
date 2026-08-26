"""Prove saved-demo playback from an installed add-on package."""

import json
from pathlib import Path
import sys

import bpy
import gpu


def main():
    module = sys.modules.get("fluidSim_warp_blender51")
    assert module is not None, "packaged FluidSim add-on is not enabled"
    module_path = Path(module.__file__).resolve()
    assert "scripts" in {part.lower() for part in module_path.parts}, module_path
    assert "addons" in {part.lower() for part in module_path.parts}, module_path
    scene = bpy.context.scene
    settings = scene.fluidsim
    assert scene.get("native_gpu_interop_demo") is True
    assert settings.native_gpu_interop is True
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}, settings.status
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    for _frame in range(3):
        runtime.solver.step_frame()
        viewport.update_outputs(scene, runtime)
    presenter = runtime.native_presenter
    assert presenter is not None, runtime.native_interop_error
    assert presenter.bridge.stats["bulk_host_position_bytes"] == 0
    area = next(area for area in bpy.context.window.screen.areas if area.type == "VIEW_3D")
    region = next(region for region in area.regions if region.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region, space_data=area.spaces.active):
        area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=3)
    report = {
        "backend": gpu.platform.backend_type_get(),
        "addon_module": str(module_path),
        "blend_file": bpy.data.filepath,
        "particles": runtime.solver.particles.host_count,
        "bridge_stats": dict(presenter.bridge.stats),
    }
    evidence = Path.home() / "native_interop_packaged_demo.json"
    evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_PACKAGED_GATE", json.dumps(report, sort_keys=True))
    bpy.ops.fluidsim.release()

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
