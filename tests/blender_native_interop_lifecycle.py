"""Exercise undo, file-load, add-on reload, viewport, and teardown lifecycle."""

import json
import os
from pathlib import Path
import sys

import bpy
import gpu
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.repo_addon_loader import enable_repo_addon


def publish_frame():
    from fluidSim_warp_blender51.blender import viewport
    from fluidSim_warp_blender51.blender.scene_bridge import session

    runtime = session()
    bpy.context.scene.fluidsim.native_gpu_interop = True
    bpy.context.scene.fluidsim.show_particles = True
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}
    runtime.solver.inject_particles(
        np.array([[-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    )
    viewport.update_outputs(bpy.context.scene, runtime)
    assert runtime.native_presenter is not None, runtime.native_interop_error
    return runtime


def main():
    gpu.init()
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    mode = os.environ["NATIVE_INTEROP_LIFECYCLE_MODE"]
    results = {}
    runtime = publish_frame()
    if mode == "undo":
        bpy.ops.ed.undo_push(message="native interop lifecycle")
        bpy.context.scene.frame_start += 1
        bpy.ops.ed.undo()
        results["undo_release"] = (
            not runtime.active and runtime.native_presenter is None)
        assert results["undo_release"]
        runtime = publish_frame()
        particle_object = bpy.data.objects["FluidSim Particles"]
        particle_object.hide_viewport = True
        bpy.context.view_layer.update()
        particle_object.hide_viewport = False
        bpy.context.view_layer.update()
        results["viewport_visibility_kept_session"] = (
            runtime.native_presenter is not None)
    elif mode == "load":
        lifecycle_file = Path.home() / "native_interop_lifecycle.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(lifecycle_file), copy=True)
        def load_after_startup():
            bpy.ops.wm.open_mainfile(filepath=str(lifecycle_file))
            results["load_release"] = (
                not runtime.active and runtime.native_presenter is None)
            assert results["load_release"]
            from fluidSim_warp_blender51.blender import scene_bridge

            results["frame_handler_persisted"] = (
                scene_bridge._on_frame_change
                in bpy.app.handlers.frame_change_post)
            assert results["frame_handler_persisted"]
            report = {
                "mode": mode,
                "results": results,
                "blend_file": bpy.data.filepath,
            }
            evidence = Path.home() / "native_interop_lifecycle_load.json"
            evidence.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("NATIVE_INTEROP_LIFECYCLE", json.dumps(report, sort_keys=True))
            bpy.app.timers.register(
                lambda: (bpy.ops.wm.quit_blender(), None)[1],
                first_interval=0.1,
            )
            return None

        if bpy.app.background:
            load_after_startup()
        else:
            bpy.app.timers.register(load_after_startup, first_interval=0.5)
        return
    elif mode == "reload":
        bpy.ops.preferences.addon_disable(module="fluidSim_warp_blender51")
        results["addon_reload_release"] = (
            not runtime.active and runtime.native_presenter is None)
        assert results["addon_reload_release"]
        bpy.ops.preferences.addon_enable(module="fluidSim_warp_blender51")
        results["addon_reenabled"] = hasattr(bpy.context.scene, "fluidsim")
        assert results["addon_reenabled"]
    else:
        raise ValueError(mode)
    if mode != "reload":
        presenter = runtime.native_presenter
        if presenter is not None:
            bpy.ops.fluidsim.release()
            results["teardown"] = (
                runtime.native_presenter is None and not presenter._slots)
            assert results["teardown"]
    report = {"mode": mode, "results": results, "blend_file": bpy.data.filepath}
    evidence = Path.home() / f"native_interop_lifecycle_{mode}.json"
    evidence.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_INTEROP_LIFECYCLE", json.dumps(report, sort_keys=True))

    def quit_after_test():
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(quit_after_test, first_interval=0.1)


if __name__ == "__main__":
    main()
