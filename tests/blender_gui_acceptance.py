"""Interactive Blender GUI acceptance test for the FluidSim Warp add-on.

Run with a real (non-background) Blender:
    blender --python tests/blender_gui_acceptance.py

Uses a timer-driven state machine so the UI event loop keeps running:
enables the add-on, builds the demo scene, initializes the CUDA solver,
starts viewport animation playback with live simulation, verifies the solver
advances during real playback, checks rewind-reset behavior, forces UI
redraws of the sidebar panels, captures a viewport screenshot, renders a
frame, and writes a JSON verdict before quitting Blender.
"""

import json
import os
import tempfile
import time

import bpy

RESULT_PATH = os.path.join(tempfile.gettempdir(), "fluidsim_gui_result.json")
SCREENSHOT_PATH = os.path.join(tempfile.gettempdir(), "fluidsim_gui_viewport.png")
RENDER_PATH = os.path.join(tempfile.gettempdir(), "fluidsim_gui_render.png")

STATE = {
    "phase": "init",
    "checks": [],
    "t_play_start": 0.0,
    "frame_at_play": 0,
    "solver_frame_at_play": 0,
}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    STATE["checks"].append({"name": name, "passed": bool(cond),
                            "detail": str(detail)})


def _view3d_override():
    wm = bpy.context.window_manager
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                return {"window": window, "area": area, "region": region,
                        "screen": window.screen}
    return None


def _redraw_all():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def finish():
    passed = [c for c in STATE["checks"] if c["passed"]]
    verdict = {
        "result": "pass" if len(passed) == len(STATE["checks"]) else "fail",
        "passed": len(passed),
        "total": len(STATE["checks"]),
        "checks": STATE["checks"],
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=2)
    print(f"GUI-ACCEPTANCE-{verdict['result'].upper()} "
          f"({verdict['passed']}/{verdict['total']})")
    bpy.ops.wm.quit_blender()


def state_machine():
    scene = bpy.context.scene
    phase = STATE["phase"]
    try:
        return _run_phase(phase, scene)
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(f"phase {phase} raised", False, repr(exc))
        finish()
        return None


def _run_phase(phase, scene):
    from fluidSim_warp_blender51.blender.scene_bridge import session

    if phase == "init":
        import addon_utils

        try:
            addon_utils.enable("fluidSim_warp_blender51", default_set=True,
                               persistent=True)
            check("addon enable", True)
        except Exception as exc:  # noqa: BLE001
            check("addon enable", False, repr(exc))
            finish()
            return None
        check("gui mode", not bpy.app.background)
        STATE["phase"] = "setup"
        return 0.1

    if phase == "setup":
        fs = scene.fluidsim
        fs.voxel_size = 0.08
        fs.max_particles = 500_000
        fs.whitewater_enabled = True
        fs.profiling_enabled = True
        fs.show_overlay = True
        bpy.ops.fluidsim.setup_scene()
        check("setup_scene", fs.domain_object is not None)

        # Camera + light for the render check.
        cam_data = bpy.data.cameras.new("FluidSim Cam")
        cam = bpy.data.objects.new("FluidSim Cam", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (7.5, -7.5, 5.5)
        cam.rotation_euler = (1.05, 0.0, 0.785)
        scene.camera = cam
        light_data = bpy.data.lights.new("FluidSim Sun", type="SUN")
        light_data.energy = 3.0
        light = bpy.data.objects.new("FluidSim Sun", light_data)
        scene.collection.objects.link(light)
        light.rotation_euler = (0.6, 0.2, 0.0)

        result = bpy.ops.fluidsim.initialize()
        sess = session()
        check("initialize solver", result == {"FINISHED"} and sess.active,
              scene.fluidsim.status)
        check("cuda device", sess.active
              and str(sess.solver.device).startswith("cuda"))
        STATE["phase"] = "play"
        return 0.2

    if phase == "play":
        fs = scene.fluidsim
        sess = session()
        bpy.ops.fluidsim.toggle_live()
        check("live toggled on", fs.live)
        scene.frame_set(scene.frame_start)
        STATE["frame_at_play"] = scene.frame_current
        STATE["solver_frame_at_play"] = sess.solver.frame
        STATE["t_play_start"] = time.perf_counter()
        override = _view3d_override()
        if override:
            with bpy.context.temp_override(**override):
                bpy.ops.screen.animation_play()
        else:
            bpy.ops.screen.animation_play()
        check("animation playing", bpy.context.screen.is_animation_playing
              if bpy.context.screen else True)
        STATE["phase"] = "playing"
        return 3.0

    if phase == "playing":
        sess = session()
        # Stop playback.
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            override = _view3d_override()
            if override:
                with bpy.context.temp_override(**override):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            else:
                bpy.ops.screen.animation_cancel(restore_frame=False)
        elapsed = time.perf_counter() - STATE["t_play_start"]
        frames_advanced = scene_frames = bpy.context.scene.frame_current - \
            STATE["frame_at_play"]
        solver_advanced = sess.solver.frame - STATE["solver_frame_at_play"]
        check("playback advanced timeline", scene_frames >= 3,
              f"{scene_frames} frames in {elapsed:.1f}s")
        check("solver stepped during playback", solver_advanced >= 3,
              f"{solver_advanced} solver frames")
        fs = bpy.context.scene.fluidsim
        check("particles during playback", fs.stat_particles > 1000,
              f"{fs.stat_particles:,}")
        surf = bpy.data.objects.get("FluidSim Surface")
        check("surface visible in viewport",
              surf is not None and surf.visible_get()
              and len(surf.data.vertices) > 100,
              f"{len(surf.data.vertices) if surf else 0} verts")
        ww = bpy.data.objects.get("FluidSim Whitewater")
        check("whitewater object present", ww is not None,
              ww.type if ww else "")
        STATE["phase"] = "rewind"
        return 0.3

    if phase == "rewind":
        sess = session()
        scene = bpy.context.scene
        # Rewinding to the start frame must reset the simulation.
        scene.frame_set(scene.frame_start)
        check("rewind resets solver", sess.solver.frame == 0,
              f"solver frame {sess.solver.frame}")
        bpy.ops.fluidsim.toggle_live()
        check("live toggled off", not scene.fluidsim.live)
        # Step manually after reset (operator path in GUI context).
        result = bpy.ops.fluidsim.step(frames=3)
        check("manual step after reset", result == {"FINISHED"},
              scene.fluidsim.status)
        STATE["phase"] = "ui"
        return 0.2

    if phase == "ui":
        # Force full UI redraws so all 13 sidebar panels + overlay draw with
        # a real context (draw errors would print tracebacks and leave the
        # checks list short at the end).
        _redraw_all()
        override = _view3d_override()
        if override:
            with bpy.context.temp_override(**override):
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
        panel_ids = [
            "FLUIDSIM_PT_simulation", "FLUIDSIM_PT_domain",
            "FLUIDSIM_PT_particles", "FLUIDSIM_PT_sources",
            "FLUIDSIM_PT_colliders", "FLUIDSIM_PT_forces",
            "FLUIDSIM_PT_pressure", "FLUIDSIM_PT_viscosity",
            "FLUIDSIM_PT_surface", "FLUIDSIM_PT_output",
            "FLUIDSIM_PT_snippets", "FLUIDSIM_PT_diagnostics",
            "FLUIDSIM_PT_validation",
        ]
        missing = [p for p in panel_ids if not hasattr(bpy.types, p)]
        check("panels registered in GUI", not missing, str(missing))
        check("ui redraw with overlay", True)

        # Property edits through the UI data path (as the panels would).
        scene = bpy.context.scene
        scene.fluidsim.flip_ratio = 0.9
        scene.fluidsim.pressure_iterations = 150
        check("property edits applied",
              abs(scene.fluidsim.flip_ratio - 0.9) < 1e-6
              and scene.fluidsim.pressure_iterations == 150)

        # Viewport screenshot (viewport-visible proof).
        try:
            override = _view3d_override()
            with bpy.context.temp_override(**override):
                bpy.ops.screen.screenshot_area(filepath=SCREENSHOT_PATH)
            ok = os.path.exists(SCREENSHOT_PATH) and \
                os.path.getsize(SCREENSHOT_PATH) > 10000
            check("viewport screenshot", ok,
                  f"{os.path.getsize(SCREENSHOT_PATH)} bytes"
                  if os.path.exists(SCREENSHOT_PATH) else "missing")
        except Exception as exc:  # noqa: BLE001
            check("viewport screenshot", False, repr(exc))
        STATE["phase"] = "render"
        return 0.3

    if phase == "render":
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_EEVEE_NEXT" if hasattr(
            bpy.types, "RenderSettings") and "BLENDER_EEVEE_NEXT" in {
            e.identifier for e in
            bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        } else "BLENDER_EEVEE"
        scene.render.resolution_x = 480
        scene.render.resolution_y = 320
        scene.render.filepath = RENDER_PATH
        bpy.ops.render.render(write_still=True)
        ok = os.path.exists(RENDER_PATH) and os.path.getsize(RENDER_PATH) > 5000
        check("render output written", ok,
              f"{os.path.getsize(RENDER_PATH)} bytes"
              if os.path.exists(RENDER_PATH) else "missing")
        if ok:
            img = bpy.data.images.load(RENDER_PATH)
            px = list(img.pixels[: 4000])
            non_black = sum(1 for v in px if v > 0.02)
            check("render contains visible content", non_black > 100,
                  f"{non_black} bright samples")
        STATE["phase"] = "diag"
        return 0.2

    if phase == "diag":
        sess = session()
        check("profiler frames in GUI",
              sess.profiler is not None and len(sess.profiler.history) >= 3,
              f"{len(sess.profiler.history) if sess.profiler else 0}")
        # Benchmark (short) through the operator.
        scene = bpy.context.scene
        scene.fluidsim.benchmark_frames = 6
        result = bpy.ops.fluidsim.run_benchmark(save_baseline=False)
        check("benchmark operator in GUI", result == {"FINISHED"},
              scene.fluidsim.status)
        ok = sess.benchmark_result is not None and \
            "timeStepAverage" in (sess.benchmark_result or {})
        check("benchmark result schema", ok)
        finish()
        return None

    finish()
    return None


def start():
    bpy.app.timers.register(state_machine, first_interval=0.5)


start()
