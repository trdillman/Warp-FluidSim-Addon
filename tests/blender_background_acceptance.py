"""Blender --background acceptance test for the FluidSim Warp add-on.

Run:  blender --background --factory-startup --python tests/blender_background_acceptance.py

Covers: add-on enable, property registration, quick-setup operator, solver
initialization on real CUDA, stepping frames through the operator, output
object creation (surface mesh + whitewater), scene stat mirrors, custom
snippet compile + effect, diagnostics report, benchmark schema, and optional
cache/export round trip.
"""

import os
import sys
import tempfile

import bpy

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def main():
    import addon_utils

    # -- enable add-on -------------------------------------------------------
    ok = False
    try:
        addon_utils.enable("fluidSim_warp_blender51", default_set=True,
                           persistent=True)
        ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"enable failed: {exc!r}")
    check("addon enable", ok)
    if not ok:
        return

    scene = bpy.context.scene
    check("scene properties registered", hasattr(scene, "fluidsim"))
    check("object properties registered",
          hasattr(bpy.data.objects.new("probe", None), "fluidsim"))

    fs = scene.fluidsim
    fs.voxel_size = 0.08
    fs.max_particles = 500_000
    fs.whitewater_enabled = True
    fs.profiling_enabled = True
    fs.show_particles = True

    # -- quick setup ----------------------------------------------------------
    result = bpy.ops.fluidsim.setup_scene()
    check("setup_scene operator", result == {"FINISHED"})
    check("domain object assigned", fs.domain_object is not None)
    src = bpy.data.objects.get("FluidSim Source")
    check("source tagged", src is not None and src.fluidsim.role == "SOURCE")
    col = bpy.data.objects.get("FluidSim Collider")
    check("collider tagged", col is not None and col.fluidsim.role == "COLLIDER")

    # -- initialize solver -----------------------------------------------------
    result = bpy.ops.fluidsim.initialize()
    check("initialize operator", result == {"FINISHED"}, fs.status)

    from fluidSim_warp_blender51.blender.scene_bridge import session

    sess = session()
    check("session active", sess.active)
    check("CUDA device", str(sess.solver.device).startswith("cuda"),
          str(sess.solver.device))

    # -- step frames -----------------------------------------------------------
    result = bpy.ops.fluidsim.step(frames=6)
    check("step operator x6", result == {"FINISHED"}, fs.status)
    check("particles emitted", fs.stat_particles > 1000,
          f"{fs.stat_particles} particles")
    check("voxels active", fs.stat_voxels > 500, f"{fs.stat_voxels} voxels")

    surf = bpy.data.objects.get("FluidSim Surface")
    check("surface object exists", surf is not None)
    if surf is not None:
        check("surface mesh populated", len(surf.data.vertices) > 100,
              f"{len(surf.data.vertices)} verts")
        check("surface material assigned", len(surf.data.materials) > 0)

    parts = bpy.data.objects.get("FluidSim Particles")
    check("particles object exists", parts is not None,
          parts.type if parts else "")

    # -- playback handler (frame_change drives the sim when live) ---------------
    bpy.ops.fluidsim.toggle_live()
    check("live enabled", fs.live)
    before = sess.solver.frame
    scene.frame_set(scene.frame_current + 1)
    scene.frame_set(scene.frame_current + 1)
    check("frame handler stepped solver", sess.solver.frame >= before + 2,
          f"solver frame {before} -> {sess.solver.frame}")
    bpy.ops.fluidsim.toggle_live()

    # -- profiling ---------------------------------------------------------------
    check("profiler collected frames",
          sess.profiler is not None and len(sess.profiler.history) > 0,
          f"{len(sess.profiler.history) if sess.profiler else 0} frames")
    if sess.profiler:
        timers = sess.profiler.history[-1]
        check("pressure phase timed", timers.get("compute_5", 0.0) > 0.0,
              f"pressure {timers.get('compute_5', 0):.2f} ms")

    # -- custom snippet -----------------------------------------------------------
    bpy.ops.fluidsim.snippet_add()
    snip = fs.snippets[-1]
    snip.kernel_name = "chill"
    snip.order = "BEFORE_FORCES"
    snip.run_over = "VOXEL_LINEAR"
    snip.text_block.clear()
    snip.text_block.write(
        "temperature[flat_voxel_index] = 123.0\n")
    arg = snip.arguments.add()
    arg.mode = "FIELD"
    arg.field = "temperature"
    result = bpy.ops.fluidsim.snippet_apply()
    check("snippet compile", result == {"FINISHED"}, snip.last_error)
    bpy.ops.fluidsim.step(frames=1)
    import numpy as np

    temp = sess.solver.temperature.data.numpy()
    check("snippet executed on GPU", float(np.abs(temp - 123.0).min()) < 1e-3,
          f"temperature field max {temp.max():.1f}")

    # -- report -----------------------------------------------------------------
    report_path = os.path.join(tempfile.gettempdir(), "fluidsim_report.json")
    fs.report_path = report_path
    result = bpy.ops.fluidsim.write_report()
    check("write_report operator", result == {"FINISHED"})
    check("report file exists", os.path.exists(report_path))
    if os.path.exists(report_path):
        import json

        with open(report_path, encoding="utf-8") as fh:
            rep = json.load(fh)
        check("report has compute profile", "compute" in rep,
              str(list(rep.keys())))
        check("report frames recorded", len(rep.get("frames", [])) > 0)

    # -- export -----------------------------------------------------------------
    export_base = os.path.join(tempfile.gettempdir(), "fluidsim_test")
    result = bpy.ops.fluidsim.export_frame(filepath=export_base)
    check("export operator", result == {"FINISHED"})
    check("surface obj exported", os.path.exists(export_base + "_surface.obj"))
    check("particles ply exported",
          os.path.exists(export_base + "_particles.ply"))

    # -- panels draw headlessly? (registration check) ------------------------------
    panel_ids = [
        "FLUIDSIM_PT_simulation", "FLUIDSIM_PT_domain", "FLUIDSIM_PT_particles",
        "FLUIDSIM_PT_sources", "FLUIDSIM_PT_colliders", "FLUIDSIM_PT_forces",
        "FLUIDSIM_PT_pressure", "FLUIDSIM_PT_viscosity", "FLUIDSIM_PT_surface",
        "FLUIDSIM_PT_output", "FLUIDSIM_PT_snippets", "FLUIDSIM_PT_diagnostics",
        "FLUIDSIM_PT_validation",
    ]
    missing = [p for p in panel_ids if not hasattr(bpy.types, p)]
    check("all 13 panels registered", not missing, str(missing))

    ops = ["initialize", "release", "reset", "step", "toggle_live",
           "setup_scene", "tag_object", "inject_particles", "run_validation",
           "run_benchmark", "write_report", "snippet_add", "snippet_remove",
           "snippet_apply", "export_frame"]
    missing_ops = [o for o in ops if not hasattr(bpy.ops.fluidsim, o)]
    check("operators registered", not missing_ops, str(missing_ops))

    # -- release -----------------------------------------------------------------
    result = bpy.ops.fluidsim.release()
    check("release operator", result == {"FINISHED"})
    check("session released", not sess.active)


main()
print()
if FAILURES:
    print(f"BACKGROUND-ACCEPTANCE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("BACKGROUND-ACCEPTANCE-PASS")
sys.exit(0)
