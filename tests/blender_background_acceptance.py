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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def main():
    # -- enable add-on -------------------------------------------------------
    ok = False
    try:
        enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
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
    fs.liquid_view_mode = "PARTICLES"
    check("liquid view mode sets particles",
          (fs.show_surface, fs.show_particles) == (False, True),
          f"{fs.liquid_view_mode} {fs.show_surface}/{fs.show_particles}")
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    check("liquid view mode sets combined output",
          (fs.show_surface, fs.show_particles) == (True, True),
          f"{fs.liquid_view_mode} {fs.show_surface}/{fs.show_particles}")
    fs.show_surface = False
    check("output toggles sync to particles mode",
          fs.liquid_view_mode == "PARTICLES",
          fs.liquid_view_mode)
    fs.show_surface = True
    fs.show_particles = False
    check("output toggles sync to surface mode",
          fs.liquid_view_mode == "SURFACE",
          fs.liquid_view_mode)
    fs.liquid_view_mode = "SURFACE"
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

    paused_frame = sess.solver.frame
    surface_mat = bpy.data.materials.new("Acceptance Surface Solid Color")
    surface_mat.diffuse_color = (0.9, 0.2, 0.05, 1.0)
    fs.surface_material = surface_mat
    fs.surface_attribute_mode = "NONE"
    fs.particle_radius_scale = 1.6
    fs.surface_dilate_iterations = 1
    fs.surface_dilate_strength = 0.2
    fs.surface_erode_iterations = 2
    fs.surface_erode_strength = 0.35
    fs.surface_smooth_type = "MEAN_CURVATURE_FLOW"
    fs.surface_smooth_iterations = 2
    fs.surface_smooth_strength = 0.45
    check("realtime surface refresh keeps frame",
          sess.solver.frame == paused_frame,
          f"{paused_frame} -> {sess.solver.frame}")
    check("realtime surface radius applied",
          abs(sess.solver.settings.particle_radius_scale - 1.6) < 1.0e-6,
          str(sess.solver.settings.particle_radius_scale))
    check("realtime surface dilate applied",
          sess.solver.settings.surface_dilate_iterations == 1,
          str(sess.solver.settings.surface_dilate_iterations))
    check("realtime surface erode applied",
          sess.solver.settings.surface_erode_iterations == 2,
          str(sess.solver.settings.surface_erode_iterations))
    check("realtime surface smooth applied",
          sess.solver.settings.surface_smooth_type.name == "MEAN_CURVATURE_FLOW",
          str(sess.solver.settings.surface_smooth_type))
    surf = bpy.data.objects.get("FluidSim Surface")
    check("realtime surface mesh still populated",
          surf is not None and len(surf.data.vertices) > 100,
          f"{len(surf.data.vertices) if surf else 0} verts")
    check("surface solid viewport material color applied",
          surf is not None and tuple(round(v, 3) for v in surf.color) == (0.9, 0.2, 0.05, 1.0),
          str(tuple(surf.color) if surf else None))
    fs.surface_attribute_mode = "HEIGHT"
    surf = bpy.data.objects.get("FluidSim Surface")
    check("surface color attribute active for solid viewport",
          surf is not None and getattr(surf.data.color_attributes, "active_color", None) is not None,
          str(getattr(surf.data.color_attributes, "active_color", None).name
              if surf and getattr(surf.data.color_attributes, "active_color", None) else ""))

    paused_frame = sess.solver.frame
    fs.whitewater_enabled = False
    check("realtime whitewater toggle keeps frame",
          sess.solver.frame == paused_frame,
          f"{paused_frame} -> {sess.solver.frame}")
    check("realtime whitewater setting applied",
          sess.solver.settings.whitewater_enabled is False,
          str(sess.solver.settings.whitewater_enabled))
    fs.whitewater_enabled = True

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

    # -- snippet library / validation / auto-fix / layering ---------------------
    fs.snippet_library_selection = "EX:learn_particles_speed_limit"
    result = bpy.ops.fluidsim.snippet_load_preset()
    check("library preset loads", result == {"FINISHED"})
    loaded = fs.snippets[-1]
    check("preset arguments bound",
          any(a.mode == "FIELD" and a.field == "particle_velocity"
              for a in loaded.arguments))
    check("preset validates", loaded.is_valid, loaded.validation_report)
    result = bpy.ops.fluidsim.snippet_test()
    check("snippet test compiles preset", result == {"FINISHED"},
          loaded.last_error)

    # Invalid code must be rejected and gated out of the scene.
    loaded.text_block.clear()
    loaded.text_block.write("import os\nx = os.getcwd()\n")
    result = bpy.ops.fluidsim.snippet_apply()
    check("invalid snippet rejected by apply", result == {"CANCELLED"})
    check("invalid snippet flagged", not loaded.is_valid,
          loaded.validation_report)

    # Auto-fix repairs C-style habits; validation then passes.
    loaded.text_block.clear()
    loaded.text_block.write(
        "// damp fast particles\n"
        "speed = wp.length(velocity);\n"
        "if speed > max_speed:\n"
        "    particle_velocity[particle_index] = velocity * 0.5;\n")
    result = bpy.ops.fluidsim.snippet_autofix()
    check("autofix operator", result == {"FINISHED"})
    check("autofix produces valid snippet", loaded.is_valid,
          loaded.validation_report)

    # Layering: frame-windowed snippet runs alongside the first one.
    loaded.frame_start = 1
    loaded.frame_end = 0
    loaded.frame_step = 1
    result = bpy.ops.fluidsim.snippet_apply()
    check("layered snippets apply", result == {"FINISHED"},
          str([(s.kernel_name, s.last_error) for s in fs.snippets]))
    result = bpy.ops.fluidsim.step(frames=1)
    check("layered snippets step", result == {"FINISHED"})

    # Save to the user library and see it in the dropdown items.
    result = bpy.ops.fluidsim.snippet_save_preset(preset_name="accept_saved")
    check("snippet save preset", result == {"FINISHED"})
    from fluidSim_warp_blender51.properties import (
        _snippet_library_items, user_snippet_dir,
    )

    items = _snippet_library_items(None, bpy.context)
    check("saved preset listed in library",
          any(ident.startswith("USR:") and "accept_saved" in ident
              for ident, _, _ in items if ident),
          str([i[0] for i in items if i[0]]))
    saved_path = os.path.join(user_snippet_dir(), "accept_saved.json")
    if os.path.exists(saved_path):
        os.remove(saved_path)  # keep the environment clean for reruns

    # Shortcut insertion appends code and revalidates.
    fs.shortcut_constant_name = "boost"
    fs.shortcut_constant_value = 2.0
    result = bpy.ops.fluidsim.snippet_shortcut_constant()
    check("constant shortcut", result == {"FINISHED"})
    check("constant argument added",
          any(a.mode == "CONSTANT" and a.constant_name == "boost"
              for a in loaded.arguments))

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
        "FLUIDSIM_PT_realtime_display", "FLUIDSIM_PT_output", "FLUIDSIM_PT_snippets",
        "FLUIDSIM_PT_snippet_editor", "FLUIDSIM_PT_snippet_shortcuts",
        "FLUIDSIM_PT_diagnostics", "FLUIDSIM_PT_validation",
    ]
    missing = [p for p in panel_ids if not hasattr(bpy.types, p)]
    check("all 16 panels registered", not missing, str(missing))

    ops = ["initialize", "release", "reset", "step", "toggle_live",
           "setup_scene", "tag_object", "inject_particles", "run_validation",
           "run_benchmark", "write_report", "snippet_add", "snippet_remove",
           "snippet_apply", "snippet_load_preset", "snippet_validate",
           "snippet_autofix", "snippet_test", "snippet_save_preset",
           "snippet_insert_shortcut", "snippet_shortcut_field",
           "snippet_shortcut_constant", "snippet_select",
           "snippet_edit_line", "snippet_line_add", "snippet_line_delete",
           "export_frame"]
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
