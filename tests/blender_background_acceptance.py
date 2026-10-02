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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _test_device import script_device  # noqa: E402

from tools.repo_addon_loader import ADDON_NAME, enable_repo_addon

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

    check("persistent addon preferences entry registered",
          ADDON_NAME in bpy.context.preferences.addons)
    check("persistent runtime refresh operator completes",
          bpy.ops.fluidsim.refresh_runtime_probe() == {"FINISHED"})
    check("persistent device redetect operator completes",
          bpy.ops.fluidsim.redetect_device() == {"FINISHED"})

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
    fs.show_particles = True

    # -- quick setup ----------------------------------------------------------
    result = bpy.ops.fluidsim.setup_scene()
    check("setup_scene operator", result == {"FINISHED"})
    check("domain object assigned", fs.domain_object is not None)

    from fluidSim_warp_blender51.blender import scene_bridge

    domain_fs = scene_bridge.active_domain_settings(scene)
    check(
        "authoritative domain settings resolved",
        domain_fs is not None and domain_fs.id_data == fs.domain_object,
    )
    domain_fs.voxel_size = 0.08
    domain_fs.max_particles = 500_000
    domain_fs.whitewater_enabled = True
    domain_fs.profiling_enabled = True
    domain_fs.directional_force = (0.0, 0.0, 0.1)
    domain_fs.omnidirectional_force_center = (0.0, 0.0, 1.0)
    domain_fs.omnidirectional_force_strength = 0.05
    domain_fs.omnidirectional_force_radius = 1.0
    domain_fs.force_diffusion = 0.05
    domain_fs.dissipation_enabled = True
    domain_fs.dissipation_attribute = "density"
    domain_fs.dissipation_rate = 0.1
    domain_fs.dissipation_target = 1.0
    domain_fs.attraction_enabled = True
    domain_fs.attraction_strength = 0.05
    domain_fs.attraction_max_distance = 0.5

    src = next(
        (
            obj for obj in scene.objects
            if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
            == "SOURCE"
        ),
        None,
    )
    check("source tagged", src is not None and src.fluidsim.role == "SOURCE")
    if src is not None:
        src.fluidsim.force_type = "SPIN"
        src.fluidsim.force_strength = 0.05
        src.fluidsim.force_direction = (0.0, 0.0, 1.0)
    implicit_colliders = [
        obj for obj in scene.objects
        if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
        == "COLLIDER"
    ]
    check(
        "quick liquid does not invent an implicit collider",
        not implicit_colliders,
        str([obj.name for obj in implicit_colliders]),
    )

    force_specs = (
        ("FORCE", -0.03), ("WIND", 0.02), ("VORTEX", 0.01),
        ("MAGNET", 0.01), ("HARMONIC", 0.01), ("CHARGE", 0.01),
        ("LENNARDJ", 0.01), ("TEXTURE", 0.01), ("GUIDE", 0.01),
        ("BOID", 0.01), ("TURBULENCE", 0.01), ("DRAG", 0.01),
        ("FLUID", 0.01),
    )
    for field_type, strength in force_specs:
        bpy.ops.object.effector_add(type=field_type, location=(0.0, 0.0, 1.0))
        field = bpy.context.object.field
        field.strength = strength
        field.use_max_distance = True
        field.distance_max = 2.0
        if field.type == "TEXTURE":
            field.texture = bpy.data.textures.new("Acceptance Texture", type="CLOUDS")

    # -- initialize solver -----------------------------------------------------
    result = bpy.ops.fluidsim.initialize()
    check("initialize operator", result == {"FINISHED"}, fs.status)

    sess = scene_bridge.session()
    check("session active", sess.active)
    check("CUDA device", str(sess.solver.device).startswith("cuda"),
          str(sess.solver.device))
    check("global force settings bridged",
          all(abs(actual - expected) < 1.0e-6 for actual, expected in zip(
              sess.solver.settings.directional_force, (0.0, 0.0, 0.1)))
          and sess.solver.settings.dissipation_attribute == "density"
          and sess.solver.settings.attraction_enabled,
          str(sess.solver.settings))
    check("native Blender force fields bridged",
          [field.force_type.name for field in sess.solver.force_fields]
          == ["FORCE", "WIND", "VORTEX", "MAGNETIC", "HARMONIC",
              "CHARGE", "LENNARD_JONES", "TEXTURE", "CURVE_GUIDE",
              "BOID", "TURBULENCE", "DRAG", "FLUID_FLOW"],
          str(sess.solver.force_fields))

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
    domain_fs.particle_radius_scale = 1.6
    domain_fs.surface_dilate_iterations = 1
    domain_fs.surface_dilate_strength = 0.2
    domain_fs.surface_erode_iterations = 2
    domain_fs.surface_erode_strength = 0.35
    domain_fs.surface_smooth_type = "MEAN_CURVATURE_FLOW"
    domain_fs.surface_smooth_iterations = 2
    domain_fs.surface_smooth_strength = 0.45
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
    domain_fs.whitewater_enabled = False
    check("realtime whitewater toggle keeps frame",
          sess.solver.frame == paused_frame,
          f"{paused_frame} -> {sess.solver.frame}")
    check("realtime whitewater setting applied",
          sess.solver.settings.whitewater_enabled is False,
          str(sess.solver.settings.whitewater_enabled))
    domain_fs.whitewater_enabled = True

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
    check("operator live disabled", not fs.live)

    before = sess.solver.frame
    fs.live = True
    check("direct live property enabled",
          fs.live and fs.status == "Live simulation enabled - press play",
          fs.status)
    scene.frame_set(scene.frame_current + 1)
    scene.frame_set(scene.frame_current + 1)
    check("direct live property drives frame handler",
          sess.solver.frame >= before + 2,
          f"solver frame {before} -> {sess.solver.frame}")
    fs.live = False
    check("direct live property disabled",
          not fs.live and fs.status == "Live simulation disabled",
          fs.status)

    # -- profiling ---------------------------------------------------------------
    check("profiler collected frames",
          sess.profiler is not None and len(sess.profiler.history) > 0,
          f"{len(sess.profiler.history) if sess.profiler else 0} frames")
    if sess.profiler:
        timers = sess.profiler.history[-1]
        check("pressure phase timed", timers.get("compute_5", 0.0) > 0.0,
              f"pressure {timers.get('compute_5', 0):.2f} ms")

    # -- Custom Effects v2 -------------------------------------------------------
    from fluidSim_warp_blender51.blender.effects import add_builtin_effect

    add_builtin_effect(scene, "fluidsim.vortex_force")
    effect_stack = scene.fluidsim_effects
    check("v2 effect added", len(effect_stack.items) == 1
          and effect_stack.items[0].effect_id == "fluidsim.vortex_force")

    result = bpy.ops.fluidsim.effect_apply_stack(device=script_device())
    check("v2 effect compile and apply", result == {"FINISHED"},
          effect_stack.blocking_reason)
    result = bpy.ops.fluidsim.step(frames=1)
    check("v2 effect step", result == {"FINISHED"})
    effect_stats = list(sess.solver.effects.stats.values())
    check("v2 effect executed on GPU",
          any(item.effect_id == "fluidsim.vortex_force" and item.calls > 0
              for item in effect_stats),
          str([(item.effect_id, item.calls) for item in effect_stats]))

    # A bad draft must fail closed without replacing the applied live stack.
    applied_signature = effect_stack.applied_signature
    kernel_text = effect_stack.items[0].kernels[0].source_text
    kernel_text.clear()
    kernel_text.write("import os\nx = os.getcwd()\n")
    try:
        result = bpy.ops.fluidsim.effect_apply_stack(device=script_device())
        rejection = effect_stack.blocking_reason
    except RuntimeError as exc:
        result = {"CANCELLED"}
        rejection = str(exc)
    check("invalid v2 draft rejected", result == {"CANCELLED"}, rejection)
    check("invalid draft preserves applied stack",
          effect_stack.applied_signature == applied_signature)
    result = bpy.ops.fluidsim.effect_revert_draft()
    check("revert restores applied effect", result == {"FINISHED"}
          and effect_stack.status == "APPLIED")

    # Layering is owned by the ordered v2 stack and must hot-swap atomically.
    add_builtin_effect(scene, "fluidsim.curl_turbulence")
    result = bpy.ops.fluidsim.effect_apply_stack(device=script_device())
    check("layered v2 effects apply", result == {"FINISHED"},
          effect_stack.blocking_reason)
    result = bpy.ops.fluidsim.step(frames=1)
    check("layered v2 effects step", result == {"FINISHED"})
    executed_ids = {
        item.effect_id for item in sess.solver.effects.stats.values()
        if item.calls > 0
    }
    check("layered v2 effects executed",
          {"fluidsim.vortex_force", "fluidsim.curl_turbulence"}
          <= executed_ids, str(sorted(executed_ids)))

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
        "FLUIDSIM_PT_scene_control", "FLUIDSIM_PT_scene_pending",
        "FLUIDSIM_PT_scene_runtime_stats", "FLUIDSIM_PT_scene_domain",
        "FLUIDSIM_PT_scene_simulation", "FLUIDSIM_PT_sim_pressure",
        "FLUIDSIM_PT_sim_viscosity", "FLUIDSIM_PT_scene_surface",
        "FLUIDSIM_PT_scene_whitewater", "FLUIDSIM_PT_scene_outputs",
        "FLUIDSIM_PT_viewport_status", "FLUIDSIM_PT_header_popover",
        "FLUIDSIM_STAGE_PT_bake",
        "FLUIDSIM_STAGE_PT_bake_storage", "FLUIDSIM_PT_scene_diagnostics",
        "FLUIDSIM_PT_object_role", "FLUIDSIM_PT_object_source",
        "FLUIDSIM_PT_object_collider", "FLUIDSIM_PT_object_force",
        "FLUIDSIM_PT_custom_effects", "FLUIDSIM_PT_effect_packages",
        "FLUIDSIM_PT_effect_warp_code",
    ]
    missing = [p for p in panel_ids if not hasattr(bpy.types, p)]
    check("all Phase 3 panels registered", not missing, str(missing))

    ops = ["initialize", "release", "reset", "step", "toggle_live",
           "setup_scene", "tag_object", "tag_liquid_emitter", "tag_liquid_sink",
           "tag_collider", "tag_force_region", "tag_field_source", "clear_object_role",
           "inject_particles", "run_validation",
           "run_benchmark", "write_report", "effect_add", "effect_remove", "effect_duplicate",
           "effect_move", "effect_test_draft", "effect_apply_stack",
           "effect_revert_draft", "effect_preview_one_frame",
           "effect_create_controller", "effect_export_package", "export_frame",
           "open_workspace"]
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
