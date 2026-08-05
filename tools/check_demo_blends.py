"""Open generated demo .blend files in Blender and verify saved scene contents.

Run with Blender:
    blender --background --python tools/check_demo_blends.py
"""

import json
import os
import sys

import bpy
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon

DEMOS = os.path.join(REPO, "demos")
SCENES = (
    "dam_break", "crown_splash", "force_fields", "adhesive_calender",
    "mesh_solid_source", "mesh_shell_source", "point_source",
    "volume_source", "liquid_sink", "height_fill",
)


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(obj)


def check_scene(name):
    path = os.path.join(DEMOS, name + ".blend")
    result = {
        "scene": name,
        "path": path,
        "exists": os.path.exists(path),
        "opened": False,
        "checks": {},
    }
    if not result["exists"]:
        return result

    bpy.ops.wm.open_mainfile(filepath=path)
    enable_repo_addon(REPO, default_set=False, persistent=False)
    scene = bpy.context.scene
    saved_frame = scene.frame_current
    result["opened"] = True

    fs = getattr(scene, "fluidsim", None)
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")

    checks = {
        "fluidsim_settings": fs is not None,
        "domain_object": bool(fs and fs.domain_object),
        "camera": bool(scene.camera),
    }
    if name == "dam_break":
        domain = bpy.data.objects.get("DamBreak Domain")
        source = bpy.data.objects.get("DamBreak Source Column")
        collider = bpy.data.objects.get("Sphere")
        near = lambda actual, expected: all(
            abs(float(a) - float(e)) < 1.0e-4
            for a, e in zip(actual, expected)
        )
        checks.update({
            "canonical_domain_scale": bool(
                domain and near(domain.dimensions, (5.6, 2.00292993, 1.6))),
            "canonical_source_transform": bool(
                source
                and near(source.location, (0.3, 0.3, 0.6))
                and near(source.dimensions, (0.6, 0.6, 1.2))
                and source.fluidsim.emission_mode == "INITIAL_FILL"
                and source.fluidsim.source_region_kind == "ANALYTIC"),
            "sphere_collider": bool(
                collider and collider.fluidsim.role == "COLLIDER"
                and near(collider.location, (2.41923952, 0.0, 0.0))),
            "canonical_friction": bool(
                collider
                and abs(float(collider.fluidsim.collider_friction) - 1.0) < 1.0e-6),
            "no_redundant_empties": not any(
                obj.type == "EMPTY" for obj in scene.objects),
        })
    elif name == "force_fields":
        force_fields = [
            obj for obj in scene.objects
            if getattr(getattr(obj, "field", None), "type", "NONE") != "NONE"
        ]
        expected_types = {
            "FORCE", "WIND", "VORTEX", "MAGNET", "HARMONIC", "CHARGE",
            "LENNARDJ", "TEXTURE", "GUIDE", "BOID", "TURBULENCE",
            "DRAG", "FLUID_FLOW",
        }
        texture = next(
            (obj for obj in force_fields if obj.field.type == "TEXTURE"), None)
        guide = next(
            (obj for obj in force_fields if obj.field.type == "GUIDE"), None)
        checks.update({
            "all_13_force_types": (
                len(force_fields) == 13
                and {obj.field.type for obj in force_fields} == expected_types),
            "force_proxy_metadata": (
                scene.get("fluidsim_force_field_types") == 13
                and bpy.data.texts.get("README_Force_Fields") is not None),
            "texture_proxy_configured": bool(
                texture and texture.field.texture
                and texture.field.texture_mode == "CURL"),
            "curve_guide_has_controls": bool(
                guide and guide.type == "CURVE" and guide.data.splines
                and len(guide.data.splines[0].points) == 4),
            "saved_prestepped_frame": scene.frame_current == 19,
        })
    elif name in {
            "mesh_solid_source", "mesh_shell_source", "point_source",
            "volume_source", "liquid_sink", "height_fill"}:
        checks["phase3_workflow_metadata"] = scene.get("fluidsim_phase3_workflow") == name
        if name == "mesh_solid_source":
            obj = bpy.data.objects.get("Watertight Mesh Solid Emitter")
            checks["real_mesh_solid_emitter"] = bool(
                obj and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "MESH_SOLID"
                and obj.fluidsim.emission_mode == "INITIAL_FILL")
        elif name == "mesh_shell_source":
            obj = bpy.data.objects.get("Open Mesh Surface Shell Emitter")
            checks["real_mesh_shell_emitter"] = bool(
                obj and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "MESH_SHELL"
                and obj.fluidsim.mesh_surface_thickness > 0.0)
        elif name == "point_source":
            obj = bpy.data.objects.get("Rasterized Point Emitter")
            checks["real_rasterized_point_source"] = bool(
                obj and obj.type == "POINTCLOUD" and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "POINTS"
                and obj.fluidsim.point_source_representation == "RASTERIZED_REGION")
        elif name == "volume_source":
            obj = bpy.data.objects.get("Fog Grid Liquid Emitter")
            checks["real_volume_source"] = bool(
                obj and obj.type == "VOLUME" and obj.fluidsim.role == "SOURCE"
                and obj.fluidsim.source_region_kind == "VOLUME_GRID"
                and obj.fluidsim.volume_mask_grid == "density")
        elif name == "liquid_sink":
            obj = bpy.data.objects.get("Mesh Solid Liquid Sink")
            checks["real_liquid_sink"] = bool(
                obj and obj.fluidsim.role == "SINK"
                and obj.fluidsim.sink_region_kind == "MESH_SOLID")
        elif name == "height_fill":
            checks["real_profile_height_fill"] = bool(
                fs and fs.height_enabled and fs.height_back_mode == "PROFILE"
                and fs.height_profile_image is not None
                and fs.height_emission_mode == "INITIAL_FILL")
    elif name == "adhesive_calender":
        low_seed = bpy.data.objects.get("Low Viscosity Adhesive Seed")
        high_seed = bpy.data.objects.get("High Viscosity Adhesive Seed")
        divider = bpy.data.objects.get("Viscosity Lane Divider")
        force_fields = [
            obj for obj in scene.objects
            if getattr(getattr(obj, "field", None), "type", "NONE") != "NONE"
        ]
        field_types = [obj.field.type for obj in force_fields]
        checks.update({
            "million_particle_contract": bool(
                fs and fs.max_particles == 1_000_000
                and scene.get("fluidsim_initial_particle_count") == 1_000_000),
            "250_frame_timeline": bool(
                scene.frame_end == 250 and saved_frame == 1
                and scene.get("fluidsim_simulated_frames") == 250),
            "four_max_substeps": bool(
                fs and fs.max_substeps == 4
                and 1 <= scene.get("fluidsim_max_observed_substeps", 0) <= 4),
            "whitewater_enabled": bool(fs and fs.whitewater_enabled),
            "mirrored_viscosity_seeds": bool(
                low_seed and high_seed
                and low_seed.type == high_seed.type == "MESH"
                and len(low_seed.data.vertices) == 500_000
                and len(high_seed.data.vertices) == 500_000
                and low_seed.fluidsim.enabled
                and high_seed.fluidsim.enabled
                and low_seed.fluidsim.auto_inject_seed
                and high_seed.fluidsim.auto_inject_seed
                and abs(float(low_seed.fluidsim.viscosity) - 0.002) < 1.0e-8
                and abs(float(high_seed.fluidsim.viscosity) - 0.5) < 1.0e-8),
            "live_playback_saved": bool(fs and fs.live),
            "fluid_flow_and_drag_fields": bool(
                len(force_fields) == 3
                and field_types.count("FLUID_FLOW") == 2
                and field_types.count("DRAG") == 1
                and not any(kind in field_types for kind in (
                    "VORTEX", "HARMONIC", "TURBULENCE"))),
            "per_particle_viscosity": bool(
                fs and fs.viscosity_enabled and fs.viscosity_from_particles
                and scene.get("fluidsim_low_viscosity") == 0.002
                and scene.get("fluidsim_high_viscosity") == 0.5),
            "separated_lanes": bool(
                divider and divider.fluidsim.role == "COLLIDER"
                and float(divider.dimensions.y) >= 0.239),
            "measured_final_state": bool(
                scene.get("fluidsim_final_particle_count", 0) > 0
                and scene.get("fluidsim_final_low_core_coupling", 0.0) >= 0.0
                and scene.get("fluidsim_final_high_core_coupling", 0.0) >= 0.0
                and scene.get("fluidsim_measured_viscosity_effect") is True),
            "review_workspace_and_clips": bool(
                bpy.data.texts.get("README_Adhesive_Calender")
                and all(bpy.data.objects.get(camera) for camera in (
                    "Camera Main", "Camera Top", "Camera Profile"))
                and "Adhesive Calender" in bpy.context.workspace.name),
        })
        if checks["mirrored_viscosity_seeds"] and checks["live_playback_saved"]:
            try:
                from fluidSim_warp_blender51.blender.scene_bridge import session

                scene.frame_set(2)
                sess = session()
                before = sess.solver.particles.position[:1024].numpy().copy()
                scene.frame_set(3)
                after = sess.solver.particles.position[:1024].numpy()
                delta = float(np.max(np.linalg.norm(after - before, axis=1)))
                checks["viewport_playback_advances"] = bool(
                    sess.active
                    and sess.solver.particles.refresh_host_count() == 1_000_000
                    and delta > 1.0e-6)
                result["playback_position_delta"] = delta
            except Exception as exc:  # noqa: BLE001
                checks["viewport_playback_advances"] = False
                result["playback_error"] = repr(exc)
            finally:
                fs.live = False
                bpy.ops.fluidsim.release()
        else:
            checks["viewport_playback_advances"] = False

    result["checks"] = checks
    result["surface_vertices"] = len(surface.data.vertices) if surface and surface.type == "MESH" else 0
    result["particle_points"] = point_count(particles)
    result["whitewater_points"] = point_count(whitewater)
    return result


def main():
    results = [check_scene(name) for name in SCENES]
    passed = all(
        item["exists"] and item["opened"] and all(item["checks"].values())
        for item in results
    )
    print(json.dumps(results, indent=2))
    print("DEMO-BLEND-CHECK-" + ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)


main()
