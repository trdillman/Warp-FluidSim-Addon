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
    "dam_break", "crown_splash", "chromatic_vortex", "force_fields",
    "adhesive_calender",
    "cache_export_playback_review", "paddle_stir",
    "mesh_solid_source", "mesh_shell_source", "point_source",
    "volume_source", "liquid_sink", "height_fill",
)
EXPECTED_AUTO_SEEDS = {
    "crown_splash": {"Crown Pool Seed", "Crown Droplet Seed"},
    "force_fields": {"Force Fields Particle Seed"},
    "adhesive_calender": {
        "Low Viscosity Adhesive Seed", "High Viscosity Adhesive Seed"},
    "paddle_stir": {"Paddle Stir Pool Seed"},
}


def point_count(obj):
    # Output clouds carry spare capacity, so len(data.points) is not the count.
    from fluidSim_warp_blender51.blender import output_publication

    return output_publication.live_point_count(obj)


def output_snapshot(limit=32):
    """Return a bounded sample of whichever owned viewport outputs exist."""
    snapshot = []
    visible_count = 0
    for name in ("FluidSim Particles", "FluidSim Whitewater", "FluidSim Surface"):
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if obj.type == "POINTCLOUD":
            count = point_count(obj)
            position = obj.data.attributes.get("position")
            values = (() if position is None else tuple(
                round(float(value), 6)
                for item in position.data[:min(limit, count)]
                for value in item.vector
            ))
        elif obj.type == "MESH":
            count = len(obj.data.vertices)
            values = tuple(
                round(float(value), 6)
                for vertex in obj.data.vertices[:min(limit, count)]
                for value in vertex.co
            )
        else:
            continue
        visible_count += count
        snapshot.append((name, count, values))
    return visible_count, tuple(snapshot)


def check_playback_contract(scene, fs, *, cache_mode=False):
    """Exercise the saved timeline contract through real frame handlers."""
    result = {"ready": False, "advances": False}
    if fs is None:
        return result
    expected_mode = bool(fs.live and bool(fs.cache_playback_enabled) == cache_mode)
    result["ready"] = bool(
        expected_mode
        and scene.frame_current == scene.frame_start
        and scene.get("fluidsim_playback_ready") is True)
    if not result["ready"]:
        return result

    from fluidSim_warp_blender51.blender.scene_bridge import session

    _before_count, before = output_snapshot()
    try:
        scene.frame_set(scene.frame_start + 1)
        middle_count, middle = output_snapshot()
        scene.frame_set(scene.frame_start + 2)
        after_count, after = output_snapshot()
        sess = session()
        if cache_mode:
            state_valid = not sess.active
        else:
            state_valid = bool(
                sess.active
                and sess.solver.particles.refresh_host_count() > 0)
        result["advances"] = bool(
            state_valid
            and max(middle_count, after_count) > 0
            and (middle != before or after != middle))
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)
    return result


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
        "valid_domain_boundary_policy": bool(
            fs and (fs.open_boundaries or fs.domain_sdf_walls)),
    }
    actual_auto_seeds = {
        obj.name for obj in scene.objects
        if getattr(getattr(obj, "fluidsim", None), "auto_inject_seed", False)
    }
    checks["owned_auto_seeds"] = (
        actual_auto_seeds == EXPECTED_AUTO_SEEDS.get(name, set()))
    if not checks["owned_auto_seeds"]:
        result["actual_auto_seeds"] = sorted(actual_auto_seeds)
    playback = check_playback_contract(scene, fs, cache_mode=False)
    checks["playback_ready"] = playback["ready"]
    checks["playback_advances"] = playback["advances"]
    if playback.get("error"):
        result["playback_contract_error"] = playback["error"]
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
    elif name == "crown_splash":
        pool_seed = bpy.data.objects.get("Crown Pool Seed")
        drop_seed = bpy.data.objects.get("Crown Droplet Seed")
        seed_contract = bool(
            pool_seed and drop_seed
            and pool_seed.type == drop_seed.type == "MESH"
            and pool_seed.fluidsim.enabled
            and drop_seed.fluidsim.enabled
            and pool_seed.fluidsim.auto_inject_seed
            and drop_seed.fluidsim.auto_inject_seed
            and abs(float(pool_seed.fluidsim.initial_velocity.z)) < 1.0e-6
            and abs(float(drop_seed.fluidsim.initial_velocity.z) + 3.5) < 1.0e-6
        )
        checks.update({
            "restartable_seed_contract": seed_contract,
            "live_playback_saved": bool(
                fs and fs.live and saved_frame == 1
                and scene.get("fluidsim_playback_ready") is True),
        })
        if seed_contract and checks["live_playback_saved"]:
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
                    and sess.solver.particles.refresh_host_count() > 0
                    and point_count(bpy.data.objects.get("FluidSim Particles")) > 0
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
    elif name == "chromatic_vortex":
        red_source = bpy.data.objects.get("Red Dye Initial Fill")
        blue_source = bpy.data.objects.get("Blue Dye Initial Fill")
        vortex = bpy.data.objects.get("Chromatic Mixing Vortex Force")
        live_surface = bpy.data.objects.get("FluidSim Surface")
        color_attr = (
            live_surface.data.color_attributes.get("display_color")
            if live_surface and live_surface.type == "MESH" else None)
        live_colors = np.asarray(
            [item.color[:3] for item in color_attr.data], dtype=np.float32
        ) if color_attr and len(color_attr.data) else np.empty((0, 3))
        checks.update({
            "color_mixing_enabled": bool(
                fs and fs.species_enabled and fs.species_mixing_rate > 0.0),
            "two_colored_initial_fills": bool(
                red_source and blue_source
                and red_source.fluidsim.role == "SOURCE"
                and blue_source.fluidsim.role == "SOURCE"
                and red_source.fluidsim.emission_mode == "INITIAL_FILL"
                and blue_source.fluidsim.emission_mode == "INITIAL_FILL"
                and red_source.fluidsim.source_color[0]
                > red_source.fluidsim.source_color[2]
                and blue_source.fluidsim.source_color[2]
                > blue_source.fluidsim.source_color[0]),
            "owned_spin_force": bool(
                vortex and vortex.fluidsim.role == "FORCE"
                and vortex.fluidsim.force_type == "SPIN"
                and vortex.fluidsim.force_strength > 0.0),
            "live_two_color_output": bool(
                len(live_colors)
                and np.any(live_colors[:, 0] > live_colors[:, 2] + 0.15)
                and np.any(live_colors[:, 2] > live_colors[:, 0] + 0.15)),
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
            "saved_live_frame": bool(
                saved_frame == 1
                and scene.get("fluidsim_playback_ready") is True),
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
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")
    result["surface_vertices"] = len(surface.data.vertices) if surface and surface.type == "MESH" else 0
    result["particle_points"] = point_count(particles)
    result["whitewater_points"] = point_count(whitewater)
    return result


def main():
    disk_scenes = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(DEMOS)
        if filename.endswith(".blend")
    }
    declared_scenes = set(SCENES)
    if disk_scenes != declared_scenes:
        print(json.dumps({
            "undeclared_demo_blends": sorted(disk_scenes - declared_scenes),
            "missing_demo_blends": sorted(declared_scenes - disk_scenes),
        }, indent=2))
        raise SystemExit("DEMO-BLEND-CHECK-FAIL (scene inventory mismatch)")
    results = [check_scene(name) for name in SCENES]
    passed = all(
        item["exists"] and item["opened"] and all(item["checks"].values())
        for item in results
    )
    print(json.dumps(results, indent=2))
    print("DEMO-BLEND-CHECK-" + ("PASS" if passed else "FAIL"))
    sys.exit(0 if passed else 1)


main()
