"""Open generated demo .blend files in Blender and verify saved scene contents.

Run with Blender:
    blender --background --python tools/check_demo_blends.py
"""

import json
import os
import sys

import bpy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon

DEMOS = os.path.join(REPO, "demos")
CACHE = os.path.join(REPO, "artifacts", "demo_cache")
SCENES = ("dam_break", "crown_splash", "slosh_tank", "tidal_lock",
          "material_response", "maelstrom_basin")


def load_meta(name):
    path = os.path.join(DEMOS, name, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def point_count(obj):
    if obj is None:
        return 0
    if obj.type == "MESH":
        return len(obj.data.vertices)
    if obj.type == "POINTCLOUD":
        return len(obj.data.points)
    return 0


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
    result["opened"] = True
    meta = load_meta(name)

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
                and near(source.location, (0.47105896, 0.0, 1.06185913))
                and near(source.scale, (0.73273778, 1.70176971, 0.99442428))),
            "sphere_collider": bool(
                collider and collider.fluidsim.role == "COLLIDER"
                and near(collider.location, (2.41923952, 0.0, 0.0))),
            "canonical_friction": bool(
                collider
                and abs(float(collider.fluidsim.collider_friction) - 1.0) < 1.0e-6),
            "no_redundant_empties": not any(
                obj.type == "EMPTY" for obj in scene.objects),
        })
    if name == "tidal_lock":
        checks.update({
            "surface_mesh": bool(surface and surface.type == "MESH" and len(surface.data.vertices) > 100),
            "particle_output": bool(particles and point_count(particles) > 100),
            "playable_live_setup": bool(meta.get("playable_live_setup")),
            "source_objects": all(
                bpy.data.objects.get(obj_name)
                for obj_name in meta.get("source_objects", [])
            ),
            "gate_objects": all(
                bpy.data.objects.get(obj_name)
                for obj_name in meta.get("gate_objects", [])
            ),
        })
    if name == "material_response":
        series_path = os.path.join(CACHE, name, "series.json")
        series = {}
        if os.path.exists(series_path):
            with open(series_path, encoding="utf-8") as fh:
                series = json.load(fh)
        source_intervals = []
        for obj_name in meta.get("source_objects", []):
            source = bpy.data.objects.get(obj_name)
            if source is not None:
                half = float(source.dimensions.y) * 0.5
                source_intervals.append(
                    (float(source.location.y) - half,
                     float(source.location.y) + half))
        source_intervals.sort()
        ramps = [
            bpy.data.objects.get(obj_name)
            for obj_name in meta.get("collider_objects", [])
            if " Ramp " in obj_name
        ]
        material_values = {
            (round(float(obj.fluidsim.collider_friction), 2),
             round(float(obj.fluidsim.collider_restitution), 2))
            for obj_name in meta.get("collider_objects", [])
            if (obj := bpy.data.objects.get(obj_name)) is not None
        }
        checks.update({
            "250_frame_timeline": scene.frame_end >= 250,
            "minimum_particle_scale": int(meta.get("injected", 0)) >= 250_000,
            "configured_particle_scale": bool(
                fs and int(fs.max_particles) >= 250_000),
            "generated_through_frame_250": bool(
                series.get("frames") and series["frames"][-1] == 250
                and len(series["frames"]) == 250),
            "four_material_combinations": material_values == {
                (0.0, 0.0), (0.9, 0.0), (0.0, 0.9), (0.75, 0.75)},
            "eight_material_colliders": len(meta.get("collider_objects", [])) == 8,
            "four_live_sources": len(meta.get("source_objects", [])) == 4,
            "non_overlapping_sources": bool(
                len(source_intervals) == 4
                and all(source_intervals[i][1] < source_intervals[i + 1][0]
                        for i in range(3))),
            "emitters_over_high_end": bool(
                len(ramps) == 4
                and all(float(ramp.rotation_euler.y) > 0.0 for ramp in ramps)),
            "timeline_live_enabled": bool(fs and fs.live),
            "visible_material_legend": bool(
                bpy.data.objects.get("Material Response Legend")),
            "particle_output": bool(particles and point_count(particles) >= 250_000),
            "playable_live_setup": bool(meta.get("playable_live_setup")),
            "no_redundant_empties": not any(
                obj.type == "EMPTY" for obj in scene.objects),
        })
    if name == "maelstrom_basin":
        seed = bpy.data.objects.get(meta.get("seed_object", ""))
        force = bpy.data.objects.get(meta.get("spin_force_object", ""))
        checks.update({
            "surface_mesh": bool(surface and surface.type == "MESH" and len(surface.data.vertices) > 100),
            "surface_material": bool(surface and surface.data.materials),
            "particle_output": bool(particles and point_count(particles) > 100),
            "whitewater_output": bool(whitewater and point_count(whitewater) > 0),
            "playable_live_setup": bool(meta.get("playable_live_setup")),
            "auto_inject_seed": bool(seed and point_count(seed) > 0),
            "spin_force_object": bool(force and meta.get("spin_strength", 0) >= 5.0),
            "million_particle_measure": int(meta.get("injected", 0)) >= 1_000_000,
        })
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
