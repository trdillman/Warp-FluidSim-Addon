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
SCENES = ("dam_break", "crown_splash", "slosh_tank", "maelstrom_basin")


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
