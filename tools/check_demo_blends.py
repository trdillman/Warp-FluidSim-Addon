"""Open generated demo .blend files in Blender and verify saved scene contents.

Run with Blender:
    blender --background --python tools/check_demo_blends.py
"""

import json
import os
import sys

import addon_utils
import bpy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMOS = os.path.join(REPO, "demos")
SCENES = ("dam_break", "crown_splash", "slosh_tank")


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
    addon_utils.enable("fluidSim_warp_blender51", default_set=False)
    scene = bpy.context.scene
    result["opened"] = True

    fs = getattr(scene, "fluidsim", None)
    surface = bpy.data.objects.get("FluidSim Surface")
    particles = bpy.data.objects.get("FluidSim Particles")
    whitewater = bpy.data.objects.get("FluidSim Whitewater")

    result["checks"] = {
        "fluidsim_settings": fs is not None,
        "domain_object": bool(fs and fs.domain_object),
        "surface_mesh": bool(surface and surface.type == "MESH" and len(surface.data.vertices) > 100),
        "surface_material": bool(surface and surface.data.materials),
        "particle_output": bool(particles and point_count(particles) > 100),
        "camera": bool(scene.camera),
        "whitewater_output": True if name == "slosh_tank" else bool(whitewater and point_count(whitewater) > 0),
    }
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
