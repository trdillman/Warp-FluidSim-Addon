"""End-to-end collider material response gate for the canonical demo."""

from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.repo_addon_loader import enable_repo_addon

enable_repo_addon(ROOT, default_set=True, persistent=True)
bpy.ops.wm.open_mainfile(filepath=os.path.join(ROOT, "demos", "material_response.blend"))

from fluidSim_warp_blender51.blender.scene_bridge import initialize_session
from tools.build_demo_scenes import lattice

scene = bpy.context.scene
scene.fluidsim.trace_mode = "NONE"
from fluidSim_warp_blender51.blender import colliders

old_targets = [obj for obj in scene.objects if "Impact Sphere" in obj.name]
for old in old_targets:
    y = old.location.y
    friction = old.fluidsim.collider_friction
    restitution = old.fluidsim.collider_restitution
    bpy.data.objects.remove(old, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(4.55, y, 1.60))
    wall = bpy.context.active_object
    wall.name = f"Lane {y:+.1f} Impact Wall"
    wall.scale = (0.20, 1.05, 3.20)
    colliders.tag_collider(wall, friction=friction, restitution=restitution)
bpy.context.view_layer.update()
for obj in scene.objects:
    pg = getattr(obj, "fluidsim", None)
    if pg is not None and pg.role == "SOURCE":
        pg.enabled = False
session = initialize_session(scene)
session.reset(scene)
ys = (-1.8, -0.6, 0.6, 1.8)
for lane, y in enumerate(ys):
    session.solver.inject_particles(
        lattice((0.45, y - 0.32, 1.85), (1.65, y + 0.32, 2.55), 0.018),
        velocity=(5.5, 0.0, -0.15), temperature=lane / 3.0)

peaks = [{"z95": 0.0, "vz95": -1.0e9} for _ in range(4)]
for frame in range(1, 49):
    session.step(scene, write_cache=False)
    if 12 <= frame <= 32:
        p = session.solver.particles
        n = p.refresh_host_count()
        pos = p.position[:n].numpy()
        vel = p.velocity[:n].numpy()
        temp = p.attr("temperature")[:n].numpy()
        for lane in range(4):
            mask = np.abs(temp - lane / 3.0) < 0.08
            peaks[lane]["z95"] = max(
                peaks[lane]["z95"], float(np.quantile(pos[mask, 2], 0.95)))
            peaks[lane]["vz95"] = max(
                peaks[lane]["vz95"], float(np.quantile(vel[mask, 2], 0.95)))

p = session.solver.particles
n = p.refresh_host_count()
positions = p.position[:n].numpy()
velocities = p.velocity[:n].numpy()
temperature = p.attr("temperature")[:n].numpy()
lanes = []
for lane in range(4):
    mask = np.abs(temperature - lane / 3.0) < 0.08
    pos = positions[mask]
    vel = velocities[mask]
    lanes.append({
        "x50": float(np.median(pos[:, 0])),
        "z95": float(np.quantile(pos[:, 2], 0.95)),
        "vx50": float(np.median(vel[:, 0])),
        "vx05": float(np.quantile(vel[:, 0], 0.05)),
        "backscatter": float(np.mean(vel[:, 0] < -0.1)),
        "vz95": float(np.quantile(vel[:, 2], 0.95)),
    })

report = {"frame": 48, "lanes": lanes, "impact_peaks": peaks}
path = os.path.join(ROOT, "artifacts", "material_response_lane_gate.json")
with open(path, "w", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)

slide, grip, bounce, combined = lanes
if slide["x50"] - grip["x50"] < 0.20:
    raise RuntimeError(f"friction response is not distinct: {report}")
if bounce["backscatter"] - slide["backscatter"] < 0.05:
    raise RuntimeError(f"restitution response is not distinct: {report}")
print("MATERIAL-RESPONSE-LANE-GATE-PASS")
