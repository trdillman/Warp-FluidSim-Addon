"""Prove the chromatic-vortex demo starts inside a real vessel collider."""

from __future__ import annotations

import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon


bpy.ops.wm.open_mainfile(filepath=os.path.join(REPO, "demos", "chromatic_vortex.blend"))
enable_repo_addon(REPO, default_set=True, persistent=True)

scene = bpy.context.scene
vessel = bpy.data.objects["Chromatic Glass Vessel"]
sources = (
    bpy.data.objects["Red Dye Initial Fill"],
    bpy.data.objects["Blue Dye Initial Fill"],
)
assert vessel.fluidsim.role == "COLLIDER", "vessel is not a FluidSim collider"
assert vessel.fluidsim.enabled, "vessel collider is disabled"

edge_users = {}
for polygon in vessel.data.polygons:
    for edge_key in polygon.edge_keys:
        edge_users[edge_key] = edge_users.get(edge_key, 0) + 1
assert edge_users and all(count == 2 for count in edge_users.values()), (
    "vessel collision mesh is not closed")

vessel_radii = [
    np.hypot(*(vessel.matrix_world @ vertex.co)[:2])
    for vertex in vessel.data.vertices
]
vessel_radius = max(vessel_radii)
inner_radius = min(vessel_radii)
from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

settings = active_domain_settings(scene)
voxel_size = float(settings.voxel_size)
assert settings.species_enabled and settings.species_mixing_rate > 0.0, (
    "authoritative domain settings do not enable color mixing")
assert vessel_radius - inner_radius >= voxel_size, (
    "vessel collision wall is thinner than one voxel")
source_corner_radius = max(
    np.hypot(*(source.matrix_world @ Vector(corner))[:2])
    for source in sources
    for corner in source.bound_box
)
assert source_corner_radius <= inner_radius - 0.5 * voxel_size, (
    "initial-fill boxes overlap the vessel collision wall")

from fluidSim_warp_blender51.blender import scene_bridge

scene.fluidsim.live = False
scene_bridge.release_session(scene)
sess = scene_bridge.initialize_session(scene)
scene_bridge.step_frame(
    scene, sess, write_cache=False, refresh_pending=False,
    timeline_frame=scene.frame_start + 1, publish_outputs=True)

particle_count = sess.solver.particles.refresh_host_count()
positions = sess.solver.particles.position[:particle_count].numpy()
particle_radii = np.linalg.norm(positions[:, :2], axis=1)
inside_sources = np.zeros(particle_count, dtype=bool)
for source in sources:
    corners = [source.matrix_world @ Vector(corner) for corner in source.bound_box]
    lower = np.min(np.asarray(corners), axis=0)
    upper = np.max(np.asarray(corners), axis=0)
    inside_sources |= np.all((positions >= lower) & (positions <= upper), axis=1)

assert particle_count > 0, "frame one emitted no liquid"
assert bpy.data.objects["FluidSim Particles"].visible_get(), (
    "frame-one liquid output is hidden in the viewport")
assert np.count_nonzero(inside_sources) >= int(particle_count * 0.9), (
    "frame-one liquid did not spawn throughout the emitter volumes")
assert float(np.max(particle_radii)) <= inner_radius + voxel_size, (
    "frame-one liquid escaped through the vessel collider")
assert not sess.error, sess.error

scene_bridge.step_frame(
    scene, sess, write_cache=False, refresh_pending=False,
    timeline_frame=scene.frame_start + 2, publish_outputs=True)
assert sess.solver.particles.has_optional_channel("color"), (
    "live color output lost its particle attribute")
point_attributes = bpy.data.objects["FluidSim Particles"].data.attributes
assert point_attributes.get("fluidsim_color") is not None, (
    "published particles lost the authored FluidSim color attribute")
assert point_attributes.get("display_color") is not None, (
    "published particles lost the viewport color attribute")
assert not sess.error, sess.error

print("CHROMATIC-VORTEX-CONTAINMENT-ACCEPTANCE-PASS")
print(json.dumps({
    "particles": particle_count,
    "inside_emitters": int(np.count_nonzero(inside_sources)),
    "max_particle_radius": float(np.max(particle_radii)),
    "inner_vessel_radius": inner_radius,
}, sort_keys=True))
