"""Source tagging, solver descriptor translation, and particle injection."""

from __future__ import annotations

import bpy
import numpy as np


def tag_source(obj, shape="BOX", mode="PARTICLES"):
    obj.fluidsim.role = "SOURCE"
    obj.fluidsim.shape = shape
    obj.fluidsim.source_mode = mode
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def tag_sink(obj, shape="BOX"):
    obj.fluidsim.role = "SINK"
    obj.fluidsim.shape = shape
    obj.display_type = "WIRE"
    obj.hide_render = True
    return obj


def gather_sources(scene, session):
    """Compatibility wrapper for the SceneInputs owner."""
    return getattr(session, "scene_inputs", session).gather_sources(scene)


def gather_height_source(scene):
    """Compatibility wrapper for callers without a live Session."""
    from .scene_inputs import SceneInputs
    return SceneInputs().gather_height_source(scene)


def object_points_world(obj, depsgraph=None):
    """Extract world-space points from a mesh / point cloud / particle system.

    Used for direct particle injection (fluidSimSourceParticles path).
    """
    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mw = np.array(obj.matrix_world, dtype=np.float32)

    # Particle system on the object takes priority.
    if eval_obj.particle_systems:
        psys = eval_obj.particle_systems[0]
        n = len(psys.particles)
        if n:
            pts = np.empty(n * 3, dtype=np.float32)
            psys.particles.foreach_get("location", pts)
            return pts.reshape(n, 3)  # already world space

    if obj.type == "POINTCLOUD":
        pc = eval_obj.data
        n = len(pc.points)
        if n:
            pts = np.empty(n * 3, dtype=np.float32)
            pc.attributes["position"].data.foreach_get("vector", pts)
            pts = pts.reshape(n, 3)
            return pts @ mw[:3, :3].T + mw[:3, 3]

    if obj.type == "MESH":
        mesh = eval_obj.to_mesh()
        n = len(mesh.vertices)
        pts = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", pts)
        pts = pts.reshape(n, 3)
        out = pts @ mw[:3, :3].T + mw[:3, 3]
        eval_obj.to_mesh_clear()
        return out

    return np.zeros((0, 3), dtype=np.float32)


def inject_object_particles(scene, obj, sess, velocity=(0.0, 0.0, 0.0)):
    """Inject an object's points directly as liquid particles (one shot)."""
    pts = object_points_world(obj)
    if not len(pts):
        return 0
    pg = obj.fluidsim
    sess.solver.inject_particles(
        pts,
        velocity=tuple(velocity) if any(velocity) else tuple(pg.initial_velocity),
        temperature=pg.temperature,
        viscosity=pg.viscosity,
        density=pg.density,
        life=(pg.life_min + pg.life_max) * 0.5 if pg.use_lifespan else 0.0,
        use_lifespan=pg.use_lifespan,
    )
    return len(pts)
