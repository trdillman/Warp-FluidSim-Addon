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
    """Build solver source descriptors from enabled tagged objects."""
    from ..sim.enums import (
        BlendMode,
        SourceShapeForceType,
        SourceShapeShape,
        SourceShapeSourceMode,
        SourceShapeType,
    )
    from ..sim.solver import SourceShapeDesc

    sources = []
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if (pg is None or pg.role not in {"SOURCE", "SINK", "FORCE"}
                or not pg.enabled or obj.hide_viewport):
            continue
        desc = SourceShapeDesc()
        desc.shape = SourceShapeShape[pg.shape]
        desc.stype = (SourceShapeType.SINK if pg.role == "SINK"
                      else SourceShapeType.SOURCE)
        desc.mode = SourceShapeSourceMode[pg.source_mode]
        matrix = obj.matrix_world
        location = matrix.to_translation()
        rotation = matrix.to_quaternion()
        desc.position = (location.x, location.y, location.z)
        desc.rotation = (rotation.x, rotation.y, rotation.z, rotation.w)
        dimensions = obj.dimensions
        desc.half_extents = (
            max(dimensions.x, 1.0e-4) / 2,
            max(dimensions.y, 1.0e-4) / 2,
            max(dimensions.z, 1.0e-4) / 2,
        )
        desc.radius = max(dimensions.x, dimensions.y, dimensions.z) / 2
        desc.height = max(dimensions.z, 1.0e-4) / 2
        desc.feather = pg.feather
        velocity = tuple(pg.initial_velocity)
        if pg.use_object_velocity:
            object_velocity = session.object_velocity(obj, scene)
            velocity = tuple(a + b for a, b in zip(velocity, object_velocity))
        desc.velocity = velocity
        desc.blend_mode = BlendMode[pg.blend_mode]
        desc.emit_rate = pg.emit_rate
        desc.temperature = pg.temperature
        desc.viscosity = pg.viscosity
        desc.density = pg.density
        desc.use_lifespan = pg.use_lifespan
        desc.life_min = pg.life_min
        desc.life_max = pg.life_max
        if pg.role == "FORCE":
            desc.mode = SourceShapeSourceMode.FIELD
            desc.emit_rate = 0.0
        desc.force_type = SourceShapeForceType[pg.force_type]
        desc.force_strength = pg.force_strength
        desc.force_dir = tuple(pg.force_direction)
        sources.append(desc)
    return sources


def gather_height_source(scene):
    """Build the optional height-source descriptor from scene settings."""
    from ..sim.enums import SourceHeightBackMode
    from ..sim.solver import HeightSourceDesc

    fs = scene.fluidsim
    if not fs.height_enabled:
        return None
    desc = HeightSourceDesc()
    desc.enabled = True
    desc.base_height = fs.height_base
    desc.back_mode = SourceHeightBackMode[fs.height_back_mode]
    desc.velocity = tuple(fs.height_velocity)
    desc.emit_particles = fs.height_emit_particles
    desc.emit_rate = fs.particles_per_voxel
    return desc


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
