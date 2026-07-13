"""Blender scene-input translation state and lifecycle."""

from __future__ import annotations

import numpy as np


def _object_identity(obj):
    as_pointer = getattr(obj, "as_pointer", None)
    return int(as_pointer()) if as_pointer is not None else id(obj)


def _data_identity(obj):
    data = getattr(obj, "data", getattr(obj, "mesh", None))
    return _object_identity(data) if data is not None else 0


def _depsgraph_updates(depsgraph):
    updated_ids = set()
    for update in getattr(depsgraph, "updates", ()):
        updated = getattr(update, "id", None)
        updated = getattr(updated, "original", updated)
        if updated is not None:
            updated_ids.add(_object_identity(updated))
    return updated_ids


def _array_signature(array):
    contiguous = np.ascontiguousarray(array)
    return contiguous.shape, str(contiguous.dtype), contiguous.tobytes()


def _cache_signature(object_id, device, verts, tris, matrix_world):
    return (object_id, device, _array_signature(verts), _array_signature(tris),
            _array_signature(np.ascontiguousarray(matrix_world, dtype=np.float32)))


class SceneInputs:
    """Own animated-object history and reusable collider mesh state."""

    def __init__(self):
        self.device = "cuda:0"
        self.clear()

    def clear(self):
        self._collider_prev = {}
        self._collider_rotation_prev = {}
        self._collider_mesh_cache = {}
        self.collider_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}

    def object_velocity(self, obj, scene):
        fps = scene.render.fps / scene.render.fps_base
        current = np.array(obj.matrix_world.translation)
        previous = self._collider_prev.get(obj.name)
        frame = scene.frame_current
        self._collider_prev[obj.name] = (current, frame)
        if previous is None or frame <= previous[1]:
            return (0.0, 0.0, 0.0)
        dt = (frame - previous[1]) / fps
        if dt == 0:
            return (0.0, 0.0, 0.0)
        return tuple(float(value) for value in (current - previous[0]) / dt)

    def object_angular_velocity(self, obj, scene):
        fps = scene.render.fps / scene.render.fps_base
        current = obj.matrix_world.to_quaternion()
        previous = self._collider_rotation_prev.get(obj.name)
        frame = scene.frame_current
        self._collider_rotation_prev[obj.name] = (current.copy(), frame)
        if previous is None or frame <= previous[1]:
            return (0.0, 0.0, 0.0)
        dt = (frame - previous[1]) / fps
        delta = previous[0].rotation_difference(current)
        scale = float(delta.angle) / dt
        axis = delta.axis
        return (float(axis.x) * scale, float(axis.y) * scale, float(axis.z) * scale)

    def gather_sources(self, scene):
        from ..sim.enums import (BlendMode, SourceShapeForceType,
                                 SourceShapeShape, SourceShapeSourceMode,
                                 SourceShapeType)
        from ..sim.solver import SourceShapeDesc

        descriptors = []
        for obj in scene.objects:
            pg = getattr(obj, "fluidsim", None)
            if (pg is None or pg.role not in {"SOURCE", "SINK", "FORCE"}
                    or not pg.enabled or obj.hide_viewport):
                continue
            desc = SourceShapeDesc()
            desc.shape = SourceShapeShape[pg.shape]
            desc.stype = SourceShapeType.SINK if pg.role == "SINK" else SourceShapeType.SOURCE
            desc.mode = SourceShapeSourceMode[pg.source_mode]
            location = obj.matrix_world.to_translation()
            rotation = obj.matrix_world.to_quaternion()
            desc.position = (location.x, location.y, location.z)
            desc.rotation = (rotation.x, rotation.y, rotation.z, rotation.w)
            dimensions = obj.dimensions
            desc.half_extents = tuple(max(v, 1.0e-4) / 2 for v in
                                      (dimensions.x, dimensions.y, dimensions.z))
            desc.radius = max(dimensions.x, dimensions.y, dimensions.z) / 2
            desc.height = max(dimensions.z, 1.0e-4) / 2
            desc.feather = pg.feather
            velocity = tuple(pg.initial_velocity)
            if pg.use_object_velocity:
                velocity = tuple(a + b for a, b in zip(
                    velocity, self.object_velocity(obj, scene)))
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
            descriptors.append(desc)
        return descriptors

    def gather_colliders(self, scene, depsgraph=None):
        import bpy
        import warp as wp
        from ..sim.solver import ColliderDesc

        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        updated_ids = _depsgraph_updates(depsgraph)
        descriptors, seen = [], set()
        objects = (o for o in scene.objects if getattr(o, "fluidsim", None)
                   and o.fluidsim.role == "COLLIDER" and o.fluidsim.enabled)
        for obj in objects:
            if obj.type != "MESH" or obj.hide_viewport:
                continue
            object_id = _object_identity(obj)
            seen.add(object_id)
            evaluated = obj.evaluated_get(depsgraph)
            matrix_world = evaluated.matrix_world
            matrix = np.array(matrix_world, dtype=np.float32)
            matrix_signature = _array_signature(matrix)
            data_id = _data_identity(obj)
            entry = self._collider_mesh_cache.get(object_id)
            if (entry is not None and entry["device"] == self.device
                    and entry["data_id"] == data_id
                    and entry["matrix_signature"] == matrix_signature
                    and object_id not in updated_ids and data_id not in updated_ids):
                warp_mesh, aabb_min, aabb_max = entry["mesh"], entry["aabb_min"], entry["aabb_max"]
                self.collider_mesh_cache_stats["reuses"] += 1
            else:
                mesh = evaluated.to_mesh()
                if mesh is None or len(mesh.polygons) == 0:
                    evaluated.to_mesh_clear()
                    self._collider_mesh_cache.pop(object_id, None)
                    continue
                mesh.calc_loop_triangles()
                verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", verts)
                verts = verts.reshape(len(mesh.vertices), 3)
                tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
                mesh.loop_triangles.foreach_get("vertices", tris)
                world_verts = verts @ matrix[:3, :3].T + matrix[:3, 3]
                evaluated.to_mesh_clear()
                signature = _cache_signature(object_id, self.device, verts, tris, matrix)
                if entry is not None and entry["signature"] == signature:
                    warp_mesh, aabb_min, aabb_max = entry["mesh"], entry["aabb_min"], entry["aabb_max"]
                    self.collider_mesh_cache_stats["reuses"] += 1
                else:
                    warp_mesh = wp.Mesh(
                        points=wp.array(world_verts, dtype=wp.vec3, device=self.device),
                        indices=wp.array(tris, dtype=wp.int32, device=self.device))
                    aabb_min, aabb_max = world_verts.min(axis=0), world_verts.max(axis=0)
                    self.collider_mesh_cache_stats["rebuilds"] += 1
                self._collider_mesh_cache[object_id] = {
                    "signature": signature, "device": self.device, "data_id": data_id,
                    "matrix_signature": matrix_signature, "mesh": warp_mesh,
                    "aabb_min": aabb_min, "aabb_max": aabb_max}
            desc = ColliderDesc()
            desc.mesh, desc.aabb_min, desc.aabb_max = warp_mesh, tuple(map(float, aabb_min)), tuple(map(float, aabb_max))
            center = matrix_world.translation
            desc.center = (center.x, center.y, center.z)
            desc.friction = float(obj.fluidsim.collider_friction)
            desc.restitution = float(obj.fluidsim.collider_restitution)
            if obj.fluidsim.transfer_velocity:
                desc.linear_velocity = self.object_velocity(evaluated, scene)
                desc.angular_velocity = self.object_angular_velocity(evaluated, scene)
            descriptors.append(desc)
        for cache_key in self._collider_mesh_cache.keys() - seen:
            del self._collider_mesh_cache[cache_key]
        return descriptors

    def gather_height_source(self, scene):
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
