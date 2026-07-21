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


def _watertight_triangles(tris):
    edge_counts = {}
    for tri in np.asarray(tris, dtype=np.int32):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


def _set_mesh_error(pg, message):
    try:
        pg.mesh_error = str(message)
    except (AttributeError, TypeError):
        pass


def _scene_contains_object(scene, obj):
    getter = getattr(scene.objects, "get", None)
    if getter is not None:
        return getter(getattr(obj, "name", "")) is obj
    return obj in scene.objects


class SceneInputs:
    """Own animated-object history and reusable collider mesh state."""

    def __init__(self):
        self.device = "cuda:0"
        self.clear()

    def clear(self):
        self._collider_prev = {}
        self._collider_rotation_prev = {}
        self._evaluated_mesh_cache = {}
        self._source_mesh_cache_keys = set()
        self._collider_mesh_cache_keys = set()
        self._sync_generation = 0
        # Compatibility alias for older diagnostics/tests.
        self._collider_mesh_cache = self._evaluated_mesh_cache
        self.evaluated_mesh_cache_stats = {"rebuilds": 0, "reuses": 0}
        self.collider_mesh_cache_stats = self.evaluated_mesh_cache_stats

    def begin_sync(self):
        """Return a token that deduplicates extraction within one scene sync."""
        self._sync_generation += 1
        return self._sync_generation

    def _prune_evaluated_mesh_cache(self):
        active = self._source_mesh_cache_keys | self._collider_mesh_cache_keys
        for cache_key in list(self._evaluated_mesh_cache):
            if cache_key not in active:
                del self._evaluated_mesh_cache[cache_key]

    def _evaluated_mesh(self, obj, depsgraph, cache_key, *, force_refresh=False,
                        sync_generation=None):
        """Return one validated, triangulated world-space Warp mesh cache entry."""
        import warp as wp

        object_id = _object_identity(obj)
        data_id = _data_identity(obj)
        updated_ids = _depsgraph_updates(depsgraph)
        entry = self._evaluated_mesh_cache.get(cache_key)
        matrix_world = obj.matrix_world
        matrix = np.asarray(matrix_world, dtype=np.float32)
        matrix_signature = _array_signature(matrix)
        if (not force_refresh and entry is not None
                and entry["device"] == self.device
                and entry["data_id"] == data_id
                and entry["matrix_signature"] == matrix_signature
                and (entry.get("sync_generation") == sync_generation
                     or (object_id not in updated_ids and data_id not in updated_ids))):
            self.collider_mesh_cache_stats["reuses"] += 1
            return entry, ""

        evaluated = obj.evaluated_get(depsgraph)
        matrix_world = evaluated.matrix_world
        matrix = np.asarray(matrix_world, dtype=np.float32)
        matrix_signature = _array_signature(matrix)
        mesh = evaluated.to_mesh()
        if mesh is None:
            return None, "evaluated mesh is unavailable"
        try:
            mesh.calc_loop_triangles()
            if len(mesh.vertices) == 0 or len(mesh.loop_triangles) == 0:
                return None, "evaluated mesh has no triangles"
            verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape(len(mesh.vertices), 3)
            tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tris)
            tris = tris.reshape(-1, 3)
            world_verts = np.ascontiguousarray(
                verts @ matrix[:3, :3].T + matrix[:3, 3],
                dtype=np.float32,
            )
            triangle_points = world_verts[np.unique(tris)]
            diagonal = float(np.linalg.norm(
                triangle_points.max(axis=0) - triangle_points.min(axis=0)))
            edges_a = world_verts[tris[:, 1]] - world_verts[tris[:, 0]]
            edges_b = world_verts[tris[:, 2]] - world_verts[tris[:, 0]]
            area2 = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
            valid = area2 > (1.0e-12 * max(1.0, diagonal * diagonal))
            if not np.all(valid):
                return None, "evaluated mesh contains degenerate triangles"
            tris = np.ascontiguousarray(tris, dtype=np.int32)
            triangle_points = world_verts[np.unique(tris)]
        finally:
            evaluated.to_mesh_clear()

        signature = _cache_signature(
            object_id, self.device, verts, tris, matrix)
        if entry is not None and entry["signature"] == signature:
            entry["sync_generation"] = sync_generation
            self.collider_mesh_cache_stats["reuses"] += 1
            return entry, ""

        points = wp.array(world_verts, dtype=wp.vec3, device=self.device)
        indices = wp.array(
            tris.reshape(-1), dtype=wp.int32, device=self.device)
        warp_mesh = wp.Mesh(points=points, indices=indices)
        entry = {
            "signature": signature,
            "device": self.device,
            "data_id": data_id,
            "matrix_signature": matrix_signature,
            "sync_generation": sync_generation,
            "mesh": warp_mesh,
            "aabb_min": triangle_points.min(axis=0),
            "aabb_max": triangle_points.max(axis=0),
            "watertight": _watertight_triangles(tris),
        }
        self._evaluated_mesh_cache[cache_key] = entry
        self.collider_mesh_cache_stats["rebuilds"] += 1
        return entry, ""

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

    def gather_sources(self, scene, depsgraph=None, *, capture_first_mesh=True,
                       sync_generation=None):
        import bpy
        from ..sim.enums import (BlendMode, MeshEvaluationMode,
                                 MeshSinkRegionMode, SourceGeometryMode,
                                 SourceShapeForceType, SourceShapeShape,
                                 SourceShapeSourceMode, SourceShapeType)
        from ..sim.solver import SourceShapeDesc

        sync_generation = sync_generation or self.begin_sync()
        descriptors, source_cache_keys = [], set()
        for obj in scene.objects:
            pg = getattr(obj, "fluidsim", None)
            if (pg is None or pg.role not in {"SOURCE", "SINK", "FORCE"}
                    or not pg.enabled or obj.hide_viewport):
                continue
            geometry_mode = getattr(
                pg, "geometry_mode", SourceGeometryMode.ANALYTIC.name)
            use_mesh = (geometry_mode == SourceGeometryMode.MESH.name
                        and pg.role in {"SOURCE", "SINK"})
            if use_mesh:
                from ..sim.sourcing import MeshSourceDesc
                desc = MeshSourceDesc()
            else:
                desc = SourceShapeDesc()
            if not use_mesh:
                desc.shape = SourceShapeShape[pg.shape]
            desc.stype = (SourceShapeType.SINK if pg.role == "SINK"
                          else SourceShapeType.SOURCE)
            desc.mode = SourceShapeSourceMode[pg.source_mode]
            location = obj.matrix_world.to_translation()
            rotation = obj.matrix_world.to_quaternion()
            if not use_mesh:
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
            if use_mesh:
                depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
                mesh_obj = getattr(pg, "mesh_object", None) or obj
                if (getattr(mesh_obj, "type", None) != "MESH"
                        or not _scene_contains_object(scene, mesh_obj)):
                    _set_mesh_error(
                        pg, "Mesh geometry requires a mesh object in this scene")
                    continue
                evaluation_mode = getattr(
                    pg, "mesh_evaluation_mode",
                    MeshEvaluationMode.FIRST_SIM_FRAME.name)
                cache_key = (
                    "EVALUATED_MESH", _object_identity(mesh_obj), evaluation_mode)
                source_cache_keys.add(cache_key)
                entry = self._evaluated_mesh_cache.get(cache_key)
                if (entry is None
                        and evaluation_mode == MeshEvaluationMode.FIRST_SIM_FRAME.name
                        and not capture_first_mesh):
                    _set_mesh_error(pg, "")
                    continue
                if (entry is None
                        or evaluation_mode == MeshEvaluationMode.EVERY_SIM_FRAME.name):
                    force_refresh = (
                        evaluation_mode == MeshEvaluationMode.EVERY_SIM_FRAME.name
                        and (entry is None
                             or entry.get("sync_generation") != sync_generation))
                    entry, error = self._evaluated_mesh(
                        mesh_obj, depsgraph, cache_key,
                        force_refresh=force_refresh,
                        sync_generation=sync_generation)
                    if entry is None:
                        self._evaluated_mesh_cache.pop(cache_key, None)
                        _set_mesh_error(pg, error)
                        continue
                desc.mesh = entry["mesh"]
                desc.aabb_min = tuple(map(float, entry["aabb_min"]))
                desc.aabb_max = tuple(map(float, entry["aabb_max"]))
                desc.surface_thickness = float(getattr(
                    pg, "mesh_surface_thickness", 0.0))
                desc.sink_region_mode = (
                    int(MeshSinkRegionMode.SURFACE)
                    if getattr(pg, "sink_region_mode", MeshSinkRegionMode.SOLID.name)
                    == MeshSinkRegionMode.SURFACE.name
                    else int(MeshSinkRegionMode.SOLID)
                )
                if (pg.role == "SINK"
                        and desc.sink_region_mode == MeshSinkRegionMode.SOLID
                        and not entry["watertight"]):
                    _set_mesh_error(
                        pg, "Solid mesh sink requires a watertight mesh")
                    continue
                _set_mesh_error(pg, "")
            else:
                _set_mesh_error(pg, "")
            if pg.role == "FORCE":
                desc.mode = SourceShapeSourceMode.FIELD
                desc.emit_rate = 0.0
            desc.force_type = SourceShapeForceType[pg.force_type]
            desc.force_strength = pg.force_strength
            desc.force_dir = tuple(pg.force_direction)
            descriptors.append(desc)
        self._source_mesh_cache_keys = source_cache_keys
        self._prune_evaluated_mesh_cache()
        return descriptors

    def gather_colliders(self, scene, depsgraph=None, *, sync_generation=None):
        import bpy
        from ..sim.enums import MeshEvaluationMode
        from ..sim.solver import ColliderDesc

        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        sync_generation = sync_generation or self.begin_sync()
        descriptors, collider_cache_keys = [], set()
        objects = (o for o in scene.objects if getattr(o, "fluidsim", None)
                   and o.fluidsim.role == "COLLIDER" and o.fluidsim.enabled)
        for obj in objects:
            if obj.type != "MESH" or obj.hide_viewport:
                continue
            object_id = _object_identity(obj)
            cache_key = (
                "EVALUATED_MESH", object_id,
                MeshEvaluationMode.EVERY_SIM_FRAME.name)
            collider_cache_keys.add(cache_key)
            entry, _error = self._evaluated_mesh(
                obj, depsgraph, cache_key, force_refresh=False,
                sync_generation=sync_generation)
            if entry is None:
                self._evaluated_mesh_cache.pop(cache_key, None)
                continue
            warp_mesh = entry["mesh"]
            aabb_min, aabb_max = entry["aabb_min"], entry["aabb_max"]
            evaluated = obj.evaluated_get(depsgraph)
            matrix_world = evaluated.matrix_world
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
        self._collider_mesh_cache_keys = collider_cache_keys
        self._prune_evaluated_mesh_cache()
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
