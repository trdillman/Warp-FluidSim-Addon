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
    """True when every triangle edge is shared by exactly two faces.

    Vectorized: the scalar Python loop this replaces cost ~450 ms at 123k
    triangles and ran on every mesh rebuild. Semantics are unchanged,
    including the degenerate cases pinned by
    tests/test_scene_bridge_collider_cache.py::test_watertight_triangles_semantics.
    """
    tris = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    if tris.size == 0:
        return False
    edges = np.concatenate((tris[:, (0, 1)], tris[:, (1, 2)], tris[:, (2, 0)]))
    edges = np.sort(edges, axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _entry_is_watertight(entry):
    """Evaluate (once, lazily) whether a cached mesh entry is watertight.

    Only SOLID mesh sinks consume this, so it must never be computed on the
    collider/source gather path. Memoized into the cache entry.
    """
    cached = entry.get("watertight")
    if cached is None:
        cached = _watertight_triangles(entry["watertight_tris"])
        entry["watertight"] = cached
    return cached


def _set_mesh_error(pg, message):
    try:
        pg.mesh_error = str(message)
    except (AttributeError, TypeError):
        pass


def _set_source_status(pg, severity="", code="", message=""):
    """Update the bounded RNA presentation snapshot; never compile from draw."""
    try:
        pg.source_status_severity = str(severity)
        pg.source_status_code = str(code)
        pg.source_status_message = str(message)
    except (AttributeError, TypeError):
        pass


def _scene_contains_object(scene, obj):
    getter = getattr(scene.objects, "get", None)
    if getter is not None:
        return getter(getattr(obj, "name", "")) is obj
    return obj in scene.objects


def _normalized_axis(matrix, column=2):
    axis = np.asarray(matrix[:3, column], dtype=np.float64)
    length = float(np.linalg.norm(axis))
    return ((0.0, 0.0, 1.0) if length < 1.0e-8
            else tuple(float(value) for value in axis / length))


def _guide_control_points(obj, matrix, direction):
    """Return up to four world-space curve controls for the GPU guide proxy."""
    points = []
    for spline in getattr(getattr(obj, "data", None), "splines", ()):
        controls = (getattr(spline, "bezier_points", ())
                    or getattr(spline, "points", ()))
        for control in controls:
            co = np.asarray(getattr(control, "co", ()), dtype=np.float64)
            if co.size < 3:
                continue
            world = matrix @ np.array((co[0], co[1], co[2], 1.0))
            points.append(tuple(float(value) for value in world[:3]))
    if len(points) > 4:
        indices = np.linspace(0, len(points) - 1, 4).round().astype(int)
        points = [points[index] for index in indices]
    if len(points) < 2:
        center = np.asarray(matrix[:3, 3], dtype=np.float64)
        points = [tuple(center), tuple(center + np.asarray(direction))]
    return tuple(points)


class SceneInputs:
    """Own animated-object history and reusable collider mesh state."""

    def __init__(self):
        self.device = "cuda:0"
        self.clear()

    def clear(self):
        # A clear denotes Initialize/Reset/load and creates a fresh host-only
        # lifecycle namespace. It is deliberately not saved in RNA.
        self._session_generation = getattr(self, "_session_generation", 0) + 1
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
                        sync_generation=None, strict_degenerate=True):
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
            if strict_degenerate and entry.get("had_degenerate", False):
                return None, "evaluated mesh contains degenerate triangles"
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
            had_degenerate = not np.all(valid)
            if strict_degenerate and had_degenerate:
                return None, "evaluated mesh contains degenerate triangles"
            if not np.any(valid):
                return None, "evaluated mesh contains only degenerate triangles"
            if had_degenerate:
                tris = tris[valid]
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
            "had_degenerate": had_degenerate,
            "mesh": warp_mesh,
            # Owned immutable adapter values for Phase 2 plans. No evaluated
            # mesh or mathutils proxy survives to persistent solver state.
            "vertices": np.ascontiguousarray(world_verts, dtype=np.float32),
            "triangles": np.ascontiguousarray(tris, dtype=np.int32),
            "aabb_min": triangle_points.min(axis=0),
            "aabb_max": triangle_points.max(axis=0),
            "force_center": tuple(float(value) for value in matrix[:3, 3]),
            # Deferred: computed on first SOLID-sink read via
            # _entry_is_watertight(). None means "not evaluated yet".
            "watertight": None,
            "watertight_tris": tris,
        }
        self._evaluated_mesh_cache[cache_key] = entry
        self.collider_mesh_cache_stats["rebuilds"] += 1
        return entry, ""

    def object_velocity(self, obj, scene):
        fps = scene.render.fps / scene.render.fps_base
        current = np.array(obj.matrix_world.translation)
        identity = _object_identity(obj)
        previous = self._collider_prev.get(identity)
        frame = scene.frame_current
        self._collider_prev[identity] = (current, frame)
        if previous is None or frame <= previous[1]:
            return (0.0, 0.0, 0.0)
        dt = (frame - previous[1]) / fps
        if dt == 0:
            return (0.0, 0.0, 0.0)
        return tuple(float(value) for value in (current - previous[0]) / dt)

    def object_angular_velocity(self, obj, scene):
        fps = scene.render.fps / scene.render.fps_base
        current = obj.matrix_world.to_quaternion()
        identity = _object_identity(obj)
        previous = self._collider_rotation_prev.get(identity)
        frame = scene.frame_current
        self._collider_rotation_prev[identity] = (current.copy(), frame)
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
                                 MeshSinkRegionMode, MeshSourceRegion,
                                 SourceEmissionMode, SourceEvaluationCadence,
                                 SourceGeometryMode, SourceShapeForceType,
                                 SourceShapeShape, SourceShapeSourceMode,
                                 SourceShapeType)
        from ..sim.solver import SourceShapeDesc

        sync_generation = sync_generation or self.begin_sync()
        descriptors, source_cache_keys = [], set()
        for obj in scene.objects:
            pg = getattr(obj, "fluidsim", None)
            role = getattr(pg, "role", "NONE") if pg is not None else "NONE"
            legacy_field = (role == "SOURCE"
                            and getattr(pg, "source_mode", "PARTICLES") == "FIELD")
            schema_version = int(getattr(pg, "source_ui_schema_version", 0) or 0)
            source_region = (getattr(pg, "source_region_kind", "ANALYTIC")
                             if schema_version >= 3 else
                             ("MESH_SHELL" if getattr(pg, "geometry_mode", "ANALYTIC") == "MESH"
                              else "ANALYTIC"))
            sink_region = (getattr(pg, "sink_region_kind", "ANALYTIC")
                           if schema_version >= 3 else
                           ("MESH_SOLID" if getattr(pg, "geometry_mode", "ANALYTIC") == "MESH"
                            and getattr(pg, "sink_region_mode", "SOLID") == "SOLID"
                            else "MESH_SHELL" if getattr(pg, "geometry_mode", "ANALYTIC") == "MESH"
                            else "ANALYTIC"))
            direct_plan_role = (legacy_field or role == "FIELD"
                                or (role == "SOURCE" and source_region in {"POINTS", "VOLUME_GRID"})
                                or (role == "SINK" and sink_region == "VOLUME_GRID"))
            if (pg is None or role not in {"SOURCE", "SINK", "FORCE", "FIELD"}
                    or not pg.enabled or obj.hide_viewport or direct_plan_role):
                continue
            geometry_mode = (SourceGeometryMode.MESH.name
                             if ((role == "SOURCE" and source_region in {"MESH_SOLID", "MESH_SHELL"})
                                 or (role == "SINK" and sink_region in {"MESH_SOLID", "MESH_SHELL"}))
                             else SourceGeometryMode.ANALYTIC.name)
            use_mesh = (geometry_mode == SourceGeometryMode.MESH.name
                        and role in {"SOURCE", "SINK"})
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
            desc.emission_mode = SourceEmissionMode[pg.emission_mode]
            pointer = getattr(obj, "as_pointer", None)
            runtime_pointer = int(pointer()) if callable(pointer) else id(obj)
            desc.source_id = f"blender-object:{runtime_pointer}"
            desc.runtime_source_key = (
                f"session:{self._session_generation}:object:{runtime_pointer}")
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
                desc.vertices = np.array(entry["vertices"], copy=True, order="C")
                desc.triangles = np.array(entry["triangles"], copy=True, order="C")
                desc.source_region = int(
                    MeshSourceRegion.SOLID_VOLUME
                    if source_region == "MESH_SOLID"
                    else MeshSourceRegion.SURFACE_SHELL)
                desc.evaluation_cadence = int(
                    SourceEvaluationCadence.EVERY_SIM_FRAME
                    if evaluation_mode == MeshEvaluationMode.EVERY_SIM_FRAME.name
                    else SourceEvaluationCadence.FIRST_SIM_FRAME)
                desc.aabb_min = tuple(map(float, entry["aabb_min"]))
                desc.aabb_max = tuple(map(float, entry["aabb_max"]))
                desc.force_center = entry["force_center"]
                desc.surface_thickness = float(getattr(
                    pg, "mesh_surface_thickness", 0.0))
                desc.sink_region_mode = (
                    int(MeshSinkRegionMode.SURFACE)
                    if sink_region == "MESH_SHELL"
                    else int(MeshSinkRegionMode.SOLID)
                )
                solid_consumer = (
                    (pg.role == "SINK"
                     and desc.sink_region_mode == MeshSinkRegionMode.SOLID)
                    or (pg.role == "SOURCE"
                        and desc.source_region == MeshSourceRegion.SOLID_VOLUME))
                if solid_consumer and not _entry_is_watertight(entry):
                    label = "source" if pg.role == "SOURCE" else "sink"
                    _set_mesh_error(
                        pg, f"Solid mesh {label} requires a watertight mesh")
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

    def _runtime_source_key(self, obj):
        return f"session:{self._session_generation}:object:{_object_identity(obj)}"

    @staticmethod
    def _analytic_region_plan(obj, pg):
        from ..sim.enums import SourceShapeShape
        from ..sim.source_regions import AnalyticRegionPlan

        location = obj.matrix_world.to_translation()
        rotation = obj.matrix_world.to_quaternion()
        dimensions = obj.dimensions
        return AnalyticRegionPlan(
            shape=SourceShapeShape[pg.shape],
            position=(location.x, location.y, location.z),
            rotation=(rotation.x, rotation.y, rotation.z, rotation.w),
            half_extents=tuple(max(float(v), 1.0e-4) / 2.0 for v in
                               (dimensions.x, dimensions.y, dimensions.z)),
            radius=max(float(dimensions.x), float(dimensions.y),
                       float(dimensions.z)) / 2.0,
            height=max(float(dimensions.z), 1.0e-4) / 2.0,
            feather=float(pg.feather),
        )

    @staticmethod
    def _volume_region_plan(obj, pg, *, include_field_payload=False):
        import bpy
        from ..sim.source_regions import VolumeRegionPlan

        volume_obj = getattr(pg, "volume_input_object", None) or obj
        if getattr(volume_obj, "type", None) != "VOLUME":
            raise ValueError("Choose a Blender Volume object")
        data = getattr(volume_obj, "data", None)
        grids = getattr(data, "grids", None)
        filepath = (getattr(grids, "frame_filepath", "")
                    or getattr(data, "filepath", ""))
        filepath = bpy.path.abspath(filepath) if filepath else ""
        if not filepath:
            raise ValueError("The selected Volume has no VDB filepath")
        use_sdf = getattr(pg, "volume_grid_type", "FOG") == "SDF"
        scalar_grid = (getattr(pg, "volume_scalar_grid", "")
                       if include_field_payload else "")
        vector_grid = (getattr(pg, "volume_vector_grid", "")
                       if include_field_payload else "")
        vector_x = (getattr(pg, "volume_vector_x_grid", "")
                    if include_field_payload and not vector_grid else "")
        vector_y = (getattr(pg, "volume_vector_y_grid", "")
                    if include_field_payload and not vector_grid else "")
        vector_z = (getattr(pg, "volume_vector_z_grid", "")
                    if include_field_payload and not vector_grid else "")
        return VolumeRegionPlan(
            filepath=filepath,
            mask_grid="" if use_sdf else getattr(pg, "volume_mask_grid", "density"),
            sdf_grid=getattr(pg, "volume_sdf_grid", "surface") if use_sdf else "",
            scalar_grid=scalar_grid,
            vector_grid=vector_grid,
            vector_x_grid=vector_x,
            vector_y_grid=vector_y,
            vector_z_grid=vector_z,
            transform=tuple(float(value) for value in
                            np.asarray(volume_obj.matrix_world).reshape(-1)),
            fog_threshold=float(getattr(pg, "volume_fog_threshold", 0.0)),
        )

    def gather_source_plans(self, scene, depsgraph=None):
        """Translate P3-only Point/Volume/Field workflows to frozen plans.

        Legacy analytic/mesh/sink/force descriptors remain owned by
        :meth:`gather_sources`; this method returns only workflows that have no
        legacy descriptor representation.
        """
        from . import sources as source_inputs
        from ..sim.enums import (
            BlendMode, SourceEmissionMode, SourceEvaluationCadence,
            SourceLifecycle, SourceOperation, SourcePayloadKind,
        )
        from ..sim.source_regions import (
            ActivationPlan, BlendPlan, PayloadPlan, PointRegionPlan,
            SourceKey, SourcePlan,
        )

        depsgraph = depsgraph or __import__("bpy").context.evaluated_depsgraph_get()
        plans = []
        for obj in scene.objects:
            pg = getattr(obj, "fluidsim", None)
            if pg is None or not getattr(pg, "enabled", False) or obj.hide_viewport:
                continue
            role = getattr(pg, "role", "NONE")
            schema_version = int(getattr(pg, "source_ui_schema_version", 0) or 0)
            legacy_field = (role == "SOURCE"
                            and getattr(pg, "source_mode", "PARTICLES") == "FIELD")
            presentation_role = "FIELD" if legacy_field else role
            source_region = (getattr(pg, "source_region_kind", "ANALYTIC")
                             if schema_version >= 3 else "ANALYTIC")
            sink_region = (getattr(pg, "sink_region_kind", "ANALYTIC")
                           if schema_version >= 3 else "ANALYTIC")
            needs_plan = (presentation_role == "FIELD"
                          or (role == "SOURCE" and source_region in {"POINTS", "VOLUME_GRID"})
                          or (role == "SINK" and sink_region == "VOLUME_GRID"))
            if not needs_plan:
                continue
            _set_source_status(pg)
            key = SourceKey(self._runtime_source_key(obj))
            cadence = (SourceEvaluationCadence.EVERY_SIM_FRAME
                       if getattr(pg, "mesh_evaluation_mode", "FIRST_SIM_FRAME")
                       == "EVERY_SIM_FRAME" else SourceEvaluationCadence.FIRST_SIM_FRAME)
            try:
                if role == "SOURCE" and source_region == "POINTS":
                    point_obj = getattr(pg, "point_input_object", None) or obj
                    positions = source_inputs.object_points_world(point_obj, depsgraph)
                    if not len(positions):
                        raise ValueError(
                            "Point input must provide mesh vertices, PointCloud points, or a particle system")
                    direct = getattr(pg, "point_source_representation", "DIRECT_APPEND") == "DIRECT_APPEND"
                    region = PointRegionPlan(
                        positions,
                        rasterize_radius=(0.0 if direct else float(
                            getattr(pg, "point_rasterize_radius", 0.1))),
                    )
                    lifecycle = (SourceLifecycle.DIRECT_APPEND if direct else
                                 (SourceLifecycle.ONCE
                                  if pg.emission_mode == SourceEmissionMode.INITIAL_FILL.name
                                  else SourceLifecycle.TOP_UP))
                    operation = SourceOperation.LIQUID
                    payload_kind = SourcePayloadKind.LIQUID_ATTRIBUTES
                elif role == "SOURCE" and source_region == "VOLUME_GRID":
                    region = self._volume_region_plan(obj, pg)
                    lifecycle = (SourceLifecycle.ONCE
                                 if pg.emission_mode == SourceEmissionMode.INITIAL_FILL.name
                                 else SourceLifecycle.TOP_UP)
                    operation = SourceOperation.LIQUID
                    payload_kind = SourcePayloadKind.LIQUID_ATTRIBUTES
                elif role == "SINK" and sink_region == "VOLUME_GRID":
                    region = self._volume_region_plan(obj, pg)
                    lifecycle = SourceLifecycle.EVERY_FRAME
                    operation = SourceOperation.SINK
                    payload_kind = SourcePayloadKind.MASK
                else:
                    operation_name = getattr(pg, "field_operation", "PRESSURE")
                    operation = {
                        "PRESSURE": SourceOperation.PRESSURE,
                        "SCALAR": SourceOperation.SCALAR_FIELD,
                        "VECTOR": SourceOperation.VECTOR_FIELD,
                        "STAGGERED_VELOCITY": SourceOperation.STAGGERED_VELOCITY,
                    }[operation_name]
                    use_volume = (getattr(pg, "field_region_kind", "ANALYTIC") == "VOLUME_GRID"
                                  or getattr(pg, "field_value_source", "CONSTANT") == "VOLUME_GRID")
                    region = (self._volume_region_plan(
                        obj, pg, include_field_payload=True) if use_volume
                        else self._analytic_region_plan(obj, pg))
                    lifecycle = SourceLifecycle.EVERY_FRAME
                    payload_kind = {
                        SourceOperation.PRESSURE: SourcePayloadKind.PRESSURE,
                        SourceOperation.SCALAR_FIELD: SourcePayloadKind.SCALAR,
                        SourceOperation.VECTOR_FIELD: SourcePayloadKind.VECTOR,
                        SourceOperation.STAGGERED_VELOCITY: SourcePayloadKind.STAGGERED_VECTOR,
                    }[operation]
                velocity = tuple(float(value) for value in pg.initial_velocity)
                if getattr(pg, "use_object_velocity", False):
                    velocity = tuple(a + b for a, b in zip(
                        velocity, self.object_velocity(obj, scene)))
                payload = PayloadPlan(
                    kind=payload_kind,
                    velocity=velocity,
                    pressure=float(getattr(pg, "field_scalar_value", 1.0)),
                    scalar=float(getattr(pg, "field_scalar_value", 1.0)),
                    vector=tuple(float(value) for value in getattr(
                        pg, "field_vector_value", (0.0, 0.0, 0.0))),
                    temperature=float(pg.temperature),
                    viscosity=float(pg.viscosity), density=float(pg.density),
                    lifespan=((float(pg.life_min) + float(pg.life_max)) * 0.5
                              if pg.use_lifespan else 0.0),
                    use_lifespan=bool(pg.use_lifespan),
                    target_particles_per_voxel=float(pg.emit_rate),
                )
                plans.append(SourcePlan(
                    key, operation, lifecycle, cadence, region, payload,
                    ActivationPlan(enabled=bool(pg.enabled)),
                    BlendPlan(mode=BlendMode[pg.blend_mode], amount=float(
                        getattr(pg, "field_amount", 1.0))),
                ))
            except Exception as exc:
                _set_source_status(pg, "ERROR", "SOURCE_INPUT_INVALID", str(exc))
        return plans

    def update_source_status(self, scene, diagnostics, apply_result=None):
        """Copy compiler/runtime results to bounded RNA status snapshots."""
        objects = {}
        for obj in scene.objects:
            pg = getattr(obj, "fluidsim", None)
            if pg is None:
                continue
            objects[self._runtime_source_key(obj)] = pg
            if getattr(pg, "source_status_code", "") != "SOURCE_INPUT_INVALID":
                _set_source_status(pg)
            try:
                pg.source_status_emitted = 0
                pg.source_status_capacity_limited = False
            except (AttributeError, TypeError):
                pass
        for diagnostic in diagnostics or ():
            pg = objects.get(str(getattr(diagnostic, "stable_id", "")))
            if pg is not None:
                _set_source_status(
                    pg, getattr(diagnostic, "severity", ""),
                    getattr(diagnostic, "code", ""),
                    getattr(diagnostic, "message", ""))
        if apply_result:
            # SourcePipeline currently reports one aggregate transaction. It is
            # truthful to attribute that count only when exactly one liquid
            # source is active; broadcasting it to every source/field panel
            # would present scene-global telemetry as per-object truth.
            active_liquid = [pg for pg in objects.values()
                             if getattr(pg, "role", "NONE") == "SOURCE"
                             and getattr(pg, "source_mode", "PARTICLES") != "FIELD"
                             and getattr(pg, "enabled", False)]
            if len(active_liquid) == 1:
                pg = active_liquid[0]
                try:
                    pg.source_status_emitted = int(apply_result.get("emitted", 0))
                    pg.source_status_capacity_limited = bool(
                        apply_result.get("capacity_limited", False))
                except (AttributeError, TypeError):
                    pass

    def gather_force_fields(self, scene):
        """Translate native Blender force-field objects into GPU proxies."""
        from ..sim.enums import (
            BlenderForceFieldShape,
            BlenderForceFieldType,
            BlenderTextureForceMode,
        )
        from ..sim.solver import ForceFieldDesc

        force_types = {
            "WIND": BlenderForceFieldType.WIND,
            "FORCE": BlenderForceFieldType.FORCE,
            "VORTEX": BlenderForceFieldType.VORTEX,
            "MAGNET": BlenderForceFieldType.MAGNETIC,
            "HARMONIC": BlenderForceFieldType.HARMONIC,
            "CHARGE": BlenderForceFieldType.CHARGE,
            "LENNARDJ": BlenderForceFieldType.LENNARD_JONES,
            "TEXTURE": BlenderForceFieldType.TEXTURE,
            "GUIDE": BlenderForceFieldType.CURVE_GUIDE,
            "BOID": BlenderForceFieldType.BOID,
            "TURBULENCE": BlenderForceFieldType.TURBULENCE,
            "DRAG": BlenderForceFieldType.DRAG,
            "FLUID_FLOW": BlenderForceFieldType.FLUID_FLOW,
        }
        shapes = {
            "POINT": BlenderForceFieldShape.POINT,
            "LINE": BlenderForceFieldShape.LINE,
            "PLANE": BlenderForceFieldShape.PLANE,
            "SURFACE": BlenderForceFieldShape.SURFACE,
            "EVERY_POINT": BlenderForceFieldShape.EVERY_POINT,
        }
        texture_modes = {
            "RGB": BlenderTextureForceMode.RGB,
            "GRAD": BlenderTextureForceMode.GRADIENT,
            "GRADIENT": BlenderTextureForceMode.GRADIENT,
            "CURL": BlenderTextureForceMode.CURL,
        }
        descriptors = []
        for obj in scene.objects:
            field = getattr(obj, "field", None)
            field_type = getattr(field, "type", "NONE")
            force_type = force_types.get(field_type)
            if (force_type is None or obj.hide_viewport
                    or not getattr(field, "apply_to_location", True)):
                continue
            strength = float(getattr(field, "strength", 0.0))
            linear_drag = float(getattr(field, "linear_drag", strength))
            quadratic_drag = float(getattr(field, "quadratic_drag", 0.0))
            texture = getattr(field, "texture", None)
            if (strength == 0.0
                    and not (field_type == "DRAG"
                             and (linear_drag != 0.0 or quadratic_drag != 0.0))
                    and field_type not in {"GUIDE", "BOID"}):
                continue
            if field_type == "TEXTURE" and texture is None:
                continue
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            direction = _normalized_axis(matrix)
            use_max = bool(getattr(field, "use_max_distance", False))
            max_distance = max(0.0, float(getattr(field, "distance_max", 0.0)))
            if use_max and max_distance == 0.0:
                continue
            min_distance = (max(0.0, float(getattr(
                field, "distance_min", 0.0)))
                if getattr(field, "use_min_distance", False) else 0.0)
            object_scale = np.linalg.norm(matrix[:3, :3], axis=0)
            size = abs(float(getattr(field, "size", 0.0)))
            if size < 1.0e-6:
                size = max(float(np.mean(object_scale)) * 0.25, 0.25)
            if field_type == "LENNARDJ" and min_distance > 0.0:
                size = min_distance

            source_velocity = direction
            source_object = getattr(field, "source_object", None)
            if source_object is not None:
                source_matrix = np.asarray(
                    getattr(source_object, "matrix_world", matrix),
                    dtype=np.float64)
                source_velocity = _normalized_axis(source_matrix)

            flags = 0
            if getattr(field, "use_global_coords", False):
                flags |= 1
            if getattr(field, "use_smoke_density", False):
                flags |= 2
            seed = int(getattr(field, "seed", 0))
            if texture is not None:
                seed += sum((index + 1) * ord(character)
                            for index, character in enumerate(
                                getattr(texture, "name", ""))) % 10007
            descriptors.append(ForceFieldDesc(
                force_type=force_type,
                shape=shapes.get(
                    getattr(field, "shape", "POINT"),
                    BlenderForceFieldShape.POINT),
                texture_mode=texture_modes.get(
                    getattr(field, "texture_mode", "RGB"),
                    BlenderTextureForceMode.RGB),
                flags=flags,
                seed=seed,
                center=tuple(float(value) for value in matrix[:3, 3]),
                direction=direction,
                source_velocity=source_velocity,
                guide_points=(_guide_control_points(obj, matrix, direction)
                              if field_type == "GUIDE" else ()),
                strength=strength,
                min_distance=min_distance,
                max_distance=max_distance if use_max else 0.0,
                falloff_power=max(
                    0.0, float(getattr(field, "falloff_power", 0.0))),
                flow=float(getattr(field, "flow", 0.0)),
                noise=max(0.0, float(getattr(field, "noise", 0.0))),
                size=size,
                rest_length=max(
                    0.0, float(getattr(field, "rest_length", 0.0))),
                damping=max(
                    0.0, float(getattr(field, "harmonic_damping", 0.0))),
                linear_drag=max(0.0, linear_drag),
                quadratic_drag=max(0.0, quadratic_drag),
                guide_minimum=max(
                    0.0, float(getattr(field, "guide_minimum", 0.0))),
                guide_free=max(
                    0.0, float(getattr(field, "guide_free", 0.0))),
                guide_clump=float(getattr(field, "guide_clump_amount", 0.0)),
            ))
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
            entry, error = self._evaluated_mesh(
                obj, depsgraph, cache_key, force_refresh=False,
                sync_generation=sync_generation, strict_degenerate=False)
            if entry is None:
                self._evaluated_mesh_cache.pop(cache_key, None)
                _set_mesh_error(obj.fluidsim, error)
                continue
            _set_mesh_error(obj.fluidsim, "")
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
        from ..sim.enums import SourceEmissionMode, SourceHeightBackMode
        from ..sim.solver import HeightSourceDesc

        fs = scene.fluidsim
        if not fs.height_enabled:
            try:
                fs.height_error = ""
            except (AttributeError, TypeError):
                pass
            return None
        mode = getattr(fs, "height_back_mode", "FLAT")
        if mode == SourceHeightBackMode.INFINITE.name:
            fs.height_error = (
                "Infinite height behavior is unsupported; choose Flat or provide a Profile image")
            return None
        profile = None
        if mode == SourceHeightBackMode.PROFILE.name:
            image = getattr(fs, "height_profile_image", None)
            if image is None:
                fs.height_error = "Choose a grayscale Height Profile image"
                return None
            width, height = map(int, image.size[:2])
            if width < 2 or height < 2:
                fs.height_error = "Height Profile image must be at least 2 x 2 pixels"
                return None
            pixels = np.empty(width * height * 4, dtype=np.float32)
            foreach_get = getattr(image.pixels, "foreach_get", None)
            if callable(foreach_get):
                foreach_get(pixels)
            else:
                pixels[:] = np.asarray(image.pixels, dtype=np.float32)
            rgba = pixels.reshape(height, width, 4)
            profile = np.ascontiguousarray(
                (rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152
                 + rgba[..., 2] * 0.0722).T
                * float(getattr(fs, "height_profile_scale", 1.0)),
                dtype=np.float32)
        fs.height_error = ""
        desc = HeightSourceDesc()
        desc.enabled = True
        desc.base_height = fs.height_base
        desc.back_mode = SourceHeightBackMode[mode]
        desc.profile = profile
        desc.velocity = tuple(fs.height_velocity)
        desc.emit_particles = fs.height_emit_particles
        desc.emit_rate = fs.particles_per_voxel
        desc.emission_mode = SourceEmissionMode[getattr(
            fs, "height_emission_mode",
            SourceEmissionMode.CONTINUOUS_INFLOW.name)]
        return desc
