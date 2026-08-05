"""Deep owner of persistent Blender simulation output.

Output Preparation creates persistent IDs and schemas. Payload adapters perform
live readback or normalize cache facts. Output Publication mutates Blender IDs
from an owned payload and cannot reacquire solver or Warp state.
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import sys

import numpy as np

# Retained as a compatibility hook for Blender-free tests. Nearest-value
# lookup now always uses the batched NumPy grid path below.
_KDTree = None

from ..sim.constants import TIMER_PHASE_SLOTS

SURFACE_NAME = "FluidSim Surface"
PARTICLES_NAME = "FluidSim Particles"
WHITEWATER_NAME = "FluidSim Whitewater"
COLOR_ATTR = "display_color"
# Live point count of a padded output cloud; see _swap_points.
POINT_COUNT_TAG = "fluidsim_live_point_count"
# Headroom keeps steady-state waste near 15% of ~32 B/point.
_POINT_CAPACITY_HEADROOM = 1.15
_POINT_CAPACITY_PAD = 256
_MIN_POINT_CAPACITY = 1024

_STRUCTURAL_REPLACEMENTS = Counter()
_COST_PHASES = (
    "live_adaptation", "cache_adaptation",
    "output_preparation", "output_publication",
)
_COST_EVENTS = (
    "invocations", "host_frame_acquisitions", "device_readbacks",
    "surface_extractions", "synchronizations", "blender_id_creations",
    "blender_id_removals", "depsgraph_updates", "relation_changes",
    "structural_replacements", "datablock_replacements",
)
_PHASE_COSTS = {phase: Counter() for phase in _COST_PHASES}
_ACTIVE_COST_PHASE = ContextVar("fluidsim_output_cost_phase", default=None)
bpy = sys.modules.get("bpy")


def output_cost_counts(*, reset=False):
    """Return stable, separately measurable Output Publication phase costs."""
    counts = {
        phase: {event: int(_PHASE_COSTS[phase][event])
                for event in _COST_EVENTS}
        for phase in _COST_PHASES
    }
    if reset:
        for counter in _PHASE_COSTS.values():
            counter.clear()
    return counts


def _record_cost(event, amount=1, *, phase=None):
    phase = phase or _ACTIVE_COST_PHASE.get()
    if phase in _PHASE_COSTS and amount:
        _PHASE_COSTS[phase][event] += int(amount)


def _host_output_counts():
    from ..sim.frame_output import host_output_operation_counts
    return host_output_operation_counts()


def _cost_phase(phase):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            token = _ACTIVE_COST_PHASE.set(phase)
            before = _host_output_counts() if phase == "live_adaptation" else None
            _record_cost("invocations")
            try:
                return function(*args, **kwargs)
            finally:
                if before is not None:
                    after = _host_output_counts()
                    _record_cost(
                        "device_readbacks",
                        after["device_readbacks"] - before["device_readbacks"])
                    _record_cost(
                        "surface_extractions",
                        after["surface_extractions"]
                        - before["surface_extractions"])
                _ACTIVE_COST_PHASE.reset(token)
        return measured
    return decorate


def _new_blender_id(factory, *args, **kwargs):
    _record_cost("blender_id_creations")
    return factory(*args, **kwargs)


def _remove_blender_id(collection, datablock, *args, **kwargs):
    _record_cost("blender_id_removals")
    return collection.remove(datablock, *args, **kwargs)


def _link_relation(linker, datablock):
    _record_cost("relation_changes")
    return linker.link(datablock)


def _replace_datablock(obj, datablock):
    _record_cost("datablock_replacements")
    obj.data = datablock


def _ensure_material(ensure, *args):
    """Count IDs created inside the imported materials owner."""
    try:
        before = len(bpy.data.materials)
    except (AttributeError, TypeError):
        before = None
    result = ensure(*args)
    if before is not None:
        _record_cost("blender_id_creations", max(
            0, len(bpy.data.materials) - before))
    return result


def _require_bpy():
    """Load Blender's API only for datablock and overlay operations."""
    global bpy
    if bpy is None:
        try:
            import bpy as blender_api
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Output publication requires Blender's Python runtime"
            ) from exc
        if not hasattr(blender_api, "data"):
            raise RuntimeError(
                "Output publication requires Blender's bpy module"
            )
        bpy = blender_api
    return bpy


# ---------------------------------------------------------------------------
# Output collection / objects
# ---------------------------------------------------------------------------


def _output_collection(scene):
    name = (getattr(scene.fluidsim, "output_collection_name", "")
            or "FluidSim Output")
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = _new_blender_id(bpy.data.collections.new, name)
    if name not in {c.name for c in scene.collection.children}:
        try:
            _link_relation(scene.collection.children, coll)
        except RuntimeError:
            pass
    return coll


OUTPUT_TAG = "fluidsim_output_kind"


def output_object(scene, kind):
    """Resolve an add-on-owned output by tag inside its output collection.

    Never claim a global object merely because its name matches. Saved files
    from older versions are adopted only when the exact legacy-named object is
    already linked to the configured output collection.
    """
    coll = _output_collection(scene)
    for obj in coll.objects:
        try:
            if obj.get(OUTPUT_TAG) == kind:
                return obj
        except AttributeError:
            pass
    legacy = bpy.data.objects.get(kind)
    if legacy is not None and legacy.name in {obj.name for obj in coll.objects}:
        try:
            legacy[OUTPUT_TAG] = kind
        except (AttributeError, TypeError):
            pass
        return legacy
    return None


def _remove_output_object(obj):
    data = getattr(obj, "data", None)
    _remove_blender_id(bpy.data.objects, obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            _remove_blender_id(bpy.data.meshes, data)
        elif (hasattr(bpy.data, "pointclouds")
              and isinstance(data, bpy.types.PointCloud)):
            _remove_blender_id(bpy.data.pointclouds, data)


def _ensure_object(scene, name, data_factory, expected_type):
    """Get/create one tagged, correctly-typed add-on output object."""
    obj = output_object(scene, name)
    if obj is not None and obj.type != expected_type:
        # Tagged/adopted objects are add-on owned, so a type mismatch is safe
        # to recreate. Untagged same-named user objects are never selected.
        _remove_output_object(obj)
        obj = None
    if obj is None:
        obj = _new_blender_id(bpy.data.objects.new, name, data_factory())
        obj[OUTPUT_TAG] = name
        _link_relation(_output_collection(scene).objects, obj)
    return obj


def _values_to_colors(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        t = np.zeros(len(values), dtype=np.float32)
    else:
        finite_values = values[finite]
        step = max(1, int(np.ceil(len(finite_values) / 50_000)))
        range_sample = finite_values[::step]
        lo = float(np.percentile(range_sample, 5.0))
        hi = float(np.percentile(range_sample, 95.0))
        if hi <= lo + 1.0e-8:
            t = np.full(len(values), 0.5, dtype=np.float32)
        else:
            t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
            t[~finite] = 0.0
    stops = np.asarray([
        (0.05, 0.12, 0.95, 1.0),
        (0.0, 0.80, 1.0, 1.0),
        (0.10, 0.85, 0.20, 1.0),
        (1.0, 0.85, 0.05, 1.0),
        (1.0, 0.05, 0.02, 1.0),
    ], dtype=np.float32)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(np.int32), 0, len(stops) - 2)
    f = (x - i).reshape(-1, 1)
    return stops[i] * (1.0 - f) + stops[i + 1] * f


def _nearest_values(points, sample_points, sample_values, cell_size):
    points = np.asarray(points, dtype=np.float32)
    sample_points = np.asarray(sample_points, dtype=np.float32)
    sample_values = np.asarray(sample_values, dtype=np.float32)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)
    if len(sample_points) == 0:
        return np.zeros(len(points), dtype=np.float32)
    # Cap CPU-side lookup work for interactive attribute debugging.
    max_samples = 120_000
    if len(sample_points) > max_samples:
        step = int(np.ceil(len(sample_points) / max_samples))
        sample_points = sample_points[::step]
        sample_values = sample_values[::step]

    # Sort sample cells once. Collision-free linear int64 keys make the
    # batched searchsorted lookup substantially faster than structured rows.
    cell_size = max(float(cell_size), 1.0e-5)
    sample_keys = np.floor(sample_points / cell_size).astype(np.int64)
    point_keys = np.floor(points / cell_size).astype(np.int64)
    key_min = np.minimum(
        sample_keys.min(axis=0), point_keys.min(axis=0) - 1)
    key_max = np.maximum(
        sample_keys.max(axis=0), point_keys.max(axis=0) + 1)
    key_dims = key_max - key_min + 1
    stride_y = int(key_dims[2])
    stride_x = int(key_dims[1]) * stride_y
    if int(key_dims[0]) * stride_x > np.iinfo(np.int64).max:
        raise ValueError("attribute lookup grid coordinate range is too large")

    def linear_keys(keys):
        shifted = keys - key_min
        return (shifted[..., 0] * stride_x
                + shifted[..., 1] * stride_y + shifted[..., 2])

    sample_linear = linear_keys(sample_keys)
    sample_order = np.argsort(sample_linear, kind="stable")
    unique_keys, sample_starts, sample_counts = np.unique(
        sample_linear[sample_order], return_index=True, return_counts=True)

    default = float(np.nanmean(sample_values)) if len(sample_values) else 0.0
    out = np.full(len(points), default, dtype=np.float32)
    offsets = np.asarray([
        (dx, dy, dz) for dx in range(-1, 2)
        for dy in range(-1, 2) for dz in range(-1, 2)
    ], dtype=np.int64)
    max_distance_pairs = 2_000_000
    query_start = 0
    query_batch = 4096
    while query_start < len(points):
        query_end = min(query_start + query_batch, len(points))
        target_keys = (
            point_keys[query_start:query_end, None, :] + offsets[None, :, :]
        ).reshape(-1, 3)
        target_linear = linear_keys(target_keys)
        bucket_ids = np.searchsorted(unique_keys, target_linear)
        valid = bucket_ids < len(unique_keys)
        valid_indices = np.flatnonzero(valid)
        valid[valid_indices] = (
            unique_keys[bucket_ids[valid_indices]]
            == target_linear[valid_indices])
        if not valid.any():
            query_start = query_end
            query_batch = 4096
            continue

        bucket_ids = bucket_ids[valid]
        target_queries = np.repeat(
            np.arange(query_end - query_start, dtype=np.intp), len(offsets)
        )[valid]
        target_counts = sample_counts[bucket_ids].astype(np.intp, copy=False)
        pair_count = int(target_counts.sum())
        if pair_count > max_distance_pairs and query_end - query_start > 1:
            query_batch = max(1, int(
                (query_end - query_start) * max_distance_pairs / pair_count))
            continue

        target_offsets = np.cumsum(target_counts) - target_counts
        repeated_targets = np.repeat(
            np.arange(len(bucket_ids), dtype=np.intp), target_counts)
        within_bucket = (
            np.arange(pair_count, dtype=np.intp)
            - np.repeat(target_offsets, target_counts))
        candidate_sorted = (
            sample_starts[bucket_ids[repeated_targets]] + within_bucket)
        candidate_indices = sample_order[candidate_sorted]
        query_indices = np.repeat(target_queries, target_counts)
        delta = (
            points[query_start + query_indices]
            - sample_points[candidate_indices])
        distances = np.einsum("ij,ij->i", delta, delta)
        best_distances = np.full(
            query_end - query_start, np.inf, dtype=np.float32)
        np.minimum.at(best_distances, query_indices, distances)
        winners = distances == best_distances[query_indices]
        out[query_start + query_indices[winners]] = sample_values[
            candidate_indices[winners]]
        query_start = query_end
        query_batch = 4096
    return out


def _write_mesh_colors(mesh, tris, vertex_colors):
    if vertex_colors is None or len(vertex_colors) == 0:
        return
    if not hasattr(mesh, "color_attributes"):
        return
    colors = np.asarray(vertex_colors, dtype=np.float32)
    attr = mesh.color_attributes.get(COLOR_ATTR)
    if len(tris) == 0:
        if attr is None:
            attr = mesh.color_attributes.new(
                name=COLOR_ATTR, type="FLOAT_COLOR", domain="POINT")
        attr.data.foreach_set("color", colors.ravel())
    else:
        loop_colors = colors[np.asarray(tris, dtype=np.int32).ravel()]
        if attr is None:
            attr = mesh.color_attributes.new(
                name=COLOR_ATTR, type="FLOAT_COLOR", domain="CORNER")
        attr.data.foreach_set("color", loop_colors.ravel())
    try:
        mesh.color_attributes.active_color = attr
    except Exception:
        pass


def _set_point_radius(obj, radius):
    """Set the display radius on the live points, leaving padding parked."""
    data = getattr(obj, "data", None)
    attributes = getattr(data, "attributes", None)
    if attributes is None:
        return
    rad_attr = attributes.get("radius")
    if rad_attr is None:
        return
    capacity = len(rad_attr.data)
    if not capacity:
        return
    values = np.zeros(capacity, dtype=np.float32)
    values[:live_point_count(obj)] = radius
    rad_attr.data.foreach_set("value", values)
    if hasattr(data, "update_tag"):
        data.update_tag()


def _swap_mesh(obj, verts, tris, colors=None):
    """Replace an object's mesh with new geometry (fast bulk transfer)."""
    previous = getattr(obj, "data", None)
    old = previous if obj.type == "MESH" else None
    reuse = (old is not None and getattr(old, "users", 1) == 1
             and hasattr(old, "clear_geometry"))
    if reuse:
        attr = getattr(old, "color_attributes", None)
        color = attr.get(COLOR_ATTR) if attr is not None else None
        if color is not None:
            attr.remove(color)
        mesh = old
        mesh.clear_geometry()
    else:
        mesh = _new_blender_id(bpy.data.meshes.new, obj.name)
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int32)
    nv = len(verts)
    nt = len(tris)
    if nt and (tris.min() < 0 or tris.max() >= nv):
        raise ValueError("surface triangle index is outside the vertex range")
    if nv:
        mesh.vertices.add(nv)
        mesh.vertices.foreach_set("co", verts.ravel())
    if nt:
        mesh.loops.add(nt * 3)
        mesh.loops.foreach_set("vertex_index",
                               tris.ravel())
        mesh.polygons.add(nt)
        mesh.polygons.foreach_set("loop_start",
                                   np.arange(0, nt * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
        mesh.shade_smooth()
    if nv:
        _write_mesh_colors(mesh, tris, colors)
    mesh.update()
    if not reuse:
        mats = [m for m in getattr(previous, "materials", ())]
        _replace_datablock(obj, mesh)
        for m in mats:
            obj.data.materials.append(m)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                _remove_blender_id(bpy.data.meshes, previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                _remove_blender_id(bpy.data.pointclouds, previous)


def _supports_pointcloud():
    return hasattr(bpy.data, "pointclouds")


def _new_prewarmed_pointcloud(name):
    """Build a point cloud with its attribute schema and headroom already in
    place, entirely before it is attached to an object.

    ``_needs_realloc`` forces a detach/reattach the first time a point cloud
    is written to if it has no "position"/"radius" attributes or its capacity
    is 0 -- both true of a bare ``pointclouds.new()``. Doing that work here,
    on a still-detached cloud, means the first live write after
    ``ensure_outputs`` can just ``foreach_set`` into existing slots instead of
    reallocating from inside ``frame_change_post`` (see ensure_outputs).
    """
    pc = _new_blender_id(bpy.data.pointclouds.new, name)
    pc.resize(_MIN_POINT_CAPACITY)
    _ensure_point_attributes(pc, wants_color=False)
    return pc


def _ensure_point_object(scene, name):
    """Get/create a point output object, native point cloud where available."""
    if _supports_pointcloud():
        return _ensure_object(
            scene, name, lambda: _new_prewarmed_pointcloud(name), "POINTCLOUD")
    return _ensure_object(
        scene, name,
        lambda: _new_blender_id(bpy.data.meshes.new, name), "MESH")


def _point_capacity_for(point_count):
    return max(_MIN_POINT_CAPACITY,
               int(point_count * _POINT_CAPACITY_HEADROOM)
               + _POINT_CAPACITY_PAD)


def live_point_count(obj):
    """Live point count of an output object.

    Its datablock is padded (see ``_swap_points``), so ``len(data.points)``
    is the capacity, not the count. Falls back to the datablock length for
    mesh outputs and for clouds written before the tag existed.
    """
    if obj is None:
        return 0
    count = obj.get(POINT_COUNT_TAG) if hasattr(obj, "get") else None
    if count is not None:
        return int(count)
    data = getattr(obj, "data", None)
    for name in ("points", "vertices"):
        elements = getattr(data, name, None)
        if elements is not None:
            return len(elements)
    return 0


def _point_replacement_reason(pc, point_count, wants_color):
    """Classify the existing detached-cloud structural fallback."""
    if (pc is None or getattr(pc, "users", 1) != 1
            or not hasattr(pc, "resize")):
        return "ownership"
    if (pc.attributes.get("position") is None
            or pc.attributes.get("radius") is None
            or (pc.attributes.get(COLOR_ATTR) is not None) != wants_color):
        return "schema"
    capacity = len(pc.points)
    if point_count > capacity:
        return "growth"
    if (point_count * 2 < capacity
            and _point_capacity_for(point_count) < capacity):
        return "shrink"
    return None


def _needs_realloc(pc, point_count, wants_color):
    """Whether this write needs the characterized detached-cloud fallback."""
    return _point_replacement_reason(pc, point_count, wants_color) is not None


def structural_replacement_counts(*, reset=False):
    """Return separately counted point-cloud structural replacements."""
    counts = {
        name: int(_STRUCTURAL_REPLACEMENTS[name])
        for name in ("growth", "shrink", "schema", "ownership", "forced")
    }
    if reset:
        _STRUCTURAL_REPLACEMENTS.clear()
    return counts


def _ensure_point_attributes(pc, wants_color):
    """Create the attribute schema. Detached clouds only -- see _needs_realloc."""
    if pc.attributes.get("position") is None:
        pc.attributes.new("position", "FLOAT_VECTOR", "POINT")
    if pc.attributes.get("radius") is None:
        pc.attributes.new("radius", "FLOAT", "POINT")
    if wants_color and pc.attributes.get(COLOR_ATTR) is None:
        attr = pc.attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
        try:
            pc.attributes.active = attr
        except Exception:
            pass


def _swap_points(obj, positions, radius, colors=None, *, reuse_datablock=True):
    """Replace a point cloud object's points (bulk attribute transfer).

    Invariant: an attached point cloud is never resized and never gains or
    loses an attribute. Every structural change is built on a detached cloud
    and attached in a single assignment.

    Point counts vary every frame, so holding that invariant without
    allocating a datablock per frame means the cloud carries
    ``capacity >= n`` points, with the tail past the live count parked at the
    origin at radius 0. ``live_point_count(obj)`` reads the real count;
    ``len(pc.points)`` is the capacity.
    """
    previous = getattr(obj, "data", None)
    pc_old = previous if obj.type == "POINTCLOUD" else None
    n = len(positions)
    wants_color = colors is not None and len(colors) != 0
    reason = ("forced" if reuse_datablock is False else
              _point_replacement_reason(pc_old, n, wants_color))
    realloc = reason is not None
    if realloc:
        _STRUCTURAL_REPLACEMENTS[reason] += 1
        _record_cost("structural_replacements")
        pc = _new_blender_id(bpy.data.pointclouds.new, obj.name)
        pc.resize(_point_capacity_for(n))
        _ensure_point_attributes(pc, wants_color)
    else:
        pc = pc_old
    capacity = len(pc.points)

    coords = np.zeros((capacity, 3), dtype=np.float32)
    radii = np.zeros(capacity, dtype=np.float32)
    if n:
        coords[:n] = np.asarray(positions, dtype=np.float32)
        radii[:n] = radius
    pc.attributes["position"].data.foreach_set("vector", coords.ravel())
    pc.attributes["radius"].data.foreach_set("value", radii)
    color_attr = pc.attributes.get(COLOR_ATTR)
    if color_attr is not None:
        channels = np.zeros((capacity, 4), dtype=np.float32)
        if wants_color:
            channels[:n] = np.asarray(colors, dtype=np.float32)
        color_attr.data.foreach_set("color", channels.ravel())
    obj[POINT_COUNT_TAG] = n
    pc.update_tag()
    if realloc:
        mats = [m for m in getattr(previous, "materials", ())]
        _replace_datablock(obj, pc)
        for m in mats:
            obj.data.materials.append(m)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                _remove_blender_id(bpy.data.meshes, previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                _remove_blender_id(bpy.data.pointclouds, previous)


def _swap_vertex_cloud(obj, positions, colors=None):
    _swap_mesh(obj, positions, np.zeros((0, 3), dtype=np.int32), colors)


def _particle_values(solver, mode, positions, velocities=None, indices=None,
                     host=None):
    if mode == "NONE":
        return None
    if mode == "HEIGHT":
        return positions[:, 2]
    if mode == "SPEED":
        if velocities is None:
            velocities = host.velocities if host is not None else solver.particle_velocities()
            if indices is not None:
                velocities = velocities[indices]
        return np.linalg.norm(velocities, axis=1)
    if mode == "TIME_OFFSET":
        values = (host.particle_attr("time_offset") if host is not None else
                  solver.particles.time_offset.numpy()[:solver.particles.host_count])
        return values[indices] if indices is not None else values
    name = mode.lower()
    if name in {"age", "life", "temperature", "viscosity", "density"}:
        vals = (host.particle_attr(name) if host is not None else
                solver.particles.attr(name).numpy()[:solver.particles.host_count])
        return vals[indices] if indices is not None else vals
    return None


def _whitewater_values(solver, mode, positions, host=None):
    if mode == "NONE":
        return None
    n = len(positions)
    if mode == "HEIGHT":
        return positions[:, 2]
    if mode == "SPEED":
        vel = (host.whitewater_attr("velocity") if host is not None else
               solver.whitewater.velocity.numpy()[:n])
        return np.linalg.norm(vel, axis=1)
    if mode == "AGE":
        return (host.whitewater_attr("age") if host is not None else
                solver.whitewater.age.numpy()[:n])
    if mode == "LIFE":
        return (host.whitewater_attr("life") if host is not None else
                solver.whitewater.life.numpy()[:n])
    return None


def _surface_values(solver, mode, verts, host=None):
    if mode == "NONE":
        return None
    if mode == "HEIGHT":
        return verts[:, 2]
    pos = host.positions if host is not None else solver.particle_positions()
    vel = (host.velocities if host is not None and mode == "SPEED" else
           solver.particle_velocities() if mode == "SPEED" else None)
    values = _particle_values(solver, mode, pos, vel, host=host)
    if values is None:
        return None
    return _nearest_values(verts, pos, values, solver.settings.voxel_size * 2.0)


def _cached_values(mode, positions, velocities=None):
    if mode == "NONE":
        return None
    if len(positions) == 0:
        return np.zeros(0, dtype=np.float32)
    if mode == "HEIGHT":
        return positions[:, 2]
    if mode == "SPEED" and velocities is not None and len(velocities) == len(positions):
        return np.linalg.norm(velocities, axis=1)
    return None


def _cached_named_values(
        frame_data,
        prefix,
        mode,
        positions,
        velocities=None,
        indices=None):
    values = _cached_values(mode, positions, velocities)
    if values is not None:
        return values
    key = f"{prefix}_{mode.lower()}"
    cached = frame_data.get(key)
    if cached is None:
        return None
    cached = np.asarray(cached, dtype=np.float32)
    if indices is not None and len(cached):
        cached = cached[indices]
    if len(cached) != len(positions):
        return None
    return cached


def _cache_array(frame_data, name, shape):
    value = frame_data.get(name)
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Main update
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationPayload:
    """Owned facts consumed by publication without solver/readback access."""

    voxel_size: float
    publish_surface: bool
    surface_visible: bool
    surface_attribute_mode: str
    surface_verts: np.ndarray
    surface_tris: np.ndarray
    surface_attributes: dict[str, np.ndarray]
    particles_visible: bool
    particle_attribute_mode: str
    particle_radius: float
    particles: np.ndarray
    particle_attributes: dict[str, np.ndarray]
    whitewater_visible: bool
    whitewater_attribute_mode: str
    whitewater_radius: float
    whitewater: np.ndarray
    whitewater_attributes: dict[str, np.ndarray]


def _attribute_map(mode, values):
    return {} if values is None else {mode: np.asarray(values)}


def _display_stride(fs):
    fraction = float(getattr(fs, "particle_display_fraction", 1.0))
    if not 0.0 < fraction < 1.0:
        return 1
    return max(1, int(round(1.0 / fraction)))


@_cost_phase("live_adaptation")
def prepare_live_payload(scene, host, publish_surface=True):
    """Perform the one live readback/adaptation stage from an acquired frame."""
    fs = scene.fluidsim
    solver = host.solver
    # particle_display_fraction decimates on the device, so omitted particles
    # never cross PCIe. Attribute values stride identically or colors misalign.
    step = _display_stride(fs)
    indices = slice(None, None, step) if step > 1 else None
    positions = (host.positions_strided(step) if fs.show_particles else
                 np.zeros((0, 3), dtype=np.float32))
    particle_values = (_particle_values(
        solver, fs.particle_attribute_mode, positions, indices=indices,
        host=host) if fs.show_particles else None)
    whitewater = (host.whitewater
                  if fs.show_whitewater and fs.whitewater_enabled else
                  np.zeros((0, 3), dtype=np.float32))
    whitewater_values = (_whitewater_values(
        solver, fs.whitewater_attribute_mode, whitewater, host=host)
        if fs.show_whitewater and fs.whitewater_enabled else None)
    if publish_surface and fs.show_surface and fs.surface_enabled:
        verts, tris = host.surface_mesh
        surface_values = _surface_values(
            solver, fs.surface_attribute_mode, verts, host=host)
    else:
        verts = np.zeros((0, 3), dtype=np.float32)
        tris = np.zeros((0, 3), dtype=np.int32)
        surface_values = None
    voxel_size = float(solver.settings.voxel_size)
    return PublicationPayload(
        voxel_size=voxel_size,
        publish_surface=publish_surface,
        surface_visible=bool(fs.show_surface and fs.surface_enabled),
        surface_attribute_mode=str(fs.surface_attribute_mode),
        surface_verts=np.asarray(verts, dtype=np.float32),
        surface_tris=np.asarray(tris, dtype=np.int32),
        surface_attributes=_attribute_map(
            fs.surface_attribute_mode, surface_values),
        particles_visible=bool(fs.show_particles),
        particle_attribute_mode=str(fs.particle_attribute_mode),
        particle_radius=(voxel_size * 0.25
                         * getattr(fs, "particle_display_scale", 1.0)),
        particles=np.asarray(positions, dtype=np.float32),
        particle_attributes=_attribute_map(
            fs.particle_attribute_mode, particle_values),
        whitewater_visible=bool(
            fs.show_whitewater and fs.whitewater_enabled),
        whitewater_attribute_mode=str(fs.whitewater_attribute_mode),
        whitewater_radius=(voxel_size * 0.15
                           * getattr(fs, "whitewater_display_scale", 1.0)),
        whitewater=np.asarray(whitewater, dtype=np.float32),
        whitewater_attributes=_attribute_map(
            fs.whitewater_attribute_mode, whitewater_values),
    )


@_cost_phase("cache_adaptation")
def prepare_cache_payload(scene, frame_data):
    """Normalize cached arrays and characterized missing-field fallbacks."""
    fs = scene.fluidsim
    positions = np.asarray(
        _cache_array(frame_data, "positions", (0, 3)), dtype=np.float32)
    velocities = np.asarray(
        _cache_array(frame_data, "velocities", (0, 3)), dtype=np.float32)
    whitewater = np.asarray(
        _cache_array(frame_data, "whitewater", (0, 3)), dtype=np.float32)
    verts = np.asarray(
        _cache_array(frame_data, "surface_verts", (0, 3)), dtype=np.float32)
    tris = np.asarray(
        _cache_array(frame_data, "surface_tris", (0, 3)), dtype=np.int32)
    voxel_size = float(np.asarray(
        frame_data.get("voxel_size", getattr(fs, "voxel_size", 0.05))).item())

    surface_values = _cached_values(fs.surface_attribute_mode, verts)
    if surface_values is None:
        particle_surface_values = _cached_named_values(
            frame_data, "particle", fs.surface_attribute_mode,
            positions, velocities)
        if particle_surface_values is not None:
            surface_values = _nearest_values(
                verts, positions, particle_surface_values, voxel_size * 2.0)
    particle_values = _cached_named_values(
        frame_data, "particle", fs.particle_attribute_mode, positions,
        velocities if len(velocities) == len(positions) else None)
    whitewater_velocities = frame_data.get("whitewater_velocities")
    if whitewater_velocities is not None:
        whitewater_velocities = np.asarray(
            whitewater_velocities, dtype=np.float32)
    whitewater_values = _cached_named_values(
        frame_data, "whitewater", fs.whitewater_attribute_mode,
        whitewater, whitewater_velocities)

    # Cached frames are full-fidelity on disk. Apply their display stride once
    # here. Live stride remains device-side in prepare_live_payload.
    step = _display_stride(fs)
    if step > 1 and len(positions):
        positions = positions[::step]
        if particle_values is not None:
            particle_values = particle_values[::step]
    return PublicationPayload(
        voxel_size=voxel_size,
        publish_surface=True,
        surface_visible=bool(fs.show_surface and fs.surface_enabled),
        surface_attribute_mode=str(fs.surface_attribute_mode),
        surface_verts=verts,
        surface_tris=tris,
        surface_attributes=_attribute_map(
            fs.surface_attribute_mode, surface_values),
        particles_visible=bool(fs.show_particles),
        particle_attribute_mode=str(fs.particle_attribute_mode),
        particle_radius=(voxel_size * 0.25
                         * getattr(fs, "particle_display_scale", 1.0)),
        particles=positions,
        particle_attributes=_attribute_map(
            fs.particle_attribute_mode, particle_values),
        whitewater_visible=bool(
            fs.show_whitewater and fs.whitewater_enabled),
        whitewater_attribute_mode=str(fs.whitewater_attribute_mode),
        whitewater_radius=(voxel_size * 0.15
                           * getattr(fs, "whitewater_display_scale", 1.0)),
        whitewater=whitewater,
        whitewater_attributes=_attribute_map(
            fs.whitewater_attribute_mode, whitewater_values),
    )


@_cost_phase("output_preparation")
def prepare_outputs(scene):
    """Output Preparation: create enabled IDs/schema without geometry writes."""
    _require_bpy()
    from . import materials

    fs = scene.fluidsim
    if fs.show_surface and fs.surface_enabled:
        _ensure_material(materials.ensure_surface_material, scene, _ensure_object(
            scene, SURFACE_NAME,
            lambda: _new_blender_id(
                bpy.data.meshes.new, SURFACE_NAME), "MESH"))
    if fs.show_particles:
        _ensure_material(
            materials.ensure_particle_material,
            scene, _ensure_point_object(scene, PARTICLES_NAME))
    if fs.show_whitewater and fs.whitewater_enabled:
        _ensure_material(
            materials.ensure_whitewater_material,
            scene, _ensure_point_object(scene, WHITEWATER_NAME))


@_cost_phase("output_publication")
def publish_payload(scene, payload):
    """Output Publication: mutate persistent IDs from owned facts only.

    This function has no solver, HostFrame, Warp-array, surface extraction, or
    synchronization access. Structural point-cloud fallback remains confined
    to ``_swap_points`` and is separately observable by reason.
    """
    _require_bpy()
    from . import materials

    if payload.surface_visible:
        if payload.publish_surface:
            values = payload.surface_attributes.get(
                payload.surface_attribute_mode)
            colors = _values_to_colors(values) if values is not None else None
            obj = _ensure_object(
                scene, SURFACE_NAME,
                lambda: _new_blender_id(
                    bpy.data.meshes.new, SURFACE_NAME), "MESH")
            _swap_mesh(
                obj, payload.surface_verts, payload.surface_tris, colors)
            _ensure_material(materials.ensure_surface_material, scene, obj)
    else:
        obj = output_object(scene, SURFACE_NAME)
        if obj is not None and obj.type == "MESH" and len(obj.data.vertices):
            _swap_mesh(
                obj,
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
            )

    if payload.particles_visible:
        values = payload.particle_attributes.get(
            payload.particle_attribute_mode)
        colors = _values_to_colors(values) if values is not None else None
        obj = _ensure_point_object(scene, PARTICLES_NAME)
        if _supports_pointcloud():
            _swap_points(obj, payload.particles, payload.particle_radius, colors)
        else:
            _swap_vertex_cloud(obj, payload.particles, colors)
        _ensure_material(materials.ensure_particle_material, scene, obj)
    else:
        _clear_points(scene, PARTICLES_NAME)

    if payload.whitewater_visible:
        values = payload.whitewater_attributes.get(
            payload.whitewater_attribute_mode)
        colors = _values_to_colors(values) if values is not None else None
        obj = _ensure_point_object(scene, WHITEWATER_NAME)
        if _supports_pointcloud():
            _swap_points(
                obj, payload.whitewater, payload.whitewater_radius, colors)
        else:
            _swap_vertex_cloud(obj, payload.whitewater, colors)
        _ensure_material(materials.ensure_whitewater_material, scene, obj)
    else:
        _clear_points(scene, WHITEWATER_NAME)

def refresh_existing_output_settings(scene):
    """Apply display/material settings to existing output objects only."""
    _require_bpy()
    from . import materials

    # This is intentionally metadata-only. Without an active solver we cannot
    # recompute attributes or surfaces, but cache/current-output objects should
    # still respond to visibility and material picker changes.
    fs = scene.fluidsim
    surface = output_object(scene, SURFACE_NAME)
    if surface is not None:
        visible = fs.show_surface and fs.surface_enabled
        surface.hide_viewport = not visible
        surface.hide_render = not visible
        if visible:
            _ensure_material(
                materials.ensure_surface_material, scene, surface)

    particles = output_object(scene, PARTICLES_NAME)
    if particles is not None:
        particles.hide_viewport = not fs.show_particles
        particles.hide_render = not fs.show_particles
        if fs.show_particles:
            radius = (getattr(fs, "voxel_size", 0.05) * 0.25 *
                      getattr(fs, "particle_display_scale", 1.0))
            _set_point_radius(particles, radius)
            _ensure_material(
                materials.ensure_particle_material, scene, particles)

    whitewater = output_object(scene, WHITEWATER_NAME)
    if whitewater is not None:
        visible = fs.show_whitewater and fs.whitewater_enabled
        whitewater.hide_viewport = not visible
        whitewater.hide_render = not visible
        if visible:
            radius = (getattr(fs, "voxel_size", 0.05) * 0.15 *
                      getattr(fs, "whitewater_display_scale", 1.0))
            _set_point_radius(whitewater, radius)
            _ensure_material(
                materials.ensure_whitewater_material, scene, whitewater)


def ensure_render_visibility(scene):
    """Apply render visibility to persistent simulation outputs."""
    fs = scene.fluidsim
    for name, show in (
        (SURFACE_NAME, fs.show_surface),
        (PARTICLES_NAME, fs.show_particles),
        (WHITEWATER_NAME, fs.show_whitewater),
    ):
        obj = output_object(scene, name)
        if obj is not None:
            obj.hide_render = not show


def configure_cycles_points(scene):
    """Resolve prepared point-cloud outputs for Cycles rendering."""
    for name in (PARTICLES_NAME, WHITEWATER_NAME):
        obj = output_object(scene, name)
        if obj is None or obj.type != "POINTCLOUD":
            continue
        # Cycles renders point clouds natively; no additional mutation needed.


def update_outputs(scene, sess, *, force_surface=False):
    """Write current solver outputs into Blender datablocks."""
    _require_bpy()
    if not sess.active:
        return
    fs = scene.fluidsim
    surface = output_object(scene, SURFACE_NAME)
    has_surface = (surface is not None and surface.type == "MESH" and
                   len(surface.data.vertices) > 0)
    solver = sess.solver
    _record_cost("host_frame_acquisitions", phase="live_adaptation")
    if hasattr(solver, "host_frame"):
        host = solver.host_frame()
    else:
        from ..sim.frame_output import HostFrame
        host = HostFrame(solver)
    publish_surface = (
        force_surface or not has_surface or
        host.should_publish_surface(
            getattr(fs, "live_surface_publish_cadence", 1))
    )
    payload = prepare_live_payload(
        scene, host, publish_surface=publish_surface)
    publish_payload(scene, payload)


def update_outputs_from_cache_frame(scene, frame_data):
    """Write a cached frame snapshot into the regular Blender output objects."""
    _require_bpy()
    publish_payload(scene, prepare_cache_payload(scene, frame_data))


def _clear_points(scene, name):
    obj = output_object(scene, name)
    if obj is None:
        return
    if obj.type == "POINTCLOUD" and live_point_count(obj):
        _swap_points(obj, np.zeros((0, 3), dtype=np.float32), 0.01)
    elif obj.type == "MESH" and len(obj.data.vertices):
        _swap_vertex_cloud(obj, np.zeros((0, 3), dtype=np.float32))


def remove_outputs(scene=None):
    _require_bpy()
    if scene is not None:
        objects = [
            output_object(scene, name)
            for name in (SURFACE_NAME, PARTICLES_NAME, WHITEWATER_NAME)
        ]
    else:
        objects = [
            obj for obj in list(bpy.data.objects)
            if getattr(obj, "get", lambda *_args: None)(OUTPUT_TAG)
            in {SURFACE_NAME, PARTICLES_NAME, WHITEWATER_NAME}
        ]
    for obj in objects:
        if obj is not None:
            _remove_output_object(obj)
