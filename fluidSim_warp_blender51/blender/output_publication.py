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

from ..sim.attributes import (
    ATTRIBUTE_ALIASES,
    ATTRIBUTE_SPECS,
    cache_attribute_name,
    enabled_attribute_keys,
    validate_attribute_values,
)
from ..sim.constants import TIMER_PHASE_SLOTS
from ..sim.whitewater_types import (
    LEGACY_UNCLASSIFIED,
    WhitewaterType,
    partition_type_indices,
    validate_type_values,
)

SURFACE_NAME = "FluidSim Surface"
PARTICLES_NAME = "FluidSim Particles"
WHITEWATER_NAME = "FluidSim Whitewater"
FOAM_NAME = "FluidSim Foam"
BUBBLES_NAME = "FluidSim Bubbles"
SPRAY_NAME = "FluidSim Spray"
WHITEWATER_OUTPUT_NAMES = (
    WHITEWATER_NAME, FOAM_NAME, BUBBLES_NAME, SPRAY_NAME,
)
COLOR_ATTR = "display_color"
_VISUALIZATION_VALUE_KEY = "__visualization__"
# Live point count of a padded output cloud; see _swap_points.
POINT_COUNT_TAG = "fluidsim_live_point_count"


@dataclass(frozen=True)
class RenderInputSyncResult:
    """Recoverable render-input synchronization outcome after publication."""

    synchronized: bool
    phase: str = "RENDER_INPUT_SYNC"
    exception: Exception | None = None

    @property
    def warning(self) -> str:
        if self.synchronized or self.exception is None:
            return ""
        return (
            f"Render input synchronization failed: "
            f"{type(self.exception).__name__}: {self.exception}"
        )


def _sync_render_inputs(scene) -> RenderInputSyncResult:
    """Synchronize render adapters without discarding publication success."""
    try:
        from .geometry_nodes import sync_render_inputs

        sync_render_inputs(scene)
    except Exception as exc:  # noqa: BLE001 - returned to recovery boundaries
        return RenderInputSyncResult(False, exception=exc)
    return RenderInputSyncResult(True)


LEGACY_UNCLASSIFIED_COUNT_TAG = "fluidsim_legacy_unclassified_count"
# Headroom keeps steady-state waste near 15% of ~32 B/point.
_POINT_CAPACITY_HEADROOM = 1.15
_POINT_CAPACITY_PAD = 256
_MIN_POINT_CAPACITY = 1024

_MANAGED_ATTRIBUTE_NAMES = frozenset(
    {spec.blender_name for spec in ATTRIBUTE_SPECS.values()}
    | set(ATTRIBUTE_ALIASES) | {COLOR_ATTR}
)


def _attribute_property(data_type):
    return {
        "FLOAT": "value",
        "INT": "value",
        "FLOAT_VECTOR": "vector",
        "FLOAT_COLOR": "color",
    }[data_type]


def publish_attribute(data, spec, values, *, live_count=None):
    """Create/update one Blender attribute with strict type/domain alignment."""
    attributes = getattr(data, "attributes", None)
    if attributes is None:
        raise TypeError("output datablock does not support attributes")
    values = np.asarray(values)
    count = len(values) if live_count is None else int(live_count)
    normalized = validate_attribute_values(spec, values, count)
    total = len(getattr(data, "vertices", getattr(data, "points", ())))
    if total < count:
        raise ValueError(
            f"{spec.blender_name} has {count} values for {total} output points")
    attr = attributes.get(spec.blender_name)
    if attr is not None and (
            str(getattr(attr, "data_type", "")) != spec.data_type
            or str(getattr(attr, "domain", "")) != spec.domain):
        attributes.remove(attr)
        attr = None
    if attr is None:
        attr = attributes.new(
            name=spec.blender_name, type=spec.data_type, domain=spec.domain)
    target_shape = ((total,) if spec.components == 1 else
                    (total, spec.components))
    padded = np.zeros(target_shape, dtype=spec.numpy_dtype)
    if count:
        padded[:count] = normalized
    attr.data.foreach_set(_attribute_property(spec.data_type), padded.ravel())
    return attr


def _canonical_attribute_key(mode):
    key = str(mode or "").lower()
    return key if key in ATTRIBUTE_SPECS else None


def _merge_selected_attribute(attributes, mode, values):
    result = {
        str(key).lower(): np.asarray(value)
        for key, value in attributes.items()
        if str(key).lower() in ATTRIBUTE_SPECS
    }
    key = _canonical_attribute_key(mode)
    if values is not None:
        if key is None:
            result[_VISUALIZATION_VALUE_KEY] = np.asarray(values)
        elif key not in result:
            result[key] = np.asarray(values)
    return result


def _selected_attribute_values(attributes, mode):
    key = _canonical_attribute_key(mode)
    if key is None:
        key = _VISUALIZATION_VALUE_KEY
    return attributes.get(key)


def _phase3_attribute_map(values):
    return _merge_selected_attribute(values, None, None)

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


# Output collection / objects


def _collection_in_tree(root, target):
    if root is target:
        return True
    return any(_collection_in_tree(child, target) for child in root.children)


def _domain_settings(scene):
    """The group owning the solver-scoped settings read in this module.

    ``surface_enabled``, ``whitewater_enabled`` and
    ``live_surface_publish_cadence`` are solver settings: post-migration the
    solver reads them from the domain object's ``fluidsim_domain``. They are
    read here beside genuinely scene-scoped display flags (``show_surface``,
    ``whitewater_output_mode``, the material pickers), which stay on
    ``scene.fluidsim`` -- hence two owners rather than one rebind.

    ``None`` without a designated domain; readers then take the disabled
    defaults, so nothing solver-backed is published.
    """
    # Lazy import: scene_bridge reaches into this module, so the accessor is
    # pulled at call time rather than closing the cycle at import.
    from .scene_bridge import active_domain_settings

    return active_domain_settings(scene)


def _output_collection(scene):
    name = (getattr(scene.fluidsim, "output_collection_name", "")
            or "FluidSim Output")
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = _new_blender_id(bpy.data.collections.new, name)
    # Guided setup nests Outputs under a FluidSim owner collection. Do not
    # add a second direct scene link when it is already reachable in the tree.
    if not _collection_in_tree(scene.collection, coll):
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
    if values.ndim == 2 and values.shape[1] in {3, 4}:
        colors = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        colors = np.clip(colors, 0.0, 1.0)
        if colors.shape[1] == 3:
            colors = np.column_stack((
                colors, np.ones(len(colors), dtype=np.float32)))
        return np.ascontiguousarray(colors, dtype=np.float32)
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
    value_shape = tuple(sample_values.shape[1:])
    if len(points) == 0:
        return np.zeros((0, *value_shape), dtype=np.float32)
    if len(sample_points) == 0:
        return np.zeros((len(points), *value_shape), dtype=np.float32)
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

    default = np.nanmean(sample_values, axis=0)
    out = np.empty((len(points), *value_shape), dtype=np.float32)
    out[...] = default
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


def _remove_managed_attributes(data):
    attributes = getattr(data, "attributes", None)
    if attributes is None:
        return
    for attr in tuple(attributes):
        if attr.name in _MANAGED_ATTRIBUTE_NAMES:
            attributes.remove(attr)


def _swap_mesh(obj, verts, tris, colors=None, attributes=None):
    """Replace an object's mesh and publish aligned render attributes."""
    previous = getattr(obj, "data", None)
    old = previous if obj.type == "MESH" else None
    reuse = (old is not None and getattr(old, "users", 1) == 1
             and hasattr(old, "clear_geometry"))
    if reuse:
        _remove_managed_attributes(old)
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
        mesh.loops.foreach_set("vertex_index", tris.ravel())
        mesh.polygons.add(nt)
        mesh.polygons.foreach_set(
            "loop_start", np.arange(0, nt * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
        mesh.shade_smooth()
    if nv:
        _write_mesh_colors(mesh, tris, colors)
    for key, values in _phase3_attribute_map(attributes or {}).items():
        publish_attribute(mesh, ATTRIBUTE_SPECS[key], values, live_count=nv)
    mesh.update()
    if not reuse:
        mats = [m for m in getattr(previous, "materials", ())]
        _replace_datablock(obj, mesh)
        for material in mats:
            obj.data.materials.append(material)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                _remove_blender_id(bpy.data.meshes, previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                _remove_blender_id(bpy.data.pointclouds, previous)


def _supports_pointcloud():
    return hasattr(bpy.data, "pointclouds")


def _new_prewarmed_pointcloud(name, attribute_keys=()):
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
    _ensure_point_attributes(pc, wants_color=False, attribute_keys=attribute_keys)
    return pc


def _ensure_point_object(scene, name, attribute_keys=()):
    """Get/create a point output object, native point cloud where available."""
    if _supports_pointcloud():
        return _ensure_object(
            scene, name, lambda: _new_prewarmed_pointcloud(
                name, attribute_keys), "POINTCLOUD")
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


def _expected_point_schema(wants_color, attribute_keys):
    expected = {"position": ("FLOAT_VECTOR", "POINT"),
                "radius": ("FLOAT", "POINT")}
    if wants_color:
        expected[COLOR_ATTR] = ("FLOAT_COLOR", "POINT")
    for key in attribute_keys:
        spec = ATTRIBUTE_SPECS[key]
        expected[spec.blender_name] = (spec.data_type, spec.domain)
    return expected


def _point_replacement_reason(pc, point_count, wants_color, attribute_keys=()):
    """Classify detached-cloud fallback, including managed schema drift."""
    if (pc is None or getattr(pc, "users", 1) != 1
            or not hasattr(pc, "resize")):
        return "ownership"
    expected = _expected_point_schema(wants_color, attribute_keys)
    for name, (data_type, domain) in expected.items():
        attr = pc.attributes.get(name)
        if (attr is None or str(getattr(attr, "data_type", "")) != data_type
                or str(getattr(attr, "domain", "")) != domain):
            return "schema"
    expected_managed = set(expected) & _MANAGED_ATTRIBUTE_NAMES
    existing_managed = {
        attr.name for attr in pc.attributes
        if attr.name in _MANAGED_ATTRIBUTE_NAMES
    }
    if existing_managed != expected_managed:
        return "schema"
    capacity = len(pc.points)
    if point_count > capacity:
        return "growth"
    if (point_count * 2 < capacity
            and _point_capacity_for(point_count) < capacity):
        return "shrink"
    return None


def _needs_realloc(pc, point_count, wants_color, attribute_keys=()):
    return _point_replacement_reason(
        pc, point_count, wants_color, attribute_keys) is not None


def structural_replacement_counts(*, reset=False):
    """Return separately counted point-cloud structural replacements."""
    counts = {
        name: int(_STRUCTURAL_REPLACEMENTS[name])
        for name in ("growth", "shrink", "schema", "ownership", "forced")
    }
    if reset:
        _STRUCTURAL_REPLACEMENTS.clear()
    return counts


def _ensure_point_attributes(pc, wants_color, attribute_keys=()):
    """Create complete schema on a detached point cloud only."""
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
    for key in attribute_keys:
        spec = ATTRIBUTE_SPECS[key]
        existing = pc.attributes.get(spec.blender_name)
        if existing is None:
            pc.attributes.new(
                spec.blender_name, spec.data_type, spec.domain)


def _swap_points(
        obj, positions, radius, colors=None, attributes=None, *,
        reuse_datablock=True):
    """Bulk-update a padded point cloud without attached schema mutation."""
    previous = getattr(obj, "data", None)
    pc_old = previous if obj.type == "POINTCLOUD" else None
    positions = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    n = len(positions)
    wants_color = colors is not None and len(colors) != 0
    if attributes is None and pc_old is not None:
        preserved = {}
        for key, spec in ATTRIBUTE_SPECS.items():
            if pc_old.attributes.get(spec.blender_name) is not None:
                shape = (n,) if spec.components == 1 else (n, spec.components)
                preserved[key] = np.zeros(shape, dtype=spec.numpy_dtype)
        attribute_values = preserved
    else:
        attribute_values = _phase3_attribute_map(attributes or {})
    attribute_keys = tuple(
        key for key in ATTRIBUTE_SPECS if key in attribute_values)
    reason = ("forced" if reuse_datablock is False else
              _point_replacement_reason(
                  pc_old, n, wants_color, attribute_keys))
    realloc = reason is not None
    if realloc:
        _STRUCTURAL_REPLACEMENTS[reason] += 1
        _record_cost("structural_replacements")
        pc = _new_blender_id(bpy.data.pointclouds.new, obj.name)
        pc.resize(_point_capacity_for(n))
        _ensure_point_attributes(
            pc, wants_color, attribute_keys=attribute_keys)
    else:
        pc = pc_old
    capacity = len(pc.points)

    coords = np.zeros((capacity, 3), dtype=np.float32)
    radii = np.zeros(capacity, dtype=np.float32)
    if n:
        coords[:n] = positions
        radii[:n] = radius
    pc.attributes["position"].data.foreach_set("vector", coords.ravel())
    pc.attributes["radius"].data.foreach_set("value", radii)
    color_attr = pc.attributes.get(COLOR_ATTR)
    if color_attr is not None:
        channels = np.zeros((capacity, 4), dtype=np.float32)
        if wants_color:
            channels[:n] = np.asarray(colors, dtype=np.float32)
        color_attr.data.foreach_set("color", channels.ravel())
    for key in attribute_keys:
        publish_attribute(
            pc, ATTRIBUTE_SPECS[key], attribute_values[key], live_count=n)
    obj[POINT_COUNT_TAG] = n
    pc.update_tag()
    if realloc:
        mats = [material for material in getattr(previous, "materials", ())]
        _replace_datablock(obj, pc)
        for material in mats:
            obj.data.materials.append(material)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                _remove_blender_id(bpy.data.meshes, previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                _remove_blender_id(bpy.data.pointclouds, previous)


def _swap_vertex_cloud(obj, positions, colors=None, attributes=None):
    _swap_mesh(
        obj, positions, np.zeros((0, 3), dtype=np.int32), colors, attributes)


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
    if mode == "COLOR":
        values = (host.particle_attr("color") if host is not None else
                  solver.particles.attr("color").numpy()[:solver.particles.host_count])
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
    if mode == "TYPE":
        return (host.whitewater_attr("type") if host is not None else
                solver.whitewater.type.numpy()[:n])
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
    whitewater_output_mode: str
    whitewater_attribute_mode: str
    whitewater_radius: float
    whitewater: np.ndarray
    whitewater_attributes: dict[str, np.ndarray]


def _display_stride(fs):
    fraction = float(getattr(fs, "particle_display_fraction", 1.0))
    if not 0.0 < fraction < 1.0:
        return 1
    return max(1, int(round(1.0 / fraction)))


@_cost_phase("live_adaptation")
def prepare_live_payload(scene, host, publish_surface=True):
    """Perform one live readback/adaptation stage from an acquired frame."""
    fs = scene.fluidsim
    ds = _domain_settings(scene)
    solver = host.solver
    step = _display_stride(fs)
    indices = slice(None, None, step) if step > 1 else None
    positions = (host.positions_strided(step) if fs.show_particles else
                 np.zeros((0, 3), dtype=np.float32))
    particle_attributes = (
        host.output_attributes(fs, "particles", stride=step)
        if fs.show_particles else {})
    particle_key = _canonical_attribute_key(fs.particle_attribute_mode)
    particle_values = None
    if (fs.show_particles
            and (particle_key is None
                 or particle_key not in particle_attributes)):
        particle_values = _particle_values(
            solver, fs.particle_attribute_mode, positions, indices=indices,
            host=host)
    particle_attributes = _merge_selected_attribute(
        particle_attributes, fs.particle_attribute_mode, particle_values)

    whitewater = (host.whitewater
                  if fs.show_whitewater and getattr(ds, "whitewater_enabled", False) else
                  np.zeros((0, 3), dtype=np.float32))
    whitewater_visible = bool(fs.show_whitewater and getattr(ds, "whitewater_enabled", False))
    whitewater_attributes = (
        host.output_attributes(fs, "whitewater")
        if whitewater_visible else {})
    whitewater_key = _canonical_attribute_key(fs.whitewater_attribute_mode)
    whitewater_values = None
    if (whitewater_visible
            and (whitewater_key is None
                 or whitewater_key not in whitewater_attributes)):
        whitewater_values = _whitewater_values(
            solver, fs.whitewater_attribute_mode, whitewater, host=host)
    whitewater_attributes = _merge_selected_attribute(
        whitewater_attributes, fs.whitewater_attribute_mode,
        whitewater_values)
    whitewater_output_mode = str(
        getattr(fs, "whitewater_output_mode", "COMBINED"))
    if (whitewater_visible
            and whitewater_output_mode.upper() in {"SEPARATE", "BOTH"}
            and "type" not in whitewater_attributes):
        whitewater_attributes["type"] = host.whitewater_attribute("type")

    if publish_surface and fs.show_surface and getattr(ds, "surface_enabled", False):
        verts, tris = host.surface_mesh
        surface_attributes = host.output_attributes(
            fs, "surface", positions=verts)
        surface_key = _canonical_attribute_key(fs.surface_attribute_mode)
        surface_values = None
        if surface_key is None or surface_key not in surface_attributes:
            surface_values = _surface_values(
                solver, fs.surface_attribute_mode, verts, host=host)
        surface_attributes = _merge_selected_attribute(
            surface_attributes, fs.surface_attribute_mode, surface_values)
    else:
        verts = np.zeros((0, 3), dtype=np.float32)
        tris = np.zeros((0, 3), dtype=np.int32)
        surface_attributes = {}
    voxel_size = float(solver.settings.voxel_size)
    return PublicationPayload(
        voxel_size=voxel_size,
        publish_surface=publish_surface,
        surface_visible=bool(fs.show_surface and getattr(ds, "surface_enabled", False)),
        surface_attribute_mode=str(fs.surface_attribute_mode),
        surface_verts=np.asarray(verts, dtype=np.float32),
        surface_tris=np.asarray(tris, dtype=np.int32),
        surface_attributes=surface_attributes,
        particles_visible=bool(fs.show_particles),
        particle_attribute_mode=str(fs.particle_attribute_mode),
        particle_radius=(voxel_size * 0.25
                         * getattr(fs, "particle_display_scale", 1.0)),
        particles=np.asarray(positions, dtype=np.float32),
        particle_attributes=particle_attributes,
        whitewater_visible=whitewater_visible,
        whitewater_output_mode=whitewater_output_mode,
        whitewater_attribute_mode=str(fs.whitewater_attribute_mode),
        whitewater_radius=(voxel_size * 0.15
                           * getattr(fs, "whitewater_display_scale", 1.0)),
        whitewater=np.asarray(whitewater, dtype=np.float32),
        whitewater_attributes=whitewater_attributes,
    )


@_cost_phase("cache_adaptation")
def prepare_cache_payload(scene, frame_data):
    """Normalize cached geometry and its declared Phase-3 attributes."""
    from ..io.cache import validate_optional_attributes

    fs = scene.fluidsim
    ds = _domain_settings(scene)
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
        frame_data.get("voxel_size", getattr(ds, "voxel_size", 0.05))).item())
    inventory = validate_optional_attributes(frame_data)

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
        whitewater_velocities = np.asarray(whitewater_velocities, dtype=np.float32)
    whitewater_values = _cached_named_values(
        frame_data, "whitewater", fs.whitewater_attribute_mode,
        whitewater, whitewater_velocities)

    surface_attributes = {
        key: np.asarray(frame_data[cache_attribute_name("surface", key)])
        for key in inventory.get("surface", ())
    }
    particle_attributes = {
        key: np.asarray(frame_data[cache_attribute_name("particles", key)])
        for key in inventory.get("particles", ())
    }
    whitewater_attributes = {
        key: np.asarray(frame_data[cache_attribute_name("whitewater", key)])
        for key in inventory.get("whitewater", ())
    }
    if "type" not in whitewater_attributes:
        # Cache-v1 and Phase-3 cache-v2 frames predate physical type state.
        # Preserve them in Combined Whitewater only; never relabel as foam.
        whitewater_attributes["type"] = np.full(
            len(whitewater), LEGACY_UNCLASSIFIED, dtype=np.int32)
    else:
        whitewater_attributes["type"] = validate_type_values(
            whitewater_attributes["type"], allow_unclassified=False)
    if (str(fs.whitewater_attribute_mode) == "TYPE"
            and whitewater_values is None):
        whitewater_values = whitewater_attributes["type"]
    surface_attributes = _merge_selected_attribute(
        surface_attributes, fs.surface_attribute_mode, surface_values)
    particle_attributes = _merge_selected_attribute(
        particle_attributes, fs.particle_attribute_mode, particle_values)
    whitewater_attributes = _merge_selected_attribute(
        whitewater_attributes, fs.whitewater_attribute_mode,
        whitewater_values)

    step = _display_stride(fs)
    if step > 1 and len(positions):
        positions = positions[::step]
        particle_attributes = {
            key: np.asarray(value)[::step]
            for key, value in particle_attributes.items()
        }
    return PublicationPayload(
        voxel_size=voxel_size, publish_surface=True,
        surface_visible=bool(fs.show_surface and getattr(ds, "surface_enabled", False)),
        surface_attribute_mode=str(fs.surface_attribute_mode),
        surface_verts=verts, surface_tris=tris,
        surface_attributes=surface_attributes,
        particles_visible=bool(fs.show_particles),
        particle_attribute_mode=str(fs.particle_attribute_mode),
        particle_radius=(voxel_size * 0.25
                         * getattr(fs, "particle_display_scale", 1.0)),
        particles=positions, particle_attributes=particle_attributes,
        whitewater_visible=bool(fs.show_whitewater and getattr(ds, "whitewater_enabled", False)),
        whitewater_output_mode=str(
            getattr(fs, "whitewater_output_mode", "COMBINED")),
        whitewater_attribute_mode=str(fs.whitewater_attribute_mode),
        whitewater_radius=(voxel_size * 0.15
                           * getattr(fs, "whitewater_display_scale", 1.0)),
        whitewater=whitewater, whitewater_attributes=whitewater_attributes,
    )


@_cost_phase("output_preparation")
def ensure_output_shells(scene):
    """Create persistent surface, particle, and whitewater output IDs.

    Guided setup needs stable render targets before a solver is initialized;
    visibility toggles remain independent and no geometry is published here.
    """
    _require_bpy()
    from . import materials

    surface = _ensure_object(
        scene, SURFACE_NAME,
        lambda: _new_blender_id(bpy.data.meshes.new, SURFACE_NAME), "MESH")
    fs = scene.fluidsim
    particles = _ensure_point_object(
        scene, PARTICLES_NAME, enabled_attribute_keys(fs, "particles"))
    whitewater = _ensure_point_object(
        scene, WHITEWATER_NAME, enabled_attribute_keys(fs, "whitewater"))
    foam = _ensure_point_object(
        scene, FOAM_NAME, enabled_attribute_keys(fs, "whitewater"))
    bubbles = _ensure_point_object(
        scene, BUBBLES_NAME, enabled_attribute_keys(fs, "whitewater"))
    spray = _ensure_point_object(
        scene, SPRAY_NAME, enabled_attribute_keys(fs, "whitewater"))
    _ensure_material(materials.ensure_surface_material, scene, surface)
    _ensure_material(materials.ensure_particle_material, scene, particles)
    _ensure_material(materials.ensure_whitewater_material, scene, whitewater)
    _ensure_material(materials.ensure_foam_material, scene, foam)
    _ensure_material(materials.ensure_bubble_material, scene, bubbles)
    _ensure_material(materials.ensure_spray_material, scene, spray)
    for obj in (surface, particles, whitewater, foam, bubbles, spray):
        try:
            obj["fluidsim_owned"] = True
            obj["fluidsim_kind"] = "OUTPUT"
        except (AttributeError, TypeError):
            pass
    return surface, particles, whitewater, foam, bubbles, spray


@_cost_phase("output_preparation")
def prepare_outputs(scene):
    """Output Preparation: create enabled IDs/schema without geometry writes."""
    _require_bpy()
    fs = scene.fluidsim
    ds = _domain_settings(scene)
    materials = None
    if (fs.show_surface and getattr(ds, "surface_enabled", False)) or fs.show_particles or (
            fs.show_whitewater and getattr(ds, "whitewater_enabled", False)):
        from . import materials
    if fs.show_surface and getattr(ds, "surface_enabled", False):
        _ensure_material(materials.ensure_surface_material, scene, _ensure_object(
            scene, SURFACE_NAME,
            lambda: _new_blender_id(
                bpy.data.meshes.new, SURFACE_NAME), "MESH"))
    if fs.show_particles:
        _ensure_material(
            materials.ensure_particle_material,
            scene, _ensure_point_object(
                scene, PARTICLES_NAME, enabled_attribute_keys(fs, "particles")))
    if fs.show_whitewater and getattr(ds, "whitewater_enabled", False):
        combined_enabled, separate_enabled = _whitewater_mode_flags(
            getattr(fs, "whitewater_output_mode", "COMBINED"))
        targets = []
        if combined_enabled:
            targets.append((WHITEWATER_NAME, materials.ensure_whitewater_material))
        if separate_enabled:
            targets.extend((
                (FOAM_NAME, materials.ensure_foam_material),
                (BUBBLES_NAME, materials.ensure_bubble_material),
                (SPRAY_NAME, materials.ensure_spray_material),
            ))
        for name, ensure_material in targets:
            _ensure_material(
                ensure_material,
                scene, _ensure_point_object(
                    scene, name, enabled_attribute_keys(fs, "whitewater")))


def _whitewater_mode_flags(mode: str) -> tuple[bool, bool]:
    mode = str(mode or "COMBINED").upper()
    if mode == "SEPARATE":
        return False, True
    if mode == "BOTH":
        return True, True
    return True, False


def _subset_aligned_attributes(attributes, indices, count):
    indices = np.asarray(indices, dtype=np.intp)
    result = {}
    for key, value in attributes.items():
        array = np.asarray(value)
        if array.ndim > 0 and len(array) == count:
            result[key] = np.ascontiguousarray(array[indices])
    return result


def _publish_whitewater_points(
        scene, name, positions, attributes, attribute_mode, radius,
        ensure_material, *, legacy_unclassified_count=0):
    values = _selected_attribute_values(attributes, attribute_mode)
    colors = _values_to_colors(values) if values is not None else None
    obj = _ensure_point_object(
        scene, name,
        tuple(key for key in ATTRIBUTE_SPECS if key in attributes))
    if _supports_pointcloud():
        _swap_points(obj, positions, radius, colors, attributes)
    else:
        _swap_vertex_cloud(obj, positions, colors, attributes)
    obj.hide_viewport = False
    obj.hide_render = False
    try:
        obj[LEGACY_UNCLASSIFIED_COUNT_TAG] = int(legacy_unclassified_count)
    except (AttributeError, TypeError):
        pass
    _ensure_material(ensure_material, scene, obj)
    return obj


def _hide_and_clear_output(scene, name):
    _clear_points(scene, name)
    obj = output_object(scene, name)
    if obj is not None:
        obj.hide_viewport = True
        obj.hide_render = True
        obj[LEGACY_UNCLASSIFIED_COUNT_TAG] = 0


@_cost_phase("output_publication")
def publish_payload(scene, payload):
    """Output Publication: mutate persistent IDs from owned facts only.

    This function has no solver, HostFrame, Warp-array, surface extraction, or
    synchronization access. Structural point-cloud fallback remains confined
    to ``_swap_points`` and is separately observable by reason.
    """
    _require_bpy()
    materials = None
    if (payload.surface_visible or payload.particles_visible
            or payload.whitewater_visible):
        from . import materials

    if payload.surface_visible:
        if payload.publish_surface:
            values = _selected_attribute_values(
                payload.surface_attributes, payload.surface_attribute_mode)
            colors = _values_to_colors(values) if values is not None else None
            obj = _ensure_object(
                scene, SURFACE_NAME,
                lambda: _new_blender_id(
                    bpy.data.meshes.new, SURFACE_NAME), "MESH")
            _swap_mesh(
                obj, payload.surface_verts, payload.surface_tris, colors,
                payload.surface_attributes)
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
        values = _selected_attribute_values(
            payload.particle_attributes, payload.particle_attribute_mode)
        colors = _values_to_colors(values) if values is not None else None
        obj = _ensure_point_object(
            scene, PARTICLES_NAME,
            tuple(key for key in ATTRIBUTE_SPECS
                  if key in payload.particle_attributes))
        if _supports_pointcloud():
            _swap_points(
                obj, payload.particles, payload.particle_radius, colors,
                payload.particle_attributes)
        else:
            _swap_vertex_cloud(
                obj, payload.particles, colors, payload.particle_attributes)
        _ensure_material(materials.ensure_particle_material, scene, obj)
    else:
        _clear_points(scene, PARTICLES_NAME)

    combined_enabled, separate_enabled = _whitewater_mode_flags(
        payload.whitewater_output_mode)
    if payload.whitewater_visible:
        particle_count = len(payload.whitewater)
        raw_types = payload.whitewater_attributes.get("type")
        if raw_types is None:
            raw_types = np.full(
                particle_count, LEGACY_UNCLASSIFIED, dtype=np.int32)
        type_values = validate_type_values(
            raw_types, allow_unclassified=True)
        if len(type_values) != particle_count:
            raise ValueError(
                "whitewater type length does not match point count")

        partitions = partition_type_indices(
            type_values, allow_unclassified=True)
        legacy_count = int(len(partitions[LEGACY_UNCLASSIFIED]))
        # Legacy cache points have no physical classification. Keep them on the
        # combined object even in Separate mode rather than dropping them or
        # falsely labelling them as foam.
        if combined_enabled or legacy_count:
            _publish_whitewater_points(
                scene, WHITEWATER_NAME, payload.whitewater,
                payload.whitewater_attributes,
                payload.whitewater_attribute_mode,
                payload.whitewater_radius, materials.ensure_whitewater_material,
                legacy_unclassified_count=legacy_count)
        else:
            _hide_and_clear_output(scene, WHITEWATER_NAME)

        if separate_enabled:
            typed_outputs = (
                (FOAM_NAME, WhitewaterType.FOAM, materials.ensure_foam_material),
                (BUBBLES_NAME, WhitewaterType.BUBBLE,
                 materials.ensure_bubble_material),
                (SPRAY_NAME, WhitewaterType.SPRAY,
                 materials.ensure_spray_material),
            )
            for name, particle_type, ensure_material in typed_outputs:
                indices = partitions[particle_type]
                attributes = _subset_aligned_attributes(
                    payload.whitewater_attributes, indices, particle_count)
                _publish_whitewater_points(
                    scene, name, np.ascontiguousarray(
                        payload.whitewater[indices]), attributes,
                    payload.whitewater_attribute_mode,
                    payload.whitewater_radius, ensure_material)
        else:
            for name in (FOAM_NAME, BUBBLES_NAME, SPRAY_NAME):
                _hide_and_clear_output(scene, name)
    else:
        for name in WHITEWATER_OUTPUT_NAMES:
            _hide_and_clear_output(scene, name)
    return _sync_render_inputs(scene)

def refresh_existing_output_settings(scene):
    """Apply display/material settings to existing output objects only."""
    _require_bpy()
    from . import materials

    # This is intentionally metadata-only. Without an active solver we cannot
    # recompute attributes or surfaces, but cache/current-output objects should
    # still respond to visibility and material picker changes.
    fs = scene.fluidsim
    ds = _domain_settings(scene)
    surface = output_object(scene, SURFACE_NAME)
    if surface is not None:
        visible = fs.show_surface and getattr(ds, "surface_enabled", False)
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
            radius = (getattr(ds, "voxel_size", 0.05) * 0.25 *
                      getattr(fs, "particle_display_scale", 1.0))
            _set_point_radius(particles, radius)
            _ensure_material(
                materials.ensure_particle_material, scene, particles)

    combined_enabled, separate_enabled = _whitewater_mode_flags(
        getattr(fs, "whitewater_output_mode", "COMBINED"))
    visible = fs.show_whitewater and getattr(ds, "whitewater_enabled", False)
    typed_materials = {
        WHITEWATER_NAME: materials.ensure_whitewater_material,
        FOAM_NAME: materials.ensure_foam_material,
        BUBBLES_NAME: materials.ensure_bubble_material,
        SPRAY_NAME: materials.ensure_spray_material,
    }
    for name, ensure_material in typed_materials.items():
        obj = output_object(scene, name)
        if obj is None:
            continue
        legacy_visible = bool(
            name == WHITEWATER_NAME
            and int(obj.get(LEGACY_UNCLASSIFIED_COUNT_TAG, 0)) > 0)
        selected = (combined_enabled or legacy_visible
                    if name == WHITEWATER_NAME else separate_enabled)
        obj_visible = visible and selected
        obj.hide_viewport = not obj_visible
        obj.hide_render = not obj_visible
        if obj_visible:
            radius = (getattr(ds, "voxel_size", 0.05) * 0.15 *
                      getattr(fs, "whitewater_display_scale", 1.0))
            _set_point_radius(obj, radius)
            _ensure_material(ensure_material, scene, obj)
    return _sync_render_inputs(scene)


def ensure_render_visibility(scene):
    """Apply render visibility to persistent simulation outputs."""
    fs = scene.fluidsim
    combined_enabled, separate_enabled = _whitewater_mode_flags(
        getattr(fs, "whitewater_output_mode", "COMBINED"))
    for name, show in (
        (SURFACE_NAME, fs.show_surface),
        (PARTICLES_NAME, fs.show_particles),
        (WHITEWATER_NAME, fs.show_whitewater and combined_enabled),
        (FOAM_NAME, fs.show_whitewater and separate_enabled),
        (BUBBLES_NAME, fs.show_whitewater and separate_enabled),
        (SPRAY_NAME, fs.show_whitewater and separate_enabled),
    ):
        obj = output_object(scene, name)
        if obj is not None:
            if (name == WHITEWATER_NAME and fs.show_whitewater
                    and int(obj.get(LEGACY_UNCLASSIFIED_COUNT_TAG, 0)) > 0):
                show = True
            obj.hide_render = not show


def configure_cycles_points(scene):
    """Resolve prepared point-cloud outputs for Cycles rendering."""
    for name in (PARTICLES_NAME, *WHITEWATER_OUTPUT_NAMES):
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
    ds = _domain_settings(scene)
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
            getattr(ds, "live_surface_publish_cadence", 1))
    )
    payload = prepare_live_payload(
        scene, host, publish_surface=publish_surface)
    return publish_payload(scene, payload)


def update_outputs_from_cache_frame(scene, frame_data):
    """Write a cached frame snapshot into the regular Blender output objects."""
    _require_bpy()
    return publish_payload(scene, prepare_cache_payload(scene, frame_data))


def _clear_points(scene, name):
    obj = output_object(scene, name)
    if obj is None:
        return
    if obj.type == "POINTCLOUD" and live_point_count(obj):
        _swap_points(obj, np.zeros((0, 3), dtype=np.float32), 0.01)
    elif obj.type == "MESH" and len(obj.data.vertices):
        _swap_vertex_cloud(obj, np.zeros((0, 3), dtype=np.float32))


def remove_output(scene, kind, *, expected_object=None):
    """Remove one identity-checked FluidSim-owned output object."""
    _require_bpy()
    allowed = {SURFACE_NAME, PARTICLES_NAME, *WHITEWATER_OUTPUT_NAMES}
    kind = str(kind)
    if kind not in allowed:
        raise ValueError(f"unsupported FluidSim output kind {kind!r}")
    obj = output_object(scene, kind)
    if obj is None:
        return False
    if expected_object is not None and obj is not expected_object:
        raise RuntimeError("FluidSim output identity changed before removal")
    if not bool(getattr(obj, "get", lambda *_args: False)("fluidsim_owned")):
        raise RuntimeError("refusing to remove an output not owned by FluidSim")
    _remove_output_object(obj)
    return True


def remove_outputs(scene=None):
    _require_bpy()
    if scene is not None:
        objects = [
            output_object(scene, name)
            for name in (SURFACE_NAME, PARTICLES_NAME, *WHITEWATER_OUTPUT_NAMES)
        ]
    else:
        objects = [
            obj for obj in list(bpy.data.objects)
            if getattr(obj, "get", lambda *_args: None)(OUTPUT_TAG)
            in {SURFACE_NAME, PARTICLES_NAME, *WHITEWATER_OUTPUT_NAMES}
        ]
    for obj in objects:
        if obj is not None:
            _remove_output_object(obj)
