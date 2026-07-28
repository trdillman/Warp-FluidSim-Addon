"""Viewport output: surface mesh, particle/whitewater point clouds, overlay.

Simulation results are written into real Blender datablocks inside a
dedicated output collection, so they are visible in the viewport and in
renders without any caching step.
"""

from __future__ import annotations

from dataclasses import dataclass
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

_OVERLAY_HANDLE = None
bpy = sys.modules.get("bpy")


def _require_bpy():
    """Load Blender's API only for datablock and overlay operations."""
    global bpy
    if bpy is None:
        try:
            import bpy as blender_api
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Viewport datablock operations require Blender's Python runtime"
            ) from exc
        if not hasattr(blender_api, "data"):
            raise RuntimeError(
                "Viewport datablock operations require Blender's bpy module"
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
        coll = bpy.data.collections.new(name)
    if name not in {c.name for c in scene.collection.children}:
        try:
            scene.collection.children.link(coll)
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
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
        elif (hasattr(bpy.data, "pointclouds")
              and isinstance(data, bpy.types.PointCloud)):
            bpy.data.pointclouds.remove(data)


def _ensure_object(scene, name, data_factory, expected_type):
    """Get/create one tagged, correctly-typed add-on output object."""
    obj = output_object(scene, name)
    if obj is not None and obj.type != expected_type:
        # Tagged/adopted objects are add-on owned, so a type mismatch is safe
        # to recreate. Untagged same-named user objects are never selected.
        _remove_output_object(obj)
        obj = None
    if obj is None:
        obj = bpy.data.objects.new(name, data_factory())
        obj[OUTPUT_TAG] = name
        _output_collection(scene).objects.link(obj)
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


def _write_point_colors(data, colors):
    if colors is None or len(colors) == 0:
        return
    attr = data.attributes.get(COLOR_ATTR)
    if attr is None:
        attr = data.attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.asarray(colors, dtype=np.float32).ravel())
    try:
        data.attributes.active = attr
    except Exception:
        pass


def _set_point_radius(data, radius):
    attr = getattr(data, "attributes", None)
    if attr is None:
        return
    rad_attr = attr.get("radius")
    if rad_attr is None:
        return
    n = len(rad_attr.data)
    if n:
        rad_attr.data.foreach_set("value", np.full(n, radius, dtype=np.float32))
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
        mesh = bpy.data.meshes.new(obj.name)
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
        obj.data = mesh
        for m in mats:
            obj.data.materials.append(m)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                bpy.data.meshes.remove(previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                bpy.data.pointclouds.remove(previous)


def _supports_pointcloud():
    return hasattr(bpy.data, "pointclouds")


def _ensure_point_object(scene, name):
    """Get/create a point output object, native point cloud where available."""
    if _supports_pointcloud():
        return _ensure_object(
            scene, name, lambda: bpy.data.pointclouds.new(name), "POINTCLOUD")
    return _ensure_object(
        scene, name, lambda: bpy.data.meshes.new(name), "MESH")


def _can_reuse_pointcloud(pc, point_count, wants_color):
    """Return whether an attached Blender point cloud is safe to overwrite."""
    if (pc is None or getattr(pc, "users", 1) != 1
            or not hasattr(pc, "resize")):
        return False
    current_count = len(pc.points)
    has_color = pc.attributes.get(COLOR_ATTR) is not None
    return current_count == point_count and has_color == wants_color


def _swap_points(obj, positions, radius, colors=None, *, reuse_datablock=True):
    """Replace a point cloud object's points (bulk attribute transfer).

    Blender 5.1 can corrupt native point-cloud state when an attached cloud is
    resized with custom attributes or when an attribute is removed/recreated
    repeatedly. Reuse is therefore limited to stable point count and attribute
    schema; all structural changes use atomic datablock replacement.
    """
    previous = getattr(obj, "data", None)
    pc_old = previous if obj.type == "POINTCLOUD" else None
    n = len(positions)
    wants_color = colors is not None and len(colors) != 0
    reuse = (reuse_datablock
             and _can_reuse_pointcloud(pc_old, n, wants_color))
    pc = pc_old if reuse else bpy.data.pointclouds.new(obj.name)
    if not reuse:
        pc.resize(n)
    if n:
        pos_attr = pc.attributes.get("position")
        if pos_attr is None:
            pos_attr = pc.attributes.new("position", "FLOAT_VECTOR", "POINT")
        pos_attr.data.foreach_set(
            "vector", np.asarray(positions, dtype=np.float32).ravel())
        rad_attr = pc.attributes.get("radius")
        if rad_attr is None:
            rad_attr = pc.attributes.new("radius", "FLOAT", "POINT")
        rad_attr.data.foreach_set(
            "value", np.full(n, radius, dtype=np.float32))
        _write_point_colors(pc, colors)
    pc.update_tag()
    if not reuse:
        mats = [m for m in getattr(previous, "materials", ())]
        obj.data = pc
        for m in mats:
            obj.data.materials.append(m)
        if previous is not None and previous.users == 0:
            if isinstance(previous, bpy.types.Mesh):
                bpy.data.meshes.remove(previous)
            elif (hasattr(bpy.data, "pointclouds")
                  and isinstance(previous, bpy.types.PointCloud)):
                bpy.data.pointclouds.remove(previous)


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
class _ViewportFrame:
    voxel_size: float
    publish_surface: bool
    surface_verts: np.ndarray
    surface_tris: np.ndarray
    surface_attributes: dict[str, np.ndarray]
    particles: np.ndarray
    particle_attributes: dict[str, np.ndarray]
    whitewater: np.ndarray
    whitewater_attributes: dict[str, np.ndarray]
    particles_decimated: bool = False


def _attribute_map(mode, values):
    return {} if values is None else {mode: np.asarray(values)}


def _display_stride(fs):
    fraction = float(getattr(fs, "particle_display_fraction", 1.0))
    if not 0.0 < fraction < 1.0:
        return 1
    return max(1, int(round(1.0 / fraction)))


def _solver_frame(scene, solver, publish_surface=True):
    from ..sim.frame_output import HostFrame

    fs = scene.fluidsim
    host = solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)
    # particle_display_fraction decimates on the device, so the particles the
    # user asked not to display never cross PCIe. HostFrame keys the readback
    # per stride, so cache/export still get the full-fidelity array from this
    # same frame. Attribute values stride identically or colors misalign.
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
    return _ViewportFrame(
        voxel_size=float(solver.settings.voxel_size),
        publish_surface=publish_surface,
        surface_verts=np.asarray(verts, dtype=np.float32),
        surface_tris=np.asarray(tris, dtype=np.int32),
        surface_attributes=_attribute_map(
            fs.surface_attribute_mode, surface_values),
        particles=np.asarray(positions, dtype=np.float32),
        particle_attributes=_attribute_map(
            fs.particle_attribute_mode, particle_values),
        whitewater=np.asarray(whitewater, dtype=np.float32),
        whitewater_attributes=_attribute_map(
            fs.whitewater_attribute_mode, whitewater_values),
        particles_decimated=step > 1,
    )


def _cached_frame(scene, frame_data):
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
            frame_data,
            "particle",
            fs.surface_attribute_mode,
            positions,
            velocities,
        )
        if particle_surface_values is not None:
            surface_values = _nearest_values(
                verts, positions, particle_surface_values, voxel_size * 2.0)
    particle_values = _cached_named_values(
        frame_data,
        "particle",
        fs.particle_attribute_mode,
        positions,
        velocities if len(velocities) == len(positions) else None,
    )
    whitewater_velocities = frame_data.get("whitewater_velocities")
    if whitewater_velocities is not None:
        whitewater_velocities = np.asarray(whitewater_velocities, dtype=np.float32)
    whitewater_values = _cached_named_values(
        frame_data,
        "whitewater",
        fs.whitewater_attribute_mode,
        whitewater,
        whitewater_velocities,
    )
    return _ViewportFrame(
        voxel_size=voxel_size,
        publish_surface=True,
        surface_verts=verts,
        surface_tris=tris,
        surface_attributes=_attribute_map(
            fs.surface_attribute_mode, surface_values),
        particles=positions,
        particle_attributes=_attribute_map(
            fs.particle_attribute_mode, particle_values),
        whitewater=whitewater,
        whitewater_attributes=_attribute_map(
            fs.whitewater_attribute_mode, whitewater_values),
    )


def ensure_outputs(scene):
    """Create the output collection, objects and materials up front.

    Everything here allocates Blender IDs and, for the collection, changes
    depsgraph relations. _render_outputs would otherwise do it on the first
    live frame -- i.e. from inside frame_change_post, while the depsgraph the
    handler was handed is still being evaluated. Doing it at initialize time
    leaves the handler writing geometry into IDs that already exist.
    """
    _require_bpy()
    from . import materials

    fs = scene.fluidsim
    if fs.show_surface and fs.surface_enabled:
        materials.ensure_surface_material(scene, _ensure_object(
            scene, SURFACE_NAME,
            lambda: bpy.data.meshes.new(SURFACE_NAME), "MESH"))
    if fs.show_particles:
        materials.ensure_particle_material(
            scene, _ensure_point_object(scene, PARTICLES_NAME))
    if fs.show_whitewater and fs.whitewater_enabled:
        materials.ensure_whitewater_material(
            scene, _ensure_point_object(scene, WHITEWATER_NAME))


def _render_outputs(scene, frame):
    _require_bpy()
    from . import materials

    fs = scene.fluidsim
    if fs.show_surface and fs.surface_enabled:
        if frame.publish_surface:
            values = frame.surface_attributes.get(fs.surface_attribute_mode)
            colors = _values_to_colors(values) if values is not None else None
            obj = _ensure_object(
                scene, SURFACE_NAME,
                lambda: bpy.data.meshes.new(SURFACE_NAME), "MESH")
            _swap_mesh(obj, frame.surface_verts, frame.surface_tris, colors)
            materials.ensure_surface_material(scene, obj)
    else:
        obj = output_object(scene, SURFACE_NAME)
        if obj is not None and obj.type == "MESH" and len(obj.data.vertices):
            _swap_mesh(
                obj,
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
            )

    if fs.show_particles:
        positions = frame.particles
        values = frame.particle_attributes.get(fs.particle_attribute_mode)
        # The live path already decimated on the device; only cached frames
        # (full-fidelity on disk) still need the host-side stride here.
        step = 1 if frame.particles_decimated else _display_stride(fs)
        if step > 1 and len(positions):
            positions = positions[::step]
            if values is not None:
                values = values[::step]
        colors = _values_to_colors(values) if values is not None else None
        radius = (frame.voxel_size * 0.25 *
                  getattr(fs, "particle_display_scale", 1.0))
        obj = _ensure_point_object(scene, PARTICLES_NAME)
        if _supports_pointcloud():
            _swap_points(obj, positions, radius, colors)
        else:
            _swap_vertex_cloud(obj, positions, colors)
        materials.ensure_particle_material(scene, obj)
    else:
        _clear_points(scene, PARTICLES_NAME)

    if fs.show_whitewater and fs.whitewater_enabled:
        values = frame.whitewater_attributes.get(fs.whitewater_attribute_mode)
        colors = _values_to_colors(values) if values is not None else None
        radius = (frame.voxel_size * 0.15 *
                  getattr(fs, "whitewater_display_scale", 1.0))
        obj = _ensure_point_object(scene, WHITEWATER_NAME)
        if _supports_pointcloud():
            # Dynamic whitewater counts make in-place PointCloud.resize plus
            # attribute replacement unsafe in Blender 5.1 (native crash after
            # repeated playback frames). Use an atomic datablock swap here.
            _swap_points(
                obj, frame.whitewater, radius, colors, reuse_datablock=False)
        else:
            _swap_vertex_cloud(obj, frame.whitewater, colors)
        materials.ensure_whitewater_material(scene, obj)
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
            materials.ensure_surface_material(scene, surface)

    particles = output_object(scene, PARTICLES_NAME)
    if particles is not None:
        particles.hide_viewport = not fs.show_particles
        particles.hide_render = not fs.show_particles
        if fs.show_particles:
            if hasattr(particles.data, "attributes"):
                radius = (getattr(fs, "voxel_size", 0.05) * 0.25 *
                          getattr(fs, "particle_display_scale", 1.0))
                _set_point_radius(particles.data, radius)
            materials.ensure_particle_material(scene, particles)

    whitewater = output_object(scene, WHITEWATER_NAME)
    if whitewater is not None:
        visible = fs.show_whitewater and fs.whitewater_enabled
        whitewater.hide_viewport = not visible
        whitewater.hide_render = not visible
        if visible:
            if hasattr(whitewater.data, "attributes"):
                radius = (getattr(fs, "voxel_size", 0.05) * 0.15 *
                          getattr(fs, "whitewater_display_scale", 1.0))
                _set_point_radius(whitewater.data, radius)
            materials.ensure_whitewater_material(scene, whitewater)


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
    if hasattr(solver, "host_frame"):
        host = solver.host_frame()
    else:
        from ..sim.frame_output import HostFrame
        host = HostFrame(solver)
    publish_surface = (
        force_surface or not has_surface or
        host.should_publish_surface(
            getattr(fs, "live_surface_publish_cadence", 4))
    )
    _render_outputs(
        scene, _solver_frame(scene, solver, publish_surface=publish_surface))


def update_outputs_from_cache_frame(scene, frame_data):
    """Write a cached frame snapshot into the regular Blender output objects."""
    _require_bpy()
    _render_outputs(scene, _cached_frame(scene, frame_data))


def _clear_points(scene, name):
    obj = output_object(scene, name)
    if obj is None:
        return
    if obj.type == "POINTCLOUD" and len(obj.data.points):
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


# ---------------------------------------------------------------------------
# Profiling overlay
# ---------------------------------------------------------------------------


def _draw_overlay():
    _require_bpy()
    import blf

    context = bpy.context
    scene = context.scene
    fs = getattr(scene, "fluidsim", None)
    if fs is None or not fs.show_overlay:
        return
    from .scene_bridge import session

    sess = session()
    if not sess.active:
        return

    font_id = 0
    dpi_scale = context.preferences.system.ui_scale
    size = int(12 * dpi_scale)
    blf.size(font_id, size)
    blf.color(font_id, 0.9, 0.95, 1.0, 1.0)
    x = 20 * dpi_scale
    y = 60 * dpi_scale
    lh = size * 1.4

    lines = [
        f"FluidSim Warp  |  {fs.status}",
        f"particles {fs.stat_particles:,}   whitewater {fs.stat_whitewater:,}   "
        f"voxels {fs.stat_voxels:,}   substeps {fs.stat_substeps}   "
        f"{fs.stat_step_ms:.1f} ms/frame",
    ]
    if sess.profiler is not None:
        summary = sess.profiler.summary(last_n=30)
        phase_names = {
            f"compute_{slot}": phase
            for phase, slot in TIMER_PHASE_SLOTS.items()
        }
        parts = []
        for slot, label in phase_names.items():
            v = summary.get(slot, 0.0)
            if v > 0.05:
                parts.append(f"{label} {v:.1f}")
        if parts:
            lines.append("timers(ms): " + "  ".join(parts))
    if sess.error:
        lines.append(f"error: {sess.error}")

    for i, line in enumerate(reversed(lines)):
        blf.position(font_id, x, y + i * lh, 0)
        blf.draw(font_id, line)


def register_overlay():
    global _OVERLAY_HANDLE
    _require_bpy()
    unregister_overlay()
    _OVERLAY_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_overlay, (), "WINDOW", "POST_PIXEL")


def unregister_overlay():
    global _OVERLAY_HANDLE
    if _OVERLAY_HANDLE is None:
        return
    _require_bpy()
    if _OVERLAY_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_OVERLAY_HANDLE, "WINDOW")
        except Exception:
            pass
        _OVERLAY_HANDLE = None
