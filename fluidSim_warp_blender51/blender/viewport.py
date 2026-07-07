"""Viewport output: surface mesh, particle/whitewater point clouds, overlay.

Simulation results are written into real Blender datablocks inside a
dedicated output collection, so they are visible in the viewport and in
renders without any caching step.
"""

from __future__ import annotations

import bpy
import numpy as np

SURFACE_NAME = "FluidSim Surface"
PARTICLES_NAME = "FluidSim Particles"
WHITEWATER_NAME = "FluidSim Whitewater"
COLOR_ATTR = "display_color"

_OVERLAY_HANDLE = None


# ---------------------------------------------------------------------------
# Output collection / objects
# ---------------------------------------------------------------------------


def _output_collection(scene):
    name = scene.fluidsim.output_collection_name or "FluidSim Output"
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if name not in {c.name for c in scene.collection.children}:
        try:
            scene.collection.children.link(coll)
        except RuntimeError:
            pass
    return coll


def _ensure_object(scene, name, data_factory):
    """Get-or-create an output object linked to the output collection."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data_factory())
        _output_collection(scene).objects.link(obj)
    else:
        coll = _output_collection(scene)
        if obj.name not in {o.name for o in coll.objects}:
            try:
                coll.objects.link(obj)
            except RuntimeError:
                pass
    return obj


def _values_to_colors(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        t = np.zeros(len(values), dtype=np.float32)
    else:
        lo = float(np.percentile(values[finite], 5.0))
        hi = float(np.percentile(values[finite], 95.0))
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
    cell_size = max(float(cell_size), 1.0e-5)
    keys = np.floor(sample_points / cell_size).astype(np.int32)
    buckets = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(i)
    default = float(np.nanmean(sample_values)) if len(sample_values) else 0.0
    out = np.full(len(points), default, dtype=np.float32)
    point_keys = np.floor(points / cell_size).astype(np.int32)
    offsets = [(dx, dy, dz) for dx in range(-1, 2)
               for dy in range(-1, 2) for dz in range(-1, 2)]
    for pi, key in enumerate(point_keys):
        best_i = -1
        best_d2 = 1.0e30
        k0 = tuple(int(v) for v in key)
        for dx, dy, dz in offsets:
            inds = buckets.get((k0[0] + dx, k0[1] + dy, k0[2] + dz))
            if not inds:
                continue
            pts = sample_points[inds]
            d2 = np.sum((pts - points[pi]) ** 2, axis=1)
            j = int(np.argmin(d2))
            if d2[j] < best_d2:
                best_d2 = float(d2[j])
                best_i = inds[j]
        if best_i >= 0:
            out[pi] = sample_values[best_i]
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
    old = obj.data if obj.type == "MESH" else None
    mesh = bpy.data.meshes.new(obj.name)
    nv = len(verts)
    nt = len(tris)
    if nv:
        mesh.vertices.add(nv)
        mesh.vertices.foreach_set("co", np.asarray(verts, dtype=np.float32).ravel())
    if nt:
        mesh.loops.add(nt * 3)
        mesh.loops.foreach_set("vertex_index",
                               np.asarray(tris, dtype=np.int32).ravel())
        mesh.polygons.add(nt)
        mesh.polygons.foreach_set("loop_start",
                                   np.arange(0, nt * 3, 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(nt, 3, dtype=np.int32))
        mesh.validate()
        mesh.shade_smooth()
    if nv:
        _write_mesh_colors(mesh, tris, colors)
    mesh.update()
    mats = [m for m in (old.materials if old else [])]
    obj.data = mesh
    for m in mats:
        obj.data.materials.append(m)
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)


def _supports_pointcloud():
    return hasattr(bpy.data, "pointclouds")


def _swap_points(obj, positions, radius, colors=None):
    """Replace a point cloud object's points (bulk attribute transfer)."""
    pc_old = obj.data
    pc = bpy.data.pointclouds.new(obj.name)
    n = len(positions)
    if n:
        pc.resize(n)
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
    mats = [m for m in (pc_old.materials if pc_old else [])]
    obj.data = pc
    for m in mats:
        obj.data.materials.append(m)
    if pc_old is not None and pc_old.users == 0:
        bpy.data.pointclouds.remove(pc_old)


def _swap_vertex_cloud(obj, positions, colors=None):
    _swap_mesh(obj, positions, np.zeros((0, 3), dtype=np.int32), colors)


def _particle_values(solver, mode, positions, velocities=None, indices=None):
    if mode == "NONE":
        return None
    if mode == "HEIGHT":
        return positions[:, 2]
    if mode == "SPEED":
        if velocities is None:
            velocities = solver.particle_velocities()
            if indices is not None:
                velocities = velocities[indices]
        return np.linalg.norm(velocities, axis=1)
    name = mode.lower()
    if name in {"age", "life", "temperature", "viscosity", "density"}:
        vals = solver.particles.attr(name).numpy()[:solver.particles.host_count]
        return vals[indices] if indices is not None else vals
    return None


def _whitewater_values(solver, mode, positions):
    if mode == "NONE":
        return None
    n = len(positions)
    if mode == "HEIGHT":
        return positions[:, 2]
    if mode == "SPEED":
        vel = solver.whitewater.velocity.numpy()[:n]
        return np.linalg.norm(vel, axis=1)
    if mode == "AGE":
        return solver.whitewater.age.numpy()[:n]
    if mode == "LIFE":
        return solver.whitewater.life.numpy()[:n]
    return None


def _surface_values(solver, mode, verts):
    if mode == "NONE":
        return None
    if mode == "HEIGHT":
        return verts[:, 2]
    pos = solver.particle_positions()
    vel = solver.particle_velocities() if mode == "SPEED" else None
    values = _particle_values(solver, mode, pos, vel)
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


def refresh_existing_output_settings(scene):
    """Apply display/material settings to existing output objects only."""
    from . import materials

    # This is intentionally metadata-only. Without an active solver we cannot
    # recompute attributes or surfaces, but cache/current-output objects should
    # still respond to visibility and material picker changes.
    fs = scene.fluidsim
    surface = bpy.data.objects.get(SURFACE_NAME)
    if surface is not None:
        visible = fs.show_surface and fs.surface_enabled
        surface.hide_viewport = not visible
        surface.hide_render = not visible
        if visible:
            materials.ensure_surface_material(scene, surface)

    particles = bpy.data.objects.get(PARTICLES_NAME)
    if particles is not None:
        particles.hide_viewport = not fs.show_particles
        particles.hide_render = not fs.show_particles
        if fs.show_particles:
            if hasattr(particles.data, "attributes"):
                radius = (getattr(fs, "voxel_size", 0.05) * 0.25 *
                          getattr(fs, "particle_display_scale", 1.0))
                _set_point_radius(particles.data, radius)
            materials.ensure_particle_material(scene, particles)

    whitewater = bpy.data.objects.get(WHITEWATER_NAME)
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


def update_outputs(scene, sess):
    """Write current solver outputs into Blender datablocks."""
    from . import materials

    if not sess.active:
        return
    fs = scene.fluidsim
    solver = sess.solver

    if fs.show_surface and solver.settings.surface_enabled:
        verts, tris = solver.surface_mesh()
        values = _surface_values(solver, fs.surface_attribute_mode, verts)
        colors = _values_to_colors(values) if values is not None else None
        obj = _ensure_object(scene, SURFACE_NAME,
                             lambda: bpy.data.meshes.new(SURFACE_NAME))
        _swap_mesh(obj, verts, tris, colors)
        materials.ensure_surface_material(scene, obj)
    else:
        obj = bpy.data.objects.get(SURFACE_NAME)
        if obj is not None and obj.type == "MESH" and len(obj.data.vertices):
            _swap_mesh(obj, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    if fs.show_particles:
        pos = solver.particle_positions()
        vel = None
        indices = None
        frac = fs.particle_display_fraction
        if frac < 1.0 and len(pos):
            step = max(1, int(round(1.0 / frac)))
            indices = np.arange(0, len(pos), step, dtype=np.int64)
            pos = pos[::step]
        if fs.particle_attribute_mode == "SPEED":
            vel = solver.particle_velocities()
            if indices is not None:
                vel = vel[indices]
        values = _particle_values(
            solver, fs.particle_attribute_mode, pos, vel, indices)
        colors = _values_to_colors(values) if values is not None else None
        radius = (solver.settings.voxel_size * 0.25 *
                  getattr(fs, "particle_display_scale", 1.0))
        if _supports_pointcloud():
            obj = _ensure_object(scene, PARTICLES_NAME,
                                  lambda: bpy.data.pointclouds.new(PARTICLES_NAME))
            _swap_points(obj, pos, radius, colors)
        else:
            obj = _ensure_object(scene, PARTICLES_NAME,
                                  lambda: bpy.data.meshes.new(PARTICLES_NAME))
            _swap_vertex_cloud(obj, pos, colors)
        materials.ensure_particle_material(scene, obj)
    else:
        _clear_points(PARTICLES_NAME)

    if fs.show_whitewater and fs.whitewater_enabled and solver.settings.whitewater_enabled:
        pos = solver.whitewater_positions()
        values = _whitewater_values(solver, fs.whitewater_attribute_mode, pos)
        colors = _values_to_colors(values) if values is not None else None
        radius = (solver.settings.voxel_size * 0.15 *
                  getattr(fs, "whitewater_display_scale", 1.0))
        if _supports_pointcloud():
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                  lambda: bpy.data.pointclouds.new(WHITEWATER_NAME))
            _swap_points(obj, pos, radius, colors)
        else:
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                  lambda: bpy.data.meshes.new(WHITEWATER_NAME))
            _swap_vertex_cloud(obj, pos, colors)
        materials.ensure_whitewater_material(scene, obj)
    else:
        _clear_points(WHITEWATER_NAME)


def update_outputs_from_cache_frame(scene, frame_data):
    """Write a cached frame snapshot into the regular Blender output objects."""
    from . import materials

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

    if fs.show_surface and fs.surface_enabled and len(verts):
        values = _cached_values(fs.surface_attribute_mode, verts)
        if values is None and fs.surface_attribute_mode == "SPEED":
            particle_values = _cached_named_values(
                frame_data, "particle", "SPEED", positions, velocities)
            if particle_values is not None:
                values = _nearest_values(
                    verts, positions, particle_values, voxel_size * 2.0)
        elif values is None:
            particle_values = _cached_named_values(
                frame_data,
                "particle",
                fs.surface_attribute_mode,
                positions,
                velocities,
            )
            if particle_values is not None:
                values = _nearest_values(
                    verts, positions, particle_values, voxel_size * 2.0)
        colors = _values_to_colors(values) if values is not None else None
        obj = _ensure_object(scene, SURFACE_NAME,
                             lambda: bpy.data.meshes.new(SURFACE_NAME))
        _swap_mesh(obj, verts, tris, colors)
        materials.ensure_surface_material(scene, obj)
    else:
        obj = bpy.data.objects.get(SURFACE_NAME)
        if obj is not None and obj.type == "MESH" and len(obj.data.vertices):
            _swap_mesh(obj, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    if fs.show_particles:
        pos = positions
        vel = velocities if len(velocities) == len(positions) else None
        indices = None
        frac = fs.particle_display_fraction
        if frac < 1.0 and len(pos):
            step = max(1, int(round(1.0 / frac)))
            indices = np.arange(0, len(pos), step, dtype=np.int64)
            pos = pos[::step]
            if vel is not None:
                vel = vel[indices]
        values = _cached_named_values(
            frame_data,
            "particle",
            fs.particle_attribute_mode,
            pos,
            vel,
            indices,
        )
        colors = _values_to_colors(values) if values is not None else None
        radius = voxel_size * 0.25 * getattr(fs, "particle_display_scale", 1.0)
        if _supports_pointcloud():
            obj = _ensure_object(scene, PARTICLES_NAME,
                                  lambda: bpy.data.pointclouds.new(PARTICLES_NAME))
            _swap_points(obj, pos, radius, colors)
        else:
            obj = _ensure_object(scene, PARTICLES_NAME,
                                  lambda: bpy.data.meshes.new(PARTICLES_NAME))
            _swap_vertex_cloud(obj, pos, colors)
        materials.ensure_particle_material(scene, obj)
    else:
        _clear_points(PARTICLES_NAME)

    if fs.show_whitewater and fs.whitewater_enabled and len(whitewater):
        ww_vel = frame_data.get("whitewater_velocities")
        if ww_vel is not None:
            ww_vel = np.asarray(ww_vel, dtype=np.float32)
        values = _cached_named_values(
            frame_data,
            "whitewater",
            fs.whitewater_attribute_mode,
            whitewater,
            ww_vel,
        )
        colors = _values_to_colors(values) if values is not None else None
        radius = voxel_size * 0.15 * getattr(
            fs, "whitewater_display_scale", 1.0)
        if _supports_pointcloud():
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                  lambda: bpy.data.pointclouds.new(WHITEWATER_NAME))
            _swap_points(obj, whitewater, radius, colors)
        else:
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                  lambda: bpy.data.meshes.new(WHITEWATER_NAME))
            _swap_vertex_cloud(obj, whitewater, colors)
        materials.ensure_whitewater_material(scene, obj)
    else:
        _clear_points(WHITEWATER_NAME)


def _clear_points(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    if obj.type == "POINTCLOUD" and len(obj.data.points):
        _swap_points(obj, np.zeros((0, 3), dtype=np.float32), 0.01)
    elif obj.type == "MESH" and len(obj.data.vertices):
        _swap_vertex_cloud(obj, np.zeros((0, 3), dtype=np.float32))


def remove_outputs(scene=None):
    for name in (SURFACE_NAME, PARTICLES_NAME, WHITEWATER_NAME):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            data = obj.data
            bpy.data.objects.remove(obj)
            if data is not None and data.users == 0:
                if isinstance(data, bpy.types.Mesh):
                    bpy.data.meshes.remove(data)
                elif hasattr(bpy.data, "pointclouds") and isinstance(
                        data, bpy.types.PointCloud):
                    bpy.data.pointclouds.remove(data)


# ---------------------------------------------------------------------------
# Profiling overlay
# ---------------------------------------------------------------------------


def _draw_overlay():
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
            "compute_0": "sourcing", "compute_1": "indexing", "compute_2": "p2g",
            "compute_3": "forces", "compute_4": "viscosity", "compute_5": "pressure",
            "compute_6": "g2p", "compute_7": "advection", "compute_8": "collision",
            "compute_10": "relax", "compute_11": "surface",
            "compute_12": "whitewater", "compute_13": "custom",
            "compute_14": "lifecycle", "compute_15": "misc",
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
    unregister_overlay()
    _OVERLAY_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
        _draw_overlay, (), "WINDOW", "POST_PIXEL")


def unregister_overlay():
    global _OVERLAY_HANDLE
    if _OVERLAY_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_OVERLAY_HANDLE, "WINDOW")
        except Exception:
            pass
        _OVERLAY_HANDLE = None
