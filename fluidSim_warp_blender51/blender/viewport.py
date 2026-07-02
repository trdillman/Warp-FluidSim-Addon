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


def _swap_mesh(obj, verts, tris):
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
    mesh.update()
    mats = [m for m in (old.materials if old else [])]
    obj.data = mesh
    for m in mats:
        obj.data.materials.append(m)
    if old is not None and old.users == 0:
        bpy.data.meshes.remove(old)


def _supports_pointcloud():
    return hasattr(bpy.data, "pointclouds")


def _swap_points(obj, positions, radius):
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
        pc.update_tag()
    mats = [m for m in (pc_old.materials if pc_old else [])]
    obj.data = pc
    for m in mats:
        obj.data.materials.append(m)
    if pc_old is not None and pc_old.users == 0:
        bpy.data.pointclouds.remove(pc_old)


def _swap_vertex_cloud(obj, positions):
    _swap_mesh(obj, positions, np.zeros((0, 3), dtype=np.int32))


# ---------------------------------------------------------------------------
# Main update
# ---------------------------------------------------------------------------


def update_outputs(scene, sess):
    """Write current solver outputs into Blender datablocks."""
    from . import materials

    if not sess.active:
        return
    fs = scene.fluidsim
    solver = sess.solver

    if fs.show_surface and solver.settings.surface_enabled:
        verts, tris = solver.surface_mesh()
        obj = _ensure_object(scene, SURFACE_NAME,
                             lambda: bpy.data.meshes.new(SURFACE_NAME))
        _swap_mesh(obj, verts, tris)
        materials.ensure_surface_material(scene, obj)
    else:
        obj = bpy.data.objects.get(SURFACE_NAME)
        if obj is not None and obj.type == "MESH" and len(obj.data.vertices):
            _swap_mesh(obj, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    if fs.show_particles:
        pos = solver.particle_positions()
        frac = fs.particle_display_fraction
        if frac < 1.0 and len(pos):
            step = max(1, int(round(1.0 / frac)))
            pos = pos[::step]
        radius = solver.settings.voxel_size * 0.25
        if _supports_pointcloud():
            obj = _ensure_object(scene, PARTICLES_NAME,
                                 lambda: bpy.data.pointclouds.new(PARTICLES_NAME))
            _swap_points(obj, pos, radius)
        else:
            obj = _ensure_object(scene, PARTICLES_NAME,
                                 lambda: bpy.data.meshes.new(PARTICLES_NAME))
            _swap_vertex_cloud(obj, pos)
    else:
        _clear_points(PARTICLES_NAME)

    if fs.show_whitewater and solver.settings.whitewater_enabled:
        pos = solver.whitewater_positions()
        radius = solver.settings.voxel_size * 0.15
        if _supports_pointcloud():
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                 lambda: bpy.data.pointclouds.new(WHITEWATER_NAME))
            _swap_points(obj, pos, radius)
        else:
            obj = _ensure_object(scene, WHITEWATER_NAME,
                                 lambda: bpy.data.meshes.new(WHITEWATER_NAME))
            _swap_vertex_cloud(obj, pos)
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
