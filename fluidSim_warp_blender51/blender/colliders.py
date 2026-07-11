"""Collider tagging, mesh translation, and Warp mesh caching."""

from __future__ import annotations

import bpy
import numpy as np


def tag_collider(obj, friction=0.0, restitution=0.0, transfer_velocity=True):
    obj.fluidsim.role = "COLLIDER"
    obj.fluidsim.collider_friction = friction
    obj.fluidsim.collider_restitution = restitution
    obj.fluidsim.transfer_velocity = transfer_velocity
    return obj


def collider_objects(scene):
    return [o for o in scene.objects
            if getattr(o, "fluidsim", None) is not None
            and o.fluidsim.role == "COLLIDER" and o.fluidsim.enabled]


def _object_identity(obj):
    as_pointer = getattr(obj, "as_pointer", None)
    return int(as_pointer()) if as_pointer is not None else id(obj)


def _array_signature(array):
    contiguous = np.ascontiguousarray(array)
    return contiguous.shape, str(contiguous.dtype), contiguous.tobytes()


def _cache_signature(object_id, device, verts, tris, matrix_world):
    return (
        object_id,
        device,
        _array_signature(verts),
        _array_signature(tris),
        _array_signature(np.ascontiguousarray(matrix_world, dtype=np.float32)),
    )


def gather_colliders(scene, session, depsgraph=None):
    """Build collider descriptors and reuse unchanged Warp meshes."""
    import warp as wp

    from ..sim.solver import ColliderDesc

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    descriptors = []
    seen_cache_keys = set()
    for obj in collider_objects(scene):
        if obj.type != "MESH" or obj.hide_viewport:
            continue
        object_id = _object_identity(obj)
        seen_cache_keys.add(object_id)
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None or len(mesh.polygons) == 0:
            evaluated.to_mesh_clear()
            session._collider_mesh_cache.pop(object_id, None)
            continue
        mesh.calc_loop_triangles()
        verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape(len(mesh.vertices), 3)
        tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", tris)
        matrix = np.array(obj.matrix_world, dtype=np.float32)
        world_verts = verts @ matrix[:3, :3].T + matrix[:3, 3]
        evaluated.to_mesh_clear()

        device = session.solver.device if session.solver else "cuda:0"
        signature = _cache_signature(object_id, device, verts, tris, matrix)
        entry = session._collider_mesh_cache.get(object_id)
        if entry is not None and entry["signature"] == signature:
            warp_mesh = entry["mesh"]
            aabb_min = entry["aabb_min"]
            aabb_max = entry["aabb_max"]
            session.collider_mesh_cache_stats["reuses"] += 1
        else:
            warp_mesh = wp.Mesh(
                points=wp.array(world_verts, dtype=wp.vec3, device=device),
                indices=wp.array(tris, dtype=wp.int32, device=device),
            )
            aabb_min = world_verts.min(axis=0)
            aabb_max = world_verts.max(axis=0)
            session._collider_mesh_cache[object_id] = {
                "signature": signature,
                "mesh": warp_mesh,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
            }
            session.collider_mesh_cache_stats["rebuilds"] += 1
        desc = ColliderDesc()
        desc.mesh = warp_mesh
        desc.aabb_min = tuple(float(value) for value in aabb_min)
        desc.aabb_max = tuple(float(value) for value in aabb_max)
        center = obj.matrix_world.translation
        desc.center = (center.x, center.y, center.z)
        if obj.fluidsim.transfer_velocity:
            desc.linear_velocity = session.object_velocity(obj, scene)
        descriptors.append(desc)
    for cache_key in list(session._collider_mesh_cache):
        if cache_key not in seen_cache_keys:
            del session._collider_mesh_cache[cache_key]
    return descriptors


def aggregate_material_response(scene):
    """Combine per-collider friction/restitution into solver-wide values.

    The solver applies a single restitution/friction pair during particle
    collision response; use the maximum across enabled colliders combined
    with the scene-level settings.
    """
    fs = scene.fluidsim
    friction = fs.friction
    restitution = fs.restitution
    for obj in collider_objects(scene):
        friction = max(friction, obj.fluidsim.collider_friction)
        restitution = max(restitution, obj.fluidsim.collider_restitution)
    return restitution, friction
