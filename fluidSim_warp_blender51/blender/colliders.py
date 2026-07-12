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
    updated_ids = _depsgraph_updates(depsgraph)
    descriptors = []
    seen_cache_keys = set()
    for obj in collider_objects(scene):
        if obj.type != "MESH" or obj.hide_viewport:
            continue
        object_id = _object_identity(obj)
        seen_cache_keys.add(object_id)
        device = session.solver.device if session.solver else "cuda:0"
        evaluated = obj.evaluated_get(depsgraph)
        evaluated_matrix = evaluated.matrix_world
        matrix = np.array(evaluated_matrix, dtype=np.float32)
        matrix_signature = _array_signature(matrix)
        data_id = _data_identity(obj)
        entry = session._collider_mesh_cache.get(object_id)
        if (entry is not None and entry.get("device") == device
                and entry.get("data_id") == data_id
                and entry.get("matrix_signature") == matrix_signature
                and object_id not in updated_ids and data_id not in updated_ids):
            warp_mesh = entry["mesh"]
            aabb_min = entry["aabb_min"]
            aabb_max = entry["aabb_max"]
            session.collider_mesh_cache_stats["reuses"] += 1
        else:
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
            world_verts = verts @ matrix[:3, :3].T + matrix[:3, 3]
            evaluated.to_mesh_clear()

            signature = _cache_signature(object_id, device, verts, tris, matrix)
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
                session.collider_mesh_cache_stats["rebuilds"] += 1
            session._collider_mesh_cache[object_id] = {
                "signature": signature,
                "device": device,
                "data_id": data_id,
                "matrix_signature": matrix_signature,
                "mesh": warp_mesh,
                "aabb_min": aabb_min,
                "aabb_max": aabb_max,
            }
        desc = ColliderDesc()
        desc.mesh = warp_mesh
        desc.aabb_min = tuple(float(value) for value in aabb_min)
        desc.aabb_max = tuple(float(value) for value in aabb_max)
        center = evaluated_matrix.translation
        desc.center = (center.x, center.y, center.z)
        desc.friction = float(obj.fluidsim.collider_friction)
        desc.restitution = float(obj.fluidsim.collider_restitution)
        if obj.fluidsim.transfer_velocity:
            desc.linear_velocity = session.object_velocity(evaluated, scene)
            desc.angular_velocity = session.object_angular_velocity(evaluated, scene)
        descriptors.append(desc)
    for cache_key in list(session._collider_mesh_cache):
        if cache_key not in seen_cache_keys:
            del session._collider_mesh_cache[cache_key]
    return descriptors
