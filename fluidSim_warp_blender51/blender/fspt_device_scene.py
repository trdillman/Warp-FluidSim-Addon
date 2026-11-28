"""Inactive GPU surface adapter; static geometry keeps evaluated scene semantics.

The optional static cache owns only immutable records and identity values. A
caller-supplied revision must change for every non-fluid dependency-graph edit.
Final and unsupported output semantics keep the original evaluated adapter.
"""
from __future__ import annotations

import numpy as np

from ..fspt.contracts import AssetSpan, Material, SceneSnapshot
from ..fspt.device_scene import DeviceGeometryReference, DeviceSceneRequest, DeviceSceneUnavailable
from . import fspt_bridge, surface_publication
from .fspt_scene import (
    _asset_key,
    _fail,
    _validate_evaluated_geometry,
    translate_material,
    world_color,
)
from .scene_access import active_domain_object


class StaticSceneCache:
    """One immutable static template; never retains Blender/evaluated wrappers."""

    def __init__(self):
        self.key = None
        self.parts = self.materials = self.world = None
        self.material_index = None

    def clear(self):
        self.__init__()


def _scene(scene):
    return getattr(scene, 'original', scene)


def _surface(scene):
    return fspt_bridge._surface_object(_scene(scene))


def surface_capability_reason(scene, view_layer):
    """Bounded output RNA checks; no instance scan, mesh arrays or solver work."""
    scene = _scene(scene)
    if getattr(getattr(scene, 'fluidsim', None), 'cache_playback_enabled', False):
        return 'cache playback requires evaluated fluid geometry'
    try:
        obj = _surface(scene)
        if obj is None:
            return 'fluid output object is not available'
        if not fspt_bridge._belongs(scene, obj) or not fspt_bridge._visible(obj, True, view_layer):
            return 'fluid output is outside the visible view layer'
        if obj.type != 'MESH':
            return 'fluid output is not a mesh'
        transform = np.asarray(obj.matrix_world, dtype=np.float64)
        if transform.shape != (4, 4) or not np.array_equal(transform, np.eye(4)):
            return 'fluid output transform requires evaluated geometry'
        if any(getattr(modifier, 'show_viewport', True) for modifier in obj.modifiers):
            return 'fluid output modifier requires evaluated geometry'
        mesh = obj.data
        if getattr(mesh, 'shape_keys', None) is not None:
            return 'fluid output shape keys require evaluated geometry'
        if getattr(mesh, 'has_custom_normals', False):
            return 'fluid output custom normals require evaluated geometry'
        if len(obj.material_slots) > 1 or len(mesh.materials) > 1:
            return 'fluid output material assignments require evaluated geometry'
    except ReferenceError:
        return 'fluid output was retired'
    return ''


def _instances(depsgraph):
    scene, layer = depsgraph.scene, depsgraph.view_layer
    token = surface_publication.id_token
    domain = token(active_domain_object(_scene(scene)))
    for instance in depsgraph.object_instances:
        obj = instance.object
        if domain is not None and token(obj.original) == domain:
            continue
        if not instance.is_instance and not obj.original.visible_get(view_layer=layer):
            continue
        role = getattr(getattr(obj.original, 'fluidsim', None), 'role', 'NONE')
        if role in {'DOMAIN', 'SOURCE', 'SINK', 'FORCE'}:
            continue
        if obj.type in {'CAMERA', 'EMPTY', 'ARMATURE', 'LATTICE'}:
            continue
        yield instance


def _selection_reason(scene, selection):
    stamp, key, pool = selection.publication, selection.key, selection.pool
    token = surface_publication.id_token
    obj = _surface(scene)
    if (stamp.channel != surface_publication.SurfaceChannel.GPU_FINAL
            or stamp.source != surface_publication.SurfaceSource.LIVE):
        return 'GPU selection requires its own live publication channel'
    if surface_publication.current_surface_publication(
            scene, surface_publication.SurfaceChannel.GPU_FINAL) != stamp:
        return 'GPU publication changed during capture'
    if (stamp.scene_token != token(_scene(scene))
            or stamp.owner_token != surface_publication.surface_owner_token(_scene(scene))
            or stamp.object_token != token(obj) or stamp.mesh_token != token(obj.data)):
        return 'GPU publication owner or output identity changed'
    # The channel and pool mint independent counters. The immutable runtime
    # selection pairs them; all source/build/count facts must still agree.
    for name in ('scene_token', 'owner_token', 'epoch', 'source_timeline_frame',
                 'state_revision', 'solver_frame', 'simulation_time'):
        if getattr(stamp, name) != getattr(key, name):
            return 'GPU publication and selected source facts differ'
    counts = selection.vertex_count, selection.triangle_count
    if (key.channel != 'GPU_FINAL' or key.source != 'LIVE'
            or stamp.build_key != key.build_stamp.build_key
            or (stamp.vertex_count, stamp.triangle_count) != counts or not stamp.visible):
        return 'GPU publication build or active geometry counts differ'
    budget = getattr(pool, '_geometry_array_budget', None)
    if budget is None or budget.surface_ready(key) != counts:
        return 'selected GPU geometry is no longer ready'
    return ''


def device_scene_reason(depsgraph, selection=None):
    """Audit one output's exact instance semantics before any mesh extraction."""
    scene, layer = depsgraph.scene, depsgraph.view_layer
    reason = surface_capability_reason(scene, layer)
    if reason:
        return reason
    token = surface_publication.id_token
    expected = token(_surface(scene))
    matches = 0
    for instance in _instances(depsgraph):
        if token(instance.object.original) != expected:
            continue
        if not getattr(instance, 'show_self', True):
            continue
        matches += 1
        if instance.is_instance:
            return 'instanced fluid output requires evaluated geometry'
        transform = np.asarray(instance.matrix_world, dtype=np.float64)
        if transform.shape != (4, 4) or not np.array_equal(transform, np.eye(4)):
            return 'evaluated fluid transform requires evaluated geometry'
    if matches != 1:
        return 'fluid output is absent or duplicated in the evaluated scene'
    if selection is not None:
        reason = _selection_reason(scene, selection)
        if reason:
            return reason
        try:
            fspt_bridge.validate_publications(
                scene, (selection.publication,), for_viewport=True,
                view_layer=layer, device=str(selection.pool.device))
        except fspt_bridge.SurfaceNotReadyError as exc:
            return str(exc)
    return ''


def viewport_capability(depsgraph):
    return device_scene_reason(depsgraph)


def _material(obj, slot):
    mesh = obj.data
    material = mesh.materials[slot] if slot < len(mesh.materials) else None
    if slot < len(obj.material_slots) and obj.material_slots[slot].link == 'OBJECT':
        material = obj.material_slots[slot].material
    return material


def _capture_parts(depsgraph, selection, world):
    token = surface_publication.id_token
    output = token(_surface(depsgraph.scene))
    materials, material_cache, parts = [Material()], {}, []
    fluid_index = 0

    def material_index(material):
        if material is None:
            return 0
        identity = material.as_pointer()
        if identity not in material_cache:
            material_cache[identity] = len(materials)
            materials.append(translate_material(material))
        return material_cache[identity]

    for instance in _instances(depsgraph):
        obj = instance.object
        if token(obj.original) == output:
            if not getattr(instance, 'show_self', True):
                continue
            if selection.triangle_count:
                fluid_index = material_index(_material(obj.original, 0))
            parts.append(None)
            continue
        if obj.type != 'MESH':
            _fail(obj, f'unsupported {obj.type} object; P1 accepts evaluated triangle meshes and mesh emitters')
        if obj.original.get('fluidsim_output_kind') == 'FluidSim Surface':
            _fail(obj, 'fluid output has no eligible publication from this scene owner')
        _validate_evaluated_geometry(obj)
        if not getattr(instance, 'show_self', True):
            continue
        mesh = obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            if not len(mesh.loop_triangles):
                continue
            points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
            mesh.vertices.foreach_get('co', points.ravel())
            transform = np.asarray(instance.matrix_world, dtype=np.float64)
            points = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
            faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
            mesh.loop_triangles.foreach_get('vertices', faces.ravel())
            if np.linalg.det(transform[:3, :3]) < 0:
                faces = faces[:, [0, 2, 1]]
            slots = np.empty(len(faces), dtype=np.int32)
            mesh.loop_triangles.foreach_get('material_index', slots)
            ids = np.empty(len(faces), dtype=np.int32)
            for slot in np.unique(slots):
                material = mesh.materials[int(slot)] if int(slot) < len(mesh.materials) else None
                if int(slot) < len(obj.material_slots) and obj.material_slots[int(slot)].link == 'OBJECT':
                    material = obj.material_slots[int(slot)].material
                ids[slots == slot] = material_index(material)
            parts.append((points, faces, ids, AssetSpan(_asset_key(instance, token), len(points), len(faces))))
        finally:
            obj.to_mesh_clear()
    materials = tuple(materials)
    frame = int(depsgraph.scene.frame_current)
    owned = tuple(None if part is None else SceneSnapshot(
        part[0], part[1], part[2], materials, world, frame, asset_spans=(part[3],)) for part in parts)
    return owned, materials, fluid_index


def capture_device_scene(depsgraph, selection, *, static_revision=None, cache=None):
    """Capture admitted static work and metadata, with no fluid mesh readback."""
    reason = device_scene_reason(depsgraph, selection)
    if reason:
        raise DeviceSceneUnavailable(reason)
    layer, scene = depsgraph.view_layer, depsgraph.scene
    if layer.material_override is not None or getattr(layer, 'world_override', None) is not None:
        _fail(layer, 'view-layer material/world overrides are not supported by P1')
    world = world_color(scene.world)
    token = surface_publication.id_token
    obj = _surface(scene)
    # Material values, not only material ID, bind a cache hit to fluid shading.
    fluid_material = translate_material(_material(obj, 0)) if selection.triangle_count else Material()
    cache_key = (static_revision, token(_scene(scene)), token(layer), token(obj), token(obj.data),
                 fluid_material, bool(selection.triangle_count), world)
    reuse = cache is not None and static_revision is not None and cache.key == cache_key
    if reuse:
        parts, materials, fluid_index = cache.parts, cache.materials, cache.material_index
    else:
        parts, materials, fluid_index = _capture_parts(depsgraph, selection, world)
    reason = device_scene_reason(depsgraph, selection)
    if reason:
        raise DeviceSceneUnavailable(reason)
    if cache is not None and static_revision is not None and not reuse:
        cache.key, cache.parts, cache.materials = cache_key, parts, materials
        cache.world, cache.material_index = world, fluid_index
    reference = DeviceGeometryReference(selection.pool, selection.key, selection.vertex_count,
                                        selection.triangle_count, fluid_index)
    ordered = tuple(reference if part is None else part for part in parts)
    return DeviceSceneRequest(ordered, materials, world, int(scene.frame_current),
                              (selection.publication,))
