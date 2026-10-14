"""Blender adapter for persistent cache scene and source identities."""

from __future__ import annotations

import numpy as np

_HASHABLE_RNA_TYPES = frozenset({"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"})
_SOURCE_SIGNATURE_IGNORED_RNA = frozenset({
    "source_status_severity",
    "source_status_code",
    "source_status_message",
    "source_status_emitted",
    "source_status_capacity_limited",
})


def _settings_record(scene) -> dict:
    from .solver_settings import build_settings

    settings = build_settings(scene)
    fields = getattr(type(settings), "__dataclass_fields__", None) or vars(settings)
    return {name: getattr(settings, name) for name in fields}


def _rna_record(group, *, ignored=frozenset()) -> dict:
    record = {}
    for name, prop in sorted(group.bl_rna.properties.items()):
        if (
            name == "rna_type"
            or name in ignored
            or prop.type not in _HASHABLE_RNA_TYPES
        ):
            continue
        value = getattr(group, name)
        record[name] = tuple(value) if getattr(prop, "is_array", False) else value
    return record


def _matrix_record(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(component) for component in row) for row in matrix)


def _authored(obj):
    """World matrix as authored: published Dynamic collider poses are solver
    output, so they never change the scene identity."""
    from .rigid_publication import authored_world

    return authored_world(obj)


def _scene_object_records(scene) -> tuple[dict, ...]:
    records = []
    for obj in scene.objects:
        group = getattr(obj, "fluidsim", None)
        if group is None or group.role == "NONE":
            continue
        records.append({
            "name": obj.name,
            "properties": _rna_record(group),
            "matrix_world": tuple(tuple(row) for row in _authored(obj)),
            "vertex_count": len(
                getattr(getattr(obj, "data", None), "vertices", ())),
        })
    return tuple(records)


def scene_key(scene) -> str:
    """Return the legacy identity for settings and the tagged object roster."""
    from ..io.cache import scene_key_from_records

    return scene_key_from_records(
        _settings_record(scene), _scene_object_records(scene))


def _geometry_record(obj, depsgraph) -> dict:
    evaluated = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    mesh = None
    try:
        if getattr(evaluated, "type", None) == "MESH":
            mesh = (
                evaluated.to_mesh()
                if hasattr(evaluated, "to_mesh")
                else evaluated.data
            )
            vertices = getattr(mesh, "vertices", ())
            coords = np.empty(len(vertices) * 3, dtype=np.float32)
            if vertices:
                vertices.foreach_get("co", coords)
            return {
                "kind": "MESH",
                "vertex_count": len(vertices),
                "vertex_bytes": coords.tobytes(),
                "polygon_count": len(getattr(mesh, "polygons", ())),
            }
        data = getattr(evaluated, "data", None)
        return {
            "kind": getattr(evaluated, "type", ""),
            "point_count": len(getattr(data, "points", ())),
        }
    finally:
        if (
            mesh is not None
            and hasattr(evaluated, "to_mesh_clear")
        ):
            try:
                evaluated.to_mesh_clear()
            except RuntimeError:
                pass


def _source_records(scene, depsgraph=None) -> tuple[dict, ...]:
    from .scene_inputs import persistent_source_key

    records = []
    for obj in scene.objects:
        group = getattr(obj, "fluidsim", None)
        role = str(getattr(group, "role", "NONE")) if group is not None else "NONE"
        if role == "NONE":
            continue
        records.append({
            "identity": persistent_source_key(obj),
            "name_full": obj.name_full,
            "library": str(
                getattr(getattr(obj, "library", None), "filepath", "") or ""),
            "role": role,
            "properties": _rna_record(
                group, ignored=_SOURCE_SIGNATURE_IGNORED_RNA),
            "matrix_world": _matrix_record(_authored(obj)),
            "geometry": _geometry_record(obj, depsgraph),
        })
    return tuple(records)


def source_signature(scene, depsgraph=None) -> str:
    """Return a persistent source/input signature for a Blender scene."""
    from ..io.cache import source_signature_from_records

    return source_signature_from_records(_source_records(scene, depsgraph))


def source_frame_stamp(scene, depsgraph, frame) -> str:
    """Return the stamp of every source and collider as evaluated at ``frame``.

    The source signature is taken once, at bake start, so a collider that only
    moves on frame 40 leaves it unchanged. The stamp is taken per frame and
    covers what animation can change (world matrix and evaluated geometry).

    The caller sets the frame and passes the depsgraph evaluated at it: this
    function reads the scene as it is and raises rather than hash another
    frame's state under ``frame``'s name.
    """
    from ..io.cache import source_frame_stamp_from_records

    frame = int(frame)
    current = int(scene.frame_current)
    if current != frame:
        raise ValueError(
            f"source_frame_stamp({frame}) needs the scene at frame {frame}, "
            f"but it is at frame {current}; call scene.frame_set first")
    return source_frame_stamp_from_records(_source_records(scene, depsgraph))


def settings_signature(scene) -> str:
    """Return the cache settings signature alone.

    Unlike the source signature it does not depend on the current frame:
    source records hash transforms and evaluated geometry, which an animated
    collider or emitter changes every frame.
    """
    from ..io.cache import settings_record

    return settings_record(_settings_record(scene))["signature"]


def legacy_settings_signatures(scene) -> tuple[str, ...]:
    """Signatures a cache baked before owner decision D1 may carry for the
    scene's current settings (that form hashed the compute device in)."""
    from ..io.cache import legacy_settings_signatures as _legacy

    return _legacy(_settings_record(scene))


def legacy_signature_bundles(scene, bundle: dict[str, str]) -> tuple[dict[str, str], ...]:
    """Pre-D1 identity bundles a cache baked from this scene may carry."""
    from ..io.cache import legacy_signature_bundles as _legacy

    return _legacy(bundle, _settings_record(scene))


def signature_bundle(scene, depsgraph=None) -> dict[str, str]:
    """Return the cache settings/source identity bundle for a scene."""
    from ..io.cache import signature_bundle_from_records

    return signature_bundle_from_records(
        _settings_record(scene), _source_records(scene, depsgraph))
