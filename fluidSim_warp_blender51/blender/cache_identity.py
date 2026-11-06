"""Blender adapter for persistent cache scene and source identities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..io.cache import INPUT_IDENTITY_SCHEMA_VERSION as INPUT_IDENTITY_SCHEMA_VERSION

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


def _geometry_record(obj, depsgraph, *, point_input=False) -> dict:
    from .evaluated_geometry import EvaluatedGeometry

    geometry = EvaluatedGeometry(obj, depsgraph)
    evaluated = geometry.evaluated
    particle_systems = getattr(evaluated, "particle_systems", ())
    if point_input and particle_systems and len(particle_systems[0].particles):
        particles = particle_systems[0].particles
        coordinates = np.empty(len(particles) * 3, dtype=np.float32)
        particles.foreach_get("location", coordinates)
        return {"kind": "PARTICLES", "point_count": len(particles),
                "point_bytes": coordinates.tobytes()}
    if getattr(evaluated, "type", None) == "MESH":
        with geometry.mesh(data_fallback=True, tolerate_release_error=True) as mesh:
            vertices = getattr(mesh, "vertices", ())
            coords = np.empty(len(vertices) * 3, dtype=np.float32)
            if vertices:
                vertices.foreach_get("co", coords)
            return {
                "kind": "MESH",
                "vertex_count": len(vertices),
                "vertex_bytes": coords.tobytes(),
                "polygon_count": len(getattr(mesh, "polygons", ())),
                "topology": _mesh_topology(mesh),
            }
    data = getattr(evaluated, "data", None)
    points = getattr(data, "points", ())
    coordinates = np.empty(len(points) * 3, dtype=np.float32)
    if len(points):
        attributes = getattr(data, "attributes", None)
        position = attributes.get("position") if attributes is not None else None
        if position is not None:
            position.data.foreach_get("vector", coordinates)
        else:
            points.foreach_get("co", coordinates)
    return {
        "kind": getattr(evaluated, "type", ""),
        "point_count": len(points),
        "point_bytes": coordinates.tobytes(),
    }


def _mesh_topology(mesh):
    """Copy the evaluated triangles before the temporary mesh is released."""
    if hasattr(mesh, "calc_loop_triangles"):
        mesh.calc_loop_triangles()
        triangles = mesh.loop_triangles
        indices = np.empty(len(triangles) * 3, dtype=np.int32)
        triangles.foreach_get("vertices", indices)
        return indices.tobytes()
    # Data-only adapters retain polygon boundaries as well as connectivity.
    return tuple(tuple(int(index) for index in polygon.vertices)
                 for polygon in getattr(mesh, "polygons", ()))


def _volume_record(obj):
    data = obj.data
    grids = getattr(data, "grids", None)
    path = str(getattr(grids, "frame_filepath", "") or data.filepath)
    try:
        import bpy
        path = bpy.path.abspath(path, library=getattr(obj, "library", None))
    except ImportError:
        path = str(Path(path).resolve())
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path,
        "content_sha256": digest.hexdigest(),
        "sequence": {name: getattr(data, name, None) for name in
                     ("is_sequence", "frame_start", "frame_duration", "frame_offset", "sequence_mode")},
    }


def _dependencies(obj, group, depsgraph):
    from .scene_inputs import persistent_source_key

    refs = []
    role = getattr(group, "role", "NONE")
    source_kind = getattr(group, "source_region_kind", "ANALYTIC")
    sink_kind = getattr(group, "sink_region_kind", "ANALYTIC")
    field_kind = getattr(group, "field_region_kind", "ANALYTIC")
    if ((role == "SOURCE" and source_kind == "VOLUME_GRID")
            or (role == "SINK" and sink_kind == "VOLUME_GRID")
            or (role == "FIELD" and field_kind == "VOLUME_GRID")):
        refs.append(("volume", getattr(group, "volume_input_object", None) or obj))
    elif role == "SOURCE" and source_kind == "POINTS":
        refs.append(("points", getattr(group, "point_input_object", None) or obj))
    elif (source_kind in {"MESH_SOLID", "MESH_SHELL"}
          or sink_kind in {"MESH_SOLID", "MESH_SHELL"}
          or getattr(group, "geometry_mode", "ANALYTIC") == "MESH"):
        refs.append(("mesh", getattr(group, "mesh_object", None) or obj))
    if getattr(group, "enabled", False) and getattr(group, "auto_inject_seed", False):
        refs.append(("points", obj))
    records = []
    for kind, ref in refs:
        record = {
            "kind": kind, "identity": persistent_source_key(ref),
            "matrix_world": _matrix_record(_authored(ref)),
        }
        if kind == "volume":
            record["payload"] = _volume_record(ref)
        else:
            record["payload"] = _geometry_record(ref, depsgraph, point_input=kind == "points")
        records.append(record)
    return tuple(records)


def _source_records(scene, depsgraph=None) -> tuple[dict, ...]:
    from .scene_inputs import persistent_source_key

    records = []
    for obj in scene.objects:
        group = getattr(obj, "fluidsim", None)
        role = str(getattr(group, "role", "NONE")) if group is not None else "NONE"
        if role == "NONE" and not (
                getattr(group, "enabled", False)
                and getattr(group, "auto_inject_seed", False)):
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
            "dependencies": _dependencies(obj, group, depsgraph),
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
