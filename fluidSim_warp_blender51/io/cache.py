"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import io
import math
import os
import pickle
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from ..sim.attributes import (
    ATTRIBUTE_SCHEMA_VERSION,
    ATTRIBUTE_SPECS,
    OUTPUT_ATTRIBUTE_KEYS,
    SUPPORTED_ATTRIBUTE_SCHEMA_VERSIONS,
    SURFACE_TRANSFER_METHOD,
    cache_attribute_name,
    enabled_attribute_keys,
    validate_attribute_values,
)
from ..sim.whitewater_types import validate_type_values

CACHE_FORMAT_VERSION = 1
CACHE_SEQUENCE_METADATA = "fluidsim_sequence.json"

SCENE_KEY_VERSION = "v1"
# Settings that say where the solve runs, not what it computes: a cache baked
# on cuda:0 replays correctly on cuda:1.
_SCENE_KEY_IGNORED_SETTINGS = frozenset({
    "device",
    # Preview-only publication cadence must not invalidate or alter an
    # authoritative Production Cache Bake.
    "live_surface_publish_cadence",
})
_CACHE_METADATA_IGNORED_SETTINGS = frozenset({
    "live_surface_publish_cadence",
})
# RNA property types with a stable value repr. POINTER/COLLECTION reprs embed
# addresses and would change the key every session.
_HASHABLE_RNA_TYPES = frozenset({"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"})


def _digest(h, value):
    """Feed one setting value into the running hash."""
    if isinstance(value, np.ndarray):
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Enum):
        h.update(repr(value.value).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            _digest(h, item)
    else:
        h.update(repr(value).encode())
    h.update(b"|")


def _digest_rna(h, group):
    for name, prop in sorted(group.bl_rna.properties.items()):
        if name == "rna_type" or prop.type not in _HASHABLE_RNA_TYPES:
            continue
        value = getattr(group, name)
        h.update(f"{name}=".encode())
        _digest(h, tuple(value) if getattr(prop, "is_array", False) else value)


def scene_key(scene) -> str:
    """Content hash of every scene input that changes what the solver computes.

    This is the identity of a baked cache. Playback refuses a sequence whose key
    does not match the current scene rather than replaying a stale result as if
    it were a correct answer.

    ``build_settings`` is hashed rather than a hand-listed field set so the key
    stays correct by construction as settings are added -- a hand-maintained
    list going stale is the exact failure being fixed here.

    ponytail: a tagged object contributes its transform and vertex count, not
    its vertex positions. Hashing evaluated geometry needs a depsgraph pass and
    would duplicate what SceneInputs already does every frame. Editing vertices
    in place, without changing the count or the transform, is the known blind
    spot; hash ``obj.evaluated_get(depsgraph).to_mesh()`` here if it bites.
    """
    from ..blender.solver_settings import build_settings

    h = hashlib.blake2b(digest_size=16)
    h.update(f"{SCENE_KEY_VERSION}|settings|".encode())
    settings = build_settings(scene)
    fields = getattr(type(settings), "__dataclass_fields__", None) or vars(settings)
    for name in sorted(fields):
        if name in _SCENE_KEY_IGNORED_SETTINGS:
            continue
        h.update(f"{name}=".encode())
        _digest(h, getattr(settings, name))
    h.update(b"roster|")
    for obj in sorted(scene.objects, key=lambda value: value.name):
        group = getattr(obj, "fluidsim", None)
        if group is None or group.role == "NONE":
            continue
        h.update(f"{obj.name}|".encode())
        _digest_rna(h, group)
        _digest(h, tuple(tuple(row) for row in obj.matrix_world))
        _digest(h, len(getattr(getattr(obj, "data", None), "vertices", ())))
    return h.hexdigest()


def source_signature(scene, depsgraph=None) -> str:
    """Return a persistent, process-independent source/input signature.

    The signature intentionally excludes Blender memory addresses. Object
    names, library identity, role RNA, transforms, and evaluated geometry are
    sufficient to keep one-shot source identity stable across reopened files
    and to reject resume after materially changing sources/colliders/forces.
    """
    h = hashlib.sha256()
    h.update(b"fluidsim-source-signature-v2|")
    for obj in sorted(scene.objects, key=lambda value: value.name_full):
        group = getattr(obj, "fluidsim", None)
        role = str(getattr(group, "role", "NONE")) if group is not None else "NONE"
        if role == "NONE":
            continue
        identity = str(obj.get("fluidsim_persistent_id", "") or obj.name_full)
        library = str(getattr(getattr(obj, "library", None), "filepath", "") or "")
        h.update(f"{identity}|{obj.name_full}|{library}|{role}|".encode("utf-8"))
        _digest_rna(h, group)
        _digest(h, tuple(tuple(float(component) for component in row)
                         for row in obj.matrix_world))
        evaluated = None
        mesh = None
        try:
            evaluated = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
            if getattr(evaluated, "type", None) == "MESH":
                mesh = evaluated.to_mesh() if hasattr(evaluated, "to_mesh") else evaluated.data
                vertices = getattr(mesh, "vertices", ())
                h.update(f"verts={len(vertices)}|".encode("utf-8"))
                if vertices:
                    coords = np.empty(len(vertices) * 3, dtype=np.float32)
                    vertices.foreach_get("co", coords)
                    h.update(coords.tobytes())
                polygons = getattr(mesh, "polygons", ())
                h.update(f"polys={len(polygons)}|".encode("utf-8"))
            else:
                data = getattr(evaluated, "data", None)
                h.update(
                    f"type={getattr(evaluated, 'type', '')}|"
                    f"points={len(getattr(data, 'points', ()))}|".encode("utf-8")
                )
        finally:
            if mesh is not None and evaluated is not None and hasattr(evaluated, "to_mesh_clear"):
                try:
                    evaluated.to_mesh_clear()
                except RuntimeError:
                    pass
    return h.hexdigest()


def signature_bundle(scene, depsgraph=None) -> dict[str, str]:
    """Return the cache-v2 scene/settings/source identity bundle."""
    from ..blender.solver_settings import build_settings

    settings = settings_record(build_settings(scene))["signature"]
    sources = source_signature(scene, depsgraph)
    digest = hashlib.sha256()
    digest.update(f"settings={settings}|sources={sources}".encode("utf-8"))
    return {
        "scene_key": digest.hexdigest(),
        "settings_signature": settings,
        "source_signature": sources,
    }


def _savez_fast_compressed(path, payload):
    """Write a standard DEFLATED NPZ without costly size compression."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=0) as archive:
        for name, value in payload.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())


def _frame_path(base_dir: str, frame: int) -> str:
    return os.path.join(base_dir, f"fluidsim_{frame:05d}.npz")


def _metadata_path(base_dir: str) -> str:
    return os.path.join(base_dir, CACHE_SEQUENCE_METADATA)


def _jsonable_setting(value):
    if isinstance(value, float) and not math.isfinite(value):
        # Preserve a deterministic diagnostic signature without emitting
        # non-standard JSON or aborting an otherwise recoverable cache export.
        label = ("NaN" if math.isnan(value) else
                 "Infinity" if value > 0.0 else "-Infinity")
        return {"__fluidsim_nonfinite_float__": label}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable_setting(asdict(value))
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, dict):
        return {str(key): _jsonable_setting(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    try:
        return [_jsonable_setting(item) for item in value]
    except TypeError:
        return repr(value)


def settings_record(settings) -> dict:
    """Canonical, JSON-safe settings snapshot plus its stable content hash."""
    fields = getattr(settings, "__dataclass_fields__", None)
    names = sorted(fields) if fields is not None else sorted(vars(settings))
    values = {name: _jsonable_setting(getattr(settings, name)) for name in names
              if (not name.startswith("_")
                  and name not in _CACHE_METADATA_IGNORED_SETTINGS
                  and hasattr(settings, name))}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return {"signature": hashlib.sha256(encoded).hexdigest(), "settings": values}


def compact_settings_segments(settings_records: dict) -> list[dict]:
    """Compact frame->record data into contiguous signature boundaries."""
    # Metadata loaded from JSON has string keys, while records assembled during
    # a bake use integer keys. Normalize once before indexing; converting only
    # in the sorted expression leaves string-keyed mappings inaccessible.
    normalized = {}
    for raw_frame, record in settings_records.items():
        frame = int(raw_frame)
        if frame in normalized:
            raise ValueError(
                f"Duplicate settings record for frame {frame} after key normalization"
            )
        normalized[frame] = record

    segments = []
    for frame in sorted(normalized):
        signature = normalized[frame]["signature"]
        if (segments and segments[-1]["signature"] == signature
                and segments[-1]["frame_end"] + 1 == frame):
            segments[-1]["frame_end"] = frame
        else:
            segments.append({"frame_start": frame, "frame_end": frame,
                             "signature": signature})
    return segments


def _attribute_inventory(settings) -> dict[str, list[str]]:
    return {
        output_kind: list(enabled_attribute_keys(settings, output_kind))
        for output_kind in OUTPUT_ATTRIBUTE_KEYS
    }


_ATTRIBUTE_KEYS_BY_SCHEMA: dict[int, dict[str, tuple[str, ...]]] = {
    # Phase 3 did not persist a whitewater type channel.  Keep its ordered
    # inventory readable while requiring type for newly written Phase-4 data.
    1: {
        "surface": OUTPUT_ATTRIBUTE_KEYS["surface"],
        "particles": OUTPUT_ATTRIBUTE_KEYS["particles"],
        "whitewater": ("velocity", "speed", "age", "lifetime"),
    },
    ATTRIBUTE_SCHEMA_VERSION: OUTPUT_ATTRIBUTE_KEYS,
}


def _decode_attribute_inventory(frame_data: dict) -> dict[str, tuple[str, ...]] | None:
    if "attribute_schema_version" not in frame_data:
        return None
    version = int(np.asarray(frame_data["attribute_schema_version"]).item())
    if version not in SUPPORTED_ATTRIBUTE_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in SUPPORTED_ATTRIBUTE_SCHEMA_VERSIONS)
        raise ValueError(
            f"unsupported attribute schema {version}; supported versions are {supported}")
    schema_keys = _ATTRIBUTE_KEYS_BY_SCHEMA.get(version)
    if schema_keys is None:
        raise ValueError(f"attribute schema {version} has no inventory contract")
    raw = frame_data.get("attribute_inventory_json")
    if raw is None:
        raise ValueError("attribute payload is missing attribute_inventory_json")
    try:
        value = json.loads(str(np.asarray(raw).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid attribute inventory: {exc}") from exc
    if not isinstance(value, dict) or set(value) != set(schema_keys):
        raise ValueError("attribute inventory must contain surface, particles, and whitewater")
    result = {}
    for output_kind, keys in value.items():
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError(f"{output_kind} attribute inventory must be a string list")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{output_kind} attribute inventory contains duplicates")
        unsupported = set(keys) - set(schema_keys[output_kind])
        if unsupported:
            raise ValueError(
                f"{output_kind} attribute inventory contains unsupported keys {sorted(unsupported)}")
        ordered = tuple(key for key in schema_keys[output_kind] if key in keys)
        if tuple(keys) != ordered:
            raise ValueError(f"{output_kind} attribute inventory is not in schema order")
        result[output_kind] = ordered
    if version >= 2 and "type" not in result["whitewater"]:
        raise ValueError("Phase-4 whitewater inventory must include type")
    transfer = str(np.asarray(frame_data.get(
        "surface_attribute_transfer_method", "")).item())
    if transfer != SURFACE_TRANSFER_METHOD:
        raise ValueError(
            f"unsupported surface attribute transfer method {transfer!r}")
    return result


def validate_optional_attributes(
        payload: dict, *, particle_count: int | None = None,
        whitewater_count: int | None = None,
        surface_count: int | None = None) -> dict[str, tuple[str, ...]]:
    """Validate Phase-3 attribute inventory, dtypes, shapes and alignment.

    Legacy cache-v1/v2 frames without an attribute schema remain readable and
    return an empty inventory. New frames fail closed on any declared channel
    that is missing, malformed, non-finite, or misaligned.
    """
    inventory = _decode_attribute_inventory(payload)
    if inventory is None:
        return {}
    counts = {
        "particles": (len(np.asarray(payload.get("positions", ())))
                      if particle_count is None else int(particle_count)),
        "whitewater": (len(np.asarray(payload.get("whitewater", ())))
                       if whitewater_count is None else int(whitewater_count)),
        "surface": (len(np.asarray(payload.get("surface_verts", ())))
                    if surface_count is None else int(surface_count)),
    }
    for output_kind, keys in inventory.items():
        count = counts[output_kind]
        if count < 0:
            raise ValueError(f"{output_kind} attribute count must be non-negative")
        for key in keys:
            field = cache_attribute_name(output_kind, key)
            if field not in payload:
                raise ValueError(
                    f"{output_kind} attribute {key!r} is missing cache field {field!r}")
            validate_attribute_values(ATTRIBUTE_SPECS[key], payload[field], count)
    if "uid" in inventory["particles"]:
        uid = np.asarray(payload[cache_attribute_name("particles", "uid")], dtype=np.int32)
        if np.any(uid <= 0) or len(np.unique(uid)) != len(uid):
            raise ValueError("particle_uid must contain unique positive values")
    if "type" in inventory["whitewater"]:
        validate_type_values(
            payload[cache_attribute_name("whitewater", "type")],
            allow_unclassified=False,
        )
    return inventory


def _frame_payload(solver, scene_key_value: str = ""):
    """Build one cache payload from the shared lazy host frame."""
    from ..sim.frame_output import HostFrame

    host = solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)
    empty_verts = np.zeros((0, 3), np.float32)
    empty_tris = np.zeros((0, 3), np.int32)
    verts, tris = (host.surface_mesh if solver.settings.surface_enabled else
                   (empty_verts, empty_tris))
    surface_metadata = (solver.surface.metadata() if hasattr(solver, "surface") else {
        "voxel_size": getattr(solver.settings, "voxel_size", 0.05)})
    record = settings_record(solver.settings)
    payload = {
        "format_version": np.array(CACHE_FORMAT_VERSION, dtype=np.int32),
        "settings_signature": np.array(record["signature"]),
        "scene_key": np.array(str(scene_key_value)),
        "positions": host.positions,
        "velocities": host.velocities,
        "whitewater": host.whitewater,
        "surface_verts": verts,
        "surface_tris": tris,
        "frame": solver.frame,
        "voxel_size": np.array(getattr(solver.settings, "voxel_size", 0.05), dtype=np.float32),
        "surface_voxel_size": np.array(surface_metadata["voxel_size"], dtype=np.float32),
        "surface_resolution_multiplier": np.array(getattr(solver.settings, "surface_resolution_multiplier", 1.0), dtype=np.float32),
        "surface_resolution_mode": np.array(int(getattr(solver.settings, "surface_resolution_mode", 0)), dtype=np.int32),
        "surface_update_cadence": np.array(int(getattr(solver.settings, "surface_update_cadence", 1)), dtype=np.int32),
        "surface_smooth_iterations": np.array(int(getattr(solver.settings, "surface_smooth_iterations", 0)), dtype=np.int32),
        "particle_age": host.particle_attr("age"),
        "particle_life": host.particle_attr("life"),
        "particle_temperature": host.particle_attr("temperature"),
        "particle_viscosity": host.particle_attr("viscosity"),
        "particle_density": host.particle_attr("density"),
        "particle_time_offset": host.particle_attr("time_offset"),
        "whitewater_velocities": host.whitewater_attr("velocity"),
        "whitewater_age": host.whitewater_attr("age"),
        "whitewater_life": host.whitewater_attr("life"),
        "whitewater_type": host.whitewater_attr("type"),
        "whitewater_required": np.array(
            int(getattr(getattr(solver, "stats", None), "whitewater_required", 0)),
            dtype=np.int32),
        "whitewater_available": np.array(
            int(getattr(getattr(solver, "stats", None), "whitewater_available", 0)),
            dtype=np.int32),
        "whitewater_admitted": np.array(
            int(getattr(getattr(solver, "stats", None), "whitewater_admitted", 0)),
            dtype=np.int32),
        "whitewater_dropped": np.array(
            int(getattr(getattr(solver, "stats", None), "whitewater_dropped", 0)),
            dtype=np.int32),
        "whitewater_policy": np.array(str(getattr(
            getattr(solver, "stats", None), "whitewater_policy",
            "FIXED_CAPACITY_DETERMINISTIC_DROPS"))),
        "whitewater_status": np.array(str(getattr(
            getattr(solver, "stats", None), "whitewater_status", "OK"))),
        "attribute_schema_version": np.array(
            ATTRIBUTE_SCHEMA_VERSION, dtype=np.int32),
        "attribute_inventory_json": np.array(json.dumps(
            _attribute_inventory(solver.settings),
            sort_keys=True, separators=(",", ":"))),
        "surface_attribute_transfer_method": np.array(
            SURFACE_TRANSFER_METHOD),
    }
    for output_kind, positions in (
            ("surface", verts), ("particles", None), ("whitewater", None)):
        attributes = host.output_attributes(
            solver.settings, output_kind, positions=positions)
        for key, values in attributes.items():
            payload[cache_attribute_name(output_kind, key)] = values
    validate_optional_attributes(
        payload, particle_count=len(host.positions),
        whitewater_count=len(host.whitewater), surface_count=len(verts))
    return payload


AUTHORITATIVE_PAYLOAD_FIELDS = {
    "particles": (
        "positions", "velocities", "particle_age", "particle_life",
        "particle_temperature", "particle_viscosity", "particle_density",
        "particle_time_offset",
    ),
    "whitewater": (
        "whitewater", "whitewater_velocities", "whitewater_age",
        "whitewater_life",
    ),
}


def authoritative_payload_counts(frame_data: dict) -> dict:
    """Return diagnostic payload lengths after schema/alignment validation.

    This helper deliberately makes no comparison with another bake.  It exists
    for cache-write seam diagnostics: every aligned field must be finite, and
    callers compare these lengths with the solver counts sampled immediately
    before that specific write.
    """
    counts = {}
    schema_version = None
    if "attribute_schema_version" in frame_data:
        schema_version = int(np.asarray(frame_data["attribute_schema_version"]).item())
    for group, base_names in AUTHORITATIVE_PAYLOAD_FIELDS.items():
        names = base_names
        if group == "whitewater" and schema_version is not None and schema_version >= 2:
            names = (*base_names, "whitewater_type")
        missing = [name for name in names if name not in frame_data]
        if missing:
            raise ValueError(f"{group} payload is missing fields {missing}")
        arrays = {name: np.asanyarray(frame_data[name]) for name in names}
        primary = arrays[names[0]]
        if primary.ndim == 0:
            raise ValueError(f"{names[0]} must have a particle dimension")
        count = len(primary)
        bad_lengths = {
            name: (None if value.ndim == 0 else len(value))
            for name, value in arrays.items()
            if value.ndim == 0 or len(value) != count
        }
        if bad_lengths:
            raise ValueError(
                f"{group} attribute lengths do not match {count}: {bad_lengths}")
        nonfinite = [
            name for name, value in arrays.items()
            if not np.all(np.isfinite(value))
        ]
        if nonfinite:
            raise ValueError(f"{group} payload has non-finite fields {nonfinite}")
        counts[group] = {name: int(len(value)) for name, value in arrays.items()}
    return counts


def resolve_frame_range(
        scene,
        *,
        use_custom: bool,
        frame_start: int,
        frame_end: int) -> tuple[int, int]:
    """Resolve a cache operation range from scene defaults or custom override."""
    if use_custom:
        start = int(frame_start)
        end = int(frame_end)
    else:
        start = int(scene.frame_start)
        end = int(scene.frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    return start, end


def write_sequence_metadata(
        base_dir: str,
        frame_start: int,
        frame_end: int,
        *,
        source: str = "simulation",
        settings_records: dict | None = None,
        scene_key_value: str = "") -> str:
    """Write sequence-level cache metadata next to per-frame .npz files."""
    os.makedirs(base_dir, exist_ok=True)
    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "source": str(source),
        "scene_key": str(scene_key_value),
    }
    if settings_records:
        normalized = {int(frame): record for frame, record in settings_records.items()
                      if start <= int(frame) <= end}
        segments = compact_settings_segments(normalized)
        records_by_signature = {
            record["signature"]: record["settings"]
            for record in normalized.values()
        }
        signatures = list(records_by_signature)
        baseline_signature = segments[0]["signature"] if segments else None
        baseline = records_by_signature.get(baseline_signature, {})
        changes = {}
        for signature, values in records_by_signature.items():
            if signature == baseline_signature:
                continue
            delta = {
                name: value for name, value in values.items()
                if baseline.get(name) != value
            }
            removed = sorted(set(baseline) - set(values))
            changes[signature] = {"changed": delta, "removed": removed}
        metadata.update({
            "settings_segments": segments,
            "settings_baseline": {
                "signature": baseline_signature,
                "settings": baseline,
            },
            # Store only deltas after the baseline. Long keyframed sequences no
            # longer repeat the entire settings object for every signature.
            "settings_changes": changes,
            "mixed_settings": len(signatures) > 1,
            "settings_signature": signatures[0] if len(signatures) == 1 else None,
        })
    path = _metadata_path(base_dir)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{CACHE_SEQUENCE_METADATA}.",
        suffix=".tmp",
        dir=base_dir,
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, sort_keys=True)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def read_sequence_metadata(base_dir: str) -> dict | None:
    # Cache-v2 keeps sequence metadata in its strict manifest. Present a small
    # compatibility view for existing playback/UI callers while preserving
    # manifest-authoritative frame discovery.
    from .cache_schema import load_manifest, manifest_path

    v2 = load_manifest(manifest_path(base_dir))
    if v2 is not None:
        return {
            "format_version": 2,
            "frame_start": v2.frame_start,
            "frame_end": v2.frame_end,
            "frame_count": len(v2.frames),
            "source": "production_bake",
            "scene_key": v2.scene_key,
            "settings_signature": v2.settings_signature,
            "source_signature": v2.source_signature,
            "last_committed_frame": v2.last_committed_frame,
        }
    path = _metadata_path(base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        # Valid JSON can still be corrupt sequence metadata. Treat it like a
        # truncated/unparseable supplemental cache rather than leaking an
        # AttributeError from the playback handler.
        return None
    if data.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cache metadata format_version "
            f"{data.get('format_version')} in {path}; "
            f"expected {CACHE_FORMAT_VERSION}."
        )
    return data


def write_frame(scene, sess, frame=None) -> str:
    """Atomically write one cache frame at an explicit timeline index.

    ``frame=None`` preserves the low-level solver-frame API for standalone
    callers. Live/timeline callers must pass ``scene.frame_current`` so cache
    names and playback use the same indexing basis.
    """
    import bpy

    fs = scene.fluidsim
    base = bpy.path.abspath(fs.cache_path)
    os.makedirs(base, exist_ok=True)
    solver = sess.solver
    frame = int(solver.frame if frame is None else frame)
    path = _frame_path(base, frame)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{os.path.basename(path)}.",
        suffix=".tmp.npz",
        dir=base,
    )
    os.close(fd)
    try:
        _savez_fast_compressed(
            tmp_path,
            _frame_payload(solver, getattr(sess, "scene_key", "")))
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


def _v2_frame_path(base_dir: str | os.PathLike[str], frame: int) -> Path:
    return Path(base_dir) / "frames" / f"frame_{int(frame):06d}.npz"


def _write_npz_atomic(path: Path, payload: dict, *, compressed=False) -> tuple[str, int]:
    """Write one NPZ durably and return its SHA-256 and byte size."""
    from .cache_schema import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            if compressed:
                np.savez_compressed(handle, **payload)
            else:
                np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return sha256_file(path), path.stat().st_size
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_frame_v2_atomic(
        base_dir: str | os.PathLike[str], solver, frame: int, *,
        scene_key_value: str, settings_signature: str,
        source_signature: str, effect_signature: str = "",
        effect_inventory=(), compressed: bool = False) -> tuple[Path, str, int]:
    """Write a strict cache-v2 frame before its manifest transaction commits."""
    frame = int(frame)
    payload = _frame_payload(solver, scene_key_value)
    payload.update({
        "format_version": np.array(2, dtype=np.int32),
        "frame": np.array(frame, dtype=np.int32),
        "solver_frame": np.array(int(solver.frame), dtype=np.int32),
        "simulation_time": np.array(float(solver.sim_time), dtype=np.float64),
        "settings_signature": np.array(str(settings_signature)),
        "source_signature": np.array(str(source_signature)),
        "effect_signature": np.array(str(effect_signature)),
        "effect_inventory_json": np.array(json.dumps(
            [dict(item) for item in effect_inventory],
            sort_keys=True, separators=(",", ":"), allow_nan=False)),
    })
    authoritative_payload_counts(payload)
    if int(solver.frame) != frame:
        raise ValueError(
            f"solver frame {int(solver.frame)} does not match cache frame {frame}")
    path = _v2_frame_path(base_dir, frame)
    digest, size = _write_npz_atomic(path, payload, compressed=compressed)
    return path, digest, size


def _read_v2_frame(base_dir: str, frame: int) -> dict | None:
    from .cache_schema import load_manifest, manifest_path, verify_artifact

    manifest = load_manifest(manifest_path(base_dir))
    if manifest is None:
        return None
    entry = manifest.frames.get(int(frame))
    if entry is None:
        # Files without entries are deliberately ignored as orphans.
        return None
    path = Path(base_dir) / entry.frame_file
    verify_artifact(
        path,
        expected_sha256=entry.frame_sha256,
        expected_size=entry.frame_size,
        label=f"frame {frame}",
    )
    try:
        loaded = np.load(path, allow_pickle=False)
    except (EOFError, OSError, ValueError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cache-v2 frame {frame} is corrupt: {exc}") from exc
    with loaded as data:
        required = {
            "format_version", "frame", "solver_frame", "simulation_time",
            "settings_signature", "source_signature", "scene_key",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"cache-v2 frame {frame} is missing {sorted(missing)}")
        if int(np.asarray(data["format_version"]).item()) != 2:
            raise ValueError(f"cache-v2 frame {frame} has the wrong format version")
        if int(np.asarray(data["frame"]).item()) != int(frame):
            raise ValueError(f"cache-v2 frame {frame} has inconsistent frame metadata")
        if int(np.asarray(data["solver_frame"]).item()) != entry.solver_frame:
            raise ValueError(f"cache-v2 frame {frame} solver-frame mismatch")
        if str(np.asarray(data["settings_signature"]).item()) != manifest.settings_signature:
            raise ValueError(f"cache-v2 frame {frame} settings-signature mismatch")
        if str(np.asarray(data["source_signature"]).item()) != manifest.source_signature:
            raise ValueError(f"cache-v2 frame {frame} source-signature mismatch")
        frame_effect_signature = (
            str(np.asarray(data["effect_signature"]).item())
            if "effect_signature" in data.files else ""
        )
        if frame_effect_signature != manifest.effect_signature:
            raise ValueError(f"cache-v2 frame {frame} effect-signature mismatch")
        try:
            frame_effect_inventory = (
                json.loads(str(np.asarray(data["effect_inventory_json"]).item()))
                if "effect_inventory_json" in data.files else []
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cache-v2 frame {frame} effect inventory is invalid: {exc}") from exc
        if frame_effect_inventory != [dict(item) for item in manifest.effect_inventory]:
            raise ValueError(f"cache-v2 frame {frame} effect-inventory mismatch")
        result = {name: data[name] for name in data.files}
    authoritative_payload_counts(result)
    validate_optional_attributes(result)
    return result


def read_frame(base_dir: str, frame: int) -> dict | None:
    if detect_cache_format(base_dir) == 2:
        return _read_v2_frame(base_dir, frame)
    path = _frame_path(base_dir, frame)
    if not os.path.exists(path):
        return None
    try:
        loaded = np.load(path, allow_pickle=False)
    except (EOFError, OSError, ValueError, pickle.UnpicklingError, zipfile.BadZipFile):
        return None

    with loaded as data:
        if "format_version" not in data.files:
            raise ValueError(
                f"Cache file {path} is missing format_version; regenerate the cache."
            )
        version = int(np.asarray(data["format_version"]).item())
        if version != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported cache format_version {version} in {path}; "
                f"expected {CACHE_FORMAT_VERSION}. Regenerate the cache."
            )
        try:
            result = {k: data[k] for k in data.files}
            validate_optional_attributes(result)
            return result
        except (ValueError, pickle.UnpicklingError):
            return None


def available_frames(base_dir: str) -> list:
    if detect_cache_format(base_dir) == 2:
        from .cache_schema import load_manifest, manifest_path

        manifest = load_manifest(manifest_path(base_dir))
        return list(manifest.committed_frames) if manifest is not None else []
    if not os.path.isdir(base_dir):
        return []
    frames = []
    for name in os.listdir(base_dir):
        if name.startswith("fluidsim_") and name.endswith(".npz"):
            try:
                frames.append(int(name[9:-4]))
            except ValueError:
                pass
    return sorted(frames)


def validate_contiguous_frames(
        base_dir: str,
        frame_start: int,
        frame_end: int) -> list[int]:
    """Return cache frames missing from the inclusive requested range."""
    start = int(frame_start)
    end = int(frame_end)
    if end < start:
        raise ValueError(f"Cache end frame {end} is before start frame {start}")
    present = set(available_frames(base_dir))
    return [frame for frame in range(start, end + 1) if frame not in present]


def detect_cache_format(base_dir: str | os.PathLike[str]) -> int:
    """Detect cache v2 first, then legacy cache v1."""
    base = Path(base_dir)
    if (base / "fluidsim_cache.json").is_file():
        return 2
    if (base / CACHE_SEQUENCE_METADATA).is_file() or any(
            base.glob("fluidsim_*.npz")):
        return 1
    return 0
