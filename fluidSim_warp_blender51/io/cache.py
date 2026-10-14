"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.

Format 3 (FluidSim 3.0): every frame also records a stamp of every source and
collider at that frame (``source_frame_stamp``) and carries the sparse
``liquid_sdf_*`` arrays, which are empty until the narrow band is enabled. A
2.x cache is refused, not read or migrated.
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
    validate_attribute_values,
)
from ..sim.whitewater_types import validate_type_values
from .cache_schema import (
    LEAF_DIM,
    LEGACY_2X_CACHE_MESSAGE,
    LEGACY_2X_FORMAT_VERSION,
    LegacyCacheFormatError,
)

CACHE_FORMAT_VERSION = 3
# Markers of a cache written by FluidSim 1.0 (format 1). They are recognized
# only to refuse the directory with a clear message. A 2.x directory (manifest
# format 2) is recognized the same way and refused with LEGACY_2X_CACHE_MESSAGE;
# only format 3 is read.
LEGACY_SEQUENCE_METADATA = "fluidsim_sequence.json"
LEGACY_FRAME_GLOB = "fluidsim_*.npz"
LEGACY_CACHE_MESSAGE = (
    "This cache was written by FluidSim 1.0, which 2.0 cannot read. "
    "Free the cache and bake again.")

# Folded into every scene key so that a key computed by 2.x ("v1") can never
# equal one computed by 3.0, whatever the settings and sources are.
SCENE_KEY_VERSION = "v2"
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
    # Where the solve ran is not cache identity (owner decision D1,
    # 2026-10-12): a cache plays back, resumes and exports on any device.
    "device",
})
# Caches written before D1 hashed ``device`` into the settings signature.
# Matching those against these spellings keeps them readable; the
# preference stores one of them.
_LEGACY_SIGNATURE_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3",
                             "cuda:4", "cuda:5", "cuda:6", "cuda:7",
                             "cuda", "cpu", "")
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


def _digest_properties(h, properties):
    for name, value in sorted(properties.items()):
        h.update(f"{name}=".encode())
        _digest(h, value)


def scene_key_from_records(settings, scene_objects) -> str:
    """Hash plain settings and object records into a legacy cache identity.

    Blender owns record collection. This low-level helper deliberately knows
    nothing about scenes, RNA, depsgraphs, or evaluated meshes.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{SCENE_KEY_VERSION}|settings|".encode())
    for name, value in sorted(settings.items()):
        if name in _SCENE_KEY_IGNORED_SETTINGS:
            continue
        h.update(f"{name}=".encode())
        _digest(h, value)
    h.update(b"roster|")
    for record in sorted(scene_objects, key=lambda value: value["name"]):
        h.update(f"{record['name']}|".encode())
        _digest_properties(h, record["properties"])
        _digest(h, record["matrix_world"])
        _digest(h, record["vertex_count"])
    return h.hexdigest()


def source_signature_from_records(source_records) -> str:
    """Hash plain source records into a process-independent signature."""
    h = hashlib.sha256()
    h.update(b"fluidsim-source-signature-v2|")
    for record in sorted(source_records, key=lambda value: value["name_full"]):
        h.update(
            f"{record['identity']}|{record['name_full']}|{record['library']}|"
            f"{record['role']}|".encode("utf-8")
        )
        _digest_properties(h, record["properties"])
        _digest(h, record["matrix_world"])
        geometry = record["geometry"]
        if geometry["kind"] == "MESH":
            h.update(f"verts={geometry['vertex_count']}|".encode("utf-8"))
            h.update(geometry["vertex_bytes"])
            h.update(f"polys={geometry['polygon_count']}|".encode("utf-8"))
        else:
            h.update(
                f"type={geometry['kind']}|points={geometry['point_count']}|".encode(
                    "utf-8")
            )
    return h.hexdigest()


def source_frame_stamp_from_records(source_records) -> str:
    """Hash every source's identity, world matrix and geometry at one frame.

    Unlike ``source_signature_from_records`` (taken once, at bake start, and
    including settings-like ``properties``), the stamp is evaluated at a frame
    and covers only what animation can change. ``properties``, display names,
    library and role are excluded on purpose: they are static, and a rename
    must not look like moved animation. The result does not depend on record
    order.
    """
    parts = []
    for record in source_records:
        h = hashlib.sha256()
        _digest(h, record["matrix_world"])
        geometry = record["geometry"]
        if geometry["kind"] == "MESH":
            h.update(f"verts={geometry['vertex_count']}|".encode("utf-8"))
            h.update(geometry["vertex_bytes"])
            h.update(f"polys={geometry['polygon_count']}|".encode("utf-8"))
        else:
            h.update(
                f"type={geometry['kind']}|points={geometry['point_count']}|".encode(
                    "utf-8"))
        parts.append((str(record["identity"]), h.hexdigest()))
    outer = hashlib.sha256()
    outer.update(b"fluidsim-source-frame-stamp-v1|")
    for identity, digest in sorted(parts):
        outer.update(f"{identity}|{digest}|".encode("utf-8"))
    return outer.hexdigest()


def _bundle(settings_signature: str, source_signature: str) -> dict[str, str]:
    digest = hashlib.sha256()
    digest.update(
        f"scene-key={SCENE_KEY_VERSION}|settings={settings_signature}|"
        f"sources={source_signature}".encode("utf-8")
    )
    return {
        "scene_key": digest.hexdigest(),
        "settings_signature": settings_signature,
        "source_signature": source_signature,
    }


def signature_bundle_from_records(settings, source_records) -> dict[str, str]:
    """Return the cache identity bundle from adapter-produced records."""
    return _bundle(settings_record(settings)["signature"],
                   source_signature_from_records(source_records))


def legacy_signature_bundles(bundle: dict[str, str], settings) -> tuple[dict[str, str], ...]:
    """Pre-D1 forms of ``bundle``: same sources, device-hashing settings."""
    return tuple(_bundle(signature, bundle["source_signature"])
                 for signature in legacy_settings_signatures(settings))


def _savez_fast_compressed(path, payload):
    """Write a standard DEFLATED NPZ without costly size compression."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=0) as archive:
        for name, value in payload.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())


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


def _settings_values(settings, ignored) -> dict:
    if isinstance(settings, dict):
        names = sorted(settings)
        value_for = settings.__getitem__
    else:
        fields = getattr(settings, "__dataclass_fields__", None)
        names = sorted(fields) if fields is not None else sorted(vars(settings))
        value_for = lambda name: getattr(settings, name)
    return {name: _jsonable_setting(value_for(name)) for name in names
            if (not name.startswith("_")
                and name not in ignored
                and (isinstance(settings, dict) or hasattr(settings, name)))}


def _signature_of(values: dict) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def settings_record(settings) -> dict:
    """Canonical, JSON-safe settings snapshot plus its stable content hash."""
    values = _settings_values(settings, _CACHE_METADATA_IGNORED_SETTINGS)
    return {"signature": _signature_of(values), "settings": values}


def legacy_settings_signatures(settings) -> tuple[str, ...]:
    """Signatures a pre-D1 cache may carry for these settings.

    Those caches hashed the compute device in; this recomputes the hash for
    each device spelling the preference can hold. Empty for settings that
    carry no ``device`` field.
    """
    has_device = (("device" in settings) if isinstance(settings, dict)
                  else hasattr(settings, "device"))
    if not has_device:
        return ()
    ignored = _CACHE_METADATA_IGNORED_SETTINGS - {"device"}
    values = _settings_values(settings, ignored)
    result = []
    for device in _LEGACY_SIGNATURE_DEVICES:
        values["device"] = device
        result.append(_signature_of(values))
    return tuple(result)


def settings_signature_matches(settings, baked: str) -> bool:
    """Whether ``baked`` names these settings, in the current or pre-D1 form."""
    baked = str(baked)
    if settings_record(settings)["signature"] == baked:
        return True
    return baked in legacy_settings_signatures(settings)


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

    Cache frames without an attribute schema remain readable and
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


LIQUID_SDF_KEYS = (
    "liquid_sdf_sparse", "liquid_sdf_leaf_coords", "liquid_sdf_voxel_size")


def empty_liquid_sdf() -> dict:
    """The ``liquid_sdf_*`` arrays of a frame baked without a narrow band."""
    return {
        "liquid_sdf_sparse": np.zeros((0, LEAF_DIM ** 3), dtype=np.float32),
        "liquid_sdf_leaf_coords": np.zeros((0, 3), dtype=np.int32),
        "liquid_sdf_voxel_size": np.array(0.0, dtype=np.float32),
    }


def validate_liquid_sdf(payload, *, enabled: bool) -> None:
    """Check the sparse liquid SDF against the manifest's narrow-band switch.

    Enabled: at least one 8^3 leaf, a positive voxel size and finite values.
    Disabled: all three arrays empty and the voxel size 0. Anything else is a
    cache that disagrees with its own manifest.
    """
    missing = [name for name in LIQUID_SDF_KEYS if name not in payload]
    if missing:
        raise ValueError(f"cache frame is missing liquid_sdf arrays {missing}")
    sparse = np.asarray(payload["liquid_sdf_sparse"])
    coords = np.asarray(payload["liquid_sdf_leaf_coords"])
    voxel = np.asarray(payload["liquid_sdf_voxel_size"])
    # Optional additive metadata: old v3 frames remain readable. The corner
    # origin (not a cell-centre offset) is bound by the frame checksum.
    if "liquid_sdf_origin" in payload:
        origin = np.asarray(payload["liquid_sdf_origin"])
        if (origin.shape != (3,) or origin.dtype != np.float32
                or not np.all(np.isfinite(origin))):
            raise ValueError("liquid_sdf_origin must be a finite float32 vector of shape (3,)")
    leaf_volume = LEAF_DIM ** 3
    if (sparse.ndim != 2 or sparse.shape[1] != leaf_volume
            or sparse.dtype != np.float32):
        raise ValueError(
            f"liquid_sdf_sparse must be float32 with shape (N, {leaf_volume}); "
            f"got {sparse.dtype} {tuple(sparse.shape)}")
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != np.int32:
        raise ValueError(
            "liquid_sdf_leaf_coords must be int32 with shape (N, 3); "
            f"got {coords.dtype} {tuple(coords.shape)}")
    if sparse.shape[0] != coords.shape[0]:
        raise ValueError(
            f"liquid_sdf leaf count mismatch: {sparse.shape[0]} sparse leaves "
            f"but {coords.shape[0]} leaf coordinates")
    if voxel.ndim != 0 or voxel.dtype.kind != "f":
        raise ValueError("liquid_sdf_voxel_size must be a float scalar")
    leaves = int(sparse.shape[0])
    size = float(voxel)
    if enabled:
        if leaves < 1:
            raise ValueError(
                "liquid_sdf arrays are empty but the narrow band is enabled")
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("liquid_sdf_voxel_size must be > 0 and finite")
        if not np.all(np.isfinite(sparse)):
            raise ValueError("liquid_sdf_sparse contains non-finite values")
    elif leaves or size != 0.0:
        raise ValueError(
            "liquid_sdf arrays must be empty (voxel size 0) when the narrow "
            "band is disabled")


def _frame_payload(frame, scene_key_value: str = "", *, liquid_sdf=None):
    """Validate and encode one materialized cache-frame payload.

    ``liquid_sdf`` is the sparse narrow-band SDF (the three ``liquid_sdf_*``
    arrays); None means the frame was baked without a narrow band.
    """
    sdf = empty_liquid_sdf() if liquid_sdf is None else {
        name: liquid_sdf[name] for name in LIQUID_SDF_KEYS}
    if liquid_sdf is not None and "liquid_sdf_origin" in liquid_sdf:
        sdf["liquid_sdf_origin"] = liquid_sdf["liquid_sdf_origin"]
    payload = {
        "format_version": np.array(CACHE_FORMAT_VERSION, dtype=np.int32),
        "settings_signature": np.array(frame.settings_signature),
        "scene_key": np.array(str(scene_key_value)),
        "positions": frame.positions,
        "velocities": frame.velocities,
        "whitewater": frame.whitewater,
        "surface_verts": frame.surface_verts,
        "surface_tris": frame.surface_tris,
        "frame": frame.frame,
        "voxel_size": np.array(frame.voxel_size, dtype=np.float32),
        "surface_voxel_size": np.array(frame.surface_voxel_size, dtype=np.float32),
        "surface_resolution_multiplier": np.array(
            frame.surface_resolution_multiplier, dtype=np.float32),
        "surface_resolution_mode": np.array(
            frame.surface_resolution_mode, dtype=np.int32),
        "surface_update_cadence": np.array(
            frame.surface_update_cadence, dtype=np.int32),
        "surface_smooth_iterations": np.array(
            frame.surface_smooth_iterations, dtype=np.int32),
        "particle_age": frame.particle_attributes["age"],
        "particle_life": frame.particle_attributes["life"],
        "particle_temperature": frame.particle_attributes["temperature"],
        "particle_viscosity": frame.particle_attributes["viscosity"],
        "particle_density": frame.particle_attributes["density"],
        "particle_time_offset": frame.particle_attributes["time_offset"],
        "whitewater_velocities": frame.whitewater_attributes["velocity"],
        "whitewater_age": frame.whitewater_attributes["age"],
        "whitewater_life": frame.whitewater_attributes["life"],
        "whitewater_type": frame.whitewater_attributes["type"],
        "whitewater_required": np.array(frame.whitewater_required, dtype=np.int32),
        "whitewater_available": np.array(frame.whitewater_available, dtype=np.int32),
        "whitewater_admitted": np.array(frame.whitewater_admitted, dtype=np.int32),
        "whitewater_dropped": np.array(frame.whitewater_dropped, dtype=np.int32),
        "whitewater_policy": np.array(frame.whitewater_policy),
        "whitewater_status": np.array(frame.whitewater_status),
        "attribute_schema_version": np.array(
            ATTRIBUTE_SCHEMA_VERSION, dtype=np.int32),
        "attribute_inventory_json": np.array(json.dumps(
            dict(frame.attribute_inventory),
            sort_keys=True, separators=(",", ":"))),
        "surface_attribute_transfer_method": np.array(
            SURFACE_TRANSFER_METHOD),
        **sdf,
    }
    for output_kind, attributes in frame.output_attributes.items():
        for key, values in attributes.items():
            payload[cache_attribute_name(output_kind, key)] = values
    validate_optional_attributes(
        payload, particle_count=len(frame.positions),
        whitewater_count=len(frame.whitewater),
        surface_count=len(frame.surface_verts))
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


def read_sequence_metadata(base_dir: str) -> dict | None:
    # The strict manifest holds the sequence metadata. Present a small view for
    # playback/UI callers while preserving manifest-authoritative frame
    # discovery. A 2.x manifest raises LegacyCacheFormatError (load_manifest
    # refuses it); any other directory has no sequence metadata.
    from .cache_schema import load_manifest, manifest_path

    manifest = load_manifest(manifest_path(base_dir))
    if manifest is not None:
        return {
            "format_version": CACHE_FORMAT_VERSION,
            "frame_start": manifest.frame_start,
            "frame_end": manifest.frame_end,
            "frame_count": len(manifest.frames),
            "source": "production_bake",
            "scene_key": manifest.scene_key,
            "settings_signature": manifest.settings_signature,
            "source_signature": manifest.source_signature,
            "last_committed_frame": manifest.last_committed_frame,
            "transfer_mode": manifest.transfer_mode,
            "narrow_band": dict(manifest.narrow_band),
            "grid_spec": dict(manifest.grid_spec),
        }
    return None


def _frame_path(base_dir: str | os.PathLike[str], frame: int) -> Path:
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
            descriptor = None  # the handle owns and closes it now
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
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_frame_v3_atomic(
        base_dir: str | os.PathLike[str], frame_payload, frame: int, *,
        scene_key_value: str, settings_signature: str,
        source_signature: str, source_frame_stamp: str,
        effect_signature: str = "",
        effect_inventory=(), simulation_time: float = 0.0,
        liquid_sdf=None, narrow_band_enabled: bool = False,
        compressed: bool = False) -> tuple[Path, str, int]:
    """Write a strict cache-v3 frame before its manifest transaction commits.

    ``source_frame_stamp`` is required: it is stored in the frame and in the
    manifest entry, and a reader rejects a frame whose two copies differ.
    """
    from .cache_schema import _require_sha256

    frame = int(frame)
    if frame_payload.settings_signature != str(settings_signature):
        raise ValueError(
            "cache-frame settings signature does not match sequence metadata"
        )
    _require_sha256(source_frame_stamp, "source_frame_stamp")
    payload = _frame_payload(frame_payload, scene_key_value, liquid_sdf=liquid_sdf)
    validate_liquid_sdf(payload, enabled=bool(narrow_band_enabled))
    payload.update({
        "source_frame_stamp": np.array(str(source_frame_stamp)),
        "frame": np.array(frame, dtype=np.int32),
        "solver_frame": np.array(frame_payload.frame, dtype=np.int32),
        "simulation_time": np.array(float(simulation_time), dtype=np.float64),
        "settings_signature": np.array(str(settings_signature)),
        "source_signature": np.array(str(source_signature)),
        "effect_signature": np.array(str(effect_signature)),
        "effect_inventory_json": np.array(json.dumps(
            [dict(item) for item in effect_inventory],
            sort_keys=True, separators=(",", ":"), allow_nan=False)),
    })
    authoritative_payload_counts(payload)
    if frame_payload.frame != frame:
        raise ValueError(
            f"solver frame {frame_payload.frame} does not match cache frame {frame}")
    path = _frame_path(base_dir, frame)
    digest, size = _write_npz_atomic(path, payload, compressed=compressed)
    return path, digest, size


def _read_v3_frame(base_dir: str, frame: int) -> dict | None:
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
        raise ValueError(f"cache-v3 frame {frame} is corrupt: {exc}") from exc
    with loaded as data:
        required = {
            "format_version", "frame", "solver_frame", "simulation_time",
            "settings_signature", "source_signature", "scene_key",
            "source_frame_stamp", *LIQUID_SDF_KEYS,
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"cache-v3 frame {frame} is missing {sorted(missing)}")
        if int(np.asarray(data["format_version"]).item()) != CACHE_FORMAT_VERSION:
            raise ValueError(f"cache-v3 frame {frame} has the wrong format version")
        if int(np.asarray(data["frame"]).item()) != int(frame):
            raise ValueError(f"cache-v3 frame {frame} has inconsistent frame metadata")
        if int(np.asarray(data["solver_frame"]).item()) != entry.solver_frame:
            raise ValueError(f"cache-v3 frame {frame} solver-frame mismatch")
        if str(np.asarray(data["settings_signature"]).item()) != manifest.settings_signature:
            raise ValueError(f"cache-v3 frame {frame} settings-signature mismatch")
        if str(np.asarray(data["source_signature"]).item()) != manifest.source_signature:
            raise ValueError(f"cache-v3 frame {frame} source-signature mismatch")
        if str(np.asarray(data["source_frame_stamp"]).item()) != entry.source_frame_stamp:
            raise ValueError(f"cache-v3 frame {frame} source_frame_stamp mismatch")
        frame_effect_signature = (
            str(np.asarray(data["effect_signature"]).item())
            if "effect_signature" in data.files else ""
        )
        if frame_effect_signature != manifest.effect_signature:
            raise ValueError(f"cache-v3 frame {frame} effect-signature mismatch")
        try:
            frame_effect_inventory = (
                json.loads(str(np.asarray(data["effect_inventory_json"]).item()))
                if "effect_inventory_json" in data.files else []
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cache-v3 frame {frame} effect inventory is invalid: {exc}") from exc
        if frame_effect_inventory != [dict(item) for item in manifest.effect_inventory]:
            raise ValueError(f"cache-v3 frame {frame} effect-inventory mismatch")
        result = {name: data[name] for name in data.files}
    counts = authoritative_payload_counts(result)
    validate_liquid_sdf(result, enabled=bool(manifest.narrow_band["enabled"]))
    validate_optional_attributes(result)
    if counts["particles"]["positions"] != entry.particle_count:
        raise ValueError(
            f"cache-v3 frame {frame} particle_count mismatch: manifest "
            f"{entry.particle_count}, payload {counts['particles']['positions']}")
    if counts["whitewater"]["whitewater"] != entry.whitewater_count:
        raise ValueError(
            f"cache-v3 frame {frame} whitewater_count mismatch: manifest "
            f"{entry.whitewater_count}, payload {counts['whitewater']['whitewater']}")
    # Cache v3.1: the Dynamic collider poses live in the manifest entry (a
    # tuple of RigidBodyRecord). Frames without one (every 3.0 frame) keep
    # their 3.0 shape: readers use ``.get("rigid_bodies", ())``.
    if entry.rigid_bodies:
        result["rigid_bodies"] = entry.rigid_bodies
    return result


def read_frame(base_dir: str, frame: int) -> dict | None:
    """Read one committed cache-v3 frame; None when no cache holds it.

    A FluidSim 1.0 or 2.x cache raises rather than reading as empty, so
    playback tells the user to re-bake instead of reporting missing frames.
    """
    version = detect_cache_format(base_dir)
    if version == 1:
        raise ValueError(f"{LEGACY_CACHE_MESSAGE} ({base_dir})")
    if version == LEGACY_2X_FORMAT_VERSION:
        raise LegacyCacheFormatError()
    if version == CACHE_FORMAT_VERSION:
        return _read_v3_frame(base_dir, frame)
    return None


def available_frames(base_dir: str) -> list:
    """Committed cache-v3 frames; empty for any other directory (2.x too)."""
    if detect_cache_format(base_dir) != CACHE_FORMAT_VERSION:
        return []
    from .cache_schema import load_manifest, manifest_path

    manifest = load_manifest(manifest_path(base_dir))
    return list(manifest.committed_frames) if manifest is not None else []


def detect_cache_format(base_dir: str | os.PathLike[str]) -> int:
    """Return 3 for a current cache, 2 for a 2.x cache, 1 for a 1.0 cache, else 0.

    Formats 1 and 2 are detected only so callers can refuse them with
    ``LEGACY_CACHE_MESSAGE`` / ``LEGACY_2X_CACHE_MESSAGE``; nothing reads them.
    A manifest that cannot be parsed counts as current, so ``load_manifest``
    reports the precise problem rather than this function guessing.
    """
    from .cache_schema import _manifest_record

    base = Path(base_dir)
    manifest_file = base / "fluidsim_cache.json"
    if manifest_file.is_file():
        try:
            # Shared with load_manifest: unchanged bytes are parsed once.
            value = _manifest_record(manifest_file)[1]
        except (OSError, ValueError):
            return CACHE_FORMAT_VERSION
        if (isinstance(value, dict)
                and value.get("format_version") == LEGACY_2X_FORMAT_VERSION):
            return LEGACY_2X_FORMAT_VERSION
        return CACHE_FORMAT_VERSION
    if (base / LEGACY_SEQUENCE_METADATA).is_file() or any(
            base.glob(LEGACY_FRAME_GLOB)):
        return 1
    return 0
