"""Optional per-frame cache (supplemental tooling only).

The primary workflow is interactive viewport simulation; caching exists so
users can optionally record a run to disk and reload it later. Frames are
stored as compressed .npz files.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import io
import os
import pickle
import tempfile
import zipfile

import numpy as np

CACHE_FORMAT_VERSION = 1
CACHE_SEQUENCE_METADATA = "fluidsim_sequence.json"

SCENE_KEY_VERSION = "v1"
# Settings that say where the solve runs, not what it computes: a cache baked
# on cuda:0 replays correctly on cuda:1.
_SCENE_KEY_IGNORED_SETTINGS = frozenset({"device"})
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
    from ..blender.scene_bridge import build_settings

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


def _frame_payload(solver, scene_key_value: str = ""):
    """Build the version-1 cache payload from the shared lazy host frame."""
    from ..sim.frame_output import HostFrame

    host = solver.host_frame() if hasattr(solver, "host_frame") else HostFrame(solver)
    empty_verts = np.zeros((0, 3), np.float32)
    empty_tris = np.zeros((0, 3), np.int32)
    verts, tris = (host.surface_mesh if solver.settings.surface_enabled else
                   (empty_verts, empty_tris))
    surface_metadata = (solver.surface.metadata() if hasattr(solver, "surface") else {
        "voxel_size": getattr(solver.settings, "voxel_size", 0.05)})
    return {
        "format_version": np.array(CACHE_FORMAT_VERSION, dtype=np.int32),
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
    }


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
    path = _metadata_path(base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
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


def read_frame(base_dir: str, frame: int) -> dict | None:
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
            return {k: data[k] for k in data.files}
        except (ValueError, pickle.UnpicklingError):
            return None


def available_frames(base_dir: str) -> list:
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
