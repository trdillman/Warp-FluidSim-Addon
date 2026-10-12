"""Built-in/user preset discovery, atomic persistence, and scene application."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..version import VERSION_STRING
from .schema import SCENE_OWNED_SETTINGS, SETTING_SPECS, PresetV1

_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources" / "presets"
_MIGRATION_MARKER = ".fluidsim_presets_migrated_v1"


@dataclass(frozen=True)
class PresetRecord:
    identifier: str
    preset: PresetV1
    source: str
    path: str


@dataclass(frozen=True)
class ApplyResult:
    preset_id: str
    applied: tuple[str, ...]
    changed: tuple[str, ...]
    requires_reset: bool
    message: str


def _root_package() -> str:
    suffix = ".presets"
    package = __package__
    if package.endswith(suffix):
        return package[: -len(suffix)]
    return package


def user_preset_dir(*, create: bool = False) -> Path:
    import bpy

    package = _root_package()
    if package.startswith("bl_ext."):
        return Path(
            bpy.utils.extension_path_user(
                package, path="presets", create=create
            )
        )
    directory = Path(bpy.utils.user_resource("CONFIG")) / package / "presets"
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read(path: Path) -> PresetV1:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read preset {path}: {exc}") from exc
    return PresetV1.from_dict(payload)


def _files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.glob("*.json") if path.is_file()))


# Parsed folders keyed by their files' stat identity. The preset dropdown's
# items callback discovers presets on every redraw; re-reading and validating
# every JSON file there cost milliseconds per draw. A folder is re-read only
# when a file is added, removed, replaced or rewritten, or after an explicit
# ``invalidate_preset_cache`` (Save Preset, legacy migration).
_FOLDER_CACHE: dict[tuple[str, str], tuple[tuple, object]] = {}


def invalidate_preset_cache() -> None:
    """Forget every parsed preset folder; the next discovery re-reads them."""
    _FOLDER_CACHE.clear()


def _folder_key(directory: Path) -> tuple:
    # ``scandir`` returns each entry's size and times with the listing on
    # Windows, so keying a folder costs one directory read, not a stat per file.
    key = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    stat = entry.stat()
                except OSError:
                    continue
                key.append((entry.name, stat.st_size, stat.st_mtime_ns))
    except OSError:  # missing or unreadable folder: nothing to offer
        return ()
    return tuple(sorted(key))


def _cached_folder(kind: str, directory: Path, scan):
    cache_key = (kind, str(directory))
    folder_key = _folder_key(directory)
    entry = _FOLDER_CACHE.get(cache_key)
    if entry is not None and entry[0] == folder_key:
        return entry[1]
    result = scan(directory)
    _FOLDER_CACHE[cache_key] = (folder_key, result)
    return result


def built_in_presets() -> tuple[PresetRecord, ...]:
    return _cached_folder("BUILTIN", _RESOURCE_DIR, _scan_built_in_presets)


def _scan_built_in_presets(directory: Path) -> tuple[PresetRecord, ...]:
    records = []
    seen: set[str] = set()
    for path in _files(directory):
        preset = _read(path)
        if preset.id in seen:
            raise ValueError(f"duplicate built-in preset id {preset.id!r}")
        seen.add(preset.id)
        records.append(PresetRecord(preset.id, preset, "BUILTIN", str(path)))
    return tuple(records)


# Why each user preset file the last discovery skipped could not be offered.
_DIAGNOSTICS: tuple[str, ...] = ()


def preset_diagnostics() -> tuple[str, ...]:
    """Messages for the user preset files the last discovery skipped."""
    return _DIAGNOSTICS


def _user_preset_scan(directory: Path | None):
    """Return (records, {path: reason}) for a user preset folder.

    One unreadable, invalid, duplicate or built-in-colliding file is skipped
    with its reason; it never hides the folder's healthy presets.
    """
    directory = Path(directory) if directory is not None else user_preset_dir()
    # The scan also depends on the built-in ids (collisions), so a built-in
    # change re-reads the user folder through a new cache key.
    builtin_ids = frozenset(record.preset.id for record in built_in_presets())
    return _cached_folder(
        "USER:" + ",".join(sorted(builtin_ids)), directory,
        lambda folder: _scan_user_folder(folder, builtin_ids))


def _scan_user_folder(directory: Path, builtin_ids):
    records = []
    skipped: dict[Path, str] = {}
    seen: set[str] = set()
    for path in _files(directory):
        try:
            preset = _read(path)
        except ValueError as exc:
            skipped[path] = str(exc)
            continue
        if preset.id in builtin_ids:
            skipped[path] = f"user preset {preset.id!r} collides with a built-in preset"
        elif preset.id in seen:
            skipped[path] = f"duplicate user preset id {preset.id!r}"
        else:
            seen.add(preset.id)
            records.append(PresetRecord(preset.id, preset, "USER", str(path)))
    return tuple(records), skipped


def _record_diagnostics(skipped) -> None:
    global _DIAGNOSTICS
    _DIAGNOSTICS = tuple(
        f"Skipped user preset {path.name}: {reason}"
        for path, reason in skipped.items())


def user_presets(directory: Path | None = None) -> tuple[PresetRecord, ...]:
    """The folder's valid user presets; skipped files go to ``preset_diagnostics``."""
    records, skipped = _user_preset_scan(directory)
    _record_diagnostics(skipped)
    return records


def discover_presets(directory: Path | None = None) -> tuple[PresetRecord, ...]:
    builtins = built_in_presets()
    try:
        users = user_presets(directory)
    except (ImportError, RuntimeError):
        _record_diagnostics({})
        users = ()
    return builtins + users


def load_preset(identifier: str, directory: Path | None = None) -> PresetV1:
    identifier = str(identifier).strip()
    source_filter = ""
    if ":" in identifier:
        prefix, identifier = identifier.split(":", 1)
        source_filter = prefix.upper()
    matches = [
        record
        for record in discover_presets(directory)
        if record.preset.id == identifier
        and (not source_filter or record.source == source_filter)
    ]
    if not matches and source_filter != "BUILTIN":
        # A file named after the preset that discovery had to skip explains
        # the miss better than "not found".
        reasons = [item for item in _DIAGNOSTICS
                   if item.startswith(f"Skipped user preset {identifier}.json:")]
        if reasons:
            raise ValueError(reasons[0])
    if len(matches) != 1:
        raise KeyError(f"expected one preset {identifier!r}, found {len(matches)}")
    return matches[0].preset


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def save_user_preset(
    preset: PresetV1,
    directory: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    preset = PresetV1.from_dict(preset.to_dict())
    if preset.id in {record.preset.id for record in built_in_presets()}:
        raise ValueError(f"cannot overwrite built-in preset {preset.id!r}")
    directory = Path(directory) if directory is not None else user_preset_dir(create=True)
    path = directory / f"{preset.id}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"user preset already exists: {preset.id}")
    payload = preset.to_dict()
    payload["source_version"] = payload.get("source_version") or VERSION_STRING
    try:
        _write_atomic(path, payload)
    finally:
        invalidate_preset_cache()
    return path


def capture_scene_preset(
    scene,
    *,
    identifier: str,
    name: str,
    description: str,
    expected_scale: str,
    performance_cost: str,
    source=None,
    field_source=None,
) -> PresetV1:
    # ``source`` is the settings group whose current values to snapshot. Callers
    # in the blender layer pass the active domain settings so "Save Current
    # Settings as Preset" captures what the solver actually uses; the default
    # keeps the legacy ``scene.fluidsim`` behaviour for pre-migration scenes and
    # tests.
    settings = source if source is not None else scene.fluidsim
    # A preset field is not always authored on the same settings group: e.g.
    # ``whitewater_output_mode`` and ``cache_compress`` stay scene-owned even on
    # a migrated scene. ``field_source`` lets callers resolve each key to its
    # actual owner instead of snapshotting every key from one group.
    resolve = field_source or (lambda _key: settings)
    values = {key: getattr(resolve(key), key) for key in SETTING_SPECS}
    return PresetV1.from_dict(
        {
            "schema_version": 1,
            "id": identifier,
            "name": name,
            "description": description,
            "expected_scale": expected_scale,
            "performance_cost": performance_cost,
            "requires_reset": True,
            "settings": values,
            "source_version": VERSION_STRING,
        }
    )


def apply_preset(scene, preset: PresetV1, *, target=None, field_target=None) -> ApplyResult:
    preset = PresetV1.from_dict(preset.to_dict())
    # ``target`` is where the preset's solver values are written. Blender-layer
    # callers pass the active domain settings (post-migration the solver builds
    # from there); the default preserves the legacy ``scene.fluidsim`` target for
    # pre-migration scenes and tests.
    if target is None:
        target = getattr(scene, "fluidsim", None)
    if target is None:
        raise ValueError("scene has no FluidSim settings")
    # Not every preset field is declared on ``target``: scene-owned fields such
    # as ``whitewater_output_mode`` and ``cache_compress`` are declared on
    # ``scene.fluidsim``, and Output Publication and Production Cache Bake read
    # them from there. ``field_target`` lets callers resolve each key to the
    # group that declares it instead of writing every key to one group; when
    # omitted, SCENE_OWNED_SETTINGS keys route to ``scene.fluidsim`` and
    # everything else routes to ``target``.
    scene_settings = getattr(scene, "fluidsim", None)

    def _default_owner(key):
        if key in SCENE_OWNED_SETTINGS and scene_settings is not None:
            return scene_settings
        return target

    resolve = field_target or _default_owner
    missing = tuple(key for key in preset.settings if not hasattr(resolve(key), key))
    if missing:
        raise ValueError(f"scene does not support preset settings: {', '.join(missing)}")
    previous = {key: getattr(resolve(key), key) for key in preset.settings}
    changed = tuple(
        key for key, value in preset.settings.items() if previous[key] != value
    )
    try:
        for key, value in preset.settings.items():
            setattr(resolve(key), key, value)
        # ``active_preset_id`` is a scene field and drives the panel's "Applied:"
        # feedback, which reads the scene-level group. It is never written to the
        # value ``target`` (the domain group, which does not declare it).
        if scene_settings is not None:
            scene_settings.active_preset_id = preset.id
    except Exception:
        for key, value in previous.items():
            try:
                setattr(resolve(key), key, value)
            except Exception:
                pass
        raise
    message = (
        f"Applied {preset.name}: {len(changed)} setting(s) changed; "
        f"cost {preset.performance_cost}; scale {preset.expected_scale}"
    )
    return ApplyResult(
        preset_id=preset.id,
        applied=tuple(preset.settings),
        changed=changed,
        requires_reset=bool(preset.requires_reset and changed),
        message=message,
    )


def migrate_legacy_presets(
    legacy_directories: Iterable[Path], destination: Path | None = None
) -> tuple[Path, ...]:
    destination = (
        Path(destination) if destination is not None else user_preset_dir(create=True)
    )
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / _MIGRATION_MARKER
    if marker.exists():
        return ()
    migrated: list[Path] = []
    builtin_ids = {record.preset.id for record in built_in_presets()}
    for directory in legacy_directories:
        for source in _files(Path(directory)):
            try:
                preset = _read(source)
            except ValueError:
                continue
            if preset.id in builtin_ids:
                continue
            target = destination / f"{preset.id}.json"
            if target.exists():
                continue
            # Re-serialize through the current schema instead of copying
            # untrusted legacy bytes.
            _write_atomic(target, preset.to_dict())
            migrated.append(target)
    _write_atomic(marker, {"schema_version": 1, "migrated": [str(p) for p in migrated]})
    if migrated:
        invalidate_preset_cache()
    return tuple(migrated)
