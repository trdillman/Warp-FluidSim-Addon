"""Built-in/user preset discovery, atomic persistence, and scene application."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .schema import PresetV1, SETTING_SPECS
from ..version import VERSION_STRING

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

    return Path(
        bpy.utils.extension_path_user(
            _root_package(), path="presets", create=create
        )
    )


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


def built_in_presets() -> tuple[PresetRecord, ...]:
    records = []
    seen: set[str] = set()
    for path in _files(_RESOURCE_DIR):
        preset = _read(path)
        if preset.id in seen:
            raise ValueError(f"duplicate built-in preset id {preset.id!r}")
        seen.add(preset.id)
        records.append(PresetRecord(preset.id, preset, "BUILTIN", str(path)))
    return tuple(records)


def user_presets(directory: Path | None = None) -> tuple[PresetRecord, ...]:
    directory = Path(directory) if directory is not None else user_preset_dir()
    records = []
    seen: set[str] = set()
    builtin_ids = {record.preset.id for record in built_in_presets()}
    for path in _files(directory):
        preset = _read(path)
        if preset.id in builtin_ids:
            raise ValueError(f"user preset {preset.id!r} collides with a built-in preset")
        if preset.id in seen:
            raise ValueError(f"duplicate user preset id {preset.id!r}")
        seen.add(preset.id)
        records.append(PresetRecord(preset.id, preset, "USER", str(path)))
    return tuple(records)


def discover_presets(directory: Path | None = None) -> tuple[PresetRecord, ...]:
    builtins = built_in_presets()
    try:
        users = user_presets(directory)
    except (ImportError, RuntimeError, ValueError):
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
    _write_atomic(path, payload)
    return path


def capture_scene_preset(
    scene,
    *,
    identifier: str,
    name: str,
    description: str,
    expected_scale: str,
    performance_cost: str,
) -> PresetV1:
    settings = scene.fluidsim
    values = {key: getattr(settings, key) for key in SETTING_SPECS}
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


def apply_preset(scene, preset: PresetV1) -> ApplyResult:
    preset = PresetV1.from_dict(preset.to_dict())
    target = getattr(scene, "fluidsim", None)
    if target is None:
        raise ValueError("scene has no FluidSim settings")
    missing = tuple(key for key in preset.settings if not hasattr(target, key))
    if missing:
        raise ValueError(f"scene does not support preset settings: {', '.join(missing)}")
    previous = {key: getattr(target, key) for key in preset.settings}
    changed = tuple(
        key for key, value in preset.settings.items() if previous[key] != value
    )
    try:
        for key, value in preset.settings.items():
            setattr(target, key, value)
        if hasattr(target, "active_preset_id"):
            target.active_preset_id = preset.id
    except Exception:
        for key, value in previous.items():
            try:
                setattr(target, key, value)
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
    return tuple(migrated)
