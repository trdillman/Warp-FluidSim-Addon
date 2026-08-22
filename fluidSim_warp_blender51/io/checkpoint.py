"""Versioned solver checkpoint payloads with atomic, pickle-free NPZ I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

CHECKPOINT_FORMAT_VERSION = 1
_META_KEY = "__fluidsim_checkpoint_metadata__"


def _json_safe(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[]") for item in value]
    if isinstance(value, set):
        return [_json_safe(item, f"{path}[]") for item in sorted(value)]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string key")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path)
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _normalized_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"checkpoint array {name!r} cannot use object dtype")
    if array.dtype.kind not in "biufc":
        raise ValueError(
            f"checkpoint array {name!r} has unsupported dtype {array.dtype}"
        )
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"checkpoint array {name!r} contains non-finite values")
    return np.ascontiguousarray(array)


def _validate_channel_alignment(arrays: Mapping[str, np.ndarray]) -> None:
    particle_names = [
        name
        for name in arrays
        if name.startswith("particle_") and name not in {"particle_count"}
    ]
    whitewater_names = [name for name in arrays if name.startswith("whitewater_")]
    for label, names in (("particle", particle_names), ("whitewater", whitewater_names)):
        lengths: dict[str, int] = {}
        for name in names:
            value = arrays[name]
            if value.ndim == 0:
                raise ValueError(f"{name} must have a {label} dimension")
            lengths[name] = int(value.shape[0])
        if lengths and len(set(lengths.values())) != 1:
            raise ValueError(f"unaligned {label} checkpoint channels: {lengths}")


def _normalize_effect_inventory(value: Any) -> tuple[dict[str, Any], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (tuple, list)):
        raise ValueError("checkpoint effect_inventory must be an array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"checkpoint effect_inventory[{index}] must be an object")
        normalized = {str(key): item[key] for key in sorted(item)}
        _json_safe(normalized, f"effect_inventory[{index}]")
        result.append(normalized)
    return tuple(result)


@dataclass(frozen=True)
class CheckpointPayload:
    frame: int
    sim_time: float
    settings_signature: str
    source_signature: str
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    scalars: Mapping[str, Any] = field(default_factory=dict)
    source_state: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = CHECKPOINT_FORMAT_VERSION
    effect_signature: str = ""
    effect_inventory: tuple[dict[str, Any], ...] = ()

    def validated(self) -> "CheckpointPayload":
        if isinstance(self.frame, bool) or not isinstance(self.frame, int):
            raise ValueError("checkpoint frame must be an integer")
        if not isinstance(self.sim_time, (int, float)) or not math.isfinite(
            float(self.sim_time)
        ):
            raise ValueError("checkpoint sim_time must be finite")
        if not isinstance(self.settings_signature, str) or not self.settings_signature:
            raise ValueError("checkpoint settings_signature is required")
        if not isinstance(self.source_signature, str) or not self.source_signature:
            raise ValueError("checkpoint source_signature is required")
        if self.effect_signature:
            if (not isinstance(self.effect_signature, str)
                    or len(self.effect_signature) != 64
                    or any(ch not in "0123456789abcdef" for ch in self.effect_signature)):
                raise ValueError("checkpoint effect_signature must be a lowercase SHA-256")
        inventory = _normalize_effect_inventory(self.effect_inventory)
        if inventory and not self.effect_signature:
            raise ValueError("checkpoint effect inventory requires an effect signature")
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format {self.format_version}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}"
            )
        arrays = {
            str(name): _normalized_array(str(name), value)
            for name, value in self.arrays.items()
        }
        if any(not name or name == _META_KEY for name in arrays):
            raise ValueError("checkpoint array names must be non-empty and reserved-key safe")
        _validate_channel_alignment(arrays)
        if "whitewater_type" in arrays:
            from ..sim.whitewater_types import validate_type_values

            arrays["whitewater_type"] = validate_type_values(
                arrays["whitewater_type"], allow_unclassified=False)
        _json_safe(dict(self.scalars), "scalars")
        _json_safe(dict(self.source_state), "source_state")
        return CheckpointPayload(
            frame=int(self.frame),
            sim_time=float(self.sim_time),
            settings_signature=self.settings_signature,
            source_signature=self.source_signature,
            effect_signature=self.effect_signature,
            effect_inventory=inventory,
            arrays=arrays,
            scalars=dict(self.scalars),
            source_state=dict(self.source_state),
            format_version=self.format_version,
        )


def _metadata(payload: CheckpointPayload) -> dict[str, Any]:
    return {
        "checkpoint_format_version": payload.format_version,
        "frame": payload.frame,
        "sim_time": payload.sim_time,
        "settings_signature": payload.settings_signature,
        "source_signature": payload.source_signature,
        "effect_signature": payload.effect_signature,
        "effect_inventory": [dict(item) for item in payload.effect_inventory],
        "scalars": _json_safe(dict(payload.scalars), "scalars"),
        "source_state": _json_safe(dict(payload.source_state), "source_state"),
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(payload.arrays.items())
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_checkpoint_atomic(path: Path | str, payload: CheckpointPayload) -> str:
    """Write, fsync, checksum, and atomically replace a checkpoint."""
    path = Path(path)
    payload = payload.validated()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            metadata = json.dumps(
                _metadata(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            np.savez(
                handle,
                **{_META_KEY: np.asarray(metadata), **dict(payload.arrays)},
            )
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(temp)
        os.replace(temp, path)
        _fsync_directory(path.parent)
        return digest
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


def read_checkpoint(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    expected_frame: int | None = None,
    expected_settings_signature: str | None = None,
    expected_source_signature: str | None = None,
    expected_effect_signature: str | None = None,
) -> CheckpointPayload:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"checkpoint is missing: {path}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError("checkpoint size mismatch")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("checkpoint checksum mismatch")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"checkpoint could not be read: {exc}") from exc
    with loaded as data:
        if _META_KEY not in data.files:
            raise ValueError("checkpoint metadata is missing")
        try:
            metadata = json.loads(str(np.asarray(data[_META_KEY]).item()))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        required_meta = {
            "checkpoint_format_version",
            "frame",
            "sim_time",
            "settings_signature",
            "source_signature",
            "scalars",
            "source_state",
            "arrays",
        }
        optional_meta = {"effect_signature", "effect_inventory"}
        if not required_meta.issubset(metadata) or set(metadata) - required_meta - optional_meta:
            raise ValueError("checkpoint metadata keys are invalid")
        arrays_meta = metadata["arrays"]
        if not isinstance(arrays_meta, dict):
            raise ValueError("checkpoint array metadata must be an object")
        array_names = set(data.files) - {_META_KEY}
        if array_names != set(arrays_meta):
            raise ValueError("checkpoint array inventory does not match metadata")
        arrays: dict[str, np.ndarray] = {}
        for name in sorted(array_names):
            value = _normalized_array(name, data[name])
            record = arrays_meta[name]
            if not isinstance(record, dict) or set(record) != {"dtype", "shape"}:
                raise ValueError(f"checkpoint metadata for {name!r} is invalid")
            if str(value.dtype) != record["dtype"] or list(value.shape) != record["shape"]:
                raise ValueError(f"checkpoint array metadata mismatch for {name!r}")
            arrays[name] = value.copy()
    payload = CheckpointPayload(
        frame=metadata["frame"],
        sim_time=metadata["sim_time"],
        settings_signature=metadata["settings_signature"],
        source_signature=metadata["source_signature"],
        effect_signature=str(metadata.get("effect_signature", "")),
        effect_inventory=_normalize_effect_inventory(metadata.get("effect_inventory", ())),
        arrays=arrays,
        scalars=metadata["scalars"],
        source_state=metadata["source_state"],
        format_version=metadata["checkpoint_format_version"],
    ).validated()
    if expected_frame is not None and payload.frame != int(expected_frame):
        raise ValueError("checkpoint frame mismatch")
    if (
        expected_settings_signature is not None
        and payload.settings_signature != expected_settings_signature
    ):
        raise ValueError("checkpoint settings signature mismatch")
    if (
        expected_source_signature is not None
        and payload.source_signature != expected_source_signature
    ):
        raise ValueError("checkpoint source signature mismatch")
    if (
        expected_effect_signature is not None
        and payload.effect_signature != expected_effect_signature
    ):
        raise ValueError("checkpoint effect signature mismatch")
    return payload


def capture_solver(
    solver,
    *,
    settings_signature: str,
    source_signature: str,
    effect_signature: str = "",
    effect_inventory: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> CheckpointPayload:
    """Capture an owned solver snapshot through its audited state seam."""
    state = solver.export_checkpoint_state()
    if not isinstance(state, Mapping):
        raise ValueError("solver export_checkpoint_state() must return a mapping")
    arrays = state.get("arrays", {})
    scalars = dict(state.get("scalars", {}))
    source_state = state.get("source_state", {})
    frame = int(scalars.pop("frame", getattr(solver, "frame", 0)))
    sim_time = float(scalars.pop("sim_time", getattr(solver, "sim_time", 0.0)))
    return CheckpointPayload(
        frame=frame,
        sim_time=sim_time,
        settings_signature=settings_signature,
        source_signature=source_signature,
        effect_signature=effect_signature,
        effect_inventory=_normalize_effect_inventory(effect_inventory),
        arrays=arrays,
        scalars=scalars,
        source_state=source_state,
    ).validated()


def restore_solver(solver, payload: CheckpointPayload) -> None:
    """Restore a validated payload through the solver-owned semantic seam."""
    payload = payload.validated()
    solver.import_checkpoint_state(
        {
            "arrays": dict(payload.arrays),
            "scalars": {
                **dict(payload.scalars),
                "frame": payload.frame,
                "sim_time": payload.sim_time,
            },
            "source_state": dict(payload.source_state),
        }
    )
