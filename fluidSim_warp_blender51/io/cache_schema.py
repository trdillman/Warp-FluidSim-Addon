"""Strict cache-v3 manifest schema and durable atomic JSON transactions.

The manifest is the sole authority for playback and resume. Frame/checkpoint
files that exist without a committed manifest entry are orphans and must never
be treated as complete simulation state.

Format 3 (FluidSim 3.0) adds, as required fields, the per-frame source stamp
and particle counts plus the grid, transfer-mode, narrow-band, platform and
source-schema facts of the bake. A format-2 manifest is refused with
``LEGACY_2X_CACHE_MESSAGE``; nothing is defaulted or migrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping

from .checkpoint import CHECKPOINT_FORMAT_VERSION

FORMAT_VERSION = 3
LEGACY_2X_FORMAT_VERSION = 2
LEGACY_2X_CACHE_MESSAGE = (
    "This cache was baked by FluidSim 2.x and cannot be played by 3.0. "
    "Re-bake, or open it in 2.1.1.")
SUPPORTED_TRANSFER_MODES = ("FLIP", "APIC")
LEAF_DIM = 8
MANIFEST_FILENAME = "fluidsim_cache.json"
FRAME_DIRECTORY = "frames"
CHECKPOINT_DIRECTORY = "checkpoints"
LOG_DIRECTORY = "logs"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class LegacyCacheFormatError(ValueError):
    """A FluidSim 2.x cache directory: refused, never read or migrated."""

    def __init__(self, message: str = LEGACY_2X_CACHE_MESSAGE):
        super().__init__(message)


def utc_now() -> str:
    """Return a stable UTC timestamp accepted by the strict schema."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _require_relative_file(value: Any, name: str, expected_parent: str) -> str:
    value = _require_nonempty_string(value, name)
    if "\\" in value:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")
    if not path.parts or path.parts[0] != expected_parent or len(path.parts) != 2:
        raise ValueError(f"{name} must be inside {expected_parent}/")
    return value


def _require_timestamp(value: Any, name: str) -> str:
    value = _require_nonempty_string(value, name)
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    return value


def _normalize_effect_inventory(value: Any) -> tuple[dict[str, Any], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, (tuple, list)):
        raise ValueError("effect_inventory must be an array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"effect_inventory[{index}] must be an object")
        normalized = {str(key): item[key] for key in sorted(item)}
        try:
            json.dumps(normalized, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"effect_inventory[{index}] is not strict JSON: {exc}") from exc
        result.append(normalized)
    return tuple(result)


def _require_exact_keys(value: Any, name: str, keys: tuple[str, ...]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    missing = set(keys) - set(value)
    unknown = set(value) - set(keys)
    if missing or unknown:
        raise ValueError(
            f"{name} keys are invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _normalize_grid_spec(value: Any) -> dict[str, Any]:
    """Canonical ``grid_spec``: the grid the bake resolved, as plain JSON."""
    keys = ("resolution_divisions", "voxel_size", "domain_size", "leaf_dim")
    value = _require_exact_keys(value, "grid_spec", keys)
    divisions = _require_int(
        value["resolution_divisions"], "grid_spec.resolution_divisions", minimum=1)
    voxel_size = _require_finite_float(value["voxel_size"], "grid_spec.voxel_size")
    if voxel_size <= 0.0:
        raise ValueError("grid_spec.voxel_size must be > 0")
    size = value["domain_size"]
    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError("grid_spec.domain_size must be three numbers")
    domain_size = [
        _require_finite_float(item, "grid_spec.domain_size") for item in size]
    if any(item <= 0.0 for item in domain_size):
        raise ValueError("grid_spec.domain_size must be > 0 on every axis")
    leaf_dim = _require_int(value["leaf_dim"], "grid_spec.leaf_dim")
    if leaf_dim != LEAF_DIM:
        raise ValueError(f"grid_spec.leaf_dim must be {LEAF_DIM}")
    return {
        "resolution_divisions": divisions,
        "voxel_size": voxel_size,
        "domain_size": domain_size,
        "leaf_dim": leaf_dim,
    }


def _normalize_narrow_band(value: Any) -> dict[str, Any]:
    value = _require_exact_keys(value, "narrow_band", ("enabled", "band_voxels"))
    enabled = _require_bool(value["enabled"], "narrow_band.enabled")
    band_voxels = _require_finite_float(value["band_voxels"], "narrow_band.band_voxels")
    if band_voxels < 0.0 or (enabled and band_voxels <= 0.0):
        raise ValueError(
            "narrow_band.band_voxels must be > 0 when the narrow band is enabled "
            "and >= 0 otherwise")
    return {"enabled": enabled, "band_voxels": band_voxels}


def _normalize_platform(value: Any) -> dict[str, str]:
    """Informational only: never part of any signature (bitwise identity is
    per GPU/driver, and a cache must stay playable on another machine)."""
    keys = ("os", "gpu_name", "driver", "warp")
    value = _require_exact_keys(value, "platform", keys)
    return {
        name: _require_nonempty_string(value[name], f"platform.{name}")
        for name in keys}


def _require_transfer_mode(value: Any) -> str:
    if value not in SUPPORTED_TRANSFER_MODES:
        raise ValueError(
            f"transfer_mode must be one of {', '.join(SUPPORTED_TRANSFER_MODES)}")
    return value


@dataclass(frozen=True)
class FrameEntry:
    """One fully committed frame/checkpoint pair."""

    frame: int
    frame_file: str
    frame_sha256: str
    frame_size: int
    checkpoint_file: str
    checkpoint_sha256: str
    checkpoint_size: int
    completed_at: str
    solver_frame: int
    simulation_time: float
    settings_signature: str
    source_signature: str
    # Hash of every source/collider's identity, world matrix and evaluated
    # geometry *at this frame*; the static source_signature above is taken once
    # at bake start and cannot see animation. Resume compares these.
    source_frame_stamp: str
    whitewater_count: int
    particle_count: int

    def validate(self, *, key: int | None = None) -> "FrameEntry":
        _require_int(self.frame, "frame")
        if key is not None and self.frame != key:
            raise ValueError(f"frame entry key {key} does not match frame {self.frame}")
        _require_relative_file(self.frame_file, "frame_file", FRAME_DIRECTORY)
        _require_sha256(self.frame_sha256, "frame_sha256")
        _require_int(self.frame_size, "frame_size", minimum=1)
        _require_relative_file(
            self.checkpoint_file, "checkpoint_file", CHECKPOINT_DIRECTORY
        )
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _require_int(self.checkpoint_size, "checkpoint_size", minimum=1)
        _require_timestamp(self.completed_at, "completed_at")
        _require_int(self.solver_frame, "solver_frame")
        if self.solver_frame != self.frame:
            raise ValueError(
                f"solver_frame {self.solver_frame} does not match frame {self.frame}"
            )
        _require_finite_float(self.simulation_time, "simulation_time")
        _require_nonempty_string(self.settings_signature, "settings_signature")
        _require_nonempty_string(self.source_signature, "source_signature")
        _require_sha256(self.source_frame_stamp, "source_frame_stamp")
        _require_int(self.whitewater_count, "whitewater_count", minimum=0)
        _require_int(self.particle_count, "particle_count", minimum=0)
        return self

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameEntry":
        if not isinstance(value, Mapping):
            raise ValueError("frame entry must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        missing = expected - actual
        unknown = actual - expected
        if missing or unknown:
            raise ValueError(
                f"invalid frame entry keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**{name: value[name] for name in expected}).validate()


@dataclass
class CacheManifestV3:
    """Strict manifest-authoritative cache-v3 state."""

    format_version: int
    checkpoint_format_version: int
    solver_version: str
    scene_key: str
    settings_signature: str
    source_signature: str
    frame_start: int
    frame_end: int
    checkpoint_interval: int
    created_at: str
    updated_at: str
    # Required, no defaults: an unstamped manifest must not read as valid.
    source_schema_version: int
    transfer_mode: str
    grid_spec: dict[str, Any]
    narrow_band: dict[str, Any]
    platform: dict[str, str]
    effect_signature: str = ""
    effect_inventory: tuple[dict[str, Any], ...] = ()
    frames: dict[int, FrameEntry] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        solver_version: str,
        scene_key: str,
        settings_signature: str,
        source_signature: str,
        frame_start: int,
        frame_end: int,
        source_schema_version: int,
        transfer_mode: str,
        grid_spec: Mapping[str, Any],
        narrow_band: Mapping[str, Any],
        platform: Mapping[str, str],
        checkpoint_interval: int = 1,
        checkpoint_format_version: int = CHECKPOINT_FORMAT_VERSION,
        effect_signature: str = "",
        effect_inventory: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
        timestamp: str | None = None,
    ) -> "CacheManifestV3":
        timestamp = timestamp or utc_now()
        manifest = cls(
            format_version=FORMAT_VERSION,
            checkpoint_format_version=checkpoint_format_version,
            solver_version=solver_version,
            scene_key=scene_key,
            settings_signature=settings_signature,
            source_signature=source_signature,
            frame_start=frame_start,
            frame_end=frame_end,
            checkpoint_interval=checkpoint_interval,
            created_at=timestamp,
            updated_at=timestamp,
            source_schema_version=source_schema_version,
            transfer_mode=transfer_mode,
            grid_spec=_normalize_grid_spec(grid_spec),
            narrow_band=_normalize_narrow_band(narrow_band),
            platform=_normalize_platform(platform),
            effect_signature=str(effect_signature),
            effect_inventory=_normalize_effect_inventory(effect_inventory),
            frames={},
        )
        return manifest.validate()

    def validate(self) -> "CacheManifestV3":
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported cache format {self.format_version}; expected {FORMAT_VERSION}"
            )
        _require_int(
            self.checkpoint_format_version,
            "checkpoint_format_version",
            minimum=1,
        )
        if self.checkpoint_format_version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint_format_version {self.checkpoint_format_version} cannot "
                f"be read; expected {CHECKPOINT_FORMAT_VERSION}")
        _require_int(self.source_schema_version, "source_schema_version", minimum=1)
        _require_transfer_mode(self.transfer_mode)
        _normalize_grid_spec(self.grid_spec)
        _normalize_narrow_band(self.narrow_band)
        _normalize_platform(self.platform)
        _require_nonempty_string(self.solver_version, "solver_version")
        _require_nonempty_string(self.scene_key, "scene_key")
        _require_nonempty_string(self.settings_signature, "settings_signature")
        _require_nonempty_string(self.source_signature, "source_signature")
        _require_int(self.frame_start, "frame_start")
        _require_int(self.frame_end, "frame_end")
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        _require_int(self.checkpoint_interval, "checkpoint_interval", minimum=1)
        _require_timestamp(self.created_at, "created_at")
        _require_timestamp(self.updated_at, "updated_at")
        if self.effect_signature:
            _require_sha256(self.effect_signature, "effect_signature")
        inventory = _normalize_effect_inventory(self.effect_inventory)
        if inventory and not self.effect_signature:
            raise ValueError("effect inventory requires an effect signature")
        if not isinstance(self.frames, dict):
            raise ValueError("frames must be an object")

        keys = sorted(self.frames)
        for frame in keys:
            _require_int(frame, "frame key")
            if frame < self.frame_start or frame > self.frame_end:
                raise ValueError(
                    f"frame {frame} is outside {self.frame_start}..{self.frame_end}"
                )
            entry = self.frames[frame]
            if not isinstance(entry, FrameEntry):
                raise ValueError(f"frame {frame} is not a FrameEntry")
            entry.validate(key=frame)
            if entry.settings_signature != self.settings_signature:
                raise ValueError(f"frame {frame} settings signature drift")
            if entry.source_signature != self.source_signature:
                raise ValueError(f"frame {frame} source signature drift")

        # Production cache commits are contiguous. A later manifest entry with
        # a gap would expose an invalid resume point even if its files exist.
        if keys:
            expected = list(range(self.frame_start, keys[-1] + 1))
            if keys != expected:
                raise ValueError(
                    f"committed frames must be contiguous from {self.frame_start}; "
                    f"found {keys}"
                )
        return self

    @property
    def committed_frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.frames))

    @property
    def last_committed_frame(self) -> int | None:
        return max(self.frames) if self.frames else None

    def truncated_before(
        self, frame: int, *, timestamp: str | None = None
    ) -> "CacheManifestV3":
        """Return a validated copy holding only the frames below ``frame``.

        This is the manifest half of "Re-bake from frame N": the caller commits
        the truncated manifest first, then deletes the stale frame/checkpoint
        files, so a crash in between leaves only unreferenced orphans.
        """
        frame = _require_int(frame, "frame")
        if frame < self.frame_start:
            raise ValueError(
                f"cannot truncate to frame {frame}, which is before the first "
                f"frame {self.frame_start}")
        kept = {key: entry for key, entry in self.frames.items() if key < frame}
        if len(kept) == len(self.frames):
            return self
        return replace(
            self, updated_at=timestamp or utc_now(), frames=kept).validate()

    def with_committed_frame(
        self, entry: FrameEntry, *, timestamp: str | None = None
    ) -> "CacheManifestV3":
        """Return a validated copy with exactly the next contiguous entry."""
        entry.validate()
        expected = (
            self.frame_start
            if self.last_committed_frame is None
            else self.last_committed_frame + 1
        )
        if entry.frame != expected:
            raise ValueError(
                f"frame {entry.frame} cannot commit; next frame is {expected}"
            )
        if entry.frame > self.frame_end:
            raise ValueError(f"frame {entry.frame} exceeds frame_end {self.frame_end}")
        updated = replace(
            self,
            updated_at=timestamp or utc_now(),
            frames={**self.frames, entry.frame: entry},
        )
        return updated.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["effect_inventory"] = [dict(item) for item in self.effect_inventory]
        result["frames"] = {
            str(frame): asdict(self.frames[frame]) for frame in sorted(self.frames)
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheManifestV3":
        if not isinstance(value, Mapping):
            raise ValueError("cache manifest must be a JSON object")
        # Decide on the version before the key check, so a 2.x manifest gets the
        # re-bake message instead of a confusing "invalid cache manifest keys".
        version = value.get("format_version")
        if isinstance(version, int) and not isinstance(version, bool):
            if version == LEGACY_2X_FORMAT_VERSION:
                raise LegacyCacheFormatError()
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported cache format {version}; expected {FORMAT_VERSION}")
        expected = set(cls.__dataclass_fields__)
        optional = {"effect_signature", "effect_inventory"}
        actual = set(value)
        missing = (expected - optional) - actual
        unknown = actual - expected
        if missing or unknown:
            raise ValueError(
                f"invalid cache manifest keys; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        frames_value = value.get("frames")
        if not isinstance(frames_value, Mapping):
            raise ValueError("frames must be a JSON object")
        frames: dict[int, FrameEntry] = {}
        for key, entry_value in frames_value.items():
            if not isinstance(key, str) or not key or not key.lstrip("-").isdigit():
                raise ValueError(f"invalid frame key {key!r}")
            frame = int(key)
            entry = FrameEntry.from_dict(entry_value)
            entry.validate(key=frame)
            frames[frame] = entry
        kwargs = {
            name: value[name]
            for name in expected
            if name not in {"frames", "effect_signature", "effect_inventory"}
        }
        kwargs["effect_signature"] = str(value.get("effect_signature", ""))
        kwargs["effect_inventory"] = _normalize_effect_inventory(
            value.get("effect_inventory", ())
        )
        kwargs["grid_spec"] = _normalize_grid_spec(value["grid_spec"])
        kwargs["narrow_band"] = _normalize_narrow_band(value["narrow_band"])
        kwargs["platform"] = _normalize_platform(value["platform"])
        return cls(**kwargs, frames=frames).validate()


# Importers written for 2.x still say CacheManifestV2; it is the same class.
CacheManifestV2 = CacheManifestV3


def first_changed_source_frame(
    manifest: CacheManifestV3, current_stamp: Callable[[int], str]
) -> int | None:
    """Return the earliest committed frame whose source stamp no longer matches.

    ``current_stamp(frame)`` must evaluate the scene at that frame. The scan is
    forward and checks every committed frame: a keyframe that moved only in the
    middle of the range is found even though the last frame still matches.
    """
    for frame in manifest.committed_frames:
        if current_stamp(frame) != manifest.frames[frame].source_frame_stamp:
            return frame
    return None


@dataclass(frozen=True)
class ResumeDecision:
    allowed: bool
    resume_frame: int | None
    code: str
    message: str
    # Set with code SOURCE_ANIMATION_CHANGED: where "Re-bake from frame N" starts.
    first_changed_frame: int | None = None


def manifest_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / MANIFEST_FILENAME


def load_manifest(path: Path | str) -> CacheManifestV3 | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cache manifest {path}: {exc}") from exc
    return CacheManifestV3.from_dict(value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest_atomic(path: Path | str, manifest: CacheManifestV3) -> None:
    """Durably replace a manifest; no partially written JSON becomes visible."""
    path = Path(path)
    manifest.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    payload = manifest.to_dict()
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{label} size mismatch for {path}: {actual_size} != {expected_size}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} checksum mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )


def verify_frame_entry(base_dir: Path | str, entry: FrameEntry) -> None:
    """Verify both committed artifacts referenced by one manifest entry."""
    base = Path(base_dir)
    entry.validate()
    verify_artifact(
        base / entry.frame_file,
        expected_sha256=entry.frame_sha256,
        expected_size=entry.frame_size,
        label=f"frame {entry.frame}",
    )
    verify_artifact(
        base / entry.checkpoint_file,
        expected_sha256=entry.checkpoint_sha256,
        expected_size=entry.checkpoint_size,
        label=f"checkpoint {entry.frame}",
    )


def verified_contiguous_frames(
    base_dir: Path | str, manifest: CacheManifestV3
) -> tuple[int, ...]:
    manifest.validate()
    verified: list[int] = []
    for frame in manifest.committed_frames:
        verify_frame_entry(base_dir, manifest.frames[frame])
        verified.append(frame)
    return tuple(verified)
