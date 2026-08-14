"""Strict cache-v2 manifest schema and durable atomic JSON transactions.

The manifest is the sole authority for playback and resume. Frame/checkpoint
files that exist without a committed manifest entry are orphans and must never
be treated as complete simulation state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

FORMAT_VERSION = 2
MANIFEST_FILENAME = "fluidsim_cache.json"
FRAME_DIRECTORY = "frames"
CHECKPOINT_DIRECTORY = "checkpoints"
LOG_DIRECTORY = "logs"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


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
class CacheManifestV2:
    """Strict manifest-authoritative cache-v2 state."""

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
        checkpoint_interval: int = 1,
        checkpoint_format_version: int = 1,
        timestamp: str | None = None,
    ) -> "CacheManifestV2":
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
            frames={},
        )
        return manifest.validate()

    def validate(self) -> "CacheManifestV2":
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported cache format {self.format_version}; expected {FORMAT_VERSION}"
            )
        _require_int(
            self.checkpoint_format_version,
            "checkpoint_format_version",
            minimum=1,
        )
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

    def with_committed_frame(
        self, entry: FrameEntry, *, timestamp: str | None = None
    ) -> "CacheManifestV2":
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
        result["frames"] = {
            str(frame): asdict(self.frames[frame]) for frame in sorted(self.frames)
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheManifestV2":
        if not isinstance(value, Mapping):
            raise ValueError("cache manifest must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        missing = expected - actual
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
        kwargs = {name: value[name] for name in expected if name != "frames"}
        return cls(**kwargs, frames=frames).validate()


@dataclass(frozen=True)
class ResumeDecision:
    allowed: bool
    resume_frame: int | None
    code: str
    message: str


def manifest_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / MANIFEST_FILENAME


def load_manifest(path: Path | str) -> CacheManifestV2 | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid cache manifest {path}: {exc}") from exc
    return CacheManifestV2.from_dict(value)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest_atomic(path: Path | str, manifest: CacheManifestV2) -> None:
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
    base_dir: Path | str, manifest: CacheManifestV2
) -> tuple[int, ...]:
    manifest.validate()
    verified: list[int] = []
    for frame in manifest.committed_frames:
        verify_frame_entry(base_dir, manifest.frames[frame])
        verified.append(frame)
    return tuple(verified)
