"""Transactional cache inspection and user-owned cache filesystem operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from ..io.cache import LEGACY_CACHE_MESSAGE, detect_cache_format
from ..io.cache_schema import (
    CacheManifestV2,
    load_manifest,
    manifest_path,
    verified_contiguous_frames,
)


@dataclass(frozen=True)
class CacheInspection:
    path: str
    format_version: int
    frame_start: int | None
    frame_end: int | None
    committed_frames: tuple[int, ...]
    last_complete_frame: int | None
    last_resumable_frame: int | None
    disk_size: int
    missing_files: tuple[str, ...]
    corrupt_files: tuple[str, ...]
    orphan_files: tuple[str, ...]
    compatible: bool
    message: str
    effect_signature: str = ""
    effect_inventory: tuple[dict[str, object], ...] = ()

    @property
    def healthy(self) -> bool:
        return not (self.missing_files or self.corrupt_files)


def assert_cache_path_mutable(path: Path | str) -> Path:
    """Return a resolved cache path or reject one owned by Production Bake."""
    path = Path(path).expanduser().resolve()
    from . import production_bake

    if production_bake.cache_path_is_active(path):
        raise RuntimeError(
            "Production Bake is actively writing this cache; wait for the "
            "bake to finish or cancel before changing cache files")
    return path


def is_managed_cache_directory(path: Path | str) -> bool:
    """Return whether ``path`` contains only recognizable FluidSim cache state.

    A manifest is authoritative.  A FluidSim 1.0 cache is recognized too, so
    it can be freed before re-baking.  The limited partial patterns cover a
    first-frame transaction interrupted before manifest commit without
    treating an arbitrary existing directory as a cache.
    """
    base = Path(path).expanduser().resolve()
    if not base.is_dir():
        return False
    if detect_cache_format(base):
        return True
    patterns = (
        ("frames", "frame_*.npz"),
        ("frames", "frame_*.tmp"),
        ("frames", "frame_*.tmp.npz"),
        ("checkpoints", "checkpoint_*.npz"),
        ("checkpoints", "checkpoint_*.tmp"),
    )
    for directory, pattern in patterns:
        root = base / directory
        if root.is_dir() and any(root.glob(pattern)):
            return True
    return False


def _require_managed_cache_directory(path: Path | str) -> Path:
    base = Path(path).expanduser().resolve()
    if not is_managed_cache_directory(base):
        raise ValueError(f"not a recognized FluidSim cache directory: {base}")
    return base


# What a Production Bake restart replaces inside a cache folder, and nothing else.
RESTART_MANAGED_ENTRIES = ("fluidsim_cache.json", "frames", "checkpoints", "logs")


def require_restartable_cache_directory(
        path: Path | str,
        entries: tuple[str, ...] = RESTART_MANAGED_ENTRIES) -> Path:
    """Refuse a restart that would replace data FluidSim did not write.

    Restart replaces ``entries`` (``frames``, ``logs``...).  Those names are
    generic, so they may only be replaced inside a recognized FluidSim cache;
    empty directories are safe because replacing them loses nothing.
    """
    base = Path(path).expanduser().resolve()
    if not base.is_dir() or is_managed_cache_directory(base):
        return base
    occupied = []
    for name in entries:
        entry = base / name
        if entry.is_symlink() or entry.is_file() or (
                entry.is_dir() and any(entry.iterdir())):
            occupied.append(name)
    if occupied:
        raise ValueError(
            f"Restart would replace {', '.join(occupied)} in {base}, but that "
            "folder is not a FluidSim cache. Choose an empty folder or an "
            "existing FluidSim cache folder.")
    return base


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        result[item.relative_to(path).as_posix()] = (item.stat().st_size, _sha256_file(item))
    return result


def _disk_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _referenced_files(manifest: CacheManifestV2) -> set[str]:
    referenced = {"fluidsim_cache.json"}
    for entry in manifest.frames.values():
        referenced.add(entry.frame_file)
        referenced.add(entry.checkpoint_file)
    return referenced


def _orphans(path: Path, manifest: CacheManifestV2) -> tuple[str, ...]:
    referenced = _referenced_files(manifest)
    candidates: set[str] = set()
    for directory, pattern in (("frames", "*.npz"), ("checkpoints", "*.npz")):
        root = path / directory
        if root.is_dir():
            candidates.update(
                item.relative_to(path).as_posix() for item in root.glob(pattern)
            )
    candidates.update(
        item.relative_to(path).as_posix()
        for item in path.rglob("*.tmp")
        if item.is_file()
    )
    return tuple(sorted(candidates - referenced))


def inspect_cache(path: Path | str) -> CacheInspection:
    path = Path(path).resolve()
    if not path.exists():
        return CacheInspection(
            str(path), 0, None, None, (), None, None, 0, (), (), (), False,
            "Cache directory does not exist",
        )
    v2_path = manifest_path(path)
    if v2_path.is_file():
        try:
            manifest = load_manifest(v2_path)
            assert manifest is not None
        except ValueError as exc:
            return CacheInspection(
                str(path), 2, None, None, (), None, None, _disk_size(path),
                (), ("fluidsim_cache.json",), (), False,
                f"Cache-v2 manifest is invalid: {exc}",
            )
        missing: list[str] = []
        corrupt: list[str] = []
        verified: list[int] = []
        for frame in manifest.committed_frames:
            entry = manifest.frames[frame]
            for relative in (entry.frame_file, entry.checkpoint_file):
                if not (path / relative).is_file():
                    missing.append(relative)
            if missing:
                break
            try:
                # Verify the contiguous prefix in manifest order and stop at
                # the first failure; later entries are not valid resume points.
                from ..io.cache_schema import verify_frame_entry

                verify_frame_entry(path, entry)
                verified.append(frame)
            except ValueError:
                corrupt.extend((entry.frame_file, entry.checkpoint_file))
                break
        compatible = not missing and not corrupt
        last = verified[-1] if verified else None
        return CacheInspection(
            path=str(path),
            format_version=2,
            frame_start=manifest.frame_start,
            frame_end=manifest.frame_end,
            committed_frames=manifest.committed_frames,
            last_complete_frame=last,
            last_resumable_frame=last,
            disk_size=_disk_size(path),
            missing_files=tuple(sorted(set(missing))),
            corrupt_files=tuple(sorted(set(corrupt))),
            orphan_files=_orphans(path, manifest),
            compatible=compatible,
            message=(
                f"Cache v2: {len(verified)} verified committed frame(s)"
                if compatible else
                "Cache v2 failed integrity verification"
            ),
            effect_signature=manifest.effect_signature,
            effect_inventory=tuple(dict(item) for item in manifest.effect_inventory),
        )

    if detect_cache_format(path) == 1:
        return CacheInspection(
            str(path), 1, None, None, (), None, None, _disk_size(path),
            (), (), (), False, LEGACY_CACHE_MESSAGE,
        )
    return CacheInspection(
        str(path), 0, None, None, (), None, None, _disk_size(path), (), (), (),
        False, "No FluidSim cache was found",
    )


def remove_orphans(path: Path | str) -> tuple[str, ...]:
    assert_cache_path_mutable(path)
    inspection = inspect_cache(path)
    removed: list[str] = []
    base = Path(inspection.path)
    for relative in inspection.orphan_files:
        target = base / relative
        try:
            target.unlink()
            removed.append(relative)
        except FileNotFoundError:
            pass
    return tuple(removed)


def free_cache(path: Path | str) -> Path | None:
    """Rename on the same volume before deletion; return retained trash on failure."""
    path = assert_cache_path_mutable(path)
    if not path.exists():
        return None
    path = _require_managed_cache_directory(path)
    trash = path.with_name(f".{path.name}.fluidsim-trash-{uuid.uuid4().hex}")
    os.replace(path, trash)
    try:
        shutil.rmtree(trash)
        return None
    except Exception:
        return trash


def _copy_and_verify(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    source_inspection = inspect_cache(source)
    destination_inspection = inspect_cache(destination)
    if source_inspection.format_version != destination_inspection.format_version:
        raise RuntimeError("cache format changed during move")
    if source_inspection.format_version == 2:
        if not destination_inspection.healthy or (
            destination_inspection.committed_frames
            != source_inspection.committed_frames
        ):
            raise RuntimeError("copied cache failed manifest/checksum verification")
        verified_contiguous_frames(
            destination,
            load_manifest(manifest_path(destination)),  # type: ignore[arg-type]
        )
    elif _file_snapshot(source) != _file_snapshot(destination):
        # No manifest to verify against: a partial or FluidSim 1.0 cache.
        raise RuntimeError("copied cache failed byte-for-byte verification")


def move_cache(source: Path | str, destination: Path | str) -> Path:
    """Copy, verify, atomically publish, then remove the source."""
    source = assert_cache_path_mutable(source)
    destination = assert_cache_path_mutable(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"cache source does not exist: {source}")
    source = _require_managed_cache_directory(source)
    if destination.exists():
        raise FileExistsError(f"cache destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.fluidsim-copy-{uuid.uuid4().hex}")
    try:
        _copy_and_verify(source, temp)
        os.replace(temp, destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    retained = free_cache(source)
    if retained is not None:
        raise RuntimeError(
            f"cache copied successfully, but source cleanup remains at {retained}")
    return destination


def validate_leaf_name(name: str) -> str:
    name = str(name).strip()
    if not name or name in {".", ".."}:
        raise ValueError("cache name must be a non-empty directory name")
    if Path(name).name != name or any(separator in name for separator in ("/", "\\")):
        raise ValueError("cache name must not contain path separators")
    return name


def rename_cache(path: Path | str, name: str) -> Path:
    path = assert_cache_path_mutable(path)
    path = _require_managed_cache_directory(path)
    name = validate_leaf_name(name)
    destination = path.with_name(name)
    assert_cache_path_mutable(destination)
    if destination.exists():
        raise FileExistsError(f"cache destination already exists: {destination}")
    os.replace(path, destination)
    return destination


def open_cache_folder(path: Path | str) -> None:
    """Open the cache parent without shell-string interpolation."""
    path = Path(path).resolve()
    target = path if path.is_dir() else path.parent
    try:
        import bpy

        bpy.ops.wm.path_open(filepath=str(target))
        return
    except (ImportError, AttributeError, RuntimeError):
        pass
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(target)], check=True)
    else:
        subprocess.run(["xdg-open", str(target)], check=True)
