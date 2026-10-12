"""Cheap, side-effect-free cache presentation state for Blender panels.

Exact checksum, signature, and checkpoint verification remains in the
production-bake execute path.  This module only chooses the one action the UI
should present from the manifest transaction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..io.cache_schema import (
    LEGACY_2X_CACHE_MESSAGE,
    LegacyCacheFormatError,
    load_manifest,
    manifest_path,
)


@dataclass(frozen=True)
class ProductionBakeAction:
    action: str
    message: str
    destructive: bool = False


def _manifest_stamp(path: Path):
    """``(path, mtime_ns, size)`` of a manifest file, or ``None`` if absent.

    Manifests are replaced atomically on every commit, so the stamp changes
    whenever the content can have changed.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


# Panels redraw on every mouse move; parsing and validating a manifest grows
# with its frame count, so the presentation is memoized per manifest stamp.
_ACTION_CACHE: dict = {}
_FORMAT_CACHE: dict = {}


def production_bake_action(cache_dir: Path | str | None) -> ProductionBakeAction:
    """Return START, RESUME, or RESTART for one mutually exclusive UI action."""
    if cache_dir is None:
        return ProductionBakeAction("START", "No production cache exists")
    path = manifest_path(Path(cache_dir))
    stamp = _manifest_stamp(path)
    if stamp is None:
        return ProductionBakeAction("START", "No cache manifest exists")
    cached = _ACTION_CACHE.get(stamp[0])
    if cached is not None and cached[0] == stamp:
        return cached[1]
    action = _production_bake_action(path)
    if len(_ACTION_CACHE) > 64:
        _ACTION_CACHE.clear()
    _ACTION_CACHE[stamp[0]] = (stamp, action)
    return action


def cache_format_for_ui(cache_dir: Path | str) -> int:
    """``io.cache.detect_cache_format`` memoized per manifest stamp for drawing."""
    from ..io.cache import detect_cache_format

    stamp = _manifest_stamp(manifest_path(Path(cache_dir)))
    if stamp is None:
        # No manifest: legacy detection lists the directory; nothing to key on.
        return detect_cache_format(cache_dir)
    cached = _FORMAT_CACHE.get(stamp[0])
    if cached is not None and cached[0] == stamp:
        return cached[1]
    value = detect_cache_format(cache_dir)
    if len(_FORMAT_CACHE) > 64:
        _FORMAT_CACHE.clear()
    _FORMAT_CACHE[stamp[0]] = (stamp, value)
    return value


def _production_bake_action(path: Path) -> ProductionBakeAction:
    try:
        manifest = load_manifest(path)
    except LegacyCacheFormatError:
        # A 2.x cache is neither resumable nor invalid: say to re-bake, in the
        # one wording every consumer shows.
        return ProductionBakeAction(
            "RESTART", LEGACY_2X_CACHE_MESSAGE, destructive=True)
    except ValueError as exc:
        return ProductionBakeAction(
            "RESTART",
            f"Existing cache manifest is invalid: {exc}",
            destructive=True,
        )
    if manifest is None:
        return ProductionBakeAction("START", "No cache manifest exists")
    last = manifest.last_committed_frame
    if last is None or last < manifest.frame_end:
        return ProductionBakeAction(
            "RESUME",
            "Resume will verify signatures, checksums, and the last committed checkpoint",
        )
    return ProductionBakeAction(
        "RESTART",
        f"Existing production cache is complete through frame {manifest.frame_end}",
        destructive=True,
    )
