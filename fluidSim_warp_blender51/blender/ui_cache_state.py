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


def production_bake_action(cache_dir: Path | str | None) -> ProductionBakeAction:
    """Return START, RESUME, or RESTART for one mutually exclusive UI action."""
    if cache_dir is None:
        return ProductionBakeAction("START", "No production cache exists")
    base = Path(cache_dir)
    path = manifest_path(base)
    if not path.is_file():
        return ProductionBakeAction("START", "No cache manifest exists")
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
