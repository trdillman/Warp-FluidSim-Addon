"""Filesystem paths usable by Windows hosts without long-path opt-in."""

import os
from pathlib import Path


def filesystem_path(path: Path) -> Path:
    """Preserve the location while allowing long local and UNC Windows paths."""
    if os.name != "nt":
        return path
    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)
