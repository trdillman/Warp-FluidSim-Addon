"""Small filesystem helpers shared by the cache writers."""

from __future__ import annotations

import os
from pathlib import Path


def apply_umask_mode(path: Path | str) -> None:
    """Give a ``mkstemp`` file the mode a normal new file would have.

    ``mkstemp`` creates files as 0600. Left alone, a frame or checkpoint would be
    private to its owner while the manifest next to it honours the umask, so a
    shared project would show a valid manifest whose frames cannot be read.
    """
    if os.name == "nt":
        return
    umask = os.umask(0)
    os.umask(umask)
    try:
        os.chmod(path, 0o666 & ~umask)
    except OSError:
        pass


def fsync_directory(path: Path | str) -> None:
    """Best-effort directory sync.

    Some network and FUSE filesystems refuse ``fsync`` on a directory. The file
    itself is already durable and renamed, so a refusal must not fail the bake.
    """
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
