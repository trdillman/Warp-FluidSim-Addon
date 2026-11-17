"""Small filesystem helpers shared by the cache writers."""

from __future__ import annotations

import errno
import os
from pathlib import Path

_UNSUPPORTED_DIRECTORY_SYNC_ERRNOS = frozenset(
    code for code in (
        getattr(errno, name, None)
        for name in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS")
    ) if code is not None
)


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
    """Directory sync that tolerates filesystems which cannot do it.

    Some network and FUSE filesystems refuse ``fsync`` on a directory. The file
    itself is already durable and renamed, so a refusal must not fail the bake.
    """
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        if error.errno not in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
            raise
        return
    try:
        os.fsync(descriptor)
    except OSError as error:
        # Only "this filesystem cannot sync a directory" is tolerated. A real
        # I/O failure (EIO, ENOSPC...) must not be reported as success.
        if error.errno not in _UNSUPPORTED_DIRECTORY_SYNC_ERRNOS:
            raise
    finally:
        os.close(descriptor)
