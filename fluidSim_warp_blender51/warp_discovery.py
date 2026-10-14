"""Explicit developer-only override for an external Warp checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_OVERRIDE_ENV = "FLUIDSIM_ALLOW_EXTERNAL_WARP"


def developer_override_allowed() -> bool:
    """Return whether the process explicitly opted into external Warp code."""
    return os.environ.get(_OVERRIDE_ENV) == "1"


def enable_developer_override(path: str) -> bool:
    """Add a local module directory only after explicit process opt-in.

    Normal extension registration never calls this function. It exists solely
    for trusted local development and performs no process, environment, site,
    package-manager, or network discovery.
    """
    if not developer_override_allowed():
        return False
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        return False
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)
    return True


def developer_override_active(path: str) -> bool:
    """Recognize the authorized configured copy already loaded in this process.

    Merely opting in or editing a preference cannot exempt an unrelated Warp.
    This is a presentation check only; the runtime version gate still applies,
    and the installer never gains ownership of an external directory.
    """
    if not developer_override_allowed() or not str(path or "").strip():
        return False
    candidate = Path(path).expanduser().resolve()
    loaded = getattr(sys.modules.get("warp"), "__file__", None)
    return bool(
        candidate.is_dir()
        and str(candidate) in sys.path
        and loaded
        and Path(loaded).resolve() == (candidate / "warp" / "__init__.py").resolve()
    )
