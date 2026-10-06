"""Helpers for loading this checkout's Blender add-on during validation."""

from __future__ import annotations

import importlib
import os
import sys


ADDON_NAME = "fluidSim_warp_blender51"


def _ensure_warp_available(repo_root: str) -> None:
    try:
        import warp  # noqa: F401
        return
    except ImportError:
        pass
    appdata = os.environ.get("APPDATA")
    if appdata:
        for ver in ("5.1", "5.2", "5.0"):
            for pyver in ("python3.13", "python3.11", "python3.10"):
                candidate = os.path.join(
                    appdata,
                    "Blender Foundation",
                    "Blender",
                    ver,
                    "extensions",
                    ".local",
                    "lib",
                    pyver,
                    "site-packages",
                )
                if os.path.isdir(candidate) and candidate not in sys.path:
                    sys.path.append(candidate)
                    try:
                        import warp  # noqa: F401
                        return
                    except ImportError:
                        pass


def enable_repo_addon(repo_root, *, default_set=True, persistent=True):
    """Enable the add-on from repo_root, never from Blender's installed copy."""
    repo_root = os.path.abspath(repo_root)
    addon_root = os.path.join(repo_root, ADDON_NAME)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    _ensure_warp_available(repo_root)

    import addon_utils

    try:
        addon_utils.disable(ADDON_NAME, default_set=False)
    except Exception:  # noqa: BLE001
        pass

    for name in list(sys.modules):
        if name == ADDON_NAME or name.startswith(ADDON_NAME + "."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()

    module = importlib.import_module(ADDON_NAME)
    module_path = os.path.abspath(getattr(module, "__file__", ""))
    if os.path.commonpath([module_path, addon_root]) != addon_root:
        raise RuntimeError(
            f"{ADDON_NAME} loaded from {module_path}, expected {addon_root}"
        )

    addon_utils.enable(
        ADDON_NAME, default_set=default_set, persistent=persistent
    )
    enabled = sys.modules.get(ADDON_NAME)
    enabled_path = os.path.abspath(getattr(enabled, "__file__", ""))
    if os.path.commonpath([enabled_path, addon_root]) != addon_root:
        raise RuntimeError(
            f"{ADDON_NAME} enabled from {enabled_path}, expected {addon_root}"
        )
    return enabled
