"""Locate Blender-compatible NVIDIA Warp installs before solver imports."""

from __future__ import annotations

import importlib.util
import os
import site
import subprocess
import sys
from pathlib import Path


def _find_warp():
    return importlib.util.find_spec("warp")


def _append_existing(candidates: list[Path], path: Path | str | None) -> None:
    if not path:
        return
    candidate = Path(path)
    if candidate.exists() and candidate not in candidates:
        candidates.append(candidate)


def _blender_version() -> str | None:
    try:
        import bpy  # type: ignore

        return ".".join(str(part) for part in bpy.app.version[:2])
    except Exception:  # noqa: BLE001
        return None


def _blender_module_candidates() -> list[Path]:
    candidates: list[Path] = []

    try:
        import bpy  # type: ignore

        for script_path in bpy.utils.script_paths():
            _append_existing(candidates, Path(script_path) / "addons" / "modules")
    except Exception:  # noqa: BLE001
        pass

    version = _blender_version()
    appdata = os.environ.get("APPDATA")
    if version and appdata:
        _append_existing(
            candidates,
            Path(appdata)
            / "Blender Foundation"
            / "Blender"
            / version
            / "scripts"
            / "addons"
            / "modules",
        )
    return candidates


def _python_site_candidates() -> list[Path]:
    candidates: list[Path] = []
    for getter in (site.getusersitepackages, site.getsitepackages):
        try:
            paths = getter()
        except Exception:  # noqa: BLE001
            continue
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            _append_existing(candidates, path)
    return candidates


def _path_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    current = f"{sys.version_info.major}.{sys.version_info.minor}"

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        exe = Path(entry) / ("python.exe" if os.name == "nt" else "python")
        if not exe.exists() or exe in seen:
            continue
        seen.add(exe)
        code = (
            "import site,sys;"
            "print(f'{sys.version_info.major}.{sys.version_info.minor}');"
            "paths=[];"
            "paths.append(site.getusersitepackages());"
            "paths.extend(site.getsitepackages());"
            "print('\\n'.join(paths))"
        )
        try:
            result = subprocess.run(
                [str(exe), "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or not lines or lines[0] != current:
            continue
        for path in lines[1:]:
            _append_existing(candidates, path)
    return candidates


def _contains_warp(path: Path) -> bool:
    return (path / "warp" / "__init__.py").exists() or (path / "warp.py").exists()


def ensure_warp_discoverable() -> str | None:
    """Add the first compatible Warp module directory to sys.path if needed."""
    spec = _find_warp()
    if spec is not None:
        return spec.origin

    # Evaluated one source at a time: _path_python_candidates spawns a
    # subprocess per python.exe on PATH, so it must not run when a cheaper
    # source already yields a usable Warp.
    for source in (_blender_module_candidates, _python_site_candidates,
                   _path_python_candidates):
        for candidate in source():
            if not _contains_warp(candidate):
                continue
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            importlib.invalidate_caches()
            spec = _find_warp()
            if spec is not None:
                return spec.origin
    return None
