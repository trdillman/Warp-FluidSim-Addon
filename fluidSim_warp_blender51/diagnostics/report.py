"""Diagnostics and support-report writer.

The support envelope is deliberately serializable and bpy-free. Blender-facing
operators pass scene-specific values while this module owns runtime probing,
privacy redaction, and deterministic JSON output.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from pathlib import Path

from ..runtime.warp_runtime import probe_runtime
from ..version import BUILD_CHANNEL, EXTENSION_ID, VERSION_STRING


def redact_path(value: str | os.PathLike[str] | None, *, home: str | None = None) -> str:
    """Replace user-home prefixes in an absolute path with ``~``."""
    if value is None:
        return ""
    text = os.fspath(value)
    if not text or not os.path.isabs(os.path.expanduser(text)):
        return text

    candidates = []
    if home:
        candidates.append(os.path.abspath(os.path.expanduser(home)))
    else:
        candidates.append(os.path.abspath(os.path.expanduser("~")))
    normalized = os.path.normpath(text)
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        try:
            if os.path.commonpath((normalized, candidate)) == candidate:
                suffix = normalized[len(candidate):].lstrip("/\\")
                return "~" + (("/" + suffix.replace("\\", "/")) if suffix else "")
        except (ValueError, OSError):
            pass

    # Reports may include a module path from another user profile (for example,
    # a copied failure report). Redact common absolute home layouts as well.
    posix = re.sub(r"^/(?:home|Users)/[^/]+", "~", normalized)
    if posix != normalized:
        return posix
    windows = re.sub(
        r"^[A-Za-z]:[\\/]Users[\\/][^\\/]+",
        "~",
        normalized,
        flags=re.IGNORECASE,
    )
    return windows


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def cache_path_writable(value: str | os.PathLike[str] | None) -> bool:
    """Report whether the cache path or its nearest existing parent is writable."""
    if not value:
        return False
    path = Path(os.path.expanduser(os.fspath(value)))
    target = path if path.is_dir() else path.parent
    existing = _nearest_existing_parent(target)
    return bool(existing and existing.is_dir() and os.access(existing, os.W_OK))


def _detected_blender_version() -> str:
    try:
        import bpy

        return ".".join(str(value) for value in bpy.app.version[:3])
    except Exception:  # noqa: BLE001 - report remains valid outside Blender
        return "unavailable"


def build_report(
    profiler=None,
    frame_stats: list | None = None,
    benchmark: dict | None = None,
    validation: list | None = None,
    extra: dict | None = None,
    *,
    package_name: str | None = None,
    blender_version: str | None = None,
    selected_device: str = "",
    cache_path: str = "",
    scene_signature: str = "",
    settings_signature: str = "",
    effect_state: dict | None = None,
) -> dict:
    """Build a privacy-safe report for users, CI, and support triage."""
    runtime = probe_runtime()
    package = package_name or __package__.rsplit(".", 1)[0]
    report = {
        "tool": "fluidSim",
        "backend": "NVIDIA Warp / CUDA",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fluidsim": {
            "version": VERSION_STRING,
            "build_channel": BUILD_CHANNEL,
            "extension_id": EXTENSION_ID,
            "package": package,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "blender": blender_version or _detected_blender_version(),
        },
        "runtime": {
            "available": runtime.available,
            "message": runtime.message,
            "warp_version": runtime.warp_version,
            "module_file": redact_path(runtime.module_file),
            "cuda_driver_version": runtime.cuda_driver_version,
            "cuda_toolkit_version": runtime.cuda_toolkit_version,
            "selected_device": selected_device,
            "devices": [
                {
                    "alias": device.alias,
                    "name": device.name,
                    "memory_bytes": device.total_memory,
                }
                for device in runtime.cuda_devices
            ],
            "last_error": runtime.last_error,
        },
        "cache": {
            "path": redact_path(cache_path),
            "writable": cache_path_writable(cache_path),
        },
        "scene": {
            "scene_signature": scene_signature,
            "settings_signature": settings_signature,
        },
        # Compatibility field retained for existing report consumers.
        "warp": runtime.warp_version or "unavailable",
    }
    if effect_state is not None:
        report["custom_effects"] = effect_state
    if profiler is not None:
        report["compute"] = profiler.compute_profile()
        report["timers_average"] = profiler.summary()
    if frame_stats:
        report["frames"] = [
            item.as_dict() if hasattr(item, "as_dict") else dict(item)
            for item in frame_stats
        ]
    if benchmark:
        report["benchmark"] = benchmark
    if validation:
        report["validation"] = validation
    if extra:
        report.update(extra)
    return report


def write_report(path: str, report: dict) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"Output parent directory does not exist: {parent}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
