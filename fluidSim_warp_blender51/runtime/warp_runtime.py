"""Lazy, cached NVIDIA Warp and CUDA compatibility probing."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..version import CERTIFIED_WARP_VERSION, MIN_WARP_VERSION

EXPECTED_WARP_VERSION = CERTIFIED_WARP_VERSION
_UNAVAILABLE_DEVICE_ID = "__unavailable__"
_DEVNULL_STREAM = None


@dataclass(frozen=True)
class DeviceRecord:
    """Stable, UI-safe description of one CUDA device."""

    alias: str
    name: str
    total_memory: int


@dataclass(frozen=True)
class RuntimeProbe:
    """Serializable result of importing Warp and enumerating CUDA devices."""

    available: bool
    message: str
    warp_version: str = ""
    module_file: str = ""
    cuda_driver_version: int = 0
    cuda_toolkit_version: int = 0
    cuda_devices: tuple[DeviceRecord, ...] = ()
    last_error: str = ""
    warning: str = ""


def parse_warp_version(text: str) -> tuple[int, int, int] | None:
    """Return the numeric release of a Warp version string, or ``None``."""
    release = str(text).split("+", 1)[0].strip()
    parts = release.split(".")
    if len(parts) < 3:
        return None
    numbers = []
    for part in parts[:3]:
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            return None
        numbers.append(int(digits))
    return (numbers[0], numbers[1], numbers[2])


def warp_version_support(version: str) -> tuple[str, str]:
    """Classify a loaded Warp version against the certified range.

    Returns ``("certified" | "untested" | "unsupported", message)``. Versions
    below the floor or on another major line are unsupported; newer 1.x
    releases load with a warning because they have not been certified.
    """
    loaded = parse_warp_version(version)
    floor = parse_warp_version(MIN_WARP_VERSION)
    certified = parse_warp_version(CERTIFIED_WARP_VERSION)
    if floor is None or certified is None:
        raise RuntimeError("FluidSim Warp version constants are malformed")
    if loaded is None or loaded < floor or loaded[0] != certified[0]:
        return (
            "unsupported",
            f"FluidSim requires NVIDIA Warp {MIN_WARP_VERSION} or newer "
            f"{certified[0]}.x (certified: {CERTIFIED_WARP_VERSION}), but loaded "
            f"{version or 'an unknown version'}. Reinstall the FluidSim extension "
            "so Blender installs its bundled Warp, and remove any older Warp "
            "from Blender's Python user site-packages.",
        )
    if loaded > certified:
        return (
            "untested",
            f"Warp {version} is newer than the certified {CERTIFIED_WARP_VERSION} "
            "and has not been tested with FluidSim.",
        )
    return "certified", ""


def _cuda_version_int(value: Any) -> int:
    """Normalize Warp CUDA versions: ints pass through, ``(12, 9)`` -> 12090."""
    if isinstance(value, (tuple, list)):
        if len(value) < 2:
            return 0
        return int(value[0]) * 1000 + int(value[1]) * 10
    return int(value)


def ensure_host_streams() -> None:
    """Give console-less Blender GUI sessions safe text output streams."""
    global _DEVNULL_STREAM

    if sys.stdout is not None and sys.stderr is not None:
        return
    if _DEVNULL_STREAM is None or _DEVNULL_STREAM.closed:
        _DEVNULL_STREAM = open(  # noqa: SIM115 - process-lifetime stream
            os.devnull, "w", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _DEVNULL_STREAM
    if sys.stderr is None:
        sys.stderr = _DEVNULL_STREAM


def _import_warp():
    ensure_host_streams()
    import warp as wp

    return wp


def _safe_int_call(module: Any, name: str) -> int:
    function = getattr(module, name, None)
    if not callable(function):
        return 0
    try:
        return _cuda_version_int(function())
    except Exception:  # noqa: BLE001 - a partial diagnostic must remain available
        return 0


def _version_of(module: Any) -> str:
    version = getattr(module, "__version__", "")
    if version:
        return str(version)
    config = getattr(module, "config", None)
    return str(getattr(config, "version", "unknown"))


def _device_record(device: Any) -> DeviceRecord:
    alias = str(getattr(device, "alias", "") or device)
    name = str(getattr(device, "name", alias))
    memory = getattr(device, "total_memory", 0)
    if callable(memory):
        memory = memory()
    try:
        total_memory = int(memory)
    except (TypeError, ValueError):
        total_memory = 0
    return DeviceRecord(alias=alias, name=name, total_memory=total_memory)


@lru_cache(maxsize=1)
def _cached_probe() -> RuntimeProbe:
    try:
        wp = _import_warp()
        wp.init()
        version = _version_of(wp)
        module_file = str(getattr(wp, "__file__", ""))
        devices = tuple(
            _device_record(device)
            for device in wp.get_devices()
            if bool(getattr(device, "is_cuda", False))
        )
        driver = _safe_int_call(wp, "get_cuda_driver_version")
        toolkit = _safe_int_call(wp, "get_cuda_toolkit_version")
        support, support_message = warp_version_support(version)
        if support == "unsupported":
            return RuntimeProbe(
                available=False,
                message=support_message,
                warp_version=version,
                module_file=module_file,
                cuda_driver_version=driver,
                cuda_toolkit_version=toolkit,
                cuda_devices=devices,
                last_error=f"unsupported Warp version: {version}",
            )
        if not devices:
            return RuntimeProbe(
                available=False,
                message=(
                    "Warp loaded, but no CUDA device is available. Check the NVIDIA "
                    "GPU, driver, and whether Blender can access the device."
                ),
                warp_version=version,
                module_file=module_file,
                cuda_driver_version=driver,
                cuda_toolkit_version=toolkit,
            )
        return RuntimeProbe(
            available=True,
            message=" ".join(
                part for part in (
                    f"Detected {len(devices)} CUDA device(s).", support_message)
                if part
            ),
            warp_version=version,
            module_file=module_file,
            cuda_driver_version=driver,
            cuda_toolkit_version=toolkit,
            cuda_devices=devices,
            warning=support_message,
        )
    except Exception as exc:  # imported by Blender UI; retain actionable state
        error = f"{type(exc).__name__}: {exc}"
        return RuntimeProbe(
            available=False,
            message=(
                "FluidSim could not load its bundled Warp runtime: "
                f"{error}. Reinstall the FluidSim extension so Blender installs its bundled Warp runtime."
            ),
            last_error=error,
        )


def probe_runtime(*, force: bool = False) -> RuntimeProbe:
    """Return the cached Warp/CUDA probe, optionally refreshing it."""
    ensure_host_streams()
    if force:
        _cached_probe.cache_clear()
    return _cached_probe()


def runtime_status_for_device(
    device_alias: str,
    *,
    probe: RuntimeProbe | None = None,
) -> tuple[bool, str]:
    """Return readiness and one Blender-poll-safe status message."""
    state = probe if probe is not None else probe_runtime()
    if not state.available:
        return False, state.message
    by_alias = {device.alias: device for device in state.cuda_devices}
    device = by_alias.get(str(device_alias))
    if device is None:
        choices = ", ".join(by_alias) or "none"
        return (
            False,
            f"The selected CUDA device {device_alias!r} is unavailable. "
            f"Refresh detection or choose one of: {choices}.",
        )
    memory = device.total_memory / 1024**3
    status = f"Warp {state.warp_version} — {device.alias}: {device.name} ({memory:.1f} GiB)"
    if state.warning:
        status += " (untested Warp version)"
    return True, status


def cuda_device_items(*, force: bool = False) -> list[tuple[str, str, str]]:
    """Return stable Blender EnumProperty items for detected CUDA devices."""
    state = probe_runtime(force=force)
    if not state.available:
        return [(_UNAVAILABLE_DEVICE_ID, "No compatible CUDA device", state.message)]
    return [
        (
            device.alias,
            f"{device.alias} — {device.name}",
            f"{device.total_memory / 1024**3:.1f} GiB",
        )
        for device in state.cuda_devices
    ]
