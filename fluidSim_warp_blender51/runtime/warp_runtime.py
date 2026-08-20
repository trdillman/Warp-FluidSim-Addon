"""Lazy, cached NVIDIA Warp and CUDA compatibility probing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

EXPECTED_WARP_VERSION = "1.16.0"
_UNAVAILABLE_DEVICE_ID = "__unavailable__"


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


def _import_warp():
    import warp as wp

    return wp


def _safe_int_call(module: Any, name: str) -> int:
    function = getattr(module, name, None)
    if not callable(function):
        return 0
    try:
        return int(function())
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
        if version != EXPECTED_WARP_VERSION:
            return RuntimeProbe(
                available=False,
                message=(
                    f"FluidSim requires external Warp {EXPECTED_WARP_VERSION}, "
                    f"but loaded {version}. Install the certified Warp runtime in Blender's Python environment."
                ),
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
            message=f"Detected {len(devices)} CUDA device(s).",
            warp_version=version,
            module_file=module_file,
            cuda_driver_version=driver,
            cuda_toolkit_version=toolkit,
            cuda_devices=devices,
        )
    except Exception as exc:  # imported by Blender UI; retain actionable state
        error = f"{type(exc).__name__}: {exc}"
        return RuntimeProbe(
            available=False,
            message=(
                "FluidSim could not load its external Warp dependency: "
                f"{error}. Install the certified Warp runtime in Blender's Python environment or provide it through the host environment."
            ),
            last_error=error,
        )


def probe_runtime(*, force: bool = False) -> RuntimeProbe:
    """Return the cached Warp/CUDA probe, optionally refreshing it."""
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
    return (
        True,
        f"Warp {state.warp_version} — {device.alias}: {device.name} ({memory:.1f} GiB)",
    )


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
