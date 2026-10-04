"""Conservative GPU allocation estimates and live resource sampling."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import subprocess
import time

from .constants import LEAF_VOLUME
from .quality_contract import resolve_quality


BASE_BYTES = 256 * 1024**2
SIM_BYTES_PER_VOXEL = 356
SURFACE_TENSION_BYTES_PER_VOXEL = 24
# SurfacePipeline._ensure_band_buffers() lazily allocates one int32
# (_band_closest) plus two vec3 arrays (_band_points, _band_points_tmp)
# per voxel once surface tension is active, alongside the solver's own
# normal buffers accounted for above.
SURFACE_TENSION_PROPAGATION_BYTES_PER_VOXEL = 28
RENDER_BYTES_PER_VOXEL = 16
SURFACE_DENSE_BYTES_PER_VOXEL = 40
PARTICLE_BYTES = 128
WHITEWATER_BYTES = 76
WHITEWATER_EMISSION_BYTES_PER_LIQUID = 8  # candidate flag + exclusive offset
# Live table + candidate table + candidate flag/expand scratch (int32 each).
TOPOLOGY_BYTES_PER_LEAF_CELL = 16
# Live and candidate vec3i coordinate arrays.
TOPOLOGY_COORD_BYTES_PER_LEAF = 24


def release_cuda_memory(device_name: str | None) -> None:
    """Collect released solver cycles and return pooled CUDA memory to the OS."""
    gc.collect()
    if not device_name:
        return

    import warp as wp

    device = wp.get_device(device_name)
    if not device.is_cuda:
        return
    old_threshold = None
    try:
        if getattr(device, "is_mempool_supported", False):
            old_threshold = wp.get_mempool_release_threshold(device)
            wp.set_mempool_release_threshold(device, 0)
        wp.synchronize_device(device)
    finally:
        if old_threshold is not None:
            wp.set_mempool_release_threshold(device, old_threshold)


def is_memory_capacity_error(message: str) -> bool:
    text = str(message).lower()
    return any(marker in text for marker in (
        "memory safeguard",
        "leaf capacity exceeded",
        "out of memory",
        "memory allocation",
        "cuda_error_out_of_memory",
        "cuda error 2",
    ))


def safe_memory_budget(
        total_bytes: int, free_bytes: int, resident_bytes: int = 0) -> int:
    """Leave room for Blender, the display driver, and allocator variance.

    When a transaction must retain owned GPU allocations, return a peak budget:
    the resident estimate is already committed, while only 90% of currently
    free VRAM is available to the candidate.
    """
    resident_bytes = max(0, int(resident_bytes))
    return int(min(
        max(0, total_bytes) * 0.85,
        resident_bytes + max(0, free_bytes) * 0.90,
    ))


def _windows_system_pressure() -> tuple[float, float]:
    if os.name != "nt":
        return 0.0, 0.0

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("avail_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_extended", ctypes.c_ulonglong),
        ]

    def cpu_times():
        idle = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        return idle.value, kernel.value, user.value

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    first = cpu_times()
    time.sleep(0.05)
    second = cpu_times()
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1] + second[2] - first[2]
    cpu = 0.0 if total_delta <= 0 else 100.0 * (1.0 - idle_delta / total_delta)
    return max(0.0, min(cpu, 100.0)), float(memory.memory_load)


def vram_snapshot(device_name: str) -> dict:
    """Sample VRAM only: no subprocess, no sleep.

    The capacity decision reads ``gpu_total_bytes`` and ``gpu_free_bytes`` and
    nothing else, so it needs none of the telemetry that costs a 50 ms sleep
    plus an nvidia-smi launch. Callers reachable from a frame-change handler
    use this; the panel's telemetry readout, which the user asked for and can
    wait for, uses ``resource_snapshot``.
    """
    import warp as wp

    device = wp.get_device(device_name)
    return {
        "gpu_total_bytes": int(getattr(device, "total_memory", 0)),
        "gpu_free_bytes": int(getattr(device, "free_memory", 0)),
        "gpu_utilization": 0.0,
        "cpu_utilization": 0.0,
        "ram_utilization": 0.0,
    }


def resource_snapshot(device_name: str) -> dict:
    """Sample live VRAM, GPU utilization, CPU load, and system RAM pressure.

    Blocks for at least 50 ms and launches nvidia-smi with a 2 s timeout, so it
    must not sit on the frame-change handler path -- use ``vram_snapshot``.
    """
    import warp as wp

    snapshot = vram_snapshot(device_name)
    device = wp.get_device(device_name)
    gpu_utilization = 0.0
    if device.is_cuda:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device.ordinal}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            gpu_utilization = float(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    cpu, ram = _windows_system_pressure()
    snapshot["gpu_utilization"] = gpu_utilization
    snapshot["cpu_utilization"] = cpu
    snapshot["ram_utilization"] = ram
    return snapshot


def _value(settings, name, default):
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _resolved_quality(settings):
    lo = _value(settings, "domain_min", (-2.0, -2.0, 0.0))
    hi = _value(settings, "domain_max", (2.0, 2.0, 4.0))
    return resolve_quality(settings, (lo, hi))


def _domain_leaf_cells(settings) -> int:
    return _resolved_quality(settings).total_domain_leaf_cells


def _surface_voxel_size(settings) -> float:
    return _resolved_quality(settings).surface_voxel_size


def _surface_dense_voxels(settings, h: float) -> int:
    lo = _value(settings, "domain_min", (-2.0, -2.0, 0.0))
    hi = _value(settings, "domain_max", (2.0, 2.0, 4.0))
    max_dim = max(8, int(_value(settings, "surface_max_dense_dim", 512)))
    dims = []
    for i in range(3):
        voxels = max(8, math.ceil(max(0.0, float(hi[i]) - float(lo[i])) / h))
        padded = math.ceil(voxels / 32.0) * 32
        dims.append(min(max_dim, padded))
    return int(dims[0] * dims[1] * dims[2])


def estimate_solver_bytes(settings) -> int:
    """Return a deliberately conservative peak-device-memory estimate."""
    leaf_cells = _domain_leaf_cells(settings)
    leaves = min(max(1, int(_value(settings, "leaf_capacity", 8192))), leaf_cells)
    particles = max(0, int(_value(settings, "max_particles", 0)))
    whitewater = max(0, int(_value(settings, "max_whitewater", 0)))
    estimate = (
        BASE_BYTES
        + leaves * LEAF_VOLUME * SIM_BYTES_PER_VOXEL
        + particles * (
            PARTICLE_BYTES + WHITEWATER_EMISSION_BYTES_PER_LIQUID)
        + whitewater * WHITEWATER_BYTES
        + leaf_cells * TOPOLOGY_BYTES_PER_LEAF_CELL
        + leaves * TOPOLOGY_COORD_BYTES_PER_LEAF
    )
    surface_enabled = bool(_value(settings, "surface_enabled", True))
    surface_h = _surface_voxel_size(settings)
    sim_h = max(1.0e-4, float(_value(settings, "voxel_size", 0.1)))
    if (bool(_value(settings, "surface_tension_enabled", False))
            and float(_value(settings, "surface_tension_coefficient", 0.0)) > 0.0):
        estimate += (
            leaves * LEAF_VOLUME
            * (SURFACE_TENSION_BYTES_PER_VOXEL + SURFACE_TENSION_PROPAGATION_BYTES_PER_VOXEL)
        )
    if surface_enabled:
        estimate += _surface_dense_voxels(settings, surface_h) * SURFACE_DENSE_BYTES_PER_VOXEL
    if surface_enabled and abs(surface_h - sim_h) > 1.0e-8:
        scale = max(1.0, float(_value(settings, "surface_leaf_capacity_scale", 8.0)))
        render_leaf_cells = _domain_leaf_cells({
            "domain_min": _value(settings, "domain_min", (-2.0, -2.0, 0.0)),
            "domain_max": _value(settings, "domain_max", (2.0, 2.0, 4.0)),
            "voxel_size": _surface_voxel_size(settings),
        })
        render_leaves = min(render_leaf_cells, max(1, math.ceil(
            max(1, int(_value(settings, "leaf_capacity", 8192))) * scale)))
        estimate += render_leaves * LEAF_VOLUME * RENDER_BYTES_PER_VOXEL
        estimate += render_leaf_cells * TOPOLOGY_BYTES_PER_LEAF_CELL
    return int(estimate)
