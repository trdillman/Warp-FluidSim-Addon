"""Conservative GPU allocation estimates and one-click safe settings."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import subprocess
import time

from .quality_contract import resolve_quality


LEAF_VOLUME = 512
BASE_BYTES = 256 * 1024**2
SIM_BYTES_PER_VOXEL = 356
RENDER_BYTES_PER_VOXEL = 16
SURFACE_DENSE_BYTES_PER_VOXEL = 40
PARTICLE_BYTES = 128
WHITEWATER_BYTES = 76
WHITEWATER_EMISSION_BYTES_PER_LIQUID = 8  # candidate flag + exclusive offset
# Live table + candidate table + candidate flag/expand scratch (int32 each).
TOPOLOGY_BYTES_PER_LEAF_CELL = 16
# Live and candidate vec3i coordinate arrays.
TOPOLOGY_COORD_BYTES_PER_LEAF = 24
SPECIES_CACHE_BYTES_PER_PARTICLE = 4 + 16


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


def assess_configuration(settings, resources: dict) -> dict:
    estimate = estimate_solver_bytes(settings)
    budget = safe_memory_budget(
        int(resources.get("gpu_total_bytes", 0)),
        int(resources.get("gpu_free_bytes", 0)),
    )
    return {
        "safe": estimate <= budget,
        "estimated_bytes": estimate,
        "budget_bytes": budget,
        "resources": dict(resources),
        "recommended": (
            {} if estimate <= budget
            else recommend_safe_settings(settings, budget, resources)
        ),
    }


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

    ``assess_configuration`` reads ``gpu_total_bytes`` and ``gpu_free_bytes``
    and nothing else, so the capacity decision needs none of the telemetry that
    costs a 50 ms sleep plus an nvidia-smi launch. Callers reachable from a
    frame-change handler use this; the panel's telemetry readout, which the
    user asked for and can wait for, uses ``resource_snapshot``.
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


def estimate_species_transport_bytes(settings) -> int:
    """Exact persistent and scratch device demand of species transport."""
    if not bool(_value(settings, "species_enabled", False)):
        return 0
    from .species import SpeciesTransport

    leaf_cells = _domain_leaf_cells(settings)
    leaves = min(max(1, int(_value(settings, "leaf_capacity", 8192))), leaf_cells)
    particles = max(0, int(_value(settings, "max_particles", 0)))
    return int(SpeciesTransport.estimate_bytes(
        particles, leaves * LEAF_VOLUME, True))


def estimate_species_cache_staging_bytes(settings) -> int:
    """Peak represented-volume and concentration cache staging demand."""
    if not bool(_value(settings, "species_enabled", False)):
        return 0
    particles = max(0, int(_value(settings, "max_particles", 0)))
    return particles * SPECIES_CACHE_BYTES_PER_PARTICLE


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
    if surface_enabled:
        estimate += _surface_dense_voxels(settings, surface_h) * SURFACE_DENSE_BYTES_PER_VOXEL
    if surface_enabled and abs(surface_h - sim_h) > 1.0e-8:
        scale = max(1.0, float(_value(settings, "surface_leaf_capacity_scale", 1.0)))
        render_leaf_cells = _domain_leaf_cells({
            "domain_min": _value(settings, "domain_min", (-2.0, -2.0, 0.0)),
            "domain_max": _value(settings, "domain_max", (2.0, 2.0, 4.0)),
            "voxel_size": _surface_voxel_size(settings),
        })
        render_leaves = min(render_leaf_cells, max(1, math.ceil(
            max(1, int(_value(settings, "leaf_capacity", 8192))) * scale)))
        estimate += render_leaves * LEAF_VOLUME * RENDER_BYTES_PER_VOXEL
        estimate += render_leaf_cells * TOPOLOGY_BYTES_PER_LEAF_CELL
    estimate += estimate_species_transport_bytes(settings)
    estimate += estimate_species_cache_staging_bytes(settings)
    return int(estimate)


def recommend_safe_settings(
    settings, budget_bytes: int, machine: dict | None = None
) -> dict[str, int | float]:
    """Fit current settings under live resource pressure with minimal changes."""
    machine = machine or {}
    current = {
        "leaf_capacity": max(64, int(_value(settings, "leaf_capacity", 8192))),
        "max_particles": max(100_000, int(_value(settings, "max_particles", 100_000))),
        "max_whitewater": max(50_000, int(_value(settings, "max_whitewater", 50_000))),
        "voxel_size": max(1.0e-4, float(_value(settings, "voxel_size", 0.1))),
        "particles_per_voxel": max(1.0, float(_value(settings, "particles_per_voxel", 8.0))),
        "topology_expand": max(0, int(_value(settings, "topology_expand", 1))),
        "surface_resolution_mode": int(_value(settings, "surface_resolution_mode", 0)),
        "surface_resolution_multiplier": float(_value(settings, "surface_resolution_multiplier", 2.0)),
        "surface_leaf_capacity_scale": float(_value(settings, "surface_leaf_capacity_scale", 1.0)),
        "surface_max_dense_dim": int(_value(settings, "surface_max_dense_dim", 256)),
        "surface_enabled": bool(_value(settings, "surface_enabled", True)),
        "surface_voxel_size": float(_value(settings, "surface_voxel_size", 0.0)),
        "species_enabled": bool(_value(settings, "species_enabled", False)),
        "species_diffusivity": float(_value(settings, "species_diffusivity", 0.0)),
        "species_max_diffusion_substeps": int(_value(
            settings, "species_max_diffusion_substeps", 16)),
        "domain_min": tuple(_value(settings, "domain_min", (-2.0, -2.0, 0.0))),
        "domain_max": tuple(_value(settings, "domain_max", (2.0, 2.0, 4.0))),
    }
    budget = max(512 * 1024**2, int(budget_bytes))
    high_gpu = float(machine.get("gpu_utilization", 0.0)) >= 85.0
    high_cpu = float(machine.get("cpu_utilization", 0.0)) >= 85.0
    high_ram = float(machine.get("ram_utilization", 0.0)) >= 85.0
    if estimate_solver_bytes(settings) <= budget and not (high_gpu or high_cpu or high_ram):
        return {}

    values = dict(current)
    if high_gpu:
        values["voxel_size"] *= 1.15
        values["particles_per_voxel"] = max(2.0, values["particles_per_voxel"] * 0.75)
        values["topology_expand"] = min(values["topology_expand"], 1)
        values["max_particles"] = max(100_000, int(values["max_particles"] * 0.8))
        values["max_whitewater"] = max(50_000, int(values["max_whitewater"] * 0.7))
    if high_cpu:
        values["voxel_size"] *= 1.1
        values["particles_per_voxel"] = max(2.0, values["particles_per_voxel"] * 0.8)
        values["topology_expand"] = min(values["topology_expand"], 1)
    if high_ram:
        values["max_particles"] = max(100_000, int(values["max_particles"] * 0.75))
        values["max_whitewater"] = max(50_000, int(values["max_whitewater"] * 0.5))

    if estimate_solver_bytes(values) > budget or high_gpu or high_cpu:
        values["surface_resolution_mode"] = 0
        values["surface_leaf_capacity_scale"] = 1.0
        values["surface_max_dense_dim"] = min(256, values["surface_max_dense_dim"])
    while estimate_solver_bytes(values) > budget:
        effective_leaves = min(values["leaf_capacity"], _domain_leaf_cells(values))
        candidates = {
            "voxel_size": (
                effective_leaves * LEAF_VOLUME * SIM_BYTES_PER_VOXEL,
                None,
            ),
            "max_particles": (values["max_particles"] * PARTICLE_BYTES, 100_000),
            "max_whitewater": (values["max_whitewater"] * WHITEWATER_BYTES, 50_000),
        }
        reducible = [
            (cost, name, floor) for name, (cost, floor) in candidates.items()
            if name == "voxel_size" or values[name] > floor
        ]
        if not reducible:
            break
        _cost, name, floor = max(reducible)
        if name == "voxel_size":
            values[name] *= 2.0 ** (1.0 / 3.0)
        else:
            values[name] = max(floor, values[name] // 2)
    return {
        name: value for name, value in values.items()
        if name in current and value != current[name]
    }
