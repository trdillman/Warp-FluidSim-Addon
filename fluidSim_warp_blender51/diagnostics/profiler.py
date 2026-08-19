"""GPU/CPU profiler implementing the fluidSim_profiler::1.0 contract.

Provides per-phase timer tracks matching the synthesized TIMER_TRACK_ORDER,
compute/device metadata fields, and per-frame stats aggregation.

The profiler is bpy-free; the Blender layer renders its output as an overlay
and the diagnostics report writes it to JSON.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from ..sim.constants import TIMER_PHASE_SLOTS

PROFILER_ID = "fluidSim_profiler::1.0"

# Timer track order from the synthesized profiler contract.
TIMER_TRACK_ORDER = (
    ["dcc", "dcc_total", "dcc_other", "input", "solver", "script", "compute", "blit"]
    + [f"compute_{i}" for i in range(16)]
    + [f"blit_{i}" for i in range(16)]
    + [f"eventWait_{i}" for i in range(16)]
    + ["queue_commit", "queue_finish", "queue_compute", "queue_blit", "queue", "output"]
    + [f"debug_{i}" for i in range(16)]
    + ["total"]
)

PROFILE_COMPUTE_FIELDS = [
    "deviceName",
    "api",
    "usedHostBufferTotalMemoryTally",
    "usedBufferTotalMemoryTally",
    "usedBufferDeviceMemoryTally",
    "usedBufferSharedMemoryTally",
    "deviceHasUnifiedMemory",
    "deviceUsedSharedMemory",
]

PROFILE_FRAME_FIELDS = [
    "stepTime",
    "activeVoxels",
    "initializedVoxels",
    "totalVoxels",
    "activeParticles",
    "totalParticles",
    "whitewaterParticles",
    "substeps",
    "pressureIterations",
    "pressureResidual",
    "voxelsPerSecond",
    "voxelsPerFrame",
    "particlesPerSecond",
    "particlesPerFrame",
    "usedParticleAllocation",
    "usedVoxelAllocation",
    "activeInitializedVoxelRatio",
    "particleInitializedVoxelRatio",
    "particleVoxelFillRatio",
    "timers",
]

def _device_info(device: str) -> dict:
    """Collect device metadata via warp without keeping a module-level import."""
    try:
        import warp as wp

        dev = wp.get_device(device)
        free, total = (0, 0)
        try:
            free, total = wp.get_mem_info(dev)
        except Exception:
            pass
        return {
            "deviceName": dev.name,
            "api": "CUDA" if dev.is_cuda else "CPU",
            "totalMemory": int(total),
            "freeMemory": int(free),
            "deviceHasUnifiedMemory": bool(getattr(dev, "is_uma", False)),
            "arch": getattr(dev, "arch", 0),
        }
    except Exception:
        return {"deviceName": "unknown", "api": "unknown"}


@dataclass
class FrameProfile:
    """One frame worth of timer data (milliseconds)."""

    timers: dict = field(default_factory=dict)

    def as_ordered(self) -> dict:
        return {name: float(self.timers.get(name, 0.0)) for name in TIMER_TRACK_ORDER}


class Profiler:
    """Wall-clock + CUDA-synchronized phase profiler.

    ``track(phase)`` wraps a solver phase; on CUDA devices the region is
    bracketed with ``wp.synchronize_device`` so timings reflect GPU work.
    Set ``sync=False`` for lower-overhead approximate timings.
    """

    def __init__(self, device: str = "cuda:0", sync: bool = True, enabled: bool = True):
        self.profiler_id = PROFILER_ID
        self.device = device
        self.sync = sync
        self.enabled = enabled
        self.frame = FrameProfile()
        self.history: list[dict] = []
        self.max_history = 600
        self._wp = None
        self._buffer_tally = 0
        self._host_tally = 0

    def _warp(self):
        if self._wp is None:
            import warp as wp

            self._wp = wp
        return self._wp

    # -- buffer accounting -------------------------------------------------
    def tally_buffer(self, nbytes: int, host: bool = False) -> None:
        if host:
            self._host_tally += int(nbytes)
        else:
            self._buffer_tally += int(nbytes)

    # -- timing ------------------------------------------------------------
    @contextmanager
    def track(self, phase: str):
        if not self.enabled:
            yield
            return
        try:
            slot = f"compute_{TIMER_PHASE_SLOTS[phase]}"
        except KeyError:
            raise KeyError(f"unknown profiler phase: {phase}") from None
        wp = self._warp()
        if self.sync and str(self.device).startswith("cuda"):
            wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync and str(self.device).startswith("cuda"):
                wp.synchronize_device(self.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timers = self.frame.timers
            timers[slot] = timers.get(slot, 0.0) + elapsed_ms
            timers["compute"] = timers.get("compute", 0.0) + elapsed_ms
            timers["solver"] = timers.get("solver", 0.0) + elapsed_ms

    @contextmanager
    def track_track(self, track_name: str):
        """Time an arbitrary named track (e.g. 'input', 'output', 'dcc')."""
        if not self.enabled:
            yield
            return
        if track_name not in TIMER_TRACK_ORDER:
            raise KeyError(f"unknown profiler track: {track_name}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.frame.timers[track_name] = (
                self.frame.timers.get(track_name, 0.0) + elapsed_ms
            )

    # -- frame lifecycle ----------------------------------------------------
    def begin_frame(self) -> None:
        self.frame = FrameProfile()

    def collect(self) -> dict:
        """Finalize and return this frame's ordered timer dict."""
        timers = self.frame.as_ordered()
        timers["total"] = timers.get("solver", 0.0) + timers.get("dcc_total", 0.0)
        self.history.append(timers)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return timers

    # -- reporting -----------------------------------------------------------
    def compute_profile(self) -> dict:
        info = _device_info(self.device)
        used_device = max(0, info.get("totalMemory", 0) - info.get("freeMemory", 0))
        return {
            "profiler": self.profiler_id,
            "deviceName": info.get("deviceName", "unknown"),
            "api": info.get("api", "unknown"),
            "usedHostBufferTotalMemoryTally": self._host_tally,
            "usedBufferTotalMemoryTally": self._buffer_tally + self._host_tally,
            "usedBufferDeviceMemoryTally": self._buffer_tally or used_device,
            "usedBufferSharedMemoryTally": 0,
            "deviceHasUnifiedMemory": info.get("deviceHasUnifiedMemory", False),
            "deviceUsedSharedMemory": 0,
        }

    def summary(self, last_n: int = 60) -> dict:
        """Average timers over the last N frames for overlay display."""
        last_n = int(last_n)
        if last_n <= 0:
            raise ValueError("last_n must be positive")
        window = self.history[-last_n:]
        if not window:
            return {}
        keys = [k for k in TIMER_TRACK_ORDER if any(f.get(k, 0.0) > 0.0 for f in window)]
        return {
            k: sum(f.get(k, 0.0) for f in window) / len(window) for k in keys
        }

    def to_json(self) -> str:
        return json.dumps(
            {"compute": self.compute_profile(), "frames": self.history}, indent=2
        )
