"""A single admitted copy of a simulator surface into renderer-owned CUDA storage.

The renderer owns slot count, byte budget, eligibility and tracing retirement.
It creates a destination on the solver's current stream *after* reserving a slot,
then gives it to SurfacePipeline.copy_final_device_surface at the publication
boundary. A destination can be submitted once. Keep it until the last renderer
read retires; the independent source lease may be released as soon as copying
finishes. These geometry arrays contain positions and triangle indices only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from threading import RLock
from typing import ClassVar

import warp as wp

from ..runtime.cuda_driver import cuda_driver

_quarantine_lock = RLock()
_failed_retirements: dict[int, set] = {}


def drain_device_surface_quarantine(device):
    """Retry failed retirement outside interactive admission, on the owning device."""
    device = wp.get_device(device)
    with _quarantine_lock:
        retained = tuple(_failed_retirements.get(id(device), ()))
    for destination in retained:
        destination.drain()


class DeviceSurfaceUnavailable(RuntimeError):
    """An evaluated capture, later reservation, or new source stamp is required."""


@dataclass(frozen=True, eq=False)
class SurfaceDeviceArrays:
    """Final producer arrays plus inputs still needed by queued CUDA kernels."""

    positions: wp.array
    indices: wp.array
    dependencies: tuple = field(default=(), repr=False)


@dataclass(frozen=True)
class DeviceSurfaceStamp:
    """Geometry build identity. A SUBSTEP build can precede the completed-frame clock.

    Consumers bind outer publication metadata separately at the frame boundary.
    """

    owner: object = field(repr=False)
    build_key: tuple
    backend: str
    revision: int
    solver_frame: int
    solver_time: float
    waves_enabled: bool
    wave_frame: int
    wave_revision: int
    wave_settings: tuple


@dataclass(frozen=True, eq=False)
class DeviceSurfacePacket:
    """Active, read-only-by-contract views; wait on copy_complete before reading."""

    channels: ClassVar[frozenset[str]] = frozenset({"positions", "triangle_indices"})
    coordinate_space: ClassVar[str] = "WORLD"
    stamp: DeviceSurfaceStamp
    device: wp.Device
    positions: wp.array
    indices: wp.array
    vertex_capacity: int
    triangle_capacity: int
    producer_ready: wp.Event
    copy_complete: wp.Event
    _destination: DeviceSurfaceDestination = field(repr=False)

    @property
    def vertex_count(self):
        return self.positions.size

    @property
    def triangle_count(self):
        return self.indices.size // 3

    def retire_source(self):
        return self._destination.retire_source()


class DeviceSurfaceDestination:
    """Fresh exclusive backing arrays for a renderer-admitted, one-shot copy."""

    def __init__(self, device, *, vertex_capacity: int, triangle_capacity: int):
        self.device = wp.get_device(device)
        if not self.device.is_cuda or self.device.is_capturing:
            raise DeviceSurfaceUnavailable("device surface needs CUDA outside graph capture")
        self._check_admission()
        self._driver = cuda_driver()
        self._partial_failure = None
        nv, nt = index(vertex_capacity), index(triangle_capacity)
        if nv < 0 or nt < 0:
            raise ValueError("negative device surface capacity")
        self._stream = wp.get_stream(self.device)
        self._positions = wp.empty(nv, dtype=wp.vec3, device=self.device)
        self._indices = wp.empty(nt * 3, dtype=wp.int32, device=self.device)
        self._ready = wp.Event(self.device)
        self._copied = wp.Event(self.device)
        self._spent = False
        self._sources = None
        self._retire_event = None

    def _check_admission(self):
        with _quarantine_lock:
            if _failed_retirements.get(id(self.device)):
                raise DeviceSurfaceUnavailable("device surface retirement is quarantined")

    def _capture(self, device, stamp, materialize, *, producer_stream, known_counts=None):
        if self._spent:
            raise DeviceSurfaceUnavailable("device surface reservation was already submitted")
        self._spent = True
        self._check_admission()
        if (wp.get_device(device) != self.device or self._stream is not producer_stream
                or wp.get_stream(device) is not producer_stream):
            raise DeviceSurfaceUnavailable("reservation requires the same CUDA device and producer stream")
        if self.device.is_capturing:
            raise DeviceSurfaceUnavailable("device surface cannot run during graph capture")
        if known_counts is not None:
            nv, nt = known_counts
            if nv > self._positions.size or nt > self._indices.size // 3:
                raise DeviceSurfaceUnavailable("device surface exceeds reserved capacity")
        # The sparse extractor produces fresh owned output. The pipeline never
        # stores these arrays for mutation/reuse on a later frame.
        try:
            self._sources = source = materialize()
        except BaseException as exc:
            # A failure can follow an enqueued kernel but precede return of its
            # owners. Drain while the failing producer's traceback still owns
            # its local inputs; this is not part of successful submission.
            self._partial_failure = exc
            self.drain()
            raise
        positions, indices = source.positions, source.indices
        self._driver.record(wp, self.device, self._stream, self._ready)
        self._retire_event = self._ready
        for array, target, stride in (
            (positions, self._positions, 12),
            (indices, self._indices, 4),
        ):
            if (array.dtype != target.dtype or array.device != self.device
                    or array.ndim != 1 or not array.is_contiguous
                    or array.strides != (stride,)):
                raise DeviceSurfaceUnavailable("incompatible device surface array layout or device")
            if array.size > target.size:
                raise DeviceSurfaceUnavailable("device surface exceeds reserved capacity")
        if (indices.size and not positions.size) or indices.size % 3:
            raise DeviceSurfaceUnavailable("device surface has an incomplete triangle")

        # Copy on the producer stream: the ready event is ordered before both
        # copies without a host wait or a dependency on the tracing stream.
        # A partial copy also needs a fence. If recording fails, keep all
        # owners until drain() has synchronized this stream.
        self._retire_event = None
        try:
            if positions.size:
                wp.copy(self._positions, positions, count=positions.size, stream=self._stream)
            if indices.size:
                wp.copy(self._indices, indices, count=indices.size, stream=self._stream)
        finally:
            self._driver.record(wp, self.device, self._stream, self._copied)
            self._retire_event = self._copied
        # Warp forbids [:0] on a zero-capacity buffer.
        active_positions = self._positions
        active_indices = self._indices
        if positions.size < active_positions.size:
            active_positions = active_positions[:positions.size]
        if indices.size < active_indices.size:
            active_indices = active_indices[:indices.size]
        return DeviceSurfacePacket(
            stamp, self.device, active_positions, active_indices,
            self._positions.size, self._indices.size // 3,
            self._ready, self._copied, self)

    def _release_sources(self):
        self._sources = self._partial_failure = None
        key = id(self.device)
        with _quarantine_lock:
            held = _failed_retirements.get(key)
            if held is not None:
                held.discard(self)
                if not held:
                    del _failed_retirements[key]

    def retire_source(self):
        """Poll a successfully recorded event once, without a host wait."""
        if self._sources is None and self._partial_failure is None:
            return True
        if self._retire_event is None:
            return False
        try:
            complete = self._driver.completed(wp, self.device, self._retire_event)
        except BaseException:
            self._quarantine()
            raise
        if not complete:
            return False
        self._release_sources()
        return True

    def _quarantine(self):
        with _quarantine_lock:
            _failed_retirements.setdefault(id(self.device), set()).add(self)

    def drain(self):
        """Blocking teardown/error recovery, restricted to this submission."""
        if self._sources is None and self._partial_failure is None:
            return
        try:
            if self._retire_event is None:
                wp.synchronize_stream(self._stream)
            else:
                wp.synchronize_event(self._retire_event)
        except BaseException:
            # Refuse new reservations on this device. Keep private arrays and
            # partial producer traceback alive even if all callers drop them.
            self._quarantine()
            raise
        self._release_sources()

    def __del__(self):
        if (getattr(self, "_sources", None) is not None
                or getattr(self, "_partial_failure", None) is not None):
            self.drain()
