"""Renderer-owned, two-slot final device geometry for one CUDA device."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from operator import index
from threading import Lock, current_thread

import warp as wp

from ..runtime.cuda_driver import cuda_driver
from ..sim.device_surface import DeviceSurfaceDestination, DeviceSurfaceStamp


def _immutable_fact(value):
    if type(value) is tuple:
        return all(_immutable_fact(item) for item in value)
    return type(value) in (int, str, bool, type(None)) or (type(value) is float and isfinite(value))


@dataclass(frozen=True)
class SurfaceKey:
    scene_token: int
    owner_token: tuple[int | None, str]
    epoch: int
    source: str
    channel: str
    generation: int
    build_stamp: DeviceSurfaceStamp
    source_timeline_frame: int
    state_revision: int
    solver_frame: int
    simulation_time: float

    def __post_init__(self):
        if (
            self.channel != "GPU_FINAL"
            or self.source != "LIVE"
            or not isinstance(self.build_stamp, DeviceSurfaceStamp)
            or type(self.owner_token) is not tuple
            or len(self.owner_token) != 2
            or type(self.owner_token[0]) not in (int, type(None))
            or type(self.owner_token[1]) is not str
        ):
            raise ValueError("expected final live surface identity")
        stamp = self.build_stamp
        counters = (
            self.scene_token,
            self.epoch,
            self.generation,
            self.source_timeline_frame,
            self.state_revision,
            self.solver_frame,
            stamp.revision,
            stamp.solver_frame,
            stamp.wave_frame,
            stamp.wave_revision,
        )
        if (
            any(type(value) is not int for value in counters)
            or self.epoch < 1
            or self.generation < 1
            or type(stamp.owner) is not object
            or type(stamp.backend) is not str
            or type(stamp.waves_enabled) is not bool
            or not all(
                type(value) in (int, float) and isfinite(value)
                for value in (self.simulation_time, stamp.solver_time)
            )
            or not all(
                type(value) is tuple and _immutable_fact(value)
                for value in (stamp.build_key, stamp.wave_settings)
            )
        ):
            raise ValueError("surface keys must contain immutable source facts")


class _CudaOps:
    @staticmethod
    def reader(device, stream):
        if not isinstance(stream, wp.Stream) or stream.device != device or device.is_capturing:
            raise ValueError(
                "surface consumer needs an explicit stream on the pool device outside capture"
            )

    def producer(self, device, stream):
        self.reader(device, stream)
        if wp.get_stream(device) is not stream:
            raise ValueError("surface reservation requires the current producer stream")

    @staticmethod
    def allocate(device, nv, nt):
        return DeviceSurfaceDestination(device, vertex_capacity=nv, triangle_capacity=nt)

    @staticmethod
    def validate(packet, dest, key, nv, nt):
        from ..sim.device_surface import DeviceSurfacePacket

        if (
            not isinstance(packet, DeviceSurfacePacket)
            or packet._destination is not dest
            or packet.device != dest.device
            or packet.stamp != key.build_stamp
            or (packet.vertex_capacity, packet.triangle_capacity) != (nv, nt)
        ):
            raise ValueError("surface packet does not match the admitted producer copy")

    @staticmethod
    def record(device, stream):
        event = wp.Event(device)
        cuda_driver().record(wp, device, stream, event)
        return event

    @staticmethod
    def completed(device, event):
        return cuda_driver().completed(wp, device, event)

    @staticmethod
    def sync_reader(device, stream):
        wp.synchronize_stream(stream)


@dataclass(eq=False)
class _Slot:
    key: SurfaceKey
    size: int
    producer: object
    destination: object = None
    packet: object = None
    ready: bool = False
    submitting: bool = True
    failure: BaseException | None = None
    readers: list = field(default_factory=list)


class SurfaceLease:
    """One stream's open read rights. Keep open while an asset may issue future reads.

    finish() permanently surrenders those rights after the final read is enqueued.
    Pin all auxiliary acceleration/input owners via retain() until its fence retires.
    Each additional stream requires its own open lease before submitting work.
    """

    def __init__(self, pool, slot, stream, owners):
        self._pool, self._slot = pool, slot
        self._stream = stream
        self._owners = list(owners)
        self._event = None

    @property
    def packet(self):
        if self._slot is None:
            raise RuntimeError("surface lease has finished")
        self._pool._check_thread()
        return self._slot.packet

    @property
    def retired(self):
        """True only after checked pool retirement released this reader's owners."""
        return self._slot is None and self._stream is None

    def retain(self, owner):
        if self._slot is None:
            raise RuntimeError("surface lease has finished")
        self._pool._check_thread()
        self._owners.append(owner)

    def finish(self):
        if self._slot is None:
            return
        pool = self._pool
        pool._check_thread()
        self._slot = self._pool = None
        try:
            self._event = pool._ops.record(pool.device, self._stream)
        except BaseException:
            pool.poisoned = True
            raise

    def _release(self):
        self._owners.clear()
        self._event = self._stream = None


# Strong ownership survives callback/view loss and enforces the device-wide limit.
# Terminal close removes the entry only after every source and reader has retired.
_pools = {}
_registry_lock = Lock()


class SurfacePool:
    """Main/publication-thread owner. Selecting a newer key removes prior eligibility.

    The byte limit covers vec3f and triangle-index destination capacities. It does
    not estimate solver scratch, events, acceleration storage or allocator caching.
    Each accepted capture currently allocates a fresh one-shot destination.
    """

    def __init__(self, device, *, byte_limit, ops=None):
        self._ops = _CudaOps() if ops is None else ops
        self.device = wp.get_device(device) if ops is None else device
        if ops is None and not self.device.is_cuda:
            raise ValueError("surface pool requires CUDA")
        self._byte_limit = index(byte_limit)
        if self._byte_limit < 0:
            raise ValueError("negative geometry byte limit")
        self._thread = current_thread()
        self._key = self._watermark = None
        self._slots = []
        self._closing = self._closed = self.poisoned = False
        self.allocations = self.denials = self.peak_bytes = 0
        self._registry_key = self.device.context if ops is None else id(self.device)
        with _registry_lock:
            if self._registry_key in _pools:
                raise RuntimeError("device already has a surface pool")
            _pools[self._registry_key] = self

    def _check_thread(self):
        if current_thread() is not self._thread:
            raise RuntimeError("use the pool on its publication thread")

    @property
    def occupied(self):
        return len(self._slots)

    @property
    def reserved_bytes(self):
        return sum(slot.size for slot in self._slots)

    def select(self, key):
        self._check_thread()
        if not isinstance(key, SurfaceKey):
            raise TypeError("expected SurfaceKey")
        if self._closing or self.poisoned:
            return False
        if key == self._key:
            return True
        previous = self._watermark
        if previous is not None and (
            key.epoch < previous.epoch
            or (
                key.epoch == previous.epoch
                and (
                    key.scene_token != previous.scene_token
                    or key.owner_token != previous.owner_token
                    or key.source != previous.source
                    or key.channel != previous.channel
                    or key.generation <= previous.generation
                )
            )
        ):
            return False
        self._key = self._watermark = key
        return True

    def invalidate(self):
        self._check_thread()
        self._key = None  # Keep the watermark to reject a late replay.

    def submit(self, key, vertex_capacity, triangle_capacity, capture, *, producer_stream):
        """Reserve, then invoke capture(destination, expected_stamp) inline exactly once."""
        self._check_thread()
        nv, nt = index(vertex_capacity), index(triangle_capacity)
        if nv < 0 or nt < 0:
            raise ValueError("negative surface capacity")
        size = 12 * (nv + nt)
        if (
            self._closing
            or self.poisoned
            or self._key is None
            or key != self._key
            or self.occupied == 2
            or self.reserved_bytes + size > self._byte_limit
            or any(slot.key == key for slot in self._slots)
        ):
            self.denials += 1
            return False
        self._ops.producer(self.device, producer_stream)
        slot = _Slot(key, size, producer_stream)
        self._slots.append(slot)
        self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
        try:
            slot.destination = self._ops.allocate(self.device, nv, nt)
            self.allocations += 1
            slot.packet = capture(slot.destination, key.build_stamp)
            self._ops.validate(slot.packet, slot.destination, key, nv, nt)
        except BaseException as exc:
            slot.failure = exc
            self.poisoned = True
            raise
        finally:
            slot.submitting = False
        return True

    def poll(self):
        """Query each recorded fence at most once. No synchronization or extraction."""
        self._check_thread()
        if self._closing or self.poisoned:
            return
        try:
            for slot in tuple(self._slots):
                if slot.submitting:
                    continue
                if not slot.ready:
                    slot.ready = slot.destination.retire_source()
                for reader in tuple(slot.readers):
                    if reader._event is not None and self._ops.completed(
                        self.device, reader._event
                    ):
                        reader._release()
                        slot.readers.remove(reader)
                if slot.ready and slot.key != self._key and not slot.readers:
                    self._slots.remove(slot)
            self._collect_geometry_retirements()
        except BaseException:
            self.poisoned = True
            raise

    def _collect_geometry_retirements(self):
        budget = getattr(self, '_geometry_array_budget', None)
        if budget is not None:
            budget.collect_retired()

    def acquire(self, key, reader_stream, *, owners=()):
        self._check_thread()
        if self._closing or self.poisoned or self._key is None or key != self._key:
            return None
        for slot in self._slots:
            if slot.key == key and slot.ready:
                self._ops.reader(self.device, reader_stream)
                reader = SurfaceLease(self, slot, reader_stream, owners)
                slot.readers.append(reader)
                return reader
        return None

    def close(self):
        """Stop admission, then explicitly drain. Open read rights must finish first."""
        self._check_thread()
        if self._closed:
            return
        self._closing = True
        self.invalidate()
        for slot in self._slots:
            if slot.submitting or any(reader._slot is not None for reader in slot.readers):
                raise RuntimeError("surface submission or reader still open")
        try:
            for slot in tuple(self._slots):
                if slot.failure is not None:
                    # Also covers allocation errors and callbacks raising between
                    # enqueues. Their traceback retains any partial local owners.
                    self._ops.sync_reader(self.device, slot.producer)
                if slot.destination is not None:
                    slot.destination.drain()
                for reader in slot.readers:
                    self._ops.sync_reader(self.device, reader._stream)
                for reader in slot.readers:
                    reader._release()
                slot.readers.clear()
                slot.failure = slot.packet = slot.destination = slot.producer = None
                self._slots.remove(slot)
            self._collect_geometry_retirements()
        except BaseException:
            self.poisoned = True
            raise
        with _registry_lock:
            del _pools[self._registry_key]
        self._closed = True
