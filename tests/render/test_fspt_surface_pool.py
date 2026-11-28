"""Fence-controlled ownership tests. No CUDA behavior is inferred from these doubles."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from fluidSim_warp_blender51.fspt.surface_pool import SurfaceKey, SurfacePool
from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceStamp


class Fence:
    done = False
    broken = False

    def query(self):
        if self.broken:
            raise RuntimeError("query")
        return self.done


class Destination:
    def __init__(self):
        self.copy = Fence()
        self.source = object()
        self.broken = False

    def retire_source(self):
        if self.copy.query():
            self.source = None
        return self.source is None

    def drain(self):
        if self.broken:
            raise RuntimeError("drain")
        self.source = None


class Ops:
    def __init__(self):
        self.allocations = 0
        self.record_fault = False
        self.sync_fault = False

    def producer(self, device, stream):
        assert stream is device

    def allocate(self, device, nv, nt):
        self.allocations += 1
        return Destination()

    def validate(self, packet, dest, key, nv, nt):
        assert packet.destination is dest and packet.stamp == key.build_stamp

    def reader(self, device, stream):
        assert stream.device is device

    def record(self, device, stream):
        if self.record_fault:
            raise RuntimeError("record")
        return stream.fence

    def completed(self, device, event):
        return event.query()

    def sync_reader(self, device, stream):
        if self.sync_fault:
            raise RuntimeError("sync")
        if stream is not device:
            stream.fence.done = True


@pytest.fixture
def setup():
    ops, device = Ops(), object()
    pool = SurfacePool(device, byte_limit=96, ops=ops)
    key = SurfaceKey(
        3,
        (1, "output"),
        1,
        "LIVE",
        "GPU_FINAL",
        1,
        DeviceSurfaceStamp(object(), (4,), "sparse", 1, 1, 0.0, False, 0, 0, ()),
        1,
        4,
        1,
        0.0,
    )
    yield pool, key, ops, lambda: SimpleNamespace(device=device, fence=Fence())
    pool.close()


def submit(pool, key, nv=3, nt=1):
    captured = []

    def capture(destination, stamp):
        captured.append(destination)
        return SimpleNamespace(destination=destination, stamp=stamp)

    assert pool.select(key)
    assert pool.submit(key, nv, nt, capture, producer_stream=pool.device)
    return captured[0]


def test_two_retired_generations_deny_before_materializing_until_all_readers_complete(
    setup,
):
    pool, key, ops, stream = setup
    a = submit(pool, key)
    assert pool.acquire(key, stream()) is None
    pool.poll()
    assert a.source is not None
    a.copy.done = True
    pool.poll()
    assert a.source is None
    ra, rb = stream(), stream()

    class Owner:
        pass

    owner = Owner()
    ref = weakref.ref(owner)
    first = pool.acquire(key, ra, owners=(owner,))
    del owner
    second = pool.acquire(key, rb)
    first.finish()
    second.finish()
    new = replace(key, generation=2)
    b = submit(pool, new)
    latest = replace(key, epoch=2, generation=1)
    assert pool.select(latest)
    blocked = Mock(side_effect=AssertionError("full pool materialized"))
    assert not pool.submit(latest, 0, 0, blocked, producer_stream=pool.device)
    ra.fence.done = True
    pool.poll()
    assert pool.occupied == 2
    assert not pool.submit(latest, 3, 1, blocked, producer_stream=pool.device)
    assert ops.allocations == 2
    rb.fence.done = True
    pool.poll()
    assert pool.occupied == 1 and ref() is None
    submit(pool, latest)
    b.copy.done = True
    pool.poll()
    assert pool.acquire(new, stream()) is None
    assert not pool.select(new)
    blocked.assert_not_called()


def test_capacity_empty_and_no_consumer(setup):
    pool, key, ops, stream = setup
    capture = Mock(side_effect=AssertionError("denied submission extracted"))
    assert not pool.submit(key, 0, 0, capture, producer_stream=pool.device)
    assert pool.select(key)
    assert not pool.submit(key, 6, 3, capture, producer_stream=pool.device)
    dest = submit(pool, key, 0, 0)
    dest.copy.done = True
    pool.poll()
    lease = pool.acquire(key, stream())
    assert lease.packet.stamp == key.build_stamp
    lease.finish()
    pool.invalidate()
    assert not pool.select(key)
    assert not pool.submit(key, 0, 0, capture, producer_stream=pool.device)
    assert ops.allocations == 1
    capture.assert_not_called()


@pytest.mark.parametrize("fault", ["capture", "query", "record", "read_query"])
def test_fault_keeps_owners_and_refuses_reuse_until_successful_teardown(setup, fault):
    pool, key, ops, stream = setup
    reader = stream()

    class Owner:
        pass

    extra = Owner()
    sentinel = weakref.ref(extra)
    if fault == "capture":

        def broken(dest, stamp):
            refs.append(weakref.ref(dest))
            raise RuntimeError("capture")

        refs = []
        pool.select(key)
        with pytest.raises(RuntimeError, match="capture"):
            pool.submit(key, 3, 1, broken, producer_stream=pool.device)
        dest = refs[0]()
    else:
        dest = submit(pool, key)
        if fault == "query":
            dest.copy.broken = True
            with pytest.raises(RuntimeError, match="query"):
                pool.poll()
        else:
            dest.copy.done = True
            pool.poll()
            lease = pool.acquire(key, reader, owners=(extra,))
            del extra
            if fault == "record":
                ops.record_fault = True
                with pytest.raises(RuntimeError, match="record"):
                    lease.finish()
            else:
                lease.finish()
                reader.fence.broken = True
                with pytest.raises(RuntimeError, match="query"):
                    pool.poll()
    assert pool.poisoned and pool.occupied == 1
    assert not pool.submit(
        key, 3, 1, Mock(side_effect=AssertionError), producer_stream=pool.device
    )
    assert pool.acquire(key, reader) is None
    if fault in ("record", "read_query"):
        ops.sync_fault = True
    else:
        dest.broken = True
    with pytest.raises(RuntimeError, match="sync|drain"):
        pool.close()
    assert pool.occupied == 1 and sentinel() is not None
    ops.sync_fault = dest.broken = False
    pool.close()
    pool.close()
    assert pool.occupied == 0
    if fault in ("record", "read_query"):
        assert sentinel() is None


def test_open_reader_prevents_teardown_and_device_pool_duplication(setup):
    pool, key, ops, stream = setup
    dest = submit(pool, key)
    dest.copy.done = True
    pool.poll()
    with pytest.raises(RuntimeError):
        SurfacePool(pool.device, byte_limit=200, ops=ops)
    lease = pool.acquire(key, stream())
    with pytest.raises(RuntimeError, match="open"):
        pool.close()
    assert pool.occupied == 1
    lease.finish()
    pool.close()
    assert SurfacePool(pool.device, byte_limit=200, ops=ops).close() is None


def test_exact_identity_duplicate_and_mismatched_stream(setup):
    pool, key, ops, stream = setup
    capture = Mock(side_effect=AssertionError("should have been denied"))
    pool.select(key)
    with pytest.raises(AssertionError):
        pool.submit(key, 1, 1, capture, producer_stream=object())
    assert pool.occupied == ops.allocations == 0
    dest = submit(pool, key)
    dest.copy.done = True
    pool.poll()
    assert not pool.submit(key, 3, 1, capture, producer_stream=pool.device)
    for mismatch in (
        replace(key, scene_token=2),
        replace(key, owner_token=(2, "output")),
        replace(key, source_timeline_frame=2),
        replace(key, solver_frame=2),
        replace(key, simulation_time=0.5),
        replace(key, state_revision=5),
        replace(key, build_stamp=replace(key.build_stamp, revision=3)),
    ):
        assert not pool.select(mismatch)
        assert pool.acquire(mismatch, stream()) is None
    wrong = stream()
    wrong.device = object()
    with pytest.raises(AssertionError):
        pool.acquire(key, wrong)
    capture.assert_not_called()


def test_later_asset_owner_retained_across_future_reads_and_resize(setup):
    pool, key, ops, stream = setup
    dest = submit(pool, key)
    dest.copy.done = True
    pool.poll()
    reader = stream()
    lease = pool.acquire(key, reader)

    class Acceleration:
        pass

    mesh = Acceleration()
    reference = weakref.ref(mesh)
    lease.retain(mesh)
    del mesh
    new = replace(key, generation=2)
    pool.select(new)
    pool.poll()
    assert reference() is not None and lease.packet.destination is dest
    capture = Mock(side_effect=AssertionError("resize exceeded combined budget"))
    assert not pool.submit(new, 4, 1, capture, producer_stream=pool.device)
    assert pool.reserved_bytes == 48
    lease.finish()
    with pytest.raises(RuntimeError, match="finished"):
        _ = lease.packet
    pool.poll()
    assert reference() is not None
    reader.fence.done = True
    pool.poll()
    assert reference() is None
    submit(pool, new, nv=4)
    assert ops.allocations == 2 and pool.peak_bytes == 60


def test_descriptor_rejects_mutable_producer_geometry(setup):
    _, key, _, _ = setup
    for stamp in (
        replace(key.build_stamp, owner=Destination()),
        replace(key.build_stamp, build_key=(Destination(),)),
        replace(key.build_stamp, wave_settings=([],)),
    ):
        with pytest.raises(ValueError, match="immutable"):
            replace(key, build_stamp=stamp)
    with pytest.raises(ValueError, match="immutable"):
        replace(key, state_revision=Destination())
