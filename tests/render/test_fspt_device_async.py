"""Fault-controlled async lifetime tests; actual CUDA qualification is separate."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_fspt_async_backend import Buffer, CheckedWarp, Stream
from test_fspt_device_backend import Ops, publish

from fluidSim_warp_blender51.fspt import async_backend
from fluidSim_warp_blender51.fspt.backend import CudaBackend
from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot, Material
from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneRequest, GeometryArrayBudget
from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter
from fluidSim_warp_blender51.fspt.surface_pool import SurfacePool


class Device:
    is_cuda = True

    def __str__(self):
        return 'cuda:0'


class GeometryWarp(CheckedWarp):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.int32 = int

    def Mesh(self, **kwargs):
        self.op('mesh')
        class Mesh:
            id = 3
        return Mesh()

    def get_device(self, device):
        return self.device

    def _buffer(self, values, dtype, device):
        result = Buffer(self, np.array(values, copy=True), device)
        result.dtype = dtype
        result.ndim = 1
        result.is_contiguous = True
        result.strides = (12 if dtype == self.vec3 else 4,)
        return result

    def array(self, values, *, dtype, device, copy=True):
        self.op('upload')
        return self._buffer(values, dtype, device)

    def empty(self, count, *, dtype, device, pinned=False):
        self.op('allocate')
        shape = (count, 3 if dtype == self.vec3 else 4) if dtype in (self.vec3, self.vec4) else (count,)
        return self._buffer(np.zeros(shape, np.float32 if dtype is not int else np.int32), dtype, device)

    def zeros(self, count, *, dtype, device):
        return self.empty(count, dtype=dtype, device=device)

    def Stream(self, device):
        self.op('prepare_stream')
        stream = Stream(self, 'prepare')
        stream.device = device
        return stream

    def launch(self, kernel, *, dim, inputs, device, stream=None):
        if stream is None:
            self.op('geometry_kernel')
            return
        super().launch(kernel, dim=dim, inputs=inputs, device=device, stream=stream)

    def copy(self, dest, src, *, count, stream=None, dest_offset=0):
        self.op('copy')
        dest.values[dest_offset:dest_offset + count] = src.values[:count]


@pytest.fixture
def setup(monkeypatch):
    from fluidSim_warp_blender51.fspt.device_backend import DeviceAsyncCudaBackend
    device, arbiter = Device(), DeviceArbiter()
    wp = GeometryWarp(device)

    def init(self, _device):
        self._initialize_resources()
        self.wp, self.device, self.kernel = wp, device, object()

    monkeypatch.setattr(CudaBackend, '__init__', init)
    monkeypatch.setattr(async_backend, 'get_arbiter', lambda _: arbiter)
    monkeypatch.setattr(async_backend, '_nonblocking_stream', lambda w, d: Stream(w, 'render'))
    pool = SurfacePool(device, byte_limit=1024 * 1024, ops=Ops())
    budget = GeometryArrayBudget(pool, byte_limit=2 * 1024 * 1024)
    backend = DeviceAsyncCudaBackend(geometry_budget=budget)
    backend._prepare_stream = SimpleNamespace(device=device)
    backend._prepare_stream = wp.Stream(device)
    yield backend, pool, budget, wp, arbiter
    wp.failure = None
    backend.close()
    pool.close()


def request(backend, pool, budget, generation=1):
    ref = publish(backend, pool, budget, np.eye(3), [[0, 1, 2]], generation)
    return DeviceSceneRequest((ref,), (Material(),), (0, 0, 0), generation)


def test_async_admission_and_checked_retirement_hold_generation_leases(setup):
    backend, pool, budget, wp, arbiter = setup
    first = request(backend, pool, budget)
    camera = CameraSnapshot(np.eye(4))
    rival = arbiter.try_acquire(object())
    before = wp.ops.copy()
    assert not backend.submit(first, camera, 2, 2)
    assert wp.ops == before and not pool._slots[0].readers
    arbiter.release(rival)
    assert backend.submit(first, camera, 2, 2)
    assert backend.collect() is None
    backend.collect(wait=True)
    current = backend._device_current
    assert current.lease._slot is not None and not backend.pending
    assert backend.submit(first, camera, 2, 2)
    backend.collect(wait=True)
    assert backend._device_current is current and backend.statistics['mesh_builds'] == 1
    second = request(backend, pool, budget, 2)
    assert backend.submit(second, camera, 2, 2)
    assert current in backend._device_retired and current.lease._slot is not None
    assert backend.collect() is None
    assert current.lease._slot is not None
    backend.collect(wait=True)
    assert current.lease._slot is None and not backend._device_retired
    pool.poll()
    assert pool.occupied == 1


def test_failed_drain_retains_packet_aggregate_mesh_and_budget_then_close_retries(setup, monkeypatch):
    backend, pool, budget, wp, _arbiter = setup
    import gc
    import weakref
    camera = CameraSnapshot(np.eye(4))
    first = request(backend, pool, budget)
    assert backend.submit(first, camera, 2, 2)
    backend.collect(wait=True)
    mesh = weakref.ref(backend.mesh)
    arrays = weakref.ref(backend.arrays['points'])
    wp.failure = 'copy'
    def fail(_):
        raise RuntimeError('unavailable drain')
    with monkeypatch.context() as patch:
        patch.setattr(wp, 'synchronize_stream', fail)
        with pytest.raises(RuntimeError, match='unavailable drain'):
            backend.submit(first, camera, 2, 2)
    backend._job.failure.__traceback__ = None
    gc.collect()
    assert backend.pending and mesh() is not None and arrays() is not None
    assert budget.reserved_bytes > pool.reserved_bytes and backend._device_current.lease._slot is not None
    wp.failure = None
    backend.close()
    pool.invalidate()
    pool.poll()
    gc.collect()
    assert mesh() is None and arrays() is None and budget.reserved_bytes == 0


def test_invalid_validation_status_discards_image_and_history(setup):
    backend, pool, budget, _wp, _arbiter = setup
    first = request(backend, pool, budget)
    assert backend.submit(first, CameraSnapshot(np.eye(4)), 2, 2)
    backend._device_current.validation_host.values[0] = 1
    with pytest.raises(RuntimeError, match='invalid device geometry'):
        backend.collect(wait=True)
    assert not backend.pending and backend.history.samples == 0 and backend.mesh is None
    assert budget.reserved_bytes > pool.reserved_bytes
    pool.poll()
    assert budget.reserved_bytes == pool.reserved_bytes


def test_failed_last_read_record_keeps_generation_and_budget_until_retry(setup, monkeypatch):
    backend, pool, budget, _wp, _arbiter = setup
    camera = CameraSnapshot(np.eye(4))
    assert backend.submit(request(backend, pool, budget), camera, 2, 2)
    backend.collect(wait=True)
    assert backend.submit(request(backend, pool, budget, 2), camera, 2, 2)
    before = budget.reserved_bytes
    def fail(*_):
        raise RuntimeError('last read record fault')
    with monkeypatch.context() as patch:
        patch.setattr(pool._ops, 'record', fail)
        with pytest.raises(RuntimeError, match='last read record fault'):
            backend.collect(wait=True)
    assert backend.pending and backend._device_retired and budget.reserved_bytes == before
    assert pool.poisoned
    backend.close()
    assert budget.reserved_bytes > pool.reserved_bytes
    pool.close()
    assert budget.reserved_bytes == 0


def test_capacity_denial_queues_no_work_and_does_not_drain_old_owned_streams(setup):
    from fluidSim_warp_blender51.fspt.device_scene import GeometryBudgetExceeded
    backend, pool, budget, wp, _arbiter = setup
    first = request(backend, pool, budget)
    camera = CameraSnapshot(np.eye(4))
    budget.byte_limit = pool.reserved_bytes  # No room for one aggregate.
    before = wp.ops.copy()
    with pytest.raises(GeometryBudgetExceeded):
        backend.submit(first, camera, 2, 2)
    assert wp.ops == before and not backend.pending
    assert not pool._slots[0].readers


@pytest.mark.parametrize('reason', ['stale', 'budget'])
def test_preflight_refusal_cancels_previously_queued_arbiter_intent(setup, reason):
    from test_fspt_device_scene import key

    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable
    backend, pool, budget, wp, arbiter = setup
    first = request(backend, pool, budget)
    camera = CameraSnapshot(np.eye(4))
    rival = arbiter.try_acquire(object())
    assert not backend.submit(first, camera, 2, 2)
    if reason == 'stale':
        assert pool.select(key(2))
    else:
        budget.byte_limit = pool.reserved_bytes
    before = wp.ops.copy()
    with pytest.raises(DeviceSceneUnavailable):
        backend.submit(first, camera, 2, 2)
    assert wp.ops == before
    arbiter.release(rival)
    peer = arbiter.try_acquire(object())
    assert peer is not None
    arbiter.release(peer)


def test_owned_geometry_does_not_silently_rebind_a_different_pool(setup):
    from dataclasses import replace

    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable
    backend, pool, budget, wp, _arbiter = setup
    first = request(backend, pool, budget)
    camera = CameraSnapshot(np.eye(4))
    assert backend.submit(first, camera, 2, 2)
    backend.collect(wait=True)
    foreign = replace(first, parts=(replace(first.reference, pool=object()),))
    assert foreign.geometry_digest == first.geometry_digest
    before = wp.ops.copy()
    with pytest.raises(DeviceSceneUnavailable, match='pool'):
        backend.submit(foreign, camera, 2, 2)
    assert wp.ops == before and not backend.pending
