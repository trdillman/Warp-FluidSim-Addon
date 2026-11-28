"""Worker/main-thread ownership regressions from independent review."""
from threading import Thread

import numpy as np
import pytest
from test_fspt_device_async import request, setup  # noqa: F401
from test_fspt_device_backend import prepared, publish  # noqa: F401

from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot
from fluidSim_warp_blender51.fspt.viewport_backend import ViewportBackendBroker


def worker_collect(backend):
    results = []
    def collect():
        try:
            results.append(backend.collect(wait=True))
        except BaseException as exc:
            results.append(exc)
    worker = Thread(target=collect)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive() and len(results) == 1
    return results[0]


def test_worker_validation_failure_keeps_geometry_claims_until_main_retirement(setup):  # noqa: F811
    backend, pool, budget, _wp, arbiter = setup
    assert backend.submit(request(backend, pool, budget), CameraSnapshot(np.eye(4)), 2, 2)
    current = backend._device_current
    current.validation_host.values[0] = 1
    charged = budget.reserved_bytes
    result = worker_collect(backend)
    assert isinstance(result, RuntimeError) and 'invalid device geometry' in str(result)
    assert not backend.pending and arbiter._active is None
    assert backend._device_current is None and current in backend._device_retired
    assert current.lease._slot is not None and current.arrays is not None
    assert budget.reserved_bytes == charged
    assert backend.retire_completed_geometry()
    assert current.arrays is not None and budget.reserved_bytes == charged
    pool.poll()
    assert current.arrays is None and budget.reserved_bytes < charged


def test_main_finish_record_failure_after_worker_keeps_claim_until_checked_close(setup, monkeypatch):  # noqa: F811
    backend, pool, budget, _wp, arbiter = setup
    camera = CameraSnapshot(np.eye(4))
    assert backend.submit(request(backend, pool, budget), camera, 2, 2)
    backend.collect(wait=True)
    old = backend._device_current
    assert backend.submit(request(backend, pool, budget, 2), camera, 2, 2)
    charged = budget.reserved_bytes
    assert not isinstance(worker_collect(backend), BaseException)
    assert arbiter._active is None and old in backend._device_retired
    def fail(*_):
        raise RuntimeError('review injected last-read record failure')
    with monkeypatch.context() as patch:
        patch.setattr(pool._ops, 'record', fail)
        with pytest.raises(RuntimeError, match='last-read record failure'):
            backend.retire_completed_geometry()
    assert old in backend._device_retired and old.arrays is not None
    assert budget.reserved_bytes == charged and pool.poisoned
    backend.close()
    assert old.arrays is not None and budget.reserved_bytes == charged
    pool.close()
    assert old.arrays is None and budget.reserved_bytes == 0


def test_main_cleanup_never_finishes_current_replacement_predecessor(setup):  # noqa: F811
    backend, pool, budget, _wp, _arbiter = setup
    camera = CameraSnapshot(np.eye(4))
    assert backend.submit(request(backend, pool, budget), camera, 2, 2)
    backend.collect(wait=True)
    old = backend._device_current
    assert backend.submit(request(backend, pool, budget, 2), camera, 2, 2)
    charged = budget.reserved_bytes
    assert backend.pending and old in backend._device_retired
    assert not backend.retire_completed_geometry()
    assert old.lease._slot is not None and old.arrays is not None
    assert budget.reserved_bytes == charged
    assert not isinstance(worker_collect(backend), BaseException)
    assert backend.retire_completed_geometry()
    pool.poll()
    assert old.arrays is None and budget.reserved_bytes < charged


def test_zero_byte_pending_packet_blocks_evaluated_mode(prepared, monkeypatch):  # noqa: F811
    backend, pool, budget = prepared
    gate = {'ready': False}
    original = pool._ops.allocate
    def allocate(device, nv, nt):
        destination = original(device, nv, nt)
        destination.retire_source = lambda: gate['ready']
        return destination
    monkeypatch.setattr(pool._ops, 'allocate', allocate)
    publish(backend, pool, budget, [], [], 1)
    assert budget.reserved_bytes == 0 and budget.generation_count == 1
    broker = ViewportBackendBroker(pool.device, budget, stop_production=pool.invalidate)
    try:
        broker.request_mode('EVALUATED')
        assert broker.pump() is None, 'zero bytes did not prove generation retirement'
        assert pool.occupied == 1 and budget.generation_count == 1
        gate['ready'] = True
        assert broker.pump() == 'EVALUATED'
        assert pool.occupied == 0 and budget.generation_count == 0
    finally:
        gate['ready'] = True
        broker.close()
