"""Runtime recovery uses real broker, pool, lease and ledger ownership.

Only CUDA operations use the established checked host fixture. These controls
make no native device-completion claim.
"""
from types import SimpleNamespace

import pytest
from test_fspt_device_async import request as device_request, setup  # noqa: F401

from fluidSim_warp_blender51.blender import fspt_device_runtime as runtime
from fluidSim_warp_blender51.blender import surface_publication as publication
from fluidSim_warp_blender51.blender.fspt_device_bridge import DeviceSurfaceBridge
from fluidSim_warp_blender51.fspt import viewport_backend
from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot


@pytest.fixture
def recovery(setup, monkeypatch, request):  # noqa: F811
    import numpy as np

    backend, pool, budget, _wp, arbiter = setup
    monkeypatch.setenv('FLUIDSIM_FSPT_DEVICE_SURFACE', '1')
    for name in ('_SCENES', '_DEVICES', '_DEVICE_ERRORS', '_HOST_EVENTS'):
        monkeypatch.setattr(runtime, name, {})
    monkeypatch.setattr(publication, '_channels', {})
    monkeypatch.setattr(publication, '_subscribers', {})
    monkeypatch.setattr(viewport_backend, 'get_arbiter', lambda _: arbiter)
    bridge = DeviceSurfaceBridge(budget)
    broker = viewport_backend.ViewportBackendBroker(
        pool.device, budget, stop_production=pool.invalidate,
        device_factory=lambda *_args, **_kwargs: backend,
        evaluated_factory=lambda _: pytest.fail('evaluated allocation preceded checked zero ownership'))
    resource = runtime.DeviceResources(pool.device, pool, budget, bridge, broker)
    runtime._DEVICES['cuda:0'] = resource
    scene = SimpleNamespace(session_uid=51, fluidsim=SimpleNamespace(
        domain_object=None, output_collection_name='Fluid'))
    class View:
        pass
    demand = View()
    runtime.subscribe(scene, demand, device='cuda:0')
    runtime.set_capability(scene, demand, 'cuda:0', reason='')
    facade = broker.new_backend()
    assert facade.submit(device_request(backend, pool, budget), CameraSnapshot(np.eye(4)), 2, 2)
    assert facade.collect(wait=True) is not None
    if getattr(request, 'param', None) == 'collect':
        assert facade.submit(device_request(backend, pool, budget, 2), CameraSnapshot(np.eye(4)), 2, 2)
        backend._job.copied.done = True
    else:
        broker.request_mode('EVALUATED')
    record, sync, close = pool._ops.record, pool._ops.sync_reader, backend.close
    calls = {'record': 0, 'drain': 0, 'close': 0}
    def failed_record(*_):
        calls['record'] += 1
        if calls['record'] == 1:
            raise RuntimeError('injected last-read record failure')
        return record(*_)
    def counted_close():
        calls['close'] += 1
        return close()
    monkeypatch.setattr(pool._ops, 'record', failed_record)
    monkeypatch.setattr(backend, 'close', counted_close)
    yield SimpleNamespace(backend=backend, pool=pool, budget=budget, broker=broker,
                          bridge=bridge, facade=facade, resource=resource, calls=calls,
                          scene=scene, demand=demand)
    monkeypatch.setattr(pool._ops, 'record', record)
    monkeypatch.setattr(pool._ops, 'sync_reader', sync)
    monkeypatch.setattr(backend, 'close', close)
    runtime.shutdown()


def test_runtime_retries_known_poisoned_retirement_once_then_reaches_checked_fallback(recovery):
    a = recovery
    runtime.pump()
    assert a.pool.poisoned and a.resource.pump_blocked
    assert a.facade._delegate is a.backend and a.budget.generation_count == 1
    assert a.budget.reserved_bytes > 0
    runtime.pump()
    assert a.facade._delegate is None
    assert a.pool.occupied == a.budget.reserved_bytes == a.budget.generation_count == 0
    assert not a.resource.pump_blocked
    runtime.pump()
    assert a.broker.ready_mode == 'EVALUATED'
    assert a.calls['record'] == 1 and a.calls['close'] == 2


def test_failed_checked_pool_drain_is_terminal_until_explicit_lifecycle_retry(recovery, monkeypatch):
    a = recovery
    def failed_drain(*_):
        a.calls['drain'] += 1
        raise RuntimeError('injected checked drain failure')
    monkeypatch.setattr(a.pool._ops, 'sync_reader', failed_drain)
    runtime.pump()
    runtime.pump()
    assert a.facade._delegate is None
    assert a.resource.pump_blocked and a.pool.occupied == 1
    assert a.budget.generation_count == 1 and a.budget.reserved_bytes > 0
    assert a.calls['drain'] == 1
    for _ in range(4):
        assert not runtime.pump()
        assert runtime.poll(a.scene, 'cuda:0') is None
    assert a.calls['drain'] == 1 and a.calls['close'] == 2
    assert 'explicit shutdown' in runtime.resource_error('cuda:0')


def test_recovery_allowance_is_not_consumed_or_executed_under_final_hold(recovery, monkeypatch):
    from fluidSim_warp_blender51.fspt.final_handoff import FinalHandoff

    a = recovery
    manager = FinalHandoff()
    monkeypatch.setattr(runtime, '_final_blocked', manager.blocked)
    runtime.pump()
    hold = manager.retain_failure('cuda:0')
    before = dict(a.calls)
    try:
        for _ in range(3):
            assert not runtime.pump()
        assert a.calls == before and a.facade._delegate is a.backend
        assert a.budget.generation_count == 1
    finally:
        hold.release()
    runtime.pump()
    assert a.facade._delegate is None and a.budget.generation_count == 0


def test_second_broker_failure_is_not_retried_by_subsequent_timer_pumps(recovery, monkeypatch):
    a = recovery
    runtime.pump()
    def failed_close():
        a.calls['close'] += 1
        raise RuntimeError('injected second delegate retirement failure')
    monkeypatch.setattr(a.backend, 'close', failed_close)
    runtime.pump()
    assert a.calls['close'] == 2
    for _ in range(4):
        assert not runtime.pump()
    assert a.calls['close'] == 2 and a.resource.pump_blocked
    assert a.facade._delegate is a.backend and a.budget.generation_count == 1


def test_non_poisoned_broker_fault_never_gains_an_automatic_retry(recovery, monkeypatch):
    a = recovery
    def fail_before_last_read():
        a.calls['close'] += 1
        raise RuntimeError('unclassified delegate close fault')
    monkeypatch.setattr(a.backend, 'close', fail_before_last_read)
    assert not runtime.pump()
    assert not a.pool.poisoned and a.resource.pump_blocked
    for _ in range(4):
        assert not runtime.pump()
    assert a.calls['close'] == 1 and a.facade._delegate is a.backend
    assert a.budget.generation_count == 1


@pytest.mark.parametrize(('size', 'generations'), [(0, 1), (1, 0)])
def test_checked_close_alone_cannot_clear_remaining_geometry_ownership(monkeypatch, size, generations):
    monkeypatch.setattr(runtime, '_DEVICE_ERRORS', {})
    resource = runtime.DeviceResources(
        'cuda:0', SimpleNamespace(poisoned=True),
        SimpleNamespace(reserved_bytes=size, generation_count=generations),
        SimpleNamespace(close=lambda: None),
        SimpleNamespace(recovery_required=True, geometry_recovery_ready=True))
    runtime._recover_failed_pool('cuda:0', resource)
    assert resource.pump_blocked and resource.recovery_attempted
    assert 'Explicit FSPT geometry remains' in runtime._DEVICE_ERRORS['cuda:0']
    assert (resource.budget.reserved_bytes, resource.budget.generation_count) == (size, generations)


@pytest.mark.parametrize('recovery', ['collect'], indirect=True)
def test_checked_failed_job_discard_finishes_recovery_without_another_unclassified_fault(recovery):
    a = recovery
    runtime.pump()
    assert a.pool.poisoned and a.resource.pump_blocked and a.backend.pending
    assert a.budget.generation_count == 2
    runtime.pump()
    assert not a.backend.pending and a.facade._delegate is None
    assert a.pool.occupied == a.budget.generation_count == a.budget.reserved_bytes == 0
    assert not a.resource.pump_blocked
    runtime.pump()
    assert a.broker.ready_mode == 'EVALUATED'


@pytest.mark.parametrize('denial', ['arbiter', 'peer_lock'])
def test_nonblocking_retirement_denial_preserves_the_unused_recovery_allowance(recovery, denial):
    from threading import Event, Thread

    a = recovery
    runtime.pump()
    assert a.resource.pump_blocked and a.pool.poisoned
    lease, worker = None, None
    entered, release = Event(), Event()
    if denial == 'arbiter':
        lease = a.backend.arbiter.try_acquire(object())
        assert lease is not None
    else:
        def hold_peer():
            with a.facade._lock:
                entered.set()
                assert release.wait(2)
        worker = Thread(target=hold_peer)
        worker.start()
        assert entered.wait(1)
    try:
        for _ in range(3):
            runtime.pump()
        assert a.facade._delegate is a.backend and a.budget.generation_count == 1
        assert a.calls['close'] == 1
    finally:
        if lease is not None:
            a.backend.arbiter.release(lease)
        if worker is not None:
            release.set()
            worker.join(2)
            assert not worker.is_alive()
    runtime.pump()
    assert a.facade._delegate is None and a.budget.generation_count == 0
    assert not a.resource.pump_blocked
