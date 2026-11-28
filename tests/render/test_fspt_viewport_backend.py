"""Route transitions retain completed work and actual geometry ownership."""
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
from test_fspt_device_async import request, setup  # noqa: F401
from test_fspt_device_scene import key
from test_fspt_viewport import PendingBackend

from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot, Material, SceneSnapshot
from fluidSim_warp_blender51.fspt.device_scene import (
    DeviceGeometryReference,
    DeviceSceneRequest,
    DeviceSceneUnavailable,
)


def host_scene():
    return SceneSnapshot(np.eye(3), [[0, 1, 2]], [0], (Material(),), (0, 0, 0), 1)


@pytest.fixture
def routed():
    from fluidSim_warp_blender51.fspt.viewport_backend import ViewportBackendBroker

    holder = {}
    delegates = []
    budget = SimpleNamespace(reserved_bytes=0, generation_count=0)
    pool = SimpleNamespace(device='cuda:0', poisoned=False, ready=False, polls=0)
    budget.pool = pool

    def poll():
        pool.polls += 1
        if pool.ready and all(d.closed for d in delegates if d.mode == 'GPU'):
            budget.reserved_bytes = 0
            budget.generation_count = 0

    pool.poll = poll
    stops = []

    def factory(device, *, mode):
        backend = PendingBackend(device, holder['broker'].arbiter)
        backend.mode = mode
        submit = backend.submit

        def charged(*args, **kwargs):
            result = submit(*args, **kwargs)
            if result and mode == 'GPU':
                budget.reserved_bytes = 100
                budget.generation_count = 1
            return result

        backend.submit = charged
        backend.retire_completed_geometry = lambda: None
        delegates.append(backend)
        return backend

    broker = ViewportBackendBroker('cuda:0', budget, stop_production=lambda: stops.append(True),
        evaluated_factory=lambda device: factory(device, mode='EVALUATED'),
        device_factory=lambda device, **_: factory(device, mode='GPU'))
    holder['broker'] = broker
    ref = DeviceGeometryReference(pool, key(1), 3, 1, 0)
    scene = DeviceSceneRequest((ref,), (Material(),), (0, 0, 0), 1)
    yield SimpleNamespace(broker=broker, pool=pool, budget=budget, delegates=delegates,
                          scene=scene, camera=CameraSnapshot(np.eye(4)), stops=stops)
    pool.ready = True
    broker.close()


def submit(backend, scene, camera):
    return backend.submit(scene, camera, 2, 2, samples=1, max_bounces=2, seed=0)


def test_new_facade_does_not_allocate_a_delegate_before_submission(routed):
    facade = routed.broker.new_backend()
    assert not facade.pending and not routed.delegates
    assert facade.device == 'cuda:0' and routed.broker.ready_mode == 'GPU'


def test_fallback_waits_nonblocking_for_job_and_charged_last_read(routed):
    a = routed
    view = a.broker.new_backend()
    assert submit(view, a.scene, a.camera)
    a.broker.request_mode('EVALUATED')
    assert a.broker.ready_mode is None and a.stops == [True]
    assert a.broker.pump() is None
    assert not a.delegates[0].closed
    assert all(call != ('collect', True) for call in a.delegates[0].calls)
    a.delegates[0].ready = True
    assert a.broker.pump() is None
    assert a.delegates[0].closed and a.budget.reserved_bytes == 100
    assert not submit(view, host_scene(), a.camera)  # prior completion is still pending collection
    assert len(a.delegates) == 1
    frame = view.collect()
    assert frame is not None and frame.scene_frame == 1
    with pytest.raises(DeviceSceneUnavailable, match='transition'):
        submit(view, host_scene(), a.camera)
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'
    assert submit(view, host_scene(), a.camera)
    assert len(a.delegates) == 2 and a.delegates[-1].mode == 'EVALUATED'


def test_one_fallback_transitions_all_views_before_evaluated_allocation(routed):
    a = routed
    left, right = a.broker.new_backend(), a.broker.new_backend()
    assert submit(left, a.scene, a.camera)
    a.delegates[0].ready = True
    assert left.collect() is not None
    assert submit(right, a.scene, a.camera)
    with pytest.raises(DeviceSceneUnavailable):
        submit(left, host_scene(), a.camera)
    assert a.broker.ready_mode is None
    a.broker.pump()
    assert not any(delegate.closed for delegate in a.delegates)
    a.delegates[1].ready = True
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'
    assert all(d.closed for d in a.delegates)


def test_idle_delegate_is_not_closed_or_reclaimed_behind_pending_peer(routed):
    a = routed
    left, right = a.broker.new_backend(), a.broker.new_backend()
    assert submit(left, a.scene, a.camera)
    a.delegates[0].ready = True
    assert left.collect() is not None
    assert submit(right, a.scene, a.camera)
    a.broker.request_mode('EVALUATED')
    retired = []
    a.delegates[0].retire_completed_geometry = lambda: retired.append(True)
    polls = a.pool.polls
    assert a.broker.pump() is None
    assert not retired and a.pool.polls == polls
    assert not any(d.closed for d in a.delegates)
    a.delegates[1].ready = True
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'
    assert retired and right.collect() is not None


def test_other_device_arbiter_owner_blocks_all_destructive_retirement(routed):
    a = routed
    view = a.broker.new_backend()
    assert submit(view, a.scene, a.camera)
    a.delegates[0].ready = True
    view.collect()
    a.broker.request_mode('EVALUATED')
    rival = a.broker.arbiter.try_acquire(object())
    polls = a.pool.polls
    try:
        assert a.broker.pump() is None
        assert not a.delegates[0].closed and a.pool.polls == polls
        assert a.broker._retirement_owner not in a.broker.arbiter._viewport_waiters
    finally:
        a.broker.arbiter.release(rival)
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'


def test_reclamation_holds_arbiter_admission_across_close_and_pool_poll(routed):
    a = routed
    view = a.broker.new_backend()
    assert submit(view, a.scene, a.camera)
    a.delegates[0].ready = True
    view.collect()
    a.broker.request_mode('EVALUATED')
    old_close, old_poll = a.delegates[0].close, a.pool.poll
    observations = []

    def check_then(function):
        def guarded():
            assert a.broker.arbiter._active.owner is a.broker._retirement_owner
            rival = object()
            assert a.broker.arbiter.try_acquire(rival) is None
            a.broker.arbiter.cancel(rival)
            observations.append(True)
            return function()
        return guarded

    a.delegates[0].close = check_then(old_close)
    a.pool.poll = check_then(old_poll)
    a.pool.ready = True
    try:
        assert a.broker.pump() == 'EVALUATED'
        assert observations == [True, True]
        assert a.broker.arbiter._active is None
    finally:
        a.delegates[0].close, a.pool.poll = old_close, old_poll


def test_unavailable_peer_lock_defers_every_close_and_pool_reclamation(routed):
    from threading import Event

    a = routed
    left, right = a.broker.new_backend(), a.broker.new_backend()
    assert submit(left, a.scene, a.camera)
    a.delegates[0].ready = True
    left.collect()
    a.broker.request_mode('EVALUATED')
    locked, release = Event(), Event()
    def hold():
        with right._lock:
            locked.set()
            assert release.wait(2)
    worker = Thread(target=hold)
    worker.start()
    try:
        assert locked.wait(2)
        polls = a.pool.polls
        assert a.broker.pump() is None
        assert not a.delegates[0].closed and a.pool.polls == polls
    finally:
        release.set()
        worker.join(2)
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'


def test_gpu_reentry_is_explicit_and_never_silently_flips_peer_mode(routed):
    a = routed
    view = a.broker.new_backend()
    a.broker.request_mode('EVALUATED')
    assert a.broker.pump() == 'EVALUATED'
    assert submit(view, host_scene(), a.camera)
    a.delegates[0].ready = True
    view.collect()
    with pytest.raises(DeviceSceneUnavailable, match='GPU'):
        submit(view, a.scene, a.camera)
    assert a.broker.ready_mode == 'EVALUATED'
    a.broker.request_mode('GPU')
    assert a.broker.pump() == 'GPU'
    assert a.delegates[0].closed
    assert submit(view, a.scene, a.camera)


def test_transition_cancels_old_queued_intent_without_touching_active_peer(routed):
    a = routed
    view = a.broker.new_backend()
    rival = a.broker.arbiter.try_acquire(object())
    assert not submit(view, a.scene, a.camera)
    old_owner = view.owner
    assert old_owner in a.broker.arbiter._viewport_waiters
    a.broker.request_mode('EVALUATED')
    assert old_owner not in a.broker.arbiter._viewport_waiters
    assert a.broker.arbiter._active is rival
    a.broker.arbiter.release(rival)


def test_typed_consumer_refusal_cancels_old_queued_intent(routed):
    a = routed
    view = a.broker.new_backend()
    rival = a.broker.arbiter.try_acquire(object())
    assert not submit(view, a.scene, a.camera)
    def stale(*_, **__):
        raise DeviceSceneUnavailable('stale reference')
    a.delegates[0].submit = stale
    with pytest.raises(DeviceSceneUnavailable, match='stale'):
        submit(view, a.scene, a.camera)
    assert view.owner not in a.broker.arbiter._viewport_waiters
    a.broker.arbiter.release(rival)


def test_poisoned_pool_requires_explicit_checked_recovery_before_fallback(routed):
    a = routed
    view = a.broker.new_backend()
    assert submit(view, a.scene, a.camera)
    a.pool.poisoned = True
    a.broker.request_mode('EVALUATED')
    assert a.broker.recovery_required and a.broker.recovery_reason
    a.pool.poll = lambda: None  # qualified pools cannot poll a poisoned fence
    assert a.broker.pump() is None and not a.broker.geometry_recovery_ready
    a.delegates[0].ready = True
    assert a.broker.pump() is None and a.broker.geometry_recovery_ready
    assert a.budget.reserved_bytes == 100
    assert view.collect() is not None
    with pytest.raises(DeviceSceneUnavailable):
        submit(view, host_scene(), a.camera)
    # Runtime explicitly closes/drains the producer pool after the ready proof.
    a.budget.reserved_bytes = 0
    a.budget.generation_count = 0
    assert a.broker.pump() == 'EVALUATED'
    assert submit(view, host_scene(), a.camera)


def test_zero_byte_pending_packet_still_blocks_evaluated_generation(routed):
    a = routed
    view = a.broker.new_backend()
    a.budget.reserved_bytes = 0
    a.budget.generation_count = 1
    a.broker.request_mode('EVALUATED')
    assert a.broker.pump() is None
    with pytest.raises(DeviceSceneUnavailable, match='transition'):
        submit(view, host_scene(), a.camera)
    assert not a.delegates
    a.pool.ready = True
    assert a.broker.pump() == 'EVALUATED'


def test_final_worker_collection_defers_pool_retirement_to_publication_thread(setup):  # noqa: F811
    backend, pool, budget, _wp, arbiter = setup
    camera = CameraSnapshot(np.eye(4))
    assert backend.submit(request(backend, pool, budget), camera, 2, 2)
    backend.collect(wait=True)
    old = backend._device_current
    assert backend.submit(request(backend, pool, budget, 2), camera, 2, 2)
    charged = budget.reserved_bytes
    result = []

    def collect():
        try:
            result.append(backend.collect(wait=True))
        except BaseException as exc:
            result.append(exc)

    worker = Thread(target=collect)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and len(result) == 1
    assert not isinstance(result[0], BaseException)
    assert not backend.pending and arbiter._active is None
    assert old in backend._device_retired and old.lease._slot is not None
    assert budget.reserved_bytes == charged
    backend.retire_completed_geometry()
    assert old not in backend._device_retired and old.lease._slot is None
    pool.poll()
    assert budget.reserved_bytes < charged


def test_worker_close_keeps_pending_delegate_owned_until_main_pump(routed):
    a = routed
    view = a.broker.new_backend()
    assert submit(view, a.scene, a.camera)
    worker = Thread(target=view.close)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and view.closed
    assert not a.delegates[0].closed and view in a.broker._views
    assert a.broker.pump() == 'GPU'
    assert not a.delegates[0].closed
    a.delegates[0].ready = True
    a.pool.ready = True
    a.broker.pump()
    assert a.delegates[0].closed and view not in a.broker._views


def test_main_facade_close_keeps_idle_delegate_until_active_peer_has_completed(routed):
    a = routed
    first, peer = a.broker.new_backend(), a.broker.new_backend()
    assert submit(first, a.scene, a.camera)
    idle = a.delegates[0]
    idle.ready = True
    first.collect()
    assert submit(peer, a.scene, a.camera)
    running = a.delegates[1]
    lease = a.broker.arbiter._active
    first.close()
    assert first.closed and first in a.broker._views
    assert first._delegate is idle and not idle.closed
    assert running.pending and a.broker.arbiter._active is lease
    a.broker.pump()
    assert not idle.closed and ('collect', True) not in running.calls
    running.ready = True
    a.pool.ready = True
    a.broker.pump()
    assert idle.closed and first not in a.broker._views
    assert not running.pending


def test_explicit_broker_close_drains_every_peer_before_any_native_retirement(routed):
    a = routed
    first, peer = a.broker.new_backend(), a.broker.new_backend()
    assert submit(first, a.scene, a.camera)
    idle = a.delegates[0]
    idle.ready = True
    first.collect()
    assert submit(peer, a.scene, a.camera)
    running = a.delegates[1]
    original = idle.close
    observed = []
    def close():
        observed.append(running.pending)
        original()
    idle.close = close
    a.pool.ready = True
    a.broker.close()
    assert observed == [False]
    assert ('collect', True) in running.calls
    assert not a.broker._views and all(delegate.closed for delegate in a.delegates)


def test_explicit_close_cannot_succeed_with_unretired_evaluated_zero_ledger_delegate(routed):
    from fluidSim_warp_blender51.fspt.viewport_backend import RouteTransitionPending

    a = routed
    a.broker.request_mode('EVALUATED')
    a.broker.pump()
    view = a.broker.new_backend()
    assert submit(view, host_scene(), a.camera)
    delegate = a.delegates[0]
    delegate.ready = True
    view.collect()
    assert a.budget.reserved_bytes == a.budget.generation_count == 0
    external = a.broker.arbiter.try_acquire(object())
    assert external is not None
    try:
        with pytest.raises(RouteTransitionPending, match='retirement'):
            a.broker.close()
        assert view._delegate is delegate and not delegate.closed
        assert view in a.broker._views
    finally:
        a.broker.arbiter.release(external)
    a.broker.close()
    assert not a.broker._views and delegate.closed
