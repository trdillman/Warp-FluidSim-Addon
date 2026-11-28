"""Final allocation requires an exact checked main-thread retirement receipt."""
from dataclasses import replace
from threading import Event, Thread, current_thread, main_thread
from time import monotonic, sleep

import pytest

from fluidSim_warp_blender51.fspt.final_handoff import (
    FinalHandoff,
    FinalHandoffError,
    FinalHandoffTimeout,
    FinalRetirementReceipt,
)


def test_failure_hold_suppresses_admission_without_issuing_a_zero_receipt():
    queue = FinalHandoff()
    hold = queue.retain_failure('cuda:0')
    assert hold.receipt is None and queue.blocked('cuda:0')
    assert queue.pending_count == 0 and queue.service() == 0
    with pytest.raises(FinalHandoffError, match='already'):
        queue.acquire('cuda:0', lambda d, i: FinalRetirementReceipt(d, i, 0, 0, True))
    queue.cancel_pending('history reset')
    assert queue.blocked('cuda:0'), 'lifecycle cancellation erased an uncertain CUDA owner'
    hold.release()
    hold.release()
    assert not queue.blocked('cuda:0')


def test_failure_holds_are_independent_of_each_other_and_successful_claims():
    queue = FinalHandoff()
    claim = queue.acquire('cuda:0', lambda d, i: FinalRetirementReceipt(d, i, 0, 0, True))
    first, second = queue.retain_failure('cuda:0'), queue.retain_failure('cuda:0')
    claim.release()
    first.release()
    assert queue.blocked('cuda:0')
    queue.cancel_pending('cannot erase failed owners')
    assert queue.blocked('cuda:0')
    second.release()
    assert not queue.blocked('cuda:0')


def cleared(device, identity):
    return FinalRetirementReceipt(device, identity, reserved_bytes=0,
                                  generation_count=0, owners_retired=True)


def wait_pending(queue):
    deadline = monotonic() + 2
    while not queue.pending_count and monotonic() < deadline:
        sleep(0.001)
    assert queue.pending_count == 1


def run_worker(function):
    result = {}
    done = Event()
    def run():
        try:
            result['value'] = function()
        except BaseException as exc:
            result['error'] = exc
        finally:
            done.set()
    thread = Thread(target=run)
    thread.start()
    return thread, result, done


def test_main_final_retires_directly_and_holds_producer_suppression():
    queue = FinalHandoff()
    calls = []
    def retire(device, identity):
        assert queue.blocked(device)
        calls.append(current_thread())
        return cleared(device, identity)
    claim = queue.acquire('cuda:0', retire)
    assert calls == [main_thread()] and queue.blocked('cuda:0')
    assert queue.pending_count == 0 and claim.receipt.reserved_bytes == 0
    claim.release()
    claim.release()
    assert not queue.blocked('cuda:0')


def test_worker_waits_for_main_and_keeps_claim_until_final_work_releases():
    queue, allow_exit = FinalHandoff(), Event()
    calls = []
    def retire(device, identity):
        calls.append(current_thread())
        return cleared(device, identity)
    def worker():
        claim = queue.acquire('cuda:0', retire, timeout=1)
        assert allow_exit.wait(1)
        claim.release()
    thread, result, done = run_worker(worker)
    wait_pending(queue)
    assert not calls and queue.blocked('cuda:0') and not done.is_set()
    assert queue.service() == 1
    assert calls == [main_thread()] and queue.blocked('cuda:0')
    allow_exit.set()
    thread.join(1)
    assert not thread.is_alive() and 'error' not in result
    assert not queue.blocked('cuda:0')


def test_timeout_cancels_unstarted_cleanup_and_never_authorizes_final():
    queue, calls = FinalHandoff(), []
    def retire(*args):
        calls.append(args)
        return cleared(*args)
    thread, result, done = run_worker(lambda: queue.acquire('cuda:0', retire, timeout=0.02))
    assert done.wait(1)
    thread.join()
    assert isinstance(result['error'], FinalHandoffTimeout)
    assert queue.service() == 0 and calls == []
    assert not queue.blocked('cuda:0')


def test_timeout_during_cleanup_holds_device_until_that_cleanup_finishes():
    queue = FinalHandoff()
    result_ready = Event()
    def retire(device, identity):
        assert result_ready.wait(1)
        assert queue.blocked(device)
        return cleared(device, identity)
    def worker():
        try:
            return queue.acquire('cuda:0', retire, timeout=0.03)
        finally:
            result_ready.set()
    thread, result, _done = run_worker(worker)
    wait_pending(queue)
    assert queue.service() == 1
    thread.join(1)
    assert isinstance(result['error'], FinalHandoffTimeout)
    assert not queue.blocked('cuda:0')


@pytest.mark.parametrize('change', [
    {'device': 'cuda:1'}, {'request_id': object()}, {'reserved_bytes': 4},
    {'generation_count': 1}, {'owners_retired': False},
])
def test_wrong_or_nonempty_receipt_never_authorizes_final(change):
    queue = FinalHandoff()
    def retire(device, identity):
        return replace(cleared(device, identity), **change)
    with pytest.raises(FinalHandoffError, match='receipt|retire'):
        queue.acquire('cuda:0', retire)
    assert not queue.blocked('cuda:0')


def test_cleanup_error_preserves_failure_and_releases_only_its_own_claim():
    queue = FinalHandoff()
    other = queue.acquire('cuda:1', cleared)
    def fail(*_args):
        raise RuntimeError('checked reader drain failed')
    with pytest.raises(FinalHandoffError, match='checked reader drain failed'):
        queue.acquire('cuda:0', fail)
    assert queue.blocked('cuda:1') and not queue.blocked('cuda:0')
    other.release()


def test_same_device_nested_or_concurrent_final_is_refused_before_cleanup():
    queue = FinalHandoff()
    claim = queue.acquire('cuda:0', cleared)
    with pytest.raises(FinalHandoffError, match='already'):
        queue.acquire('cuda:0', lambda *_args: pytest.fail('second cleanup executed'))
    claim.release()
    newer = queue.acquire('cuda:0', cleared)
    claim.release()
    assert queue.blocked('cuda:0')
    newer.release()


def test_only_main_thread_can_service_and_pending_request_survives_wrong_thread():
    queue = FinalHandoff()
    thread, result, _done = run_worker(lambda: queue.acquire('cuda:0', cleared, timeout=1))
    wait_pending(queue)
    wrong, wrong_result, wrong_done = run_worker(queue.service)
    assert wrong_done.wait(1)
    wrong.join()
    assert isinstance(wrong_result['error'], RuntimeError) and queue.pending_count == 1
    assert queue.service() == 1
    thread.join(1)
    result['value'].release()


def test_unregister_cancels_pending_receipt_without_running_or_releasing_active_final():
    queue = FinalHandoff()
    active = queue.acquire('cuda:1', cleared)
    thread, result, done = run_worker(lambda: queue.acquire('cuda:0', cleared, timeout=1))
    wait_pending(queue)
    queue.cancel_pending('engine unregistered')
    assert done.wait(1)
    thread.join()
    assert isinstance(result['error'], FinalHandoffError)
    assert 'unregistered' in str(result['error'])
    assert queue.service() == 0 and queue.blocked('cuda:1')
    active.release()


def test_worker_does_not_keep_coordinator_admission_lock_while_waiting_for_main():
    from fluidSim_warp_blender51.fspt.viewport import ViewportCoordinator
    coordinator, queue = ViewportCoordinator(), FinalHandoff()
    def retire(device, identity):
        lock = coordinator._admission_lock(device)
        assert lock.acquire(blocking=False), 'worker kept admission lock across main wait'
        lock.release()
        assert coordinator.blocked(device)
        return cleared(device, identity)
    def worker():
        claim = None
        try:
            with coordinator.final('cuda:0'):
                claim = queue.acquire('cuda:0', retire, timeout=1)
        finally:
            if claim is not None:
                claim.release()
    thread, result, _done = run_worker(worker)
    wait_pending(queue)
    queue.service()
    thread.join(1)
    assert not thread.is_alive() and 'error' not in result
    assert not coordinator.blocked('cuda:0') and not queue.blocked('cuda:0')
