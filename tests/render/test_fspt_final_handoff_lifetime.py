"""Independent cancellation, ownership, and multi-device Final handoff controls."""
import gc
from threading import Event, Thread
from time import monotonic, sleep
from weakref import ref

import pytest

from fluidSim_warp_blender51.fspt.final_handoff import (
    FinalHandoff, FinalHandoffError, FinalHandoffTimeout, FinalRetirementReceipt,
)


def receipt(device, identity):
    return FinalRetirementReceipt(device, identity, 0, 0, True)


def start(function):
    outcome = {}
    done = Event()
    def work():
        try:
            outcome['value'] = function()
        except BaseException as exc:
            # Do not let an exception traceback create the lifetime being tested.
            outcome['error_type'], outcome['error'] = type(exc), str(exc)
        finally:
            done.set()
    thread = Thread(target=work)
    thread.start()
    return thread, done, outcome


def pending(queue, count=1):
    until = monotonic() + 2
    while queue.pending_count != count and monotonic() < until:
        sleep(0.001)
    assert queue.pending_count == count


def test_unstarted_timeout_releases_callback_owners_without_a_future_main_service():
    queue = FinalHandoff()
    class Owner:
        pass
    owner = Owner()
    alive = ref(owner)
    def build_callback(owned):
        def cleanup(device, identity):
            assert owned is not None
            return receipt(device, identity)
        return cleanup
    callback = build_callback(owner)
    thread, done, outcome = start(lambda owned_callback=callback: queue.acquire('cuda:0', owned_callback, timeout=0.02))
    assert done.wait(2)
    thread.join()
    assert outcome['error_type'] is FinalHandoffTimeout
    assert not queue.blocked('cuda:0') and queue.pending_count == 0
    del callback, owner
    gc.collect()
    assert alive() is None, 'cancelled queue entry retained its unstarted callback owner'


def test_running_cancel_keeps_hold_until_callback_returns_and_cannot_authorize_final():
    queue = FinalHandoff()
    observations = []
    thread, done, outcome = None, None, None
    def cleanup(device, identity):
        queue.cancel_pending('review teardown')
        assert done.wait(2)
        observations.append(queue.blocked(device))
        with pytest.raises(FinalHandoffError, match='already'):
            queue.acquire(device, receipt)
        return receipt(device, identity)
    thread, done, outcome = start(lambda: queue.acquire('cuda:0', cleanup, timeout=1))
    pending(queue)
    assert queue.service() == 1
    thread.join(2)
    assert not thread.is_alive() and observations == [True]
    assert outcome['error_type'] is FinalHandoffError and 'review teardown' in outcome['error']
    assert 'value' not in outcome and not queue.blocked('cuda:0')


def test_failed_queued_device_does_not_starve_another_devices_retirement():
    queue = FinalHandoff()
    def fail(*_):
        raise RuntimeError('review reader drain fault')
    first, first_done, first_result = start(lambda: queue.acquire('cuda:0', fail, timeout=1))
    second, second_done, second_result = start(lambda: queue.acquire('cuda:1', receipt, timeout=1))
    pending(queue, 2)
    assert queue.service() == 2
    assert first_done.wait(2) and second_done.wait(2)
    first.join()
    second.join()
    assert first_result['error_type'] is FinalHandoffError
    claim = second_result['value']
    assert not queue.blocked('cuda:0') and queue.blocked('cuda:1')
    claim.release()
    assert not queue.blocked('cuda:1')


@pytest.mark.parametrize('which', ['bytes', 'generations'])
def test_false_boolean_is_not_an_integer_zero_resource_receipt(which):
    queue = FinalHandoff()
    def cleanup(device, identity):
        return FinalRetirementReceipt(device, identity,
                                      False if which == 'bytes' else 0,
                                      False if which == 'generations' else 0, True)
    with pytest.raises(FinalHandoffError, match='receipt'):
        queue.acquire('cuda:0', cleanup)
    assert not queue.blocked('cuda:0')
