"""Behavior of one device's nonblocking FSPT work admission."""
import pytest


def _arbiter():
    from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter

    return DeviceArbiter()


def test_another_view_cannot_submit_until_delayed_completion_releases_lease():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert active is not None

    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-b") is None
    arbiter.release(active)

    lease = arbiter.try_acquire("view-b")
    assert lease is not None


def test_stale_lease_cannot_retire_a_new_quantum_even_for_same_owner():
    arbiter = _arbiter()
    old = arbiter.try_acquire("view-a")
    arbiter.release(old)
    current = arbiter.try_acquire("view-a")

    with pytest.raises(ValueError):
        arbiter.release(old)
    assert arbiter.try_acquire("view-b") is None
    arbiter.release(current)
    assert arbiter.try_acquire("view-b") is not None


def test_no_lease_cannot_be_released_when_device_is_idle():
    arbiter = _arbiter()

    with pytest.raises(ValueError):
        arbiter.release(None)


def test_final_intent_takes_the_next_slot_and_preserves_the_viewport_waiters():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("final", final=True) is None
    arbiter.release(active)

    assert arbiter.try_acquire("view-b") is None
    final = arbiter.try_acquire("final", final=True)
    assert final is not None
    assert arbiter.try_acquire("view-b") is None
    arbiter.release(final)
    assert arbiter.try_acquire("view-b") is not None


def test_withdrawal_removes_final_priority_without_cancelling_other_requests():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("first-final", final=True) is None
    assert arbiter.try_acquire("second-final", final=True) is None
    arbiter.withdraw_final("first-final")
    arbiter.release(active)

    assert arbiter.try_acquire("view-b") is None
    final = arbiter.try_acquire("second-final", final=True)
    assert final is not None
    arbiter.release(final)
    assert arbiter.try_acquire("view-b") is not None


def test_withdrawing_active_final_only_changes_intent():
    arbiter = _arbiter()
    final = arbiter.try_acquire("final", final=True)
    arbiter.withdraw_final("final")

    assert arbiter.try_acquire("view") is None
    arbiter.release(final)
    assert arbiter.try_acquire("view") is not None


def test_repeated_viewport_requests_coalesce_and_three_views_stay_fifo():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-c") is None
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-b") is None
    arbiter.release(active)

    assert arbiter.try_acquire("view-a") is None
    assert arbiter.try_acquire("view-c") is None
    second = arbiter.try_acquire("view-b")
    assert second is not None
    arbiter.release(second)

    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-a") is None
    third = arbiter.try_acquire("view-c")
    assert third is not None
    arbiter.release(third)
    assert arbiter.try_acquire("view-a") is not None


def test_cancel_drops_only_that_owner_pending_requests_and_preserves_active_lease():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-c") is None
    assert arbiter.try_acquire("view-a", final=True) is None

    arbiter.cancel("view-a")
    arbiter.cancel("view-b")
    assert arbiter.try_acquire("view-c") is None
    arbiter.release(active)

    assert arbiter.try_acquire("view-c") is not None


def test_same_device_users_share_admission_and_other_devices_are_independent():
    from fluidSim_warp_blender51.fspt.scheduler import get_arbiter

    first = get_arbiter("test-cuda:0")
    same = get_arbiter("test-cuda:0")
    other = get_arbiter("test-cuda:1")
    lease = first.try_acquire("view-a")
    independent = other.try_acquire("view-a")
    assert lease is not None and independent is not None
    assert same.try_acquire("view-b") is None

    first.release(lease)
    next_lease = same.try_acquire("view-b")
    assert next_lease is not None
    same.release(next_lease)
    other.release(independent)


def test_owner_matching_uses_identity_without_invoking_user_equality():
    class Owner:
        def __eq__(self, _other):
            raise AssertionError("Owner equality invokes caller code")

    arbiter = _arbiter()
    first, second, third = Owner(), Owner(), Owner()
    active = arbiter.try_acquire(first)
    assert arbiter.try_acquire(second) is None
    assert arbiter.try_acquire(third) is None
    arbiter.cancel(second)
    arbiter.release(active)
    assert arbiter.try_acquire(third) is not None


def test_two_threads_trying_to_acquire_the_same_device_grant_one_lease():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    arbiter = _arbiter()
    start = Barrier(2)
    owners = object(), object()

    def acquire(owner):
        start.wait(timeout=5)
        return arbiter.try_acquire(owner)

    with ThreadPoolExecutor(max_workers=2) as workers:
        leases = tuple(workers.map(acquire, owners, timeout=10))

    assert sum(lease is not None for lease in leases) == 1
    owner = next(owner for owner, lease in zip(owners, leases, strict=True) if lease is None)
    arbiter.release(next(lease for lease in leases if lease is not None))
    assert arbiter.try_acquire(owner) is not None


def test_withdrawing_only_final_intent_preserves_same_owners_queued_viewport():
    arbiter = _arbiter()
    active = arbiter.try_acquire("view-a")
    assert arbiter.try_acquire("view-b") is None
    assert arbiter.try_acquire("view-b", final=True) is None
    arbiter.withdraw_final("view-b")
    arbiter.release(active)

    assert arbiter.try_acquire("view-c") is None
    assert arbiter.try_acquire("view-b") is not None
