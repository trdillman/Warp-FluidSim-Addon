"""Independent failure-only admission hold controls."""
import pytest

from fluidSim_warp_blender51.fspt.final_handoff import (
    FinalHandoff,
    FinalHandoffError,
    FinalRetirementReceipt,
)


def ready(device, identity):
    return FinalRetirementReceipt(device, identity, 0, 0, True)


def test_failure_tokens_are_independent_and_never_receipts():
    manager = FinalHandoff()
    first = manager.retain_failure('cuda:0')
    second = manager.retain_failure('cuda:0')
    other = manager.retain_failure('cuda:1')
    assert first.receipt is second.receipt is other.receipt is None
    manager.cancel_pending('history')
    assert manager.service() == 0
    with pytest.raises(FinalHandoffError):
        manager.acquire('cuda:0', ready)
    first.release()
    first.release()
    assert manager.blocked('cuda:0') and manager.blocked('cuda:1')
    second.release()
    assert not manager.blocked('cuda:0') and manager.blocked('cuda:1')
    claim = manager.acquire('cuda:0', ready)
    first.release()
    assert manager.blocked('cuda:0')
    claim.release()
    other.release()
    assert not manager.blocked('cuda:0') and not manager.blocked('cuda:1')


def test_a_late_failure_hold_cannot_receive_a_final_allocation_claim():
    manager = FinalHandoff()
    tokens = []
    def retiring(device, identity):
        # Another already-owned backend fails checked retirement while this
        # main-thread handoff is running. A zero receipt for its own drain does
        # not retire that independent failure token.
        tokens.append(manager.retain_failure(device))
        return ready(device, identity)
    try:
        with pytest.raises(FinalHandoffError):
            claim = manager.acquire('cuda:0', retiring)
            claim.release()
        assert manager.blocked('cuda:0')
    finally:
        for token in tokens:
            token.release()
    assert not manager.blocked('cuda:0')
