"""Lifecycle cleanup reaches failed views after their Blender engine disappears."""
import gc
import weakref

import pytest
from test_fspt_engine_boundary import _optional_runtime, adapter  # noqa: F401
from test_fspt_final_engine_transitions import _peer


def _failed_orphan(a, device, *, recoverable):
    state = {'recoverable': False, 'attempts': 0, 'closed': False}
    def close():
        state['attempts'] += 1
        if not state['recoverable']:
            raise RuntimeError('retained orphan is still busy')
        state['closed'] = True
    view = _peer(a, device, close=close)
    with pytest.raises(RuntimeError, match='orphan'):
        view.close()
    reference = weakref.ref(view)
    assert not any(engine._view is view for engine in a.module._ENGINES)
    state['recoverable'] = recoverable
    return reference, state


@pytest.mark.parametrize('boundary', ['history', 'unregister'])
def test_lifecycle_retries_orphan_without_any_external_view_or_engine_owner(adapter, monkeypatch, boundary):  # noqa: F811
    a = adapter
    a.engine.view_update(a.context, a.graph)
    _optional_runtime(a, monkeypatch, enabled=lambda: False)
    reference, state = _failed_orphan(a, 'cuda:0', recoverable=True)
    gc.collect()
    assert reference() is not None and reference() in a.module._COORDINATOR.retirement_failures
    try:
        cleanup = a.module._on_fspt_view_history_pre if boundary == 'history' else a.module.unregister
        cleanup()
        gc.collect()
        assert state['closed'] and state['attempts'] == 2
        assert reference() is None
        assert not a.module._COORDINATOR.retirement_failures
    finally:
        if reference() is not None:
            state['recoverable'] = True
            reference().retire_backend()


@pytest.mark.parametrize('boundary', ['history', 'unregister'])
def test_one_orphan_fault_keeps_its_owner_but_does_not_skip_other_device(adapter, monkeypatch, boundary):  # noqa: F811
    a = adapter
    a.engine.view_update(a.context, a.graph)
    _optional_runtime(a, monkeypatch, enabled=lambda: False)
    blocked, blocked_state = _failed_orphan(a, 'cuda:0', recoverable=False)
    recovered, recovered_state = _failed_orphan(a, 'cuda:1', recoverable=True)
    cleanup = a.module._on_fspt_view_history_pre if boundary == 'history' else a.module.unregister
    try:
        cleanup()
        gc.collect()
        assert recovered_state['closed'] and recovered() is None
        assert blocked() is not None and blocked().backend.device == 'cuda:0'
        assert blocked() in a.module._COORDINATOR.retirement_failures
        # A second explicit unregister also retries after registration ended.
        blocked_state['recoverable'] = True
        cleanup()
        gc.collect()
        assert blocked_state['closed'] and blocked() is None
    finally:
        for reference, state in ((blocked, blocked_state), (recovered, recovered_state)):
            if reference() is not None:
                state['recoverable'] = True
                reference().retire_backend()


def test_repeated_unregister_finishes_runtime_retirement_after_orphan_retry(adapter, monkeypatch):  # noqa: F811
    a = adapter
    a.engine.view_update(a.context, a.graph)
    reference, state = _failed_orphan(a, 'cuda:0', recoverable=False)
    remaining = {'resource': True}
    def shutdown():
        # The runtime cannot finish its own drain while the failed facade
        # still owns geometry. It needs another explicit call after surrender.
        held = reference() is not None and reference().backend is not None
        remaining['resource'] = held
        return ('cuda:0',) if held else ()
    _optional_runtime(a, monkeypatch, enabled=lambda: False, shutdown=shutdown)
    try:
        a.module.unregister()
        assert remaining['resource'] and reference() is not None
        state['recoverable'] = True
        a.module.unregister()
        assert state['closed'] and not remaining['resource']
    finally:
        if reference() is not None:
            state['recoverable'] = True
            reference().retire_backend()


def test_lifecycle_recovery_of_a_still_live_engine_finishes_engine_cleanup(adapter, monkeypatch):  # noqa: F811
    a = adapter
    a.engine.view_update(a.context, a.graph)
    _optional_runtime(a, monkeypatch, enabled=lambda: False)
    original = a.backend.close
    attempts = []
    def transient():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError('first explicit close could not finish')
        original()
    monkeypatch.setattr(a.backend, 'close', transient)
    a.module._on_fspt_view_history_pre()
    # Retiring only the linked ViewportState would strand this engine with a
    # closed view that can never accept a new request after history returns.
    assert a.engine._view is None
    assert not a.module._COORDINATOR.retirement_failures
    a.engine.view_update(a.context, a.graph)
    assert a.engine._view.request is not None and not a.engine._view.closed
