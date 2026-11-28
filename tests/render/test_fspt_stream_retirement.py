"""Explicit native-handle ownership controls, independent of Python refcounts."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_fspt_async_backend import backend_setup  # noqa: F401

from fluidSim_warp_blender51.fspt import async_backend as module


@pytest.fixture
def owned(monkeypatch):
    calls, results = [], [0]
    failures = SimpleNamespace(constructor=False, unregister=False)

    def unregister(context, handle):
        calls.append(('unregister', handle))
        if failures.unregister:
            raise RuntimeError('unregister fault')

    device = SimpleNamespace(context=9, runtime=SimpleNamespace(core=SimpleNamespace(
        wp_cuda_stream_unregister=unregister)))

    class Borrowed:
        is_blocking = False
        name = 'render'

        def __init__(self, device, cuda_stream):
            self.device, self.cuda_stream = device, cuda_stream
            calls.append(('register', cuda_stream))
            if failures.constructor:
                raise RuntimeError('constructor fault')

        def __del__(self):
            pytest.fail('borrowed destructor must not unregister an explicitly retired stream')

    def destroy(wp, device, handle):
        calls.append(('destroy', handle))
        return results.pop(0) if len(results) > 1 else results[0]

    driver = SimpleNamespace(create_nonblocking=lambda *_: 9876, destroy=destroy)
    wp = SimpleNamespace(Stream=Borrowed, ScopedDevice=lambda _: nullcontext())
    monkeypatch.setattr(module, 'cuda_driver', lambda: driver)
    before = set(module._UNRETIRED_STREAMS)
    yield SimpleNamespace(wp=wp, device=device, calls=calls, results=results, failures=failures)
    failures.unregister = False
    results[:] = [0]
    for stream in set(module._UNRETIRED_STREAMS) - before:
        stream.retire()


def test_explicit_retirement_works_with_late_borrowed_references_and_is_idempotent(owned):
    stream = module._nonblocking_stream(owned.wp, owned.device)
    borrowed, receipt = stream, stream.retirement
    stream.retire()
    assert receipt.result == 0 and receipt.attempts == 1
    assert borrowed.cuda_stream is None
    stream.retire()
    stream.__del__()
    assert owned.calls == [('register', 9876), ('unregister', 9876), ('destroy', 9876)]


def test_wrapper_allocation_failure_happens_before_any_native_handle_exists(owned):
    driver = module.cuda_driver()
    creations = []
    driver.create_nonblocking = lambda *_: creations.append(True) or 9876

    class FailedWrapper:
        def __new__(cls):
            raise RuntimeError('wrapper allocation failed')

    owned.wp.Stream = FailedWrapper
    with pytest.raises(RuntimeError, match='wrapper allocation'):
        module._nonblocking_stream(owned.wp, owned.device)
    assert creations == [] and owned.calls == []


def test_failed_destroy_retains_handle_and_retries_without_double_unregister(owned):
    owned.results[:] = [201, 0]
    stream = module._nonblocking_stream(owned.wp, owned.device)
    with pytest.raises(RuntimeError, match='cuStreamDestroy.*201'):
        stream.retire()
    assert stream.cuda_stream == stream._native_handle == 9876
    assert stream in module._UNRETIRED_STREAMS and stream.retirement.result == 201
    with pytest.raises(RuntimeError, match='retiring'):
        stream.completed(None)
    stream.retire()
    assert stream.retirement.result == 0 and stream.retirement.attempts == 2
    assert stream not in module._UNRETIRED_STREAMS
    assert owned.calls.count(('unregister', 9876)) == 1
    assert owned.calls.count(('destroy', 9876)) == 2


def test_failed_unregister_never_destroys_before_checked_unregister_retry(owned):
    stream = module._nonblocking_stream(owned.wp, owned.device)
    owned.failures.unregister = True
    with pytest.raises(RuntimeError, match='unregister fault'):
        stream.retire()
    assert stream.retirement.result is None and stream.retirement.attempts == 0
    assert ('destroy', 9876) not in owned.calls
    owned.failures.unregister = False
    stream.retire()
    assert stream.retirement.result == 0 and stream.cuda_stream is None


@pytest.mark.parametrize('destroy_fails', [False, True])
def test_constructor_failure_cleanup_retains_a_retryable_owner_if_destroy_fails(owned, destroy_fails):
    owned.failures.constructor = True
    owned.results[:] = [201, 0] if destroy_fails else [0]
    with pytest.raises(RuntimeError) as caught:
        module._nonblocking_stream(owned.wp, owned.device)
    if destroy_fails:
        assert isinstance(caught.value, module._StreamCreationError)
        stream = caught.value.stream
        assert stream in module._UNRETIRED_STREAMS and stream.cuda_stream == 9876
        stream.retire()
        assert stream.retirement.result == 0
    else:
        assert str(caught.value) == 'constructor fault'
    assert owned.calls.count(('unregister', 9876)) == 1
    assert owned.calls.count(('destroy', 9876)) == (2 if destroy_fails else 1)


def test_destructor_fault_preserves_uncertain_handle_for_explicit_retirement(owned):
    import gc
    import weakref

    owned.results[:] = [201, 0]
    stream = module._nonblocking_stream(owned.wp, owned.device)
    weak = weakref.ref(stream)
    del stream
    gc.collect()
    retained = weak()
    assert retained is not None and retained in module._UNRETIRED_STREAMS
    assert retained.cuda_stream == 9876
    retained.retire()
    assert retained.retirement.result == 0 and retained not in module._UNRETIRED_STREAMS


def test_backend_close_cannot_report_closed_or_drop_wrapper_before_native_success(
        owned, backend_setup):  # noqa: F811
    backend, _, _, scene, camera = backend_setup
    # Restore the actual constructor after backend_setup's dispatch double.
    stream = _ACTUAL_FACTORY(owned.wp, owned.device)
    backend._render_stream = stream
    owned.results[:] = [201, 0]
    with pytest.raises(RuntimeError, match='cuStreamDestroy.*201'):
        backend.close()
    assert not backend.closed and backend._render_stream is stream
    with pytest.raises(RuntimeError, match='closed'):
        backend.submit(scene, camera, 2, 2)
    backend.close()
    assert backend.closed and backend._render_stream is None
    assert stream.retirement.result == 0
    backend.close()
    assert owned.calls.count(('destroy', 9876)) == 2


def test_failed_constructor_handle_transfers_to_backend_for_explicit_close_retry(
        owned, backend_setup, monkeypatch):  # noqa: F811
    backend, _, _, scene, camera = backend_setup
    owned.failures.constructor = True
    owned.results[:] = [201, 201, 0]
    monkeypatch.setattr(module, '_nonblocking_stream',
                        lambda *_: _ACTUAL_FACTORY(owned.wp, owned.device))
    with pytest.raises(module._StreamCreationError):
        backend.submit(scene, camera, 2, 2)
    stream = backend._render_stream
    assert stream is not None and stream.cuda_stream == 9876
    assert not backend.pending and not backend.closed
    with pytest.raises(RuntimeError, match='cuStreamDestroy.*201'):
        backend.close()
    assert backend._render_stream is stream and not backend.closed
    backend.close()
    assert backend.closed and stream.retirement.result == 0
    assert owned.calls.count(('unregister', 9876)) == 1
    assert owned.calls.count(('destroy', 9876)) == 3


@pytest.mark.parametrize('result', [None, 201])
def test_legacy_unproven_receipt_never_becomes_closed_on_retry(backend_setup, result):  # noqa: F811
    backend, _, _, _, _ = backend_setup
    stream = SimpleNamespace(retirement=module._StreamRetirement(result=result))
    backend._render_stream = stream
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='receipt|cuStreamDestroy'):
                backend.close()
            assert backend._render_stream is stream and not backend.closed
    finally:
        stream.retirement.result = 0
        backend.close()


_ACTUAL_FACTORY = module._nonblocking_stream
