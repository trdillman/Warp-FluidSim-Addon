"""Read-only engine integration review controls; native GPU proof is separate."""
from types import SimpleNamespace
from threading import Event, Thread, current_thread, main_thread
from time import monotonic, sleep

import pytest

from test_fspt_engine_boundary import adapter as engine_adapter, _optional_runtime

adapter = engine_adapter


def test_opt_in_cannot_allocate_beside_a_failed_default_off_final(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable

    a = adapter
    runtime = _optional_runtime(a, monkeypatch, enabled=lambda: False)
    real_close = a.backend.close
    allocations = []

    def fail_close():
        raise RuntimeError('review owned Final stream did not retire')

    monkeypatch.setattr(a.backend, 'close', fail_close)
    try:
        with pytest.raises(RuntimeError, match='owned Final'):
            with a.module._final_backend('cuda:0') as final:
                assert final is a.backend
        assert len(a.module._FAILED_FINAL_OWNERS) == 1
        assert a.module._FAILED_FINAL_OWNERS[0][0] is a.backend

        runtime.enabled = lambda: True

        def new_resource(_):
            allocations.append('optional device resource')
            return SimpleNamespace(broker=SimpleNamespace(new_backend=lambda: object()))

        runtime.device_resources = new_resource
        with pytest.raises(DeviceSceneUnavailable):
            a.module._viewport_backend('cuda:0')
        assert allocations == [], 'optional geometry allocated beside uncertain Final owners'
    finally:
        monkeypatch.setattr(a.backend, 'close', real_close)
        a.module._retire_failed_finals()


def test_late_opt_in_final_cannot_issue_zero_owner_receipt_with_failed_default_final(
        adapter, monkeypatch):
    a = adapter
    runtime = _optional_runtime(a, monkeypatch, enabled=lambda: False)
    real_close = a.backend.close

    def fail_close():
        raise RuntimeError('review retained Final geometry')

    monkeypatch.setattr(a.backend, 'close', fail_close)
    try:
        with pytest.raises(RuntimeError, match='retained Final'):
            with a.module._final_backend('cuda:0'):
                pass
        runtime.enabled = lambda: True
        with pytest.raises(RuntimeError):
            a.module._retire_device_for_final('cuda:0', object())
    finally:
        monkeypatch.setattr(a.backend, 'close', real_close)
        a.module._retire_failed_finals()


@pytest.mark.parametrize('retirement_fault', [False, True])
def test_engine_final_worker_requires_main_retirement_before_constructor(
        adapter, monkeypatch, retirement_fault):
    from fluidSim_warp_blender51.fspt import final_handoff

    a = adapter
    a.engine.view_update(a.context, a.graph)
    _optional_runtime(a, monkeypatch)
    manager = final_handoff.MAIN_FINAL_HANDOFF
    events, outcome, done = [], {}, Event()
    real_close = a.backend.close

    def close_view():
        events.append(('view-close', current_thread()))
        if retirement_fault:
            raise RuntimeError('review checked viewport retirement fault')
        real_close()

    monkeypatch.setattr(a.backend, 'close', close_view)

    class FinalBackend:
        def close(self):
            events.append(('final-close', current_thread()))
            assert manager.blocked('cuda:0')

    def construct(_):
        events.append(('final-construct', current_thread()))
        assert manager.blocked('cuda:0')
        assert a.engine._view.backend is None
        return FinalBackend()

    monkeypatch.setattr(a.module, '_backend', construct)

    def render():
        try:
            with a.module._final_backend('cuda:0'):
                events.append(('final-work', current_thread()))
        except BaseException as exc:
            outcome['error'] = str(exc)
        finally:
            done.set()

    worker = Thread(target=render)
    worker.start()
    try:
        deadline = monotonic() + 2.0
        while not manager.pending_count and monotonic() < deadline:
            sleep(0.001)
        assert manager.pending_count == 1 and not done.is_set()
        assert not any(name == 'final-construct' for name, _ in events)
        assert manager.service() == 1
        assert done.wait(2.0)
        worker.join()
        assert events[0] == ('view-close', main_thread())
        if retirement_fault:
            assert 'retirement fault' in outcome['error']
            assert events == [('view-close', main_thread())]
            assert a.engine._view.backend is a.backend
        else:
            assert not outcome
            assert events[1:] == [('final-construct', worker), ('final-work', worker),
                                  ('final-close', worker)]
        assert not manager.blocked('cuda:0')
        assert not a.module._COORDINATOR.blocked('cuda:0')
    finally:
        monkeypatch.setattr(a.backend, 'close', real_close)
        manager.cancel_pending('review cleanup')
        worker.join(2.0)


def test_first_optional_demand_retires_existing_plain_geometry_before_subscribing(
        adapter, monkeypatch):
    from fluidSim_warp_blender51.blender import fspt_device_scene

    a = adapter
    a.engine.view_update(a.context, a.graph)
    view = a.engine._view
    assert view.backend is a.backend
    events = []
    real_close = a.backend.close

    def close_plain():
        events.append('close-plain')
        real_close()

    def subscribe(*_, **__):
        assert view.backend is None, 'ordinary geometry still retained at GPU subscription'
        events.append('subscribe')
        return lambda: None

    monkeypatch.setattr(a.backend, 'close', close_plain)
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    _optional_runtime(a, monkeypatch, subscribe=subscribe, set_capability=lambda *_, **__: None)
    a.engine._update_device_demand(a.scene, a.graph.view_layer, a.graph)
    assert events == ['close-plain', 'subscribe']


def test_first_optional_demand_during_final_defers_before_subscribe(adapter, monkeypatch):
    from fluidSim_warp_blender51.blender import fspt_device_scene
    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable

    a = adapter
    a.engine.view_update(a.context, a.graph)
    subscriptions = []
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    _optional_runtime(a, monkeypatch,
                      subscribe=lambda *_, **__: subscriptions.append('subscribe') or (lambda: None),
                      set_capability=lambda *_, **__: None)
    with a.module._COORDINATOR.final('cuda:0'):
        with pytest.raises(DeviceSceneUnavailable):
            a.engine._update_device_demand(a.scene, a.graph.view_layer, a.graph)
        assert subscriptions == []


def test_baseexception_during_final_close_keeps_strong_owner_and_admission_hold(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt import final_handoff

    a = adapter
    _optional_runtime(a, monkeypatch)
    real_close = a.backend.close
    manager = final_handoff.MAIN_FINAL_HANDOFF

    def interrupted_close():
        raise KeyboardInterrupt('review interrupted checked CUDA retirement')

    monkeypatch.setattr(a.backend, 'close', interrupted_close)
    try:
        with pytest.raises(KeyboardInterrupt, match='interrupted checked'):
            with a.module._final_backend('cuda:0'):
                pass
        assert manager.blocked('cuda:0'), 'interrupted close released admission before proof'
        assert a.module._FAILED_FINAL_OWNERS[0][0] is a.backend
    finally:
        monkeypatch.setattr(a.backend, 'close', real_close)
        a.module._retire_failed_finals()


def _peer(a, device, *, optional=False, close=None):
    from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter

    view = a.module.ViewportState(a.module._COORDINATOR, lambda _: None)
    view.request = a.engine._view.request
    view.backend = SimpleNamespace(device=device, arbiter=DeviceArbiter(), owner=object(),
                                   pending=False, close=close or (lambda: None))
    if optional:
        view.backend.broker = object()
    return view


def test_optional_activation_retires_every_raw_peer_only_on_target_device(adapter, monkeypatch):
    from fluidSim_warp_blender51.blender import fspt_device_scene

    a = adapter
    a.engine.view_update(a.context, a.graph)
    events = []
    matching = _peer(a, 'cuda:0', close=lambda: events.append('matching-close'))
    other = _peer(a, 'cuda:1', close=lambda: events.append('other-close'))
    other_backend = other.backend

    def subscribe(*_, **__):
        assert matching.backend is None and a.engine._view.backend is None
        assert other.backend is other_backend
        events.append('subscribe')
        return lambda: None

    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    _optional_runtime(a, monkeypatch, subscribe=subscribe, set_capability=lambda *_, **__: None)
    try:
        a.engine._update_device_demand(a.scene, a.graph.view_layer, a.graph)
        assert events == ['matching-close', 'subscribe']
    finally:
        matching.close()
        other.close()


def test_optional_activation_close_fault_cannot_subscribe_or_create_pool_and_retry_is_possible(
        adapter, monkeypatch):
    from fluidSim_warp_blender51.blender import fspt_device_scene

    a = adapter
    a.engine.view_update(a.context, a.graph)
    events = []

    def failed():
        raise RuntimeError('review old evaluated owner cannot retire')

    matching = _peer(a, 'cuda:0', close=failed)
    retained = matching.backend
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    _optional_runtime(a, monkeypatch,
                      subscribe=lambda *_, **__: events.append('subscribe') or (lambda: None),
                      set_capability=lambda *_, **__: None,
                      device_resources=lambda _: pytest.fail('pool creation crossed failed close'))
    try:
        with pytest.raises(RuntimeError, match='cannot retire'):
            a.engine._update_device_demand(a.scene, a.graph.view_layer, a.graph)
        assert events == [] and matching.backend is retained
        assert matching in a.module._COORDINATOR.retirement_failures
        retained.close = lambda: events.append('retry-close')
        a.engine._update_device_demand(a.scene, a.graph.view_layer, a.graph)
        assert events == ['retry-close', 'subscribe']
        assert matching.backend is None
        assert matching not in a.module._COORDINATOR.retirement_failures
    finally:
        retained.close = lambda: None
        matching.close()


def test_opt_out_timer_detaches_all_optional_peers_before_runtime_retirement(adapter, monkeypatch):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    events = []
    a.backend.broker = object()
    original_close = a.backend.close
    monkeypatch.setattr(a.backend, 'close', lambda: events.append('first-close') or original_close())
    peer = _peer(a, 'cuda:0', optional=True, close=lambda: events.append('second-close'))
    resources = [True]

    def retire():
        assert a.engine._view.backend is None and peer.backend is None
        events.append('runtime-retire')
        resources[0] = False

    _optional_runtime(a, monkeypatch, enabled=lambda: False,
                      has_resources=lambda *_: resources[0], pump=retire)
    try:
        assert a.module._pump_device_runtime() is None
        assert set(events[:-1]) == {'first-close', 'second-close'}
        assert events[-1] == 'runtime-retire'
    finally:
        del a.backend.broker
        peer.close()


def test_opt_out_while_final_claimed_keeps_facades_until_final_owner_retires_them(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt import final_handoff

    a = adapter
    a.engine.view_update(a.context, a.graph)
    a.backend.broker = object()
    manager = final_handoff.MAIN_FINAL_HANDOFF
    # The runtime's real pump also skips held devices. Its spy verifies that
    # the engine does not close the facade ahead of that explicit ownership.
    _optional_runtime(a, monkeypatch, enabled=lambda: False, has_resources=lambda *_: True,
                      pump=lambda: None)
    hold = manager.retain_failure('cuda:0')
    try:
        a.calls.clear()
        assert a.module._pump_device_runtime() > 0
        assert a.engine._view.backend is a.backend and 'close' not in a.calls
    finally:
        hold.release()
        del a.backend.broker
