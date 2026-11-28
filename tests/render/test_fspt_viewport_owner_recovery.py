"""Independent root-integration probes: failed ownership and first subscription."""
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from test_fspt_engine_boundary import Pointer, _optional_runtime, adapter  # noqa: F401


def test_failed_view_close_remains_visible_to_final_cleanup(adapter, monkeypatch):  # noqa: F811
    a = adapter
    a.engine.view_update(a.context, a.graph)
    view = a.engine._view
    original_close = a.backend.close
    def failed_close():
        raise RuntimeError('review: viewport owner retirement failed')
    monkeypatch.setattr(a.backend, 'close', failed_close)
    _optional_runtime(a, monkeypatch)
    try:
        with pytest.raises(RuntimeError, match='viewport owner retirement failed'):
            a.engine.close()
        assert view.backend is a.backend
        with pytest.raises(RuntimeError, match='viewport owner retirement failed'):
            a.module._retire_device_for_final('cuda:0', object())
    finally:
        monkeypatch.setattr(a.backend, 'close', original_close)
        a.engine.close()


def test_first_subscription_audits_the_same_ambiguous_fluid_batch(adapter, monkeypatch):  # noqa: F811
    from fluidSim_warp_blender51.blender import fspt_device_scene
    from fluidSim_warp_blender51.blender import surface_publication as publication

    a = adapter
    assert a.engine._device_subscription_key is None
    mesh = Pointer(9, bl_rna=SimpleNamespace(identifier='Mesh'))
    surface = Pointer(8, type='MESH', data=mesh, matrix_world=np.eye(4), material_slots=[],
                      fluidsim=SimpleNamespace(role='NONE'), bl_rna=SimpleNamespace(identifier='Object'))
    stamp = publication.SurfacePublication(
        1, (None, 'FluidSim Output'), 1, publication.SurfaceChannel.EVALUATED_MESH,
        5, publication.SurfaceSource.LIVE, 1, 1, 1, 0.1, None, None, None,
        8, 9, 3, 1, True, monotonic(), source_timeline_frame=1)
    monkeypatch.setattr(publication, 'current_surface_publication', lambda _: stamp)
    monkeypatch.setattr(a.bridge, '_surface_object', lambda _: surface)
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    reasons = []
    _optional_runtime(a, monkeypatch,
                      subscribe=lambda *_args, **_kwargs: lambda: None,
                      set_capability=lambda *_, reason: reasons.append(reason),
                      poll=lambda *_: None)
    a.graph.updates = [SimpleNamespace(id=surface), SimpleNamespace(id=mesh)]
    a.engine.view_update(a.context, a.graph)
    assert not a.engine._view.error
    assert a.engine._device_subscription_key == (1, 2, 'cuda:0')
    assert a.engine._edited_surface_generation == stamp.generation
    assert reasons and reasons[-1]


def test_pending_receipt_fluid_edit_is_latched_before_sync_only_ack(adapter, monkeypatch):  # noqa: F811
    from fluidSim_warp_blender51.blender import fspt_device_scene
    from fluidSim_warp_blender51.blender import surface_publication as publication

    a = adapter
    a.engine.view_update(a.context, a.graph)
    mesh = Pointer(9, bl_rna=SimpleNamespace(identifier='Mesh'))
    surface = Pointer(8, type='MESH', data=mesh, matrix_world=np.eye(4), material_slots=[],
                      fluidsim=SimpleNamespace(role='NONE'), bl_rna=SimpleNamespace(identifier='Object'))
    # A real pending host publication has no eligible receipt. Its later
    # synchronization-only acknowledgement need not rewrite the mesh, so a
    # manual edit in this interval cannot be forgotten by GPU route selection.
    monkeypatch.setattr(publication, 'current_surface_publication', lambda _: None)
    monkeypatch.setattr(a.bridge, '_surface_object', lambda _: surface)
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    a.engine._device_subscription_key = (1, 2, 'cuda:0')
    a.engine._device_unsubscribe = lambda: None
    reasons = []
    _optional_runtime(a, monkeypatch,
                      set_capability=lambda *_, reason: reasons.append(reason),
                      poll=lambda *_: None)
    a.graph.updates = [SimpleNamespace(id=surface), SimpleNamespace(id=mesh)]
    a.engine.view_update(a.context, a.graph)
    assert not a.engine._view.error
    assert a.engine._edited_surface_generation is not None
    assert reasons and reasons[-1]


def test_failed_close_registry_owns_view_until_explicit_checked_retry():
    import gc
    import weakref

    from test_fspt_viewport import PendingBackend
    from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot, Material, SceneSnapshot
    from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter
    from fluidSim_warp_blender51.fspt.viewport import ViewportCoordinator, ViewportState, ViewRequest

    coordinator = ViewportCoordinator()
    backend = PendingBackend('cuda:0', DeviceArbiter())
    original_close = backend.close
    def failed_close():
        raise RuntimeError('review: close fault')
    backend.close = failed_close
    def make_failed():
        view = ViewportState(coordinator, lambda _: backend)
        view.request_update(ViewRequest(CameraSnapshot(np.eye(4)), 4, 3, 1, 6, 1, 'cuda:0'))
        scene = SceneSnapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0],
                              (Material(),), (0.1, 0.2, 0.3), frame=1)
        view.capture(lambda: scene)
        reference = weakref.ref(view)
        try:
            view.close()
        except RuntimeError:
            pass
        return reference
    reference = make_failed()
    gc.collect()
    assert reference() is not None
    assert reference() in coordinator.views
    backend.close = original_close
    reference().retire_backend()
    gc.collect()
    assert reference() is None
    assert not coordinator.views and not coordinator.retirement_failures
