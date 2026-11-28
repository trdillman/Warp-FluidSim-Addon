"""Adapter controls for asynchronous work and presentation boundaries.

These fakes cover callback ownership; native Blender/CUDA acceptance is separate.
"""
import importlib
import sys
from dataclasses import replace
from time import monotonic
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt.contracts import (
    Material,
    RenderFrame,
    SceneSnapshot,
    frame_key,
)
from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter


class Pointer:
    def __init__(self, token, **values):
        self.session_uid = token
        self.__dict__.update(values)

    def as_pointer(self):
        return self.session_uid


@pytest.fixture
def adapter(monkeypatch):
    calls = []
    bpy = ModuleType("bpy")
    bpy.props = ModuleType("bpy.props")
    for name in ("IntProperty", "PointerProperty", "StringProperty"):
        setattr(bpy.props, name, lambda **_: None)

    class RenderEngine:
        def tag_redraw(self): calls.append("redraw")
        def tag_update(self): calls.append("update")
        def bind_display_space_shader(self, scene): calls.append("bind")
        def unbind_display_space_shader(self): calls.append("unbind")
        def update_stats(self, *args): calls.append("stats")

    bpy.types = SimpleNamespace(RenderEngine=RenderEngine, PropertyGroup=object,
                                Panel=object, Scene=type("Scene", (), {}))
    timers = set()
    bpy.app = SimpleNamespace(
        timers=SimpleNamespace(is_registered=lambda fn: fn in timers,
                               register=lambda fn, **_: timers.add(fn),
                               unregister=timers.remove),
        handlers=SimpleNamespace(persistent=lambda fn: fn, load_pre=[], undo_pre=[], redo_pre=[]))
    bpy.utils = SimpleNamespace(register_class=lambda cls: calls.append(("register", cls)),
                                unregister_class=lambda cls: calls.append(("unregister", cls)))
    for name, value in (("bpy", bpy), ("bpy.props", bpy.props)):
        monkeypatch.setitem(sys.modules, name, value)
    module_name = "fluidSim_warp_blender51.blender.fspt_engine"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    engine = module.FSPTRenderEngine()
    monkeypatch.setattr(engine, "_draw_error", lambda message: calls.append(("display", message)))
    layer = Pointer(2, material_override=None, world_override=None)
    scene = Pointer(1, world=None, frame_current=1,
                    render=SimpleNamespace(engine=module.ENGINE_ID),
                    fspt=SimpleNamespace(viewport_samples=1, max_bounces=6, seed=31,
                                         device="cuda:0", resolution_percentage=100))
    region = Pointer(5, width=4, height=3)
    area = Pointer(4, type="VIEW_3D", regions=[region],
                   spaces=SimpleNamespace(active=SimpleNamespace(shading=SimpleNamespace(type="RENDERED"))))
    window = Pointer(3, scene=scene, view_layer=layer, screen=SimpleNamespace(areas=[area]))
    context = SimpleNamespace(window=window, area=area, region=region,
                              region_data=SimpleNamespace(perspective_matrix=SimpleNamespace(
                                  inverted=lambda: np.eye(4))))
    bpy.context = SimpleNamespace(window_manager=SimpleNamespace(windows=[window]))
    graph = SimpleNamespace(scene=scene, view_layer=layer, updates=[])
    snapshot = SceneSnapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0],
                             (Material(),), (0.1, 0.2, 0.3), frame=1)

    class Backend:
        device = "cuda:0"
        arbiter = DeviceArbiter()
        owner = object()
        pending = False

        def close(self):
            calls.append("close")

        def submit(self, *args, **kwargs):
            calls.append("submit")
            return True

        def collect(self, **kwargs):
            calls.append(("collect", kwargs))
            return None

    backend = Backend()
    monkeypatch.setattr(module, "_backend", lambda device: backend)
    bridge = importlib.import_module("fluidSim_warp_blender51.blender.fspt_bridge")
    extraction = importlib.import_module("fluidSim_warp_blender51.blender.fspt_scene")
    monkeypatch.setattr(bridge, "validate_publications", lambda *_, **__: None)
    monkeypatch.setattr(extraction, "extract_scene", lambda *_, **__: calls.append("extract") or snapshot)
    gpu = ModuleType("gpu")
    gpu.types = SimpleNamespace(
        Buffer=lambda *args: calls.append("buffer") or object(),
        GPUTexture=lambda *args, **kwargs: calls.append("texture") or object())
    gpu.state = SimpleNamespace(blend_get=lambda: "NONE", blend_set=lambda mode: None)
    presets = ModuleType("gpu_extras.presets")
    presets.draw_texture_2d = lambda *args: calls.append("draw")
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setitem(sys.modules, "gpu_extras.presets", presets)
    module.register()
    yield SimpleNamespace(module=module, engine=engine, bpy=bpy, calls=calls, scene=scene,
                          context=context, graph=graph, snapshot=snapshot, backend=backend,
                          bridge=bridge, extraction=extraction, timers=timers)
    engine.close()
    module.unregister()
    monkeypatch.delitem(sys.modules, module_name, raising=False)


def _present(adapter):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    view = a.engine._view
    request = view.request
    frame = RenderFrame(np.ones((3, 4, 4)),
                        frame_key(a.snapshot, request.camera, 4, 3, 6, 31),
                        1, "cuda:0", "boundary-test", "host", 1)
    view.presentable, view.presented_snapshot = frame, a.snapshot
    return frame


def test_repeated_draws_only_upload_a_new_completed_image_once(adapter):
    a = adapter
    _present(a)
    a.calls.clear()
    a.engine.view_draw(a.context, a.graph)
    a.engine.view_draw(a.context, a.graph)
    assert a.calls.count("draw") == 2
    assert a.calls.count("texture") == a.calls.count("buffer") == 1
    assert not {"submit", "extract", "close"}.intersection(a.calls)
    assert not any(isinstance(c, tuple) and c[0] == "collect" for c in a.calls)
    assert a.module.LAST_VIEWPORT_INFO["status"] == "COMPLETE"


def test_draw_without_completion_schedules_work_without_capturing_or_building_backend(adapter, monkeypatch):
    a = adapter
    monkeypatch.setattr(a.module, "_backend", lambda _: pytest.fail("draw constructed CUDA backend"))
    a.engine.view_draw(a.context, a.graph)
    assert a.engine._view.backend is None
    assert a.module._pump_views in a.timers
    assert "extract" not in a.calls and "texture" not in a.calls
    assert a.module.LAST_VIEWPORT_INFO["status"] == "WAITING"


def _optional_runtime(adapter, monkeypatch, **overrides):
    runtime = SimpleNamespace(enabled=lambda: True, has_resources=lambda *_: False,
                              resource_error=lambda _: '', device_resources=lambda _: None,
                              pump=lambda: False, shutdown=lambda: (),
                              retire_for_final=lambda _: {'reserved_bytes': 0, 'generation_count': 0})
    for name, value in overrides.items():
        setattr(runtime, name, value)
    monkeypatch.setattr(adapter.module, '_device_runtime', lambda: runtime, raising=False)
    return runtime


def test_optional_resource_pending_never_allocates_untracked_evaluated_backend(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable

    a = adapter
    _optional_runtime(a, monkeypatch, has_resources=lambda *_: True)
    monkeypatch.setattr(a.module, '_backend', lambda _: pytest.fail('untracked geometry allocation'))
    with pytest.raises(DeviceSceneUnavailable):
        a.module._viewport_backend('cuda:0')


def test_disabled_optional_route_still_retains_old_geometry_boundary(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable

    a = adapter
    _optional_runtime(a, monkeypatch, enabled=lambda: False, has_resources=lambda *_: True)
    monkeypatch.setattr(a.module, '_backend', lambda _: pytest.fail('old owners bypassed'))
    with pytest.raises(DeviceSceneUnavailable):
        a.module._viewport_backend('cuda:0')


def test_optional_route_uses_shared_broker_facade(adapter, monkeypatch):
    a = adapter
    facade = object()
    _optional_runtime(a, monkeypatch, device_resources=lambda _: SimpleNamespace(
        broker=SimpleNamespace(new_backend=lambda: facade)))
    monkeypatch.setattr(a.module, '_backend', lambda _: pytest.fail('untracked geometry allocation'))
    assert a.module._viewport_backend('cuda:0') is facade


def test_main_device_timer_services_final_even_without_viewport_work(adapter, monkeypatch):
    a = adapter
    order = []
    _optional_runtime(a, monkeypatch, pump=lambda: order.append('pump'))
    from fluidSim_warp_blender51.fspt import final_handoff
    monkeypatch.setattr(final_handoff.MAIN_FINAL_HANDOFF, 'service', lambda: order.append('handoff'))
    assert a.module._pump_device_runtime() > 0
    assert order == ['handoff', 'pump']


def test_final_cleanup_receipt_follows_all_matching_view_retirements(adapter, monkeypatch):
    a = adapter
    order = []
    a.engine.view_update(a.context, a.graph)
    view = a.engine._view
    close = a.backend.close
    monkeypatch.setattr(a.backend, 'close', lambda: (order.append('view'), close()))
    _optional_runtime(a, monkeypatch, retire_for_final=lambda _: (
        order.append('runtime') or {'reserved_bytes': 0, 'generation_count': 0}))
    identity = object()
    result = a.module._retire_device_for_final('cuda:0', identity)
    assert order == ['view', 'runtime']
    assert result.request_id is identity and result.owners_retired is True
    assert result.reserved_bytes == result.generation_count == 0
    assert view.backend is None and view.request is not None


def test_failed_view_cleanup_does_not_issue_final_zero_receipt(adapter, monkeypatch):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    def failed():
        raise RuntimeError('owned trace could not retire')
    close = a.backend.close
    monkeypatch.setattr(a.backend, 'close', failed)
    _optional_runtime(a, monkeypatch, retire_for_final=lambda _: pytest.fail('early zero receipt'))
    with pytest.raises(RuntimeError, match='owned trace'):
        a.module._retire_device_for_final('cuda:0', object())
    assert a.engine._view.backend is a.backend
    monkeypatch.setattr(a.backend, 'close', close)


def test_checked_final_close_failure_holds_device_until_explicit_retry(adapter, monkeypatch):
    from fluidSim_warp_blender51.fspt import final_handoff
    from fluidSim_warp_blender51.fspt.device_scene import DeviceSceneUnavailable

    a = adapter
    runtime = _optional_runtime(a, monkeypatch)
    manager = final_handoff.MAIN_FINAL_HANDOFF
    closed = []
    def failed_close():
        closed.append('failed')
        raise RuntimeError('CUDA retirement failed')
    monkeypatch.setattr(a.backend, 'close', failed_close)
    with pytest.raises(RuntimeError, match='CUDA retirement'):
        with a.module._final_backend('cuda:0') as backend:
            assert backend is a.backend and manager.blocked('cuda:0')
    assert manager.blocked('cuda:0')
    assert a.module._FAILED_FINAL_OWNERS[0][0] is a.backend
    # Disabling the experiment cannot bypass its still-owned Final claim.
    runtime.enabled = lambda: False
    with pytest.raises(DeviceSceneUnavailable, match='Final rendering'):
        a.module._viewport_backend('cuda:0')
    monkeypatch.setattr(a.backend, 'close', lambda: closed.append('retired'))
    a.module._retire_failed_finals()
    assert not manager.blocked('cuda:0') and not a.module._FAILED_FINAL_OWNERS
    assert closed == ['failed', 'retired']


def test_coalesced_fluid_update_latches_evaluated_route_across_later_publications(adapter, monkeypatch):
    from fluidSim_warp_blender51.blender import fspt_device_scene
    from fluidSim_warp_blender51.blender import surface_publication as publication

    a = adapter
    a.engine.view_update(a.context, a.graph)
    mesh = Pointer(9, bl_rna=SimpleNamespace(identifier='Mesh'))
    surface = Pointer(8, type='MESH', data=mesh, matrix_world=np.eye(4), material_slots=[],
                      fluidsim=SimpleNamespace(role='NONE'), bl_rna=SimpleNamespace(identifier='Object'))
    old = publication.SurfacePublication(
        1, (None, 'FluidSim Output'), 1, publication.SurfaceChannel.EVALUATED_MESH,
        1, publication.SurfaceSource.LIVE, 1, 1, 1, 0.1, None, None, None,
        8, 9, 3, 1, True, monotonic(), source_timeline_frame=1)
    current = [old]
    monkeypatch.setattr(publication, 'current_surface_publication', lambda _: current[0])
    monkeypatch.setattr(a.bridge, '_surface_object', lambda _: surface)
    monkeypatch.setattr(fspt_device_scene, 'viewport_capability', lambda _: '')
    a.engine._device_subscription_key = (1, 2, 'cuda:0')
    a.engine._native_publication = old
    a.engine._device_unsubscribe = lambda: None
    reasons = []
    _optional_runtime(a, monkeypatch,
                      set_capability=lambda *_, reason: reasons.append(reason),
                      poll=lambda *_: None)
    a.graph.updates = [SimpleNamespace(id=surface), SimpleNamespace(id=mesh)]
    # Even an exact owner notification cannot assign every coalesced edit to it.
    newer = replace(old, generation=2, solver_frame=2, state_revision=2,
                    simulation_time=0.2, source_timeline_frame=2)
    current[0] = newer
    a.engine._published(publication.SurfaceChannel.EVALUATED_MESH,
                        publication.ChannelState(status=publication.SurfaceStatus.READY, committed=newer))
    a.engine.view_update(a.context, a.graph)
    assert a.engine._edited_surface_generation == 2
    assert a.engine._view.presentable is None and reasons[-1]
    a.graph.updates = []
    current[0] = replace(newer, generation=3, solver_frame=3, state_revision=3,
                         simulation_time=0.3, source_timeline_frame=3)
    a.engine._published(publication.SurfaceChannel.EVALUATED_MESH,
                        publication.ChannelState(status=publication.SurfaceStatus.READY, committed=current[0]))
    a.engine.view_update(a.context, a.graph)
    assert a.engine._edited_surface_generation == 2 and reasons[-1] == reasons[-2]


def test_busy_device_defers_actual_depsgraph_extraction(adapter):
    a = adapter
    lease = a.backend.arbiter.try_acquire(object())
    a.engine.view_update(a.context, a.graph)
    assert "extract" not in a.calls
    a.backend.arbiter.release(lease)
    a.engine.view_update(a.context, a.graph)
    assert a.calls.count("extract") == 1
    assert a.engine._view.snapshot is a.snapshot


def test_scene_world_reassignment_with_scene_only_update_invalidates_capture(adapter):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    a.scene.world = Pointer(7)
    a.graph.updates = [SimpleNamespace(id=SimpleNamespace(bl_rna=SimpleNamespace(identifier="Scene")))]
    a.engine.view_update(a.context, a.graph)
    assert a.calls.count("extract") == 2


def test_sample_goal_scene_update_does_not_extract_all_triangles_again(adapter):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    a.scene.fspt.viewport_samples = 8
    a.graph.updates = [SimpleNamespace(id=SimpleNamespace(bl_rna=SimpleNamespace(identifier="Scene")))]
    a.engine.view_update(a.context, a.graph)
    assert a.calls.count("extract") == 1
    assert a.engine._view.request.target_samples == 8


def test_publication_retired_between_collect_and_draw_requests_recapture(adapter, monkeypatch):
    a = adapter
    _present(a)
    def stale(*args, **kwargs):
        raise a.bridge.SurfaceNotReadyError("surface replaced")
    monkeypatch.setattr(a.bridge, "validate_viewport_publications", stale)
    monkeypatch.setattr(a.bridge, "validate_publications", stale)
    a.engine.view_draw(a.context, a.graph)
    assert "draw" not in a.calls
    assert a.engine._view.needs_tick and not a.engine._view.error
    assert a.module._pump_views in a.timers
    a.engine._pump_view()
    assert a.engine._view.dirty


def test_registered_history_callbacks_retire_views_and_are_removed(adapter):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    for name in ("load_pre", "undo_pre", "redo_pre"):
        assert getattr(a.bpy.app.handlers, name) == [a.module._on_fspt_view_history_pre]
    a.module._on_fspt_view_history_pre()
    assert a.engine._view is None and "close" in a.calls
    a.module.unregister()
    assert not a.timers
    for name in ("load_pre", "undo_pre", "redo_pre"):
        assert not getattr(a.bpy.app.handlers, name)


@pytest.mark.parametrize("failure", ["bind_display_space_shader", "unbind_display_space_shader"])
def test_draw_failure_restores_blend_state_even_when_shader_binding_fails(adapter, monkeypatch, failure):
    a = adapter
    _present(a)
    blend = ["NONE"]
    gpu = sys.modules["gpu"]
    monkeypatch.setattr(gpu.state, "blend_get", lambda: blend[0])
    monkeypatch.setattr(gpu.state, "blend_set", lambda value: blend.__setitem__(0, value))
    def fail(*args):
        raise RuntimeError("shader binding failure")
    monkeypatch.setattr(a.engine, failure, fail)
    a.engine.view_draw(a.context, a.graph)
    assert a.module.LAST_VIEWPORT_INFO["status"] == "ERROR"
    assert blend == ["NONE"], "FSPT leaked its blend state into the surrounding Blender UI"


@pytest.mark.parametrize("extra_edit", [None, "static_object", "material", "surface_material_binding"])
def test_surface_publication_then_matching_depsgraph_update_preserves_only_compatible_history(
        adapter, monkeypatch, extra_edit):
    from fluidSim_warp_blender51.blender import surface_publication as publication

    a = adapter
    mesh = Pointer(9, bl_rna=SimpleNamespace(identifier="Mesh"))
    surface = Pointer(8, bl_rna=SimpleNamespace(identifier="Object"), type="MESH", data=mesh,
                      matrix_world=np.eye(4), material_slots=[],
                      fluidsim=SimpleNamespace(role="NONE"))
    old = publication.SurfacePublication(
        1, (None, "FluidSim Output"), 1, publication.SurfaceChannel.EVALUATED_MESH,
        1, publication.SurfaceSource.LIVE, 1, 1, 1, 0.1, None, None, None,
        8, 9, 3, 1, True, monotonic(), source_timeline_frame=1)
    new = replace(old, generation=2, timeline_frame=2, source_timeline_frame=2,
                  solver_frame=2, state_revision=2, simulation_time=0.2, published_at=monotonic())
    current = [old]
    monkeypatch.setattr(publication, "current_surface_publication", lambda _: current[0])
    monkeypatch.setattr(a.bridge, "_surface_object", lambda _: surface)
    captured = replace(a.snapshot, publications=(old,))
    monkeypatch.setattr(a.extraction, "extract_scene", lambda *_, **__: captured)
    a.engine.view_update(a.context, a.graph)
    view = a.engine._view
    request = view.request
    frame = RenderFrame(np.ones((3, 4, 4)), frame_key(captured, request.camera, 4, 3, 6, 31),
                        1, "cuda:0", "boundary-test", "host", 1, (old,))
    def submit(*args, **kwargs):
        a.backend.pending = True
        return True
    def collect(**kwargs):
        a.backend.pending = False
        return frame
    monkeypatch.setattr(a.backend, "submit", submit)
    monkeypatch.setattr(a.backend, "collect", collect)
    assert view.tick(lambda _: True) == "submitted"
    current[0] = new
    a.engine._published(publication.SurfaceChannel.EVALUATED_MESH,
                        publication.ChannelState(status=publication.SurfaceStatus.READY, committed=new))
    a.graph.updates = [SimpleNamespace(id=surface), SimpleNamespace(id=mesh)]
    if extra_edit == "static_object":
        a.graph.updates.append(SimpleNamespace(id=Pointer(20, type="MESH",
                                  bl_rna=SimpleNamespace(identifier="Object"))))
    elif extra_edit == "material":
        a.graph.updates.append(SimpleNamespace(id=Pointer(21, bl_rna=SimpleNamespace(identifier="Material"))))
    elif extra_edit == "surface_material_binding":
        surface.material_slots = [SimpleNamespace(link="OBJECT", material=Pointer(22))]
    a.engine.view_update(a.context, a.graph)
    assert view.tick(lambda _: False, presentation_eligible=lambda _: True) == "capture"
    if extra_edit is None:
        assert view.presentable is frame, "matching fluid updates erased every late completed image"
        assert view.presented_snapshot is captured and view.presentable.scene_frame == 1
    else:
        assert view.presentable is None, f"{extra_edit} did not invalidate incompatible pixels"


@pytest.mark.parametrize("role", ["SOURCE", "SINK", "FORCE"])
def test_rendered_mesh_role_change_rechecks_scene_membership(adapter, role):
    a = adapter
    a.engine.view_update(a.context, a.graph)
    obj = Pointer(30, type="MESH", fluidsim=SimpleNamespace(role=role),
                  bl_rna=SimpleNamespace(identifier="Object"))
    a.graph.updates = [SimpleNamespace(id=obj)]
    a.engine.view_update(a.context, a.graph)
    assert a.calls.count("extract") == 2, "excluded role assignment retained the old visible mesh"


def test_retained_display_expiry_does_not_discard_a_newer_pending_job(adapter, monkeypatch):
    a = adapter
    _present(a)
    a.backend.pending = True
    revision = a.engine._view._scene_revision
    def expired(*args, **kwargs):
        raise a.bridge.SurfaceNotReadyError("Retained image publication age exceeded 250 ms")
    monkeypatch.setattr(a.bridge, "validate_viewport_publications", expired)
    a.engine.view_draw(a.context, a.graph)
    assert a.engine._view.presentable is None
    assert a.backend.pending and a.engine._view._scene_revision == revision
    assert a.engine._view.needs_tick
