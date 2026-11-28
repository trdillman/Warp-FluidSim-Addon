"""Host controls for opt-in FSPT publication coordination and eligibility."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import fspt_device_runtime as runtime
from fluidSim_warp_blender51.blender import output_publication as output
from fluidSim_warp_blender51.blender import surface_publication as surface
from fluidSim_warp_blender51.blender.fspt_device_bridge import DeviceSurfaceOffer
from fluidSim_warp_blender51.fspt.surface_pool import SurfaceKey
from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceStamp


class View:
    pass


class DeviceBridge:
    def __init__(self, case):
        self.case = case
        self.keys = []
        self.current = None
        self.committed = None
        self.status = "PREPARED"
        self.ready = False
        self.fail_close = False
        self.operations = []

    def subscribe(self, token, *, required_channels):
        self.operations.append(("subscribe", token, required_channels))
        return lambda: self.operations.append(("withdraw", token))

    def prepare(self, host, source, *, producer_stream, host_required):
        self.operations.append(("prepare", host_required))
        assert host is self.case.host and producer_stream is self.case.stream
        if self.status != "PREPARED":
            if self.status == "FAIL_EXCEPTION":
                raise RuntimeError("injected optional fault")
            return DeviceSurfaceOffer(self.status, "controlled optional decision")
        key = SurfaceKey(source.scene_token, source.owner_token, source.epoch, "LIVE", "GPU_FINAL",
                         len(self.keys) + 1, self.case.stamp, source.source_timeline_frame,
                         source.state_revision, source.solver_frame, source.simulation_time)
        self.keys.append(key)
        self.current = key
        self.source = source
        return DeviceSurfaceOffer("PREPARED", key=key, source=source)

    def commit(self, key):
        self.operations.append(("commit", key))
        assert self.current == key
        self.committed = key
        return True

    def poll(self, token):
        self.operations.append(("poll", token))
        if self.ready and self.current is not None and self.current == self.committed:
            return DeviceSurfaceOffer("READY", key=self.current, vertex_count=3, triangle_count=1,
                                      source=self.source)
        return DeviceSurfaceOffer("PENDING")

    def invalidate(self, scene_token=None):
        self.operations.append(("invalidate", scene_token))
        if scene_token is None or (self.current and self.current.scene_token == scene_token):
            self.current = self.committed = None

    def close(self):
        self.operations.append(("close",))
        if self.fail_close:
            raise RuntimeError("reader still open")


class Broker:
    def __init__(self):
        self.ready_mode = "GPU"
        self.requests = []
        self.recovery_required = False
        self.geometry_recovery_ready = False
        self.pumps = 0

    def request_mode(self, mode):
        self.requests.append(mode)
        # Deterministic completed transitions in this isolated host control.
        self.ready_mode = mode

    def close(self):
        pass

    def pump(self):
        self.pumps += 1


@pytest.fixture
def case(monkeypatch):
    from fluidSim_warp_blender51 import blender

    # Host boundary tests can reload publication after this fixture's module
    # was collected. Runtime resolves it through the package at call time;
    # bind both import routes to the exact output_object stub owned here and
    # let monkeypatch restore the prior modules when this fixture ends.
    monkeypatch.setitem(sys.modules, output.__name__, output)
    monkeypatch.setattr(blender, 'output_publication', output)
    monkeypatch.setenv("FLUIDSIM_FSPT_DEVICE_SURFACE", "1")
    monkeypatch.setattr(runtime, "_SCENES", {})
    monkeypatch.setattr(runtime, "_DEVICES", {})
    monkeypatch.setattr(runtime, "_DEVICE_ERRORS", {})
    monkeypatch.setattr(runtime, "_HOST_EVENTS", {})
    monkeypatch.setattr(runtime, "_final_blocked", lambda _alias: False)
    monkeypatch.setattr(surface, "_channels", {})
    monkeypatch.setattr(surface, "_subscribers", {})
    obj = SimpleNamespace(session_uid=45, data=SimpleNamespace(session_uid=46))
    domain = SimpleNamespace(session_uid=31, fluidsim_domain=SimpleNamespace(surface_enabled=True))
    scene = SimpleNamespace(session_uid=21, as_pointer=lambda: 917, frame_current=5,
                            fluidsim=SimpleNamespace(domain_object=domain, output_collection_name="Fluid",
                                                     show_surface=True, cache_playback_enabled=False))
    stamp = DeviceSurfaceStamp(object(), (9,), "sparse", 1, 4, 0.25, False, 0, 0, ())
    completion = SimpleNamespace(scene_token=917, owner_token=(31, "Fluid"), epoch=5,
                                 source_timeline_frame=5, solver_frame=4, state_revision=9,
                                 simulation_time=0.25, ready_at=12.5)
    stream = object()
    pipe = SimpleNamespace(mesh_cache=None, mesh_key=None, _surface_source_stream=stream)
    solver = SimpleNamespace(device="cuda:0", surface=pipe)
    session = SimpleNamespace(owns_scene=lambda current: current is scene, solver=solver,
                              current_solver_completion=lambda _scene: completion)
    result = SimpleNamespace(scene=scene, obj=obj, domain=domain, stamp=stamp, completion=completion,
                             stream=stream, host=SimpleNamespace(solver=solver), session=session,
                             view=View(), created=[])
    pipe.device_surface_stamp = lambda: result.stamp
    # Mirror actual per-device attributes without initializing Warp/CUDA.
    device = SimpleNamespace(alias="cuda:0", stream=stream, is_capturing=False)
    result.bridge = DeviceBridge(result)
    result.broker = Broker()
    result.pool = SimpleNamespace(device=device, poisoned=False)
    result.budget = SimpleNamespace(
        surface_ready=lambda key: (3, 1) if result.bridge.ready and result.bridge.current == key else None,
        reserved_bytes=0, generation_count=0,
    )

    def new_device(alias):
        result.created.append(alias)
        return runtime.DeviceResources(device, result.pool, result.budget, result.bridge, result.broker)

    monkeypatch.setattr(runtime, "_new_device", new_device)
    monkeypatch.setattr(output, "output_object", lambda *_args, **_kwargs: result.obj)
    yield result
    result.bridge.fail_close = False
    runtime.shutdown()


def subscribe(case, *, reason=""):
    withdraw = runtime.subscribe(case.scene, case.view, device="cuda:0")
    runtime.set_capability(case.scene, case.view, "cuda:0", reason=reason)
    return withdraw


def prepare(case, *, host_required=True, publish_surface=True):
    return runtime.prepare_live(case.scene, case.session, case.host, case.completion,
                                host_required=host_required, publish_surface=publish_surface)


@contextmanager
def host_attempt(case, *, sync=True):
    c = case.completion
    with surface.surface_attempt(case.scene, source=surface.SurfaceSource.LIVE, epoch=c.epoch,
                                 source_timeline_frame=c.source_timeline_frame, solver_frame=c.solver_frame,
                                 state_revision=c.state_revision, simulation_time=c.simulation_time,
                                 state_ready_at=c.ready_at) as attempt:
        attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=case.obj)
        attempt.succeed(synchronized=sync, build_key=case.stamp.build_key)
        yield attempt


def test_opt_out_and_no_demand_never_create_or_poll(case, monkeypatch):
    assert prepare(case) is None and runtime.poll(case.scene, "cuda:0") is None
    assert runtime.device_resources("cuda:0") is None
    monkeypatch.delenv("FLUIDSIM_FSPT_DEVICE_SURFACE")
    subscribe(case)
    assert not runtime.has_demand(case.scene)
    assert prepare(case) is None and runtime.poll(case.scene, "cuda:0") is None
    assert case.created == case.bridge.operations == []


def test_final_worker_defers_without_touching_main_thread_resources(case):
    subscribe(case)
    with ThreadPoolExecutor(max_workers=1) as worker:
        assert worker.submit(prepare, case).result() is None
        assert worker.submit(runtime.poll, case.scene, "cuda:0").result() is None
        assert worker.submit(runtime.device_resources, "cuda:0").result() is None
    assert case.created == case.bridge.operations == []


def test_gpu_publication_requires_outer_host_commit_and_checked_copy(case):
    subscribe(case)
    ticket = prepare(case)
    assert ticket is not None and ticket.key.scene_token == 21  # publication UID, not pointer 917
    assert runtime.selection(case.scene, "cuda:0") is None
    with host_attempt(case):
        runtime.finish_live(case.scene, ticket)
        assert case.bridge.committed is None
    runtime.finish_live(case.scene, ticket)
    assert case.bridge.committed == ticket.key
    assert runtime.poll(case.scene, "cuda:0") is None
    case.bridge.ready = True
    selected = runtime.poll(case.scene, "cuda:0")
    assert selected.key == ticket.key and selected.pool is runtime.device_resources("cuda:0").pool
    assert selected.publication.channel == surface.SurfaceChannel.GPU_FINAL
    assert selected.publication.state_ready_at == case.completion.ready_at
    assert selected.publication.captured_at == ticket.captured_at
    assert (selected.publication.object_token, selected.publication.mesh_token) == (45, 46)
    assert runtime.poll(case.scene, "cuda:0") == selected
    assert runtime.selection(case.scene, "cuda:0") == selected
    case.obj.data = SimpleNamespace(session_uid=99)
    assert runtime.selection(case.scene, "cuda:0") is None


def test_host_pending_or_failed_commit_immediately_withdraws_gpu_eligibility(case):
    subscribe(case)
    ticket = prepare(case)
    with host_attempt(case):
        pass
    runtime.finish_live(case.scene, ticket)
    case.bridge.ready = True
    assert runtime.poll(case.scene, "cuda:0") is not None
    with host_attempt(case, sync=False):
        assert runtime.selection(case.scene, "cuda:0") is None
    assert runtime.selection(case.scene, "cuda:0") is None


def test_sync_only_retry_acks_only_the_exact_pending_host_receipt(case):
    subscribe(case)
    candidate = prepare(case)
    with host_attempt(case, sync=False):
        pass
    runtime.finish_live(case.scene, candidate)
    case.bridge.ready = True
    assert runtime.poll(case.scene, "cuda:0") is None
    assert case.bridge.committed is None
    pending = surface.get_surface_state(case.scene).pending
    assert pending is not None
    surface.acknowledge_surface_sync(case.scene, case.obj)
    selected = runtime.poll(case.scene, "cuda:0")
    assert selected.key == candidate.key and case.bridge.committed == candidate.key
    assert surface.current_surface_publication(case.scene) is pending


def test_replaced_pending_mesh_never_certifies_the_candidate(case):
    subscribe(case)
    candidate = prepare(case)
    with host_attempt(case, sync=False):
        pass
    runtime.finish_live(case.scene, candidate)
    case.bridge.ready = True
    old_mesh = case.obj.data
    case.obj.data = SimpleNamespace(session_uid=222)
    surface.acknowledge_surface_sync(case.scene, case.obj)
    assert runtime.poll(case.scene, "cuda:0") is None
    case.obj.data = old_mesh
    surface.acknowledge_surface_sync(case.scene, case.obj)
    assert runtime.poll(case.scene, "cuda:0") is None
    assert case.bridge.committed is None


def test_final_worker_notification_is_replayed_only_on_main_and_keeps_view_demand(case):
    subscribe(case)
    candidate = prepare(case)
    with host_attempt(case):
        pass
    runtime.finish_live(case.scene, candidate)
    case.bridge.ready = True
    assert runtime.poll(case.scene, "cuda:0") is not None
    before = list(case.bridge.operations)

    def publish_cache():
        with surface.surface_attempt(case.scene, source=surface.SurfaceSource.CACHE,
                                     cache_frame=5, cache_identity=("cache", "123")) as attempt:
            attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=case.obj)
            attempt.succeed(synchronized=True)

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(publish_cache).result()
        assert not worker.submit(runtime.pump).result()
    assert case.bridge.operations == before
    assert runtime.selection(case.scene, "cuda:0") is None
    runtime.pump()
    assert ("invalidate", 21) in case.bridge.operations
    assert runtime.has_demand(case.scene)
    assert runtime.device_resources("cuda:0").broker is case.broker
    assert case.broker.ready_mode == "EVALUATED"
    assert "cache playback" in runtime.fallback_reason(case.scene, "cuda:0")


def test_actor_reset_retires_the_offer_but_keeps_the_current_view_subscription(case):
    subscribe(case)
    candidate = prepare(case)
    with host_attempt(case):
        pass
    runtime.finish_live(case.scene, candidate)
    case.bridge.ready = True
    assert runtime.poll(case.scene, "cuda:0") is not None
    surface.invalidate_surfaces(case.scene, source=surface.SurfaceSource.LIVE)
    assert runtime.selection(case.scene, "cuda:0") is None
    assert runtime.has_demand(case.scene)
    assert case.broker.ready_mode == "EVALUATED"
    # New actual build/epoch can request the GPU path without re-subscribing.
    case.completion.epoch += 1
    case.stamp = replace(case.stamp, revision=case.stamp.revision + 1)
    assert prepare(case) is not None


def test_source_latched_fallback_cannot_be_reset_each_draw(case):
    subscribe(case)
    case.bridge.status = "UNSUPPORTED"
    assert prepare(case) is None
    assert case.broker.ready_mode == "EVALUATED"
    for _ in range(3):
        runtime.set_capability(case.scene, case.view, "cuda:0", reason="")
        assert prepare(case) is None
    assert case.broker.ready_mode == "EVALUATED"
    assert [op for op in case.bridge.operations if op[0] == "prepare"] == [("prepare", True)]
    case.stamp = replace(case.stamp, revision=2)
    case.bridge.status = "PREPARED"
    assert prepare(case) is not None
    assert case.broker.ready_mode == "GPU"


def test_warm_mandatory_cache_falls_back_without_calling_the_producer(case):
    subscribe(case)
    resource = runtime.device_resources("cuda:0")
    pipe = case.session.solver.surface
    pipe.mesh_cache, pipe.mesh_key = (object(), object()), case.stamp.build_key
    assert prepare(case) is None
    assert "already materialized" in runtime.fallback_reason(case.scene, "cuda:0")
    assert resource.broker.ready_mode == "EVALUATED"
    assert not [op for op in case.bridge.operations if op[0] == "prepare"]


def test_unsupported_native_demand_does_not_create_a_producer(case):
    subscribe(case)
    assert prepare(case, host_required=False) is None
    assert case.created == case.bridge.operations == []
    assert "exact native host publication" in runtime.fallback_reason(case.scene, "cuda:0")


def test_one_explicitly_incompatible_view_selects_the_device_wide_fallback(case):
    subscribe(case)
    resource = runtime.device_resources("cuda:0")
    peer = View()
    withdraw = runtime.subscribe(case.scene, peer, device="cuda:0")
    runtime.set_capability(case.scene, peer, "cuda:0", reason="requires an evaluated modifier")
    assert resource.broker.ready_mode == "EVALUATED"
    assert "modifier" in runtime.fallback_reason(case.scene, "cuda:0")
    assert prepare(case) is None
    withdraw()
    assert resource.broker.ready_mode == "GPU" and not runtime.fallback_reason(case.scene, "cuda:0")


def test_collected_view_stops_admission_before_explicit_retirement(case):
    subscribe(case)
    runtime.device_resources("cuda:0")
    del case.view
    assert not runtime.has_demand(case.scene)
    assert prepare(case) is None and runtime.poll(case.scene, "cuda:0") is None
    assert case.bridge.operations == []
    assert not runtime.pump()
    assert case.bridge.operations == [("close",)]


def test_pressure_never_requests_evaluated_mode(case):
    subscribe(case)
    case.bridge.status = "DEFERRED"
    assert prepare(case) is None
    assert case.broker.ready_mode == "GPU"
    assert not runtime.fallback_reason(case.scene, "cuda:0")


def test_optional_exception_keeps_native_host_available(case):
    subscribe(case)
    case.bridge.status = "FAIL_EXCEPTION"
    assert prepare(case) is None
    assert "injected optional fault" in runtime.fallback_reason(case.scene, "cuda:0")
    with host_attempt(case):
        pass
    assert surface.current_surface_publication(case.scene).vertex_count == 3
    assert runtime.selection(case.scene, "cuda:0") is None


def test_invalidation_and_close_failure_keep_owned_resources_for_retry(case):
    withdraw = subscribe(case)
    resource = runtime.device_resources("cuda:0")
    case.bridge.fail_close = True
    withdraw()
    assert not runtime.has_demand(case.scene)
    assert runtime.poll(case.scene, "cuda:0") is None
    assert runtime.shutdown()
    case.bridge.fail_close = False
    assert not runtime.shutdown()
    assert resource.bridge.operations.count(("close",)) >= 2


def test_failed_pool_drains_once_only_after_broker_surrenders_geometry(case):
    subscribe(case)
    runtime.device_resources("cuda:0")
    case.pool.poisoned = case.broker.recovery_required = True
    runtime.pump()
    assert case.broker.ready_mode == "EVALUATED"
    assert ("close",) not in case.bridge.operations
    case.broker.geometry_recovery_ready = True
    runtime.pump()
    runtime.pump()
    assert case.bridge.operations.count(("close",)) == 1
    assert "quarantined" in runtime.fallback_reason(case.scene, "cuda:0")


def test_failed_zero_byte_generation_recovery_does_not_request_recurring_work(case):
    subscribe(case)
    resource = runtime.device_resources("cuda:0")
    # An empty packet can still retain a generation and a checked fence.
    case.budget.generation_count = 1
    case.broker.ready_mode = None
    case.broker.request_mode = lambda _mode: None
    case.pool.poisoned = case.broker.recovery_required = case.broker.geometry_recovery_ready = True
    case.bridge.fail_close = True
    assert not runtime.pump()
    assert not runtime.pump()
    assert case.bridge.operations.count(("close",)) == 1
    assert runtime.device_resources("cuda:0") is resource
    assert "explicit shutdown" in runtime.fallback_reason(case.scene, "cuda:0")
    case.budget.generation_count = 0


def test_final_reservation_blocks_runtime_creation_and_poll_until_released(case, monkeypatch):
    subscribe(case)
    assert runtime.device_resources("cuda:0") is not None
    candidate = prepare(case)
    with host_attempt(case):
        pass
    runtime.finish_live(case.scene, candidate)
    case.bridge.ready = True
    assert runtime.poll(case.scene, "cuda:0") is not None
    old_operations = list(case.bridge.operations)
    monkeypatch.setattr(runtime, "_final_blocked", lambda _alias: True)
    assert runtime.device_resources("cuda:0") is None
    assert runtime.selection(case.scene, "cuda:0") is None
    assert prepare(case) is None and runtime.poll(case.scene, "cuda:0") is None
    runtime.pump()
    assert case.bridge.operations == old_operations and case.broker.pumps == 0
    assert runtime.retire_for_final("cuda:0") == {"reserved_bytes": 0, "generation_count": 0}
    assert runtime.has_demand(case.scene) and runtime.device_resources("cuda:0") is None
    monkeypatch.setattr(runtime, "_final_blocked", lambda _alias: False)
    assert runtime.device_resources("cuda:0") is not None
    assert case.created == ["cuda:0", "cuda:0"]


def test_unretired_geometry_does_not_issue_a_final_receipt(case):
    subscribe(case)
    resource = runtime.device_resources("cuda:0")
    case.budget.reserved_bytes = 128
    case.budget.generation_count = 1
    with pytest.raises(RuntimeError, match="remains after owner retirement"):
        runtime.retire_for_final("cuda:0")
    assert runtime._DEVICES["cuda:0"] is resource and resource.closing
    assert runtime.has_resources() and runtime.has_resources("cuda:0")
    assert "remains" in runtime.resource_error("cuda:0")
    case.budget.reserved_bytes = case.budget.generation_count = 0
    assert runtime.retire_for_final("cuda:0") == {"reserved_bytes": 0, "generation_count": 0}
    assert not runtime.has_resources() and not runtime.resource_error("cuda:0")


def test_disabling_opt_in_cannot_hide_old_resource_ownership(case, monkeypatch):
    subscribe(case)
    runtime.device_resources("cuda:0")
    case.bridge.fail_close = True
    monkeypatch.delenv("FLUIDSIM_FSPT_DEVICE_SURFACE")
    assert runtime.device_resources("cuda:0") is None and runtime.has_resources("cuda:0")
    assert "retiring" in runtime.resource_error("cuda:0")
    assert not runtime.pump()
    assert runtime.has_resources("cuda:0") and "reader still open" in runtime.resource_error("cuda:0")
    case.bridge.fail_close = False
    assert not runtime.shutdown() and not runtime.has_resources("cuda:0")


def test_real_final_handoff_holds_device_while_accepting_the_runtime_receipt(case, monkeypatch):
    from fluidSim_warp_blender51.fspt import final_handoff

    subscribe(case)
    assert runtime.device_resources("cuda:0") is not None
    handoff = final_handoff.FinalHandoff()
    monkeypatch.setattr(runtime, "_final_blocked", handoff.blocked)

    def retire(device, request_id):
        assert handoff.blocked(device) and runtime.device_resources(device) is None
        receipt = runtime.retire_for_final(device)
        return final_handoff.FinalRetirementReceipt(device, request_id, **receipt, owners_retired=True)

    claim = handoff.acquire("cuda:0", retire)
    try:
        assert claim.receipt.reserved_bytes == claim.receipt.generation_count == 0
        assert runtime.has_demand(case.scene)
        assert runtime.device_resources("cuda:0") is None
        assert prepare(case) is None
    finally:
        claim.release()
    assert runtime.device_resources("cuda:0") is not None


@pytest.mark.parametrize(("force_surface", "failure"), [
    (False, None), (True, None), (False, "prepare"), (False, "finish"), (False, "rigid"),
])
def test_output_hook_keeps_rigid_and_outer_commit_before_the_gpu(case, monkeypatch, force_surface, failure):
    from fluidSim_warp_blender51.blender import liquid_display, rigid_publication

    subscribe(case)
    solver = case.session.solver
    solver.frame = case.completion.solver_frame
    solver.liquid_state_revision = case.completion.state_revision
    solver.sim_time = case.completion.simulation_time
    solver.host_frame = lambda: case.host
    case.host.should_publish_surface = lambda _cadence: True
    case.session.active = True
    case.session.surface_epoch = case.completion.epoch
    case.session.solved_timeline_frame = case.completion.source_timeline_frame
    case.obj.type = "MESH"
    case.obj.data.vertices = (object(),)
    record = []

    def prepare_payload(_scene, host, *, publish_surface):
        assert host is case.host and publish_surface
        assert bool(case.bridge.keys) is (not force_surface and failure != "prepare")
        assert case.bridge.committed is None
        solver.surface.mesh_key = case.stamp.build_key
        record.append("payload")
        return object()

    def publish_payload(scene, _payload):
        attempt = surface.active_surface_attempt(scene)
        attempt.record(vertex_count=3, triangle_count=1, visible=True, obj=case.obj)
        record.append("publish")
        return output.RenderInputSyncResult(True)

    def publish_rigid(scene, _session):
        assert surface.active_surface_attempt(scene) is not None
        assert case.bridge.committed is None
        record.append("rigid")
        if failure == "rigid":
            raise RuntimeError("ordinary rigid publication failed")

    def fail_optional(*_args, **_kwargs):
        raise RuntimeError("optional publication hook failed")

    monkeypatch.setattr(output, "_require_bpy", lambda: None)
    monkeypatch.setattr(output, "prepare_live_payload", prepare_payload)
    monkeypatch.setattr(output, "publish_payload", publish_payload)
    monkeypatch.setattr(liquid_display, "viewport_families", lambda *_: (False, False))
    monkeypatch.setattr(rigid_publication, "publish_live", publish_rigid)
    if failure in ("prepare", "finish"):
        monkeypatch.setattr(runtime, failure + "_live", fail_optional)
    if failure == "rigid":
        with pytest.raises(RuntimeError, match="ordinary rigid publication failed"):
            output.update_outputs(case.scene, case.session, force_surface=force_surface)
        assert runtime.device_resources("cuda:0").ticket is None
    else:
        assert output.update_outputs(case.scene, case.session, force_surface=force_surface).synchronized
        assert surface.current_surface_publication(case.scene).vertex_count == 3
    if failure in ("prepare", "finish"):
        assert "optional publication hook failed" in runtime.fallback_reason(case.scene, "cuda:0")
    assert record == ["payload", "publish", "rigid"]
    assert bool(case.bridge.committed) is (not force_surface and failure is None)
