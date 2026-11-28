"""Producer admission and host sharing with real pipeline and pool ownership."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.blender.fspt_device_bridge import (
    DeviceSurfaceBridge,
    LiveSurfaceSource,
)
from fluidSim_warp_blender51.fspt.surface_pool import SurfacePool
from fluidSim_warp_blender51.sim.device_surface import (
    DeviceSurfaceDestination,
    SurfaceDeviceArrays,
)
from fluidSim_warp_blender51.sim.frame_output import HostFrame
from fluidSim_warp_blender51.sim.surface import SurfacePipeline

VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
TRIANGLES = np.array([[0, 1, 2]], dtype=np.int32)


class Destination(DeviceSurfaceDestination):
    def __init__(self, ops):
        self.ops = ops
        self.done = self.broken = False

    def _capture(self, _device, stamp, materialize, *, producer_stream, known_counts):
        assert producer_stream is self.ops.producer_stream
        arrays = materialize()
        if self.ops.fail_copy:
            raise RuntimeError("injected copy failure after extraction")
        return SimpleNamespace(stamp=stamp, destination=self, positions=arrays.positions,
                               indices=arrays.indices,
                               vertex_count=arrays.positions.size,
                               triangle_count=arrays.indices.size // 3)

    def retire_source(self):
        if self.broken:
            raise RuntimeError("injected fence fault")
        return self.done

    def drain(self):
        if self.ops.fail_drain:
            raise RuntimeError("injected drain failure")
        self.done = True


class PoolOps:
    def __init__(self):
        self.producer_stream = object()
        self.allocations = []
        self.fail_copy = self.fail_drain = False

    def producer(self, _device, stream):
        assert stream is self.producer_stream

    def allocate(self, _device, nv, nt):
        destination = Destination(self)
        self.allocations.append((nv, nt, destination))
        return destination

    def validate(self, packet, destination, key, nv, nt):
        assert packet.destination is destination and packet.stamp == key.build_stamp
        assert packet.vertex_count <= nv and packet.triangle_count <= nt

    def reader(self, device, stream):
        assert stream.device is device

    def record(self, _device, stream):
        return stream.fence

    def completed(self, _device, event):
        return event.done

    def sync_reader(self, _device, _stream):
        return None


class Budget:
    """Consumer-side boundary double. Exercise the qualified pool beneath it."""

    def __init__(self, pool):
        self.pool = pool
        self.deny = False
        self.submissions = []

    def submit_surface(self, key, nv, nt, capture, *, producer_stream):
        self.submissions.append((nv, nt))
        return not self.deny and self.pool.submit(
            key, nv, nt, capture, producer_stream=producer_stream)

    def surface_ready(self, key):
        for slot in self.pool._slots:
            if slot.key == key and slot.ready and self.pool._key == key and not self.pool.poisoned:
                return slot.packet.vertex_count, slot.packet.triangle_count
        return None


@pytest.fixture
def bridge_case():
    ops = PoolOps()
    pool = SurfacePool(object(), byte_limit=2_000_000, ops=ops)
    budget = Budget(pool)
    bridge = DeviceSurfaceBridge(budget)
    yield SimpleNamespace(bridge=bridge, pool=pool, budget=budget, ops=ops)
    ops.fail_drain = False
    bridge.close()


@dataclass
class WaveSettings:
    strength: float = 0.5


def make_host(case, *, waves=False, detail=False, frame=7):
    """Real HostFrame and SurfacePipeline methods with controlled geometry operations."""
    calls = []
    grid = SimpleNamespace(active_leaf_count=2)
    pipe = SurfacePipeline.__new__(SurfacePipeline)
    pipe.render_sdf = object()
    pipe.mesh_cache = pipe.mesh_key = None
    pipe.build_key = lambda: (frame, int(detail))
    pipe.built_key = pipe.build_key()
    pipe._surface_source_frame = frame - 1  # Substep build identity can be earlier.
    pipe._surface_source_time = 0.25
    pipe._surface_source_stream = case.ops.producer_stream
    expected = VERTICES + (0.125 if detail else 0.0)

    def readback(value):
        return SimpleNamespace(numpy=lambda: value.copy(), size=len(value), test_values=value)

    def extract(field):
        assert field is pipe.render_sdf
        calls.append("extract")
        return expected.copy(), TRIANGLES.copy()

    def extract_device(field):
        verts, tris = extract(field)
        return readback(verts), readback(tris.ravel())

    def displace_device(positions, indices):
        return SurfaceDeviceArrays(readback(positions.test_values + 0.25), indices)

    pipe.render_extractor = SimpleNamespace(backend="sparse", grid=grid,
                                           extract=extract, extract_sparse_device=extract_device)
    pipe.waves = SimpleNamespace(count=3, frame=frame, state_revision=2, settings=WaveSettings(),
                                 displace=lambda verts, _tris: verts + 0.25,
                                 displace_device=displace_device) if waves else None
    settings = SimpleNamespace(surface_enabled=True, surface_waves_enabled=waves)
    solver = SimpleNamespace(frame=frame, sim_time=0.5, liquid_state_revision=frame,
                             device=case.pool.device, settings=settings, surface=pipe,
                             surface_mesh=pipe.mesh)
    pipe.solver = solver
    host = solver._host_frame_cache = HostFrame(solver)
    return host, calls


def source(frame=7):
    return LiveSurfaceSource(17, (3, "surface"), 2, 10 + frame, frame, frame, 0.5,
                             state_ready_at=1032.125)


def prepare(case, host, *, host_required=False, live=None, unsupported_reason=""):
    return case.bridge.prepare(host, live or source(), producer_stream=case.ops.producer_stream,
                               host_required=host_required, unsupported_reason=unsupported_reason)


def test_no_demand_does_not_check_host_or_poll_pool(bridge_case, monkeypatch):
    def unexpected_poll():
        pytest.fail("a dormant producer polled device work")

    monkeypatch.setattr(bridge_case.pool, "poll", unexpected_poll)
    assert prepare(bridge_case, object()).status == "NO_DEMAND"
    assert bridge_case.bridge.poll(17).status == "NO_DEMAND"
    assert bridge_case.ops.allocations == []


def test_pinned_sparse_marching_cubes_table_capacity():
    tables = wp.geometry.IsoSurfaceMarchingCubes
    counts = np.diff(tables.CASE_TO_TRI_RANGE)
    assert len(tables.EDGE_TO_CORNERS) == 12
    assert len(counts) == 256 and max(counts) == 15
    assert all(counts % 3 == 0)


@pytest.mark.parametrize("waves,detail", [(False, False), (False, True), (True, False), (True, True)])
def test_displaced_final_capture_shares_one_extraction_with_host(bridge_case, waves, detail):
    case = bridge_case
    host, extractions = make_host(case, waves=waves, detail=detail)
    baseline, baseline_extractions = make_host(case, waves=waves, detail=detail)
    case.bridge.subscribe(17)

    prepared = prepare(case, host, host_required=True)
    assert prepared.status == "PREPARED"
    assert case.bridge.poll(17).status == "PENDING"
    assert case.budget.submissions == [(17_496, 7290)]
    # Both calls use real HostFrame.surface_mesh -> SurfacePipeline.mesh.
    for actual, expected in zip(host.surface_mesh, baseline.surface_mesh, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert extractions == baseline_extractions == ["extract"]
    assert prepared.key.build_stamp.waves_enabled is waves
    assert prepared.key.solver_frame == 7 and prepared.key.build_stamp.solver_frame == 6

    assert case.bridge.commit(prepared.key)
    assert case.bridge.poll(17).status == "PENDING"
    case.ops.allocations[0][2].done = True
    ready = case.bridge.poll(17)
    assert (ready.status, ready.key, ready.vertex_count, ready.triangle_count) == (
        "READY", prepared.key, 3, 1)
    assert ready.source == source() and ready.source.state_ready_at == 1032.125
    assert prepare(case, host, host_required=True).key == prepared.key
    assert extractions == ["extract"]


@pytest.mark.parametrize(("waves", "host_required"), [(True, False), (True, True), (False, True)])
def test_preexisting_mandatory_host_cache_does_not_trigger_another_extraction(
        bridge_case, waves, host_required):
    case = bridge_case
    case.bridge.subscribe(17)
    host, extractions = make_host(case, waves=waves)
    host.surface_mesh
    offer = prepare(case, host, host_required=host_required)
    assert offer.status == "UNSUPPORTED" and "already materialized" in offer.reason
    assert extractions == ["extract"]
    assert case.budget.submissions == []


def test_current_optional_cache_is_exact_but_invalid_cache_uses_growth_bound(bridge_case):
    case = bridge_case
    case.bridge.subscribe(17)
    host, extractions = make_host(case)
    host.surface_mesh
    case.budget.deny = True
    assert prepare(case, host).status == "DEFERRED"
    assert case.budget.submissions == [(3, 1)]

    host.solver.surface.mesh_key = ("old",)
    host.solver.surface.render_extractor.grid.active_leaf_count = 5
    assert prepare(case, host).status == "DEFERRED"
    assert case.budget.submissions[-1] == (43_740, 18_225)
    assert extractions == ["extract"]
    assert case.ops.allocations == []


def test_unsupported_and_stale_inputs_refuse_before_reserving(bridge_case):
    case = bridge_case
    host, extractions = make_host(case)
    withdraw = case.bridge.subscribe(17, required_channels={"normals"})
    assert prepare(case, host).status == "UNSUPPORTED"
    withdraw()
    case.bridge.subscribe(17)
    assert prepare(case, host, unsupported_reason="modifier not supported").reason == (
        "modifier not supported")
    assert prepare(case, host, live=source(frame=8)).status == "UNAVAILABLE"
    invalid_clock = replace(source(), state_ready_at=float("nan"))
    assert prepare(case, host, live=invalid_clock).reason == "invalid solver completion clock"
    host.solver.surface.render_extractor.backend = "dense"
    assert prepare(case, host).status == "UNSUPPORTED"
    assert extractions == case.budget.submissions == []


def test_unverified_warp_uses_evaluated_route_before_reading_host(bridge_case, monkeypatch):
    bridge_case.bridge.subscribe(17)
    monkeypatch.setattr(wp, "__version__", "1.18.0")
    result = prepare(bridge_case, object())
    assert result.status == "UNSUPPORTED" and "Warp 1.19.0" in result.reason
    assert bridge_case.budget.submissions == []


@pytest.mark.parametrize(("alias", "capturing"), [
    ("cuda:0", True), ("cuda:1", False), ("invalid", False),
])
def test_solver_alias_cannot_submit_on_captured_foreign_or_invalid_device(
        bridge_case, alias, capturing):
    class Device:
        is_capturing = capturing

        def __eq__(self, other):
            return self is other or (isinstance(other, str) and other == "cuda:0")

    case = bridge_case
    case.pool.device = Device()
    case.bridge.subscribe(17)
    host, calls = make_host(case)
    # FluidSolver keeps a device alias, while the pool owns the Warp device.
    host.solver.device = alias
    offer = prepare(case, host, host_required=True)
    assert offer.status == "UNSUPPORTED" and "CUDA device" in offer.reason
    assert not case.pool.poisoned and case.pool.occupied == 0
    assert case.ops.allocations == case.budget.submissions == calls == []
    np.testing.assert_array_equal(host.surface_mesh[0], VERTICES)
    assert calls == ["extract"]


def test_copy_ready_does_not_certify_unsynchronized_scene_inputs(bridge_case):
    case = bridge_case
    case.bridge.subscribe(17)
    host, _ = make_host(case)
    offer = prepare(case, host)
    assert host.solver.surface.mesh_cache is None  # GPU-only preparation never filled the host cache.
    case.ops.allocations[0][2].done = True
    pending = case.bridge.poll(17)
    assert pending.status == "PENDING" and pending.key is None
    assert pending.reason == "scene inputs have not committed"
    case.bridge.invalidate(17)
    assert not case.bridge.commit(offer.key)
    assert case.bridge.poll(17).status == "UNAVAILABLE"


def test_retained_readers_deny_next_capture_before_extraction(bridge_case):
    case = bridge_case
    case.bridge.subscribe(17)
    leases = []
    try:
        for frame in (7, 8):
            host, calls = make_host(case, frame=frame)
            live = replace(source(frame), epoch=frame)
            offer = prepare(case, host, live=live)
            assert case.bridge.commit(offer.key)
            case.ops.allocations[-1][2].done = True
            assert case.bridge.poll(17).status == "READY"
            reader = SimpleNamespace(device=case.pool.device, fence=SimpleNamespace(done=False))
            leases.append(case.pool.acquire(offer.key, reader))
            assert calls == ["extract"]
        host, calls = make_host(case, frame=9)
        live = replace(source(9), epoch=9)
        assert prepare(case, host, live=live).status == "DEFERRED"
        assert calls == [] and len(case.ops.allocations) == 2
        assert all(lease.packet.triangle_count == 1 for lease in leases)
    finally:
        for lease in leases:
            lease.finish()


def test_optional_failure_quarantines_pool_and_host_still_publishes(bridge_case):
    case = bridge_case
    case.bridge.subscribe(17)
    host, extractions = make_host(case, waves=True, detail=True)
    baseline, _ = make_host(case, waves=True, detail=True)
    case.ops.fail_copy = True
    result = prepare(case, host, host_required=True)
    assert result.status == "FAILED" and "injected copy failure" in result.reason
    assert case.pool.poisoned and case.pool.occupied == 1
    for actual, expected in zip(host.surface_mesh, baseline.surface_mesh, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert extractions == ["extract", "extract"]
    assert prepare(case, host, host_required=True).status == "FAILED"
    assert case.bridge.poll(17).status == "FAILED"
    assert len(case.ops.allocations) == 1

    case.ops.fail_drain = True
    with pytest.raises(RuntimeError, match="drain failure"):
        case.bridge.close()
    assert case.pool.occupied == 1


def test_failed_fence_query_never_exposes_a_ready_offer(bridge_case):
    case = bridge_case
    case.bridge.subscribe(17)
    host, _ = make_host(case)
    prepared = prepare(case, host)
    assert case.bridge.commit(prepared.key)
    case.ops.allocations[0][2].broken = True
    failed = case.bridge.poll(17)
    assert failed.status == "FAILED" and "fence fault" in failed.reason
    assert failed.key is None and case.pool.poisoned and case.pool.occupied == 1


def test_withdrawal_retires_eligibility_but_keeps_open_reader_rights(bridge_case, monkeypatch):
    case = bridge_case
    withdraw = case.bridge.subscribe(17)
    host, _ = make_host(case)
    key = prepare(case, host).key
    assert case.bridge.commit(key)
    case.ops.allocations[0][2].done = True
    assert case.bridge.poll(17).status == "READY"
    reader = SimpleNamespace(device=case.pool.device, fence=SimpleNamespace(done=False))
    lease = case.pool.acquire(key, reader)
    withdraw()
    withdraw()
    assert case.pool.acquire(key, reader) is None
    assert lease.packet.stamp == key.build_stamp
    assert not case.bridge.commit(key)
    monkeypatch.setattr(case.pool, "poll", lambda: pytest.fail("withdrew all consumer demand"))
    assert case.bridge.poll(17).status == "NO_DEMAND"
    with pytest.raises(RuntimeError, match="reader still open"):
        case.bridge.close()
    assert case.pool.occupied == 1
    lease.finish()
    case.bridge.close()
    assert case.pool.occupied == 0


def test_publication_control_requires_main_thread(bridge_case):
    with ThreadPoolExecutor(max_workers=1) as worker:
        for action, argument in ((bridge_case.bridge.subscribe, 17), (bridge_case.bridge.poll, 17),
                                 (bridge_case.bridge.invalidate, 17)):
            with pytest.raises(RuntimeError, match="main thread"):
                worker.submit(action, argument).result()
