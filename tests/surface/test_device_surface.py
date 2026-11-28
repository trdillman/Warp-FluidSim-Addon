"""The opt-in surface producer, separate from Blender route and pool admission."""
import weakref
from dataclasses import FrozenInstanceError
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import surface
from fluidSim_warp_blender51.sim.surface_waves import (
    SurfaceWaves,
    SurfaceWaveSettings,
)


def _pipeline(device, extractor, field):
    pipe = surface.SurfacePipeline.__new__(surface.SurfacePipeline)
    pipe.solver = SimpleNamespace(
        device=device, frame=4, sim_time=0.125, liquid_state_revision=15,
        settings=SimpleNamespace(surface_enabled=True, surface_waves_enabled=False))
    pipe.build_key = lambda: (15, 3, ("render",))
    pipe.built_key = pipe.build_key()
    pipe._surface_source_frame = pipe.solver.frame
    pipe._surface_source_time = pipe.solver.sim_time
    pipe._surface_source_stream = getattr(device, "stream", None)
    pipe.render_extractor = extractor
    pipe.render_sdf = field
    pipe.mesh_cache = pipe.mesh_key = pipe.waves = None
    return pipe


def test_stamp_changes_on_identical_key_rebuild_and_wave_state_replacement():
    pipe = _pipeline("cpu", SimpleNamespace(backend="sparse"), object())
    first = pipe.device_surface_stamp()
    assert first.build_key == pipe.built_key and first.solver_frame == 4
    assert _pipeline("cpu", pipe.render_extractor, pipe.render_sdf).device_surface_stamp() != first
    with pytest.raises(FrozenInstanceError):
        first.solver_frame = 5
    pipe.invalidate()
    second = pipe.device_surface_stamp()
    assert second != first
    pipe.solver.settings.surface_waves_enabled = True
    pipe.waves = SurfaceWaves(device="cpu")
    pipe._wave_settings = lambda: pipe.waves.settings
    third = pipe.device_surface_stamp()
    pipe.waves.settings.spacing *= 2
    fourth = pipe.device_surface_stamp()
    assert second != third != fourth
    arrays, scalars = pipe.waves.export_state()
    pipe.import_wave_state(
        {f"surface_waves_{k}": v for k, v in arrays.items()},
        {f"surface_waves_{k}": v for k, v in scalars.items()})
    assert pipe.device_surface_stamp() != fourth


@pytest.mark.parametrize("backend", ["dense", "nets"])
def test_unsupported_route_refuses_before_extraction_or_rebuild(backend):
    from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceUnavailable

    extractor = Mock(backend=backend)
    pipe = _pipeline("cpu", extractor, object())
    pipe.build = Mock(side_effect=AssertionError("renderer rebuilt surface"))
    with pytest.raises(DeviceSurfaceUnavailable, match=backend):
        pipe.copy_final_device_surface(None, expected_stamp=pipe.device_surface_stamp())
    extractor.extract.assert_not_called()
    extractor.extract_sparse_device.assert_not_called()


def test_sparse_world_vertices_and_existing_wave_displacement(cuda_device, monkeypatch):
    # These controls can also run on Warp's CPU backend in source CI.
    from test_surface_extractor_parity import CASES, _sphere_field

    grid, field = _sphere_field(cuda_device, *CASES["interior"])
    extractor = surface.SurfaceExtractor(grid, cuda_device, backend="sparse")
    pipe = _pipeline(cuda_device, extractor, field)
    # Nonzero translation catches a raw grid-index or origin-less packet.
    grid.origin = wp.vec3(3.0, -2.0, 1.0)
    verts, _tris = extractor.extract(field)
    center = np.array([3.32, -1.68, 1.32], np.float32)
    np.testing.assert_allclose(verts.mean(0), center, atol=0.005)
    normals = verts - center
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    waves = SurfaceWaves(SurfaceWaveSettings(spacing=0.015), device=cuda_device)
    arrays = {"pos": verts, "vel": np.zeros_like(verts), "nrm": normals,
              "h": np.full(len(verts), 0.004, np.float32),
              "hv": np.zeros(len(verts), np.float32), "phase": np.zeros(len(verts), np.float32)}
    waves.import_state(arrays, {"time": 0.125, "frame": 4})
    pipe.waves = waves
    pipe.solver.settings.surface_waves_enabled = True
    expected, expected_tri = pipe.mesh()
    assert np.abs(expected - verts).max() > 0.003
    stamp = pipe.device_surface_stamp()
    original_numpy = wp.array.numpy
    reads = []

    def count_reads(array):
        reads.append((array.size, array.dtype))
        return original_numpy(array)

    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", count_reads)
        output = pipe._final_device_arrays()
        positions, indices = output.positions, output.indices
    # Sparse compaction and Warp MC read back scan totals and min/max coordinates.
    assert reads and all(size <= 3 for size, _dtype in reads), reads
    assert positions.dtype == wp.vec3 and indices.dtype == wp.int32
    np.testing.assert_allclose(positions.numpy(), expected, rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(indices.numpy().reshape(-1, 3), expected_tri)
    assert pipe.device_surface_stamp() == stamp
    assert pipe.mesh()[0] is expected


def test_empty_sparse_field_has_typed_empty_device_arrays(cuda_device):
    from fluidSim_warp_blender51.sim.fields import ScalarField, SparseGrid

    grid = SparseGrid(origin=(2, -1, 3), size=(0.2, 0.2, 0.2),
                      voxel_size=0.025, leaf_capacity=1, device=cuda_device)
    extractor = surface.SurfaceExtractor(grid, cuda_device, backend="sparse")
    pipe = _pipeline(cuda_device, extractor, ScalarField(grid))
    output = pipe._final_device_arrays()
    pos, idx = output.positions, output.indices
    assert pos.shape == idx.shape == (0,)
    assert pos.dtype == wp.vec3 and idx.dtype == wp.int32


@pytest.fixture
def ordered_copy(monkeypatch):
    from fluidSim_warp_blender51.sim import device_surface as ds

    device = SimpleNamespace(is_cuda=True, is_capturing=False)
    stream = Mock(device=device)
    device.stream = stream
    events = []

    def record(event=None):
        event = event or SimpleNamespace(device=device, is_complete=False)
        events.append(event)
        return event

    stream.record_event.side_effect = record
    copy = Mock()
    # Only the CUDA ordering/ownership protocol is simulated in this control.
    def allocate(n, dtype, device):
        return Mock(dtype=dtype, device=device, size=n, ndim=1, is_contiguous=True,
                    shape=(n,), strides=(12 if dtype is wp.vec3 else 4,))

    fake = SimpleNamespace(get_device=lambda _: device, get_stream=lambda _: stream,
                           empty=allocate, vec3=wp.vec3, int32=wp.int32, copy=copy,
                           Event=lambda device: SimpleNamespace(device=device, is_complete=False))
    monkeypatch.setattr(ds, "wp", fake)
    driver = SimpleNamespace(record=lambda wp, dev, stream, event: stream.record_event(event),
                             completed=lambda wp, dev, event: event.is_complete)
    monkeypatch.setattr(ds, "cuda_driver", lambda: driver)
    return ds, device, stream, events, copy, allocate


def test_destination_retains_private_source_until_copy_completion(ordered_copy):
    ds, device, stream, events, copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    held = weakref.ref(src.positions)
    positions, indices = src.positions, src.indices
    packet = dest._capture(device, "source", lambda source=src: source, producer_stream=stream)
    assert packet.positions is not positions and packet.indices is not indices
    assert packet.vertex_count == 3 and packet.triangle_count == 1
    assert packet.producer_ready is events[0] and packet.copy_complete is events[1]
    assert copy.call_count == 2
    assert copy.call_args_list[0].kwargs["stream"] is stream
    copy.reset_mock()
    del src, positions, indices
    assert not dest.retire_source() and held() is not None
    again = Mock(side_effect=AssertionError("spent reservation re-extracted geometry"))
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="reservation"):
        dest._capture(device, "other", again, producer_stream=stream)
    again.assert_not_called()
    packet.copy_complete.is_complete = True
    assert dest.retire_source() and held() is None
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="reservation"):
        dest._capture(device, "other", again, producer_stream=stream)
    again.assert_not_called()

def test_partial_copy_failure_is_fenced_and_cannot_reopen_reservation(ordered_copy):
    ds, device, stream, events, copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    copy.side_effect = [None, RuntimeError("injected index copy failure")]
    with pytest.raises(RuntimeError, match="index copy failure"):
        dest._capture(device, "source", lambda: src, producer_stream=stream)
    assert len(events) == 2 and copy.call_count == 2
    assert not dest.retire_source()
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="reservation"):
        dest._capture(device, "other", Mock(side_effect=AssertionError("extracted")),
                      producer_stream=stream)
    events[-1].is_complete = True
    assert dest.retire_source()


def test_capacity_refusal_keeps_producer_until_ready_and_submits_no_copy(ordered_copy):
    ds, device, stream, events, copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=2, triangle_capacity=1)
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="capacity"):
        dest._capture(device, "source", lambda: src, producer_stream=stream)
    assert len(events) == 1 and not dest.retire_source()
    copy.assert_not_called()
    events[0].is_complete = True
    assert dest.retire_source()


def test_stale_or_missing_demand_and_unsupported_channel_do_no_extraction(ordered_copy):
    ds, device, _stream, _events, copy, _allocate = ordered_copy
    extractor = Mock(backend="sparse")
    pipe = _pipeline(device, extractor, object())
    stamp = pipe.device_surface_stamp()
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="channels"):
        pipe.copy_final_device_surface(dest, expected_stamp=stamp, required_channels=["uv"])
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="reservation"):
        pipe.copy_final_device_surface(None, expected_stamp=stamp)
    pipe.invalidate()
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="stale"):
        pipe.copy_final_device_surface(dest, expected_stamp=stamp)
    extractor.extract_sparse_device.assert_not_called()
    copy.assert_not_called()


def test_failed_wave_or_render_update_refuses_an_old_matching_key(monkeypatch):
    from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceUnavailable

    extractor = Mock(backend="sparse")
    pipe = _pipeline("cpu", extractor, object())
    pipe.device_surface_stamp()
    # The build may partially overwrite a still-equal render SDF before failing.
    pipe.solver._surface_positions = lambda: None
    pipe.ensure_resources = Mock(side_effect=RuntimeError("injected build failure"))
    with pytest.raises(RuntimeError, match="injected build"):
        pipe.build(build_render=True, build_lowres=False)
    with pytest.raises(DeviceSurfaceUnavailable, match="unavailable"):
        pipe.device_surface_stamp()

    pipe = _pipeline("cpu", extractor, object())
    pipe.solver.settings.surface_waves_enabled = True
    pipe.solver.settings.frame_rate, pipe.solver.settings.time_scale = 24, 1.0
    pipe._ensure_waves = Mock()
    pipe.waves = SurfaceWaves(device="cpu")
    pipe.waves.step = Mock(side_effect=RuntimeError("injected wave failure"))
    pipe._vertex_velocity = Mock()
    extractor.extract.return_value = (np.zeros((1, 3)), np.zeros((1, 3)))
    pipe.device_surface_stamp()
    with pytest.raises(RuntimeError, match="injected wave"):
        pipe.step_waves()
    with pytest.raises(DeviceSurfaceUnavailable, match="unavailable"):
        pipe.device_surface_stamp()


def test_stale_host_frame_and_detached_solver_refuse_before_extraction():
    from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceUnavailable
    from fluidSim_warp_blender51.sim.frame_output import HostFrame
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    pipe = _pipeline("cpu", Mock(backend="sparse"), object())
    pipe.solver.owns_host_frame = MethodType(FluidSolver.owns_host_frame, pipe.solver)
    host = HostFrame(pipe.solver)
    stamp = pipe.device_surface_stamp()
    # Same frame number alone is insufficient after reset/replacement.
    pipe.solver._host_frame_cache = HostFrame(pipe.solver)
    with pytest.raises(DeviceSurfaceUnavailable, match="stale"):
        host.copy_final_device_surface(None, expected_stamp=stamp)
    host.solver = None
    pipe.solver = None
    with pytest.raises(DeviceSurfaceUnavailable, match="stale"):
        host.copy_final_device_surface(None, expected_stamp=stamp)
    with pytest.raises(DeviceSurfaceUnavailable, match="unavailable"):
        pipe.device_surface_stamp()


def test_sheet_wave_host_surface_and_owned_cuda_packet(cuda_device):
    """Real solver frame, emitted sheet children, stepped waves and separate destination."""
    from test_sheet_detail import _solver_with_a_sheet

    from fluidSim_warp_blender51.sim.device_surface import (
        DeviceSurfaceDestination,
        DeviceSurfaceUnavailable,
    )

    solver = _solver_with_a_sheet(cuda_device, surface_sheet_detail_enabled=True,
                                  surface_waves_enabled=True, surface_waves_strength=4.0)
    try:
        solver.step_frame()
        pipe = solver.surface
        assert pipe.render_extractor.backend == "sparse"
        assert pipe.sheet.count > 0 and pipe.waves.frame == solver.frame
        assert pipe.last_sdf_splat["sheetDetail"]["fineParticles"] == pipe.sheet.count
        host = solver.host_frame()
        verts, tris = host.surface_mesh
        assert len(tris) and len(verts)
        dest = DeviceSurfaceDestination(cuda_device, vertex_capacity=len(verts) + 23,
                                       triangle_capacity=len(tris) + 17)
        stamp = pipe.device_surface_stamp()
        packet = host.copy_final_device_surface(dest, expected_stamp=stamp)
        assert packet.stamp == stamp and packet.vertex_count == len(verts)
        assert packet.triangle_count == len(tris) and packet.device.is_cuda
        assert packet.positions.strides == (12,) and packet.indices.strides == (4,)
        # A different stream must be able to read using only the copy event.
        reader = wp.Stream(cuda_device)
        reader.wait_event(packet.copy_complete)
        with wp.ScopedStream(reader, sync_enter=False, sync_exit=False):
            copied_pos = wp.clone(packet.positions)
            copied_idx = wp.clone(packet.indices)
        wp.synchronize_stream(reader)
        assert packet.retire_source()
        # Reuse producer scratch and wave state; snapshot bytes must persist.
        solver.step_frame()
        with pytest.raises(DeviceSurfaceUnavailable, match="stale"):
            host.copy_final_device_surface(dest, expected_stamp=stamp)
        np.testing.assert_allclose(copied_pos.numpy(), verts, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(copied_idx.numpy().reshape(-1, 3), tris)
        np.testing.assert_array_equal(packet.positions.numpy(), copied_pos.numpy())
        np.testing.assert_array_equal(packet.indices.numpy(), copied_idx.numpy())
    finally:
        solver.release()


def test_stamp_retains_source_clock_and_observes_same_frame_wave_import():
    pipe = _pipeline("cpu", Mock(backend="sparse"), object())
    pipe.solver.settings.surface_waves_enabled = True
    pipe.waves = SurfaceWaves(device="cpu")
    stamped = pipe.device_surface_stamp()
    pipe.solver.frame += 1
    pipe.solver.sim_time += 0.5
    # Reading an unchanged existing field cannot relabel its actual source.
    assert pipe.device_surface_stamp() == stamped
    arrays, scalars = pipe.waves.export_state()
    pipe.waves.import_state(arrays, scalars)
    assert pipe.device_surface_stamp() != stamped


def test_combined_admitted_demand_serves_host_from_one_final_extraction(ordered_copy):
    ds, device, _stream, events, copy, allocate = ordered_copy
    pos, idx = allocate(3, wp.vec3, device), allocate(3, wp.int32, device)
    pos.numpy.return_value = np.ones((3, 3), np.float32)
    idx.numpy.return_value = np.arange(3, dtype=np.int32)
    extractor = Mock(backend="sparse")
    extractor.extract_sparse_device.return_value = (pos, idx)
    extractor.extract.side_effect = AssertionError("second extraction for host output")
    pipe = _pipeline(device, extractor, object())
    stamp = pipe.device_surface_stamp()
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    def copy_values(target, source, **kwargs):
        target.numpy.return_value = source.numpy.return_value.copy()

    copy.side_effect = copy_values
    packet = pipe.copy_final_device_surface(dest, expected_stamp=stamp, materialize_host=True)
    verts, tris = pipe.mesh()
    np.testing.assert_array_equal(verts, pos.numpy.return_value)
    np.testing.assert_array_equal(tris, [[0, 1, 2]])
    assert packet.stamp == stamp
    extractor.extract_sparse_device.assert_called_once_with(pipe.render_sdf)
    packet.positions.numpy.assert_called_once()
    packet.indices.numpy.assert_called_once()
    pos.numpy.assert_not_called()
    idx.numpy.assert_not_called()
    events[-1].is_complete = True
    assert packet.retire_source()


def test_known_host_count_does_not_extract_into_insufficient_capacity(ordered_copy):
    ds, device, _stream, events, _copy, _allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=2, triangle_capacity=1)
    extractor = Mock(backend="sparse")
    extractor.extract_sparse_device.side_effect = AssertionError("unnecessary extraction")
    pipe = _pipeline(device, extractor, object())
    pipe.mesh_key = pipe.built_key
    pipe.mesh_cache = (np.zeros((3, 3), np.float32), np.array([[0, 1, 2]], np.int32))
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="capacity"):
        pipe.copy_final_device_surface(dest, expected_stamp=pipe.device_surface_stamp())
    extractor.extract_sparse_device.assert_not_called()
    assert not events


def test_changed_state_during_capture_does_not_return_a_packet(ordered_copy):
    ds, device, _stream, events, _copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    extractor = Mock(backend="sparse")
    pipe = _pipeline(device, extractor, object())
    stamp = pipe.device_surface_stamp()

    def changed_field(field):
        pipe.invalidate()
        return allocate(3, wp.vec3, device), allocate(3, wp.int32, device)

    extractor.extract_sparse_device.side_effect = changed_field
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="stale"):
        pipe.copy_final_device_surface(dest, expected_stamp=stamp)
    assert not dest.retire_source()
    events[-1].is_complete = True
    assert dest.retire_source()


def test_unrecorded_copy_fence_needs_stream_specific_drain(ordered_copy, monkeypatch):
    ds, device, stream, _events, copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    stream.record_event.side_effect = [None, RuntimeError("injected fence failure")]
    with pytest.raises(RuntimeError, match="fence failure"):
        dest._capture(device, "source", lambda: src, producer_stream=stream)
    assert copy.call_count == 2 and not dest.retire_source()
    sync = Mock()
    monkeypatch.setattr(ds.wp, "synchronize_stream", sync, raising=False)
    dest.drain()
    sync.assert_called_once_with(stream)
    assert dest.retire_source()


def test_async_wave_displacement_retains_hash_grid_and_input_allocations(monkeypatch):
    from test_surface_waves import _plane

    verts, tris = _plane(n=3)
    waves = SurfaceWaves(device="cpu")
    n = len(verts)
    zeros = np.zeros(n, np.float32)
    waves.import_state(
        {"pos": verts, "vel": np.zeros_like(verts),
         "nrm": np.tile(np.float32([0, 0, 1]), (n, 1)),
         "h": zeros + 0.01, "hv": zeros, "phase": zeros},
        {"frame": 2, "time": 0.125})
    coarse = wp.array(verts, dtype=wp.vec3, device="cpu")
    indices = wp.array(tris.ravel(), dtype=wp.int32, device="cpu")
    source_refs = [weakref.ref(coarse), weakref.ref(waves.h)]
    make_grid = waves._grid

    def observe_grid(*args):
        grid = make_grid(*args)
        source_refs.append(weakref.ref(grid))
        return grid

    monkeypatch.setattr(waves, "_grid", observe_grid)
    output = waves.displace_device(coarse, indices)
    del coarse
    waves.reset()
    assert all(ref() is not None for ref in source_refs)
    del output
    assert all(ref() is None for ref in source_refs)


def test_materialization_failure_drains_producer_stream_before_unwinding(ordered_copy, monkeypatch):
    ds, device, stream, _events, copy, _allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    sync = Mock()
    monkeypatch.setattr(ds.wp, "synchronize_stream", sync, raising=False)
    partial = Mock(side_effect=RuntimeError("injected kernel failure"))
    with pytest.raises(RuntimeError, match="kernel failure"):
        dest._capture(device, "source", partial, producer_stream=stream)
    sync.assert_called_once_with(stream)
    copy.assert_not_called()


def test_wrong_stream_and_cpu_fail_before_materializing(ordered_copy):
    ds, device, stream, events, _copy, _allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    ds.wp.get_stream = lambda _: Mock(device=device)
    unused = Mock(side_effect=AssertionError("unadmitted extraction"))
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="producer stream"):
        dest._capture(device, "source", unused, producer_stream=stream)
    unused.assert_not_called()
    assert not events
    device.is_cuda = False
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="needs CUDA"):
        ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)


def test_reservation_on_unrelated_current_stream_cannot_capture_solver_storage(ordered_copy):
    ds, device, _stream, _events, _copy, _allocate = ordered_copy
    extractor = Mock(backend="sparse")
    extractor.extract_sparse_device.side_effect = AssertionError("wrong CUDA stream read solver state")
    pipe = _pipeline(device, extractor, object())
    pipe._surface_source_stream = Mock(device=device)  # actual producer of this field
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="producer stream"):
        pipe.copy_final_device_surface(dest, expected_stamp=pipe.device_surface_stamp())
    extractor.extract_sparse_device.assert_not_called()


def test_cancelled_partial_materialization_survives_failed_drain_and_external_drop(ordered_copy, monkeypatch):
    import gc

    ds, device, stream, _events, _copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    refs = []

    def cancelled():
        partial = allocate(3, wp.vec3, device)
        refs.append(weakref.ref(partial))
        raise KeyboardInterrupt("cancelled after enqueue")

    failing = Mock(side_effect=RuntimeError("unretired producer stream"))
    monkeypatch.setattr(ds.wp, "synchronize_stream", failing, raising=False)
    with pytest.raises(BaseException) as interrupted:
        dest._capture(device, "source", cancelled, producer_stream=stream)
    assert "unretired" in str(interrupted.value)
    del interrupted
    del dest
    gc.collect()
    assert refs[0]() is not None
    with pytest.raises(ds.DeviceSurfaceUnavailable, match="quarantin"):
        ds.DeviceSurfaceDestination(device, vertex_capacity=1, triangle_capacity=1)
    monkeypatch.setattr(ds.wp, "synchronize_stream", lambda _: None)
    ds.drain_device_surface_quarantine(device)
    failing.reset_mock(side_effect=True)
    gc.collect()
    assert refs[0]() is None


def test_checked_copy_record_rejects_unrecorded_event_that_queries_complete(ordered_copy, monkeypatch):
    ds, device, stream, _events, copy, allocate = ordered_copy
    dest = ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)
    # A fresh CUDA event can query complete until it has actually been recorded.
    dest._copied.is_complete = True
    stream.record_event.side_effect = lambda event: None  # void Warp wrapper would hide the failure
    dest._driver.record = Mock(side_effect=[None, RuntimeError("cuEventRecord failed with CUDA result 201")])
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    with pytest.raises(RuntimeError, match="cuEventRecord.*201"):
        dest._capture(device, "source", lambda: src, producer_stream=stream)
    assert copy.call_count == 2 and not dest.retire_source()
    sync = Mock()
    monkeypatch.setattr(ds.wp, "synchronize_stream", sync, raising=False)
    dest.drain()
    sync.assert_called_once_with(stream)


def test_query_error_closes_new_and_preallocated_admission_until_checked_drain(ordered_copy, monkeypatch):
    ds, device, stream, _events, _copy, allocate = ordered_copy
    def make():
        return ds.DeviceSurfaceDestination(device, vertex_capacity=3, triangle_capacity=1)

    bad, waiting = make(), make()
    src = ds.SurfaceDeviceArrays(allocate(3, wp.vec3, device), allocate(3, wp.int32, device))
    packet = bad._capture(device, "source", lambda: src, producer_stream=stream)
    bad._driver.completed = Mock(side_effect=RuntimeError("checked event query failed"))
    monkeypatch.setattr(ds.wp, "synchronize_event", Mock(), raising=False)
    try:
        with pytest.raises(RuntimeError, match="query failed"):
            packet.retire_source()
        for reserve in (make, lambda: waiting._capture(device, "new", Mock(), producer_stream=stream)):
            with pytest.raises(ds.DeviceSurfaceUnavailable, match="quarantin"):
                reserve()
        assert bad._sources is src
    finally:
        bad.drain()
    assert bad.retire_source() and make() is not None


def test_waves_off_substep_stamp_preserves_field_build_clock_after_completed_frame_advance():
    pipe = _pipeline("cpu", Mock(backend="sparse"), object())
    stamp = pipe.device_surface_stamp()
    pipe.solver.frame, pipe.solver.sim_time = 5, 0.15
    assert not stamp.waves_enabled and pipe.device_surface_stamp() == stamp
    assert (stamp.solver_frame, stamp.solver_time) == (4, 0.125)
