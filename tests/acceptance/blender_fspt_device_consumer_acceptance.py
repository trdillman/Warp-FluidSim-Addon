"""Native pooled GPU_FINAL -> ordered aggregate -> actual image qualification.

This explicit development backend has no publication/viewport wiring. The live
fixture retains ordinary Blender output and uses its independently evaluated
geometry as the oracle. Synthetic tie/fault rows are labelled separately.
"""
from __future__ import annotations

import gc
import importlib
import sys
import time
import traceback
import weakref
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as base
import blender_fspt_device_surface_acceptance as producer_fixture
import blender_fspt_persistent_assets as flat_fixture
import blender_fspt_surface_pool_acceptance as pool_fixture


def poll_frame(backend):
    until = time.monotonic() + 60
    polls = 0
    while time.monotonic() < until:
        polls += 1
        frame = backend.collect()
        if frame is not None:
            return frame, polls
        time.sleep(0.001)
    raise TimeoutError('device consumer did not finish within 60 seconds')


def guarded_render(backend, request, camera, width, height, seed):
    wp = backend.wp
    read, copy_array = wp.array.numpy, wp.copy
    def allowed_copy(dest, src, *args, **kwargs):
        if src.device.is_cuda and dest.device.is_cpu:
            generation = backend._device_current
            image = src is backend.output and dest is backend._pinned
            status = (generation is not None and src is generation.arrays['validation']
                      and dest is generation.validation_host)
            assert image or status, 'consumer performed geometry readback through wp.copy'
        return copy_array(dest, src, *args, **kwargs)
    def host_only(array, *args, **kwargs):
        assert array.device.is_cpu, 'consumer performed device geometry readback'
        return read(array, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('consumer performed a device-wide wait')
    with ExitStack() as guards:
        guards.enter_context(patch.object(wp.array, 'numpy', host_only))
        guards.enter_context(patch.object(wp, 'copy', allowed_copy))
        guards.enter_context(patch.object(wp, 'synchronize_device', forbidden))
        assert backend.submit(request, camera, width, height, samples=1, max_bounces=1, seed=seed)
        return poll_frame(backend)


def ordered_request(modules, original, fluid_asset_key, reference):
    contracts = modules('fspt.contracts')
    parts, nv, nt = [], 0, 0
    for span in original.asset_spans:
        end, stop = nv + span.vertex_count, nt + span.triangle_count
        if span.key == fluid_asset_key:
            assert (span.vertex_count, span.triangle_count) == (reference.vertex_count, reference.triangle_count)
            assert np.all(original.material_indices[nt:stop] == reference.material_index)
            parts.append(reference)
        else:
            parts.append(contracts.SceneSnapshot(
                original.vertices[nv:end], original.triangles[nt:stop] - nv,
                original.material_indices[nt:stop], original.materials, original.world, original.frame))
        nv, nt = end, stop
    return modules('fspt.device_scene').DeviceSceneRequest(
        tuple(parts), original.materials, original.world, original.frame, original.publications)


def live_rows(modules, report, folder, path, *, waves=False):
    wp = modules('runtime.warp_runtime')._import_warp()
    device = wp.get_device(base.script_device())
    pool = modules('fspt.surface_pool').SurfacePool(device, byte_limit=32 * 1024 * 1024)
    budget = modules('fspt.device_scene').GeometryArrayBudget(pool, byte_limit=128 * 1024 * 1024)
    backend = modules('fspt.device_backend').DeviceAsyncCudaBackend(str(device), geometry_budget=budget)
    session = modules('blender.scene_bridge').session()
    source_stream = wp.get_stream(device)
    try:
        original_factory = producer_fixture.make_solver
        def make_solver(*args, **kwargs):
            kwargs['waves'] = waves
            return original_factory(*args, **kwargs)
        with patch.object(producer_fixture, 'make_solver', make_solver):
            scene, solver = flat_fixture.fixture(bpy, base, producer_fixture, modules, device, session)
        camera = modules('fspt.contracts').CameraSnapshot(
            [[0.28, 0, 0, 0.25], [0, 0.28, 0, 0.25], [0, 0, 1, 0], [0, 0, 0, 1]])
        width, height, seed = 80, 64, 91
        oracle = flat_fixture.oracle_kernel(wp)
        rows = []
        for generation in (1, 2):
            base.checkpoint(path, report, ('waves-sheet-' if waves else 'live-') + str(generation))
            original, asset_key = flat_fixture.advance(bpy, modules, scene, solver, session, generation + 1)
            if waves:
                assert solver.surface.sheet.count > 0
                assert solver.surface.waves.frame == solver.frame
                assert np.max(np.abs(solver.surface.waves.h.numpy())) > 0
            host_stamp, = original.publications
            asset = next(a for a in original.assets if a.key == asset_key)
            start, nv, _ = asset.position
            face, scalar_count, _ = asset.index
            nt = scalar_count // 3
            material_index = int(original.material_indices[face // 3])
            stamp = solver.surface.device_surface_stamp()
            key = modules('fspt.surface_pool').SurfaceKey(
                host_stamp.scene_token, host_stamp.owner_token, host_stamp.epoch, 'LIVE', 'GPU_FINAL',
                generation, stamp, host_stamp.source_timeline_frame, host_stamp.state_revision,
                host_stamp.solver_frame, host_stamp.simulation_time)
            assert pool.select(key)
            pool.poll()
            clocks = (solver.frame, solver.sim_time, solver.liquid_state_revision)
            assert budget.submit_surface(
                key, nv, nt,
                lambda dest, expected: solver.host_frame().copy_final_device_surface(dest, expected_stamp=expected),
                producer_stream=source_stream)
            # The verifier may wait for its own source preparation. Rendering
            # below is guarded against global waits and geometry readback.
            wp.synchronize_stream(source_stream)
            pool.poll()
            reference = modules('fspt.device_scene').DeviceGeometryReference(pool, key, nv, nt, material_index)
            request = ordered_request(modules, original, asset_key, reference)
            expected = flat_fixture.flat_pixels(wp, oracle, device, original, camera, width, height, seed)
            frame, polls = guarded_render(backend, request, camera, width, height, seed)
            assert clocks == (solver.frame, solver.sim_time, solver.liquid_state_revision)
            assert frame.publications == original.publications and frame.scene_frame == original.frame
            assert frame.device == str(device)
            np.testing.assert_array_equal(frame.pixels, expected)
            # Explicit verifier-only readbacks independently check all geometry.
            positions = backend.arrays['points'].numpy()
            triangles = backend.arrays['indices'].numpy().reshape(-1, 3)
            np.testing.assert_allclose(positions, original.vertices, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(triangles, original.triangles)
            np.testing.assert_array_equal(backend.arrays['materials'].numpy(), original.material_indices)
            saved = folder / f'live-waves-{int(waves)}-{generation}.npz'
            np.savez(saved, pixels=frame.pixels, positions=positions, triangles=triangles,
                     evaluated_positions=original.vertices, evaluated_triangles=original.triangles)
            rows.append(dict(scope='actual live producer with ordinary Blender output',
                             waves_and_sheet_detail=waves, source_frame=original.frame, solver_clock=clocks, polls=polls,
                             original=str(saved), sha256=base.sha(saved),
                             geometry_digest_kind='GPU identity; archive hashes actual arrays',
                             positions_sha256=producer_fixture.digest(positions),
                             pixels_sha256=producer_fixture.digest(frame.pixels),
                             fluid_vertices=nv, fluid_triangles=nt,
                             statistics=dict(frame.statistics)))
            pool.poll()
        assert rows[0]['positions_sha256'] != rows[1]['positions_sha256']
        report.setdefault('live_rows', []).extend(rows)
        report.setdefault('live_budgets', []).append(dict(waves_and_sheet_detail=waves, peak_explicit_array_bytes=budget.peak_bytes,
                                   peak_geometry_generations=budget.peak_generations,
                                   byte_limit=budget.byte_limit, pool_peak_bytes=pool.peak_bytes,
                                   excludes='BVH internals, shading, image/history, solver and allocator caching'))
        assert budget.peak_generations <= 2 and budget.peak_bytes <= budget.byte_limit
    finally:
        backend.close()
        pool.close()
        session._clear_runtime()


def synthetic_rows(modules, report, folder, path):
    wp = modules('runtime.warp_runtime')._import_warp()
    device = wp.get_device(base.script_device())
    pool_api = modules('fspt.surface_pool')
    scene_api = modules('fspt.device_scene')
    producer = modules('sim.device_surface')
    contracts = modules('fspt.contracts')
    pool = pool_api.SurfacePool(device, byte_limit=32 * 1024 * 1024)
    budget = scene_api.GeometryArrayBudget(pool, byte_limit=128 * 1024 * 1024)
    backend = modules('fspt.device_backend').DeviceAsyncCudaBackend(str(device), geometry_budget=budget)
    stream = wp.get_stream(device)
    owner, generation = object(), 0
    source_key = None
    active_gates = []

    def publish(vertices, faces, material=0):
        nonlocal generation, source_key
        generation += 1
        vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
        faces = np.asarray(faces, np.int32).reshape(-1, 3)
        stamp = producer.DeviceSurfaceStamp(owner, (generation,), 'sparse', generation,
                                            generation, float(generation), False, 0, 0, ())
        source_key = pool_api.SurfaceKey(7, (9, 'synthetic verifier'), 1, 'LIVE', 'GPU_FINAL',
                                         generation, stamp, generation, generation, generation, float(generation))
        assert pool.select(source_key)
        if backend._prepare_stream is not None:
            wp.synchronize_stream(backend._prepare_stream)
        pool.poll()
        def capture(destination, expected):
            def materialize():
                return producer.SurfaceDeviceArrays(
                    wp.array(vertices, dtype=wp.vec3, device=device),
                    wp.array(faces.ravel(), dtype=wp.int32, device=device))
            return destination._capture(device, expected, materialize, producer_stream=stream,
                                        known_counts=(len(vertices), len(faces)))
        assert budget.submit_surface(source_key, len(vertices), len(faces), capture, producer_stream=stream)
        wp.synchronize_stream(stream)
        pool.poll()
        return scene_api.DeviceGeometryReference(pool, source_key, len(vertices), len(faces), material)

    try:
        oracle = flat_fixture.oracle_kernel(wp)
        rows = []
        cases = list(flat_fixture.tie_scenes(contracts))
        # A topology change with identical counts; no position/count heuristic
        # may reuse the prior face mapping.
        label, flat, camera = cases[0]
        rewired = flat.triangles.copy()
        rewired[:2] = [[0, 1, 3], [1, 2, 3]]
        cases.append(('equal_count_topology_change', replace(flat, triangles=rewired), camera))
        for label, flat, camera in cases:
            base.checkpoint(path, report, 'synthetic-' + label)
            ref = publish(flat.vertices[:4], flat.triangles[:2])
            tail = contracts.SceneSnapshot(flat.vertices[4:], flat.triangles[2:] - 4,
                                           flat.material_indices[2:], flat.materials, flat.world, flat.frame)
            request = scene_api.DeviceSceneRequest((ref, tail), flat.materials, flat.world, flat.frame)
            expected = flat_fixture.flat_pixels(wp, oracle, device, flat, camera, 32, 24, 2)
            frame, polls = guarded_render(backend, request, camera, 32, 24, 2)
            np.testing.assert_array_equal(frame.pixels, expected, err_msg=label)
            rows.append(dict(label=label, scope='adversarial synthetic geometry', polls=polls,
                             pixels_sha256=producer_fixture.digest(frame.pixels)))
            pool.poll()
        # Counts grow, shrink, then authoritatively empty. The independent
        # oracle has no knowledge of device request layouts or copy kernels.
        camera = contracts.CameraSnapshot(np.eye(4))
        materials = (contracts.Material(kind='emission', emission=(0.7, 0.2, 0.1)),)
        for count in (12, 1, 0):
            vertices = np.tile([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], (count, 1))
            faces = np.arange(3 * count, dtype=np.int32).reshape(-1, 3)
            ref = publish(vertices, faces)
            request = scene_api.DeviceSceneRequest((ref,), materials, (0.2, 0.3, 0.4), generation)
            flat = contracts.SceneSnapshot(vertices, faces, np.zeros(count, np.int32),
                                           materials, request.world, generation)
            expected = flat_fixture.flat_pixels(wp, oracle, device, flat, camera, 32, 24, 2)
            frame, polls = guarded_render(backend, request, camera, 32, 24, 2)
            np.testing.assert_array_equal(frame.pixels, expected)
            assert (backend.mesh is None) == (count == 0)
            rows.append(dict(label='active_triangle_count_' + str(count),
                             scope='adversarial synthetic geometry', polls=polls,
                             pixels_sha256=producer_fixture.digest(frame.pixels)))
            pool.poll()
        invalid = []
        for label, vertices, faces in (
            ('nonfinite_position', [[float('nan'), 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
            ('invalid_index', np.eye(3), [[0, -1, 99]]),
        ):
            ref = publish(vertices, faces)
            request = scene_api.DeviceSceneRequest((ref,), materials, (0.2, 0.3, 0.4), generation)
            try:
                guarded_render(backend, request, camera, 32, 24, 2)
            except RuntimeError as exc:
                assert 'invalid device geometry' in str(exc)
            else:
                raise AssertionError('invalid geometry produced a presented frame')
            assert not backend.pending and backend.mesh is None and backend.history.samples == 0
            wp.synchronize_stream(backend._prepare_stream)
            pool.poll()
            assert budget.reserved_bytes == pool.reserved_bytes
            invalid.append(label)
            pool.poll()
        report['invalid_geometry_controls'] = invalid
        ref = publish(np.eye(3), [[0, 1, 2]])
        request = scene_api.DeviceSceneRequest((ref,), materials, (0.2, 0.3, 0.4), generation)
        guarded_render(backend, request, camera, 32, 24, 2)
        pool.poll()
        # The readback guard must reject an explicit staging copy, not merely
        # the convenience array.numpy() API. No forbidden transfer is enqueued.
        leaked_host = wp.empty(request.vertex_count, dtype=wp.vec3, device='cpu', pinned=True)
        def wrong_transfer(reader_stream):
            wp.copy(leaked_host, backend.arrays['points'], count=request.vertex_count, stream=reader_stream)
        try:
            with patch.object(backend, '_queue_geometry_validation', wrong_transfer):
                guarded_render(backend, request, camera, 32, 24, 2)
        except AssertionError as exc:
            assert 'geometry readback through wp.copy' in str(exc)
        else:
            raise AssertionError('explicit geometry readback escaped the verifier guard')
        assert not backend.pending
        report['wrong_geometry_readback_control'] = 'rejected before CUDA transfer'
        # Same packet, different static scene: a retired aggregate must stay
        # charged until its independently delayed last-read fence is checked.
        old_mesh = weakref.ref(backend.mesh)
        host = contracts.SceneSnapshot(np.eye(3) + 3, [[0, 1, 2]], [0], materials,
                                       request.world, request.frame)
        edited = replace(request, parts=(host, request.reference))
        old_record = pool._ops.record
        retirement_gates = []
        def delayed_retirement(dev, reader_stream):
            gate = pool_fixture.StreamGate(modules, wp, dev, reader_stream)
            retirement_gates.append(gate)
            active_gates.append(gate)
            gate.arm()
            return old_record(dev, reader_stream)
        try:
            with patch.object(pool._ops, 'record', delayed_retirement):
                guarded_render(backend, edited, camera, 32, 24, 2)
            assert len(retirement_gates) == 1
            charged = budget.reserved_bytes
            pool.poll()
            assert budget.generation_count == 2 and budget.reserved_bytes == charged
            assert old_mesh() is not None
            newer_host = replace(host, vertices=host.vertices + 0.25)
            third = replace(edited, parts=(newer_host, request.reference))
            try:
                backend.submit(third, camera, 32, 24, max_bounces=1, seed=2)
            except scene_api.GeometryBudgetExceeded:
                pass
            else:
                raise AssertionError('third aggregate generation entered a two-generation ledger')
            assert not backend.pending and budget.reserved_bytes == charged
        finally:
            for gate in retirement_gates:
                gate.open()
        wp.synchronize_stream(backend._prepare_stream)
        for gate in retirement_gates:
            gate.verify()
        pool.poll()
        gc.collect()
        assert old_mesh() is None and budget.generation_count == 1 and budget.reserved_bytes < charged
        report['delayed_aggregate_retirement'] = dict(bytes_held_until_fence=True,
                                                     third_static_revision_denied=True)
        request = edited
        # Delayed trace: a paused accepted aggregate retains its packet even
        # after a newer packet is ready; third generation is denied pre-capture.
        current = request
        publish(np.eye(3), [[0, 1, 2]])
        gate = pool_fixture.StreamGate(modules, wp, device, backend._render_stream)
        active_gates.append(gate)
        try:
            gate.arm()
            assert backend.submit(current, camera, 32, 24, max_bounces=1, seed=2)
            assert backend.collect() is None and pool.occupied == 2
            future = replace(source_key, generation=source_key.generation + 1,
                             state_revision=source_key.state_revision + 1)
            assert pool.select(future)
            called = []
            assert not budget.submit_surface(future, 3, 1, lambda *a: called.append(True), producer_stream=stream)
            assert not called and budget.generation_count == 2
        finally:
            gate.open()
        poll_frame(backend)
        gate.verify()
        # Restore is forbidden by the pool watermark; use the still-owned
        # current aggregate for the following fault and close the candidate.
        report['delayed_trace_control'] = dict(pending_observed=True, third_capture_denied=True,
                                               generations=budget.generation_count)
        # A failed drain must retain packet, aggregate arrays, and BVH even
        # after the exception traceback is cleared.
        mesh_ref, points_ref = weakref.ref(backend.mesh), weakref.ref(backend.arrays['points'])
        real_wp = backend.wp
        class FailedRetirement:
            def __getattr__(self, name):
                return getattr(real_wp, name)
            def copy(self, dest, src, *args, **kwargs):
                if dest.device.is_cpu and src.device.is_cuda:
                    raise RuntimeError('injected device consumer transfer fault')
                return real_wp.copy(dest, src, *args, **kwargs)
            def synchronize_stream(self, stream):
                raise RuntimeError('injected device consumer drain fault')
        backend.wp = FailedRetirement()
        try:
            backend.submit(current, camera, 32, 24, max_bounces=1, seed=2)
        except RuntimeError as exc:
            assert str(exc) == 'injected device consumer drain fault'
        else:
            raise AssertionError('retirement fault did not fire')
        backend._job.failure.__traceback__ = None
        gc.collect()
        assert backend.pending and mesh_ref() is not None and points_ref() is not None
        assert budget.reserved_bytes > pool.reserved_bytes
        backend.wp = real_wp
        backend.close()
        pool.close()
        gc.collect()
        assert mesh_ref() is None and points_ref() is None
        assert budget.reserved_bytes == 0
        report['failed_drain_control'] = 'owners held without traceback; retry close safely released'
        report['synthetic_rows'] = rows
        report['synthetic_budget'] = dict(peak_explicit_array_bytes=budget.peak_bytes,
                                         peak_geometry_generations=budget.peak_generations,
                                         bounded_denials=budget.denials)
        assert budget.peak_generations <= 2
    finally:
        for gate in active_gates:
            gate.open()
        backend.wp = wp
        backend.close()
        pool.close()


def main():
    args = base.arguments('artifacts/fspt-p2/device-consumer.json')
    path = Path(args.json_out).resolve()
    folder = path.with_name(path.stem + '-originals-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', scope='development device consumer, no viewport/publication integration',
                  verifier_sha256=base.sha(Path(__file__)))
    addon = None
    try:
        base.checkpoint(path, report, 'load')
        addon = base.load_addon(args, report)
        report['runtime'] = base.runtime_info(addon)
        assert report['runtime']['warp'] == '1.19.0'
        def modules(name):
            return importlib.import_module(addon.__name__ + '.' + name)
        package = Path(addon.__file__).resolve().parent
        report['loaded_sources'] = {}
        for name in ('fspt.device_scene', 'fspt.device_backend', 'fspt.device_scene_kernels',
                     'fspt.async_backend', 'fspt.surface_pool', 'sim.device_surface', 'sim.surface'):
            loaded = Path(modules(name).__file__).resolve()
            assert loaded.is_relative_to(package)
            report['loaded_sources'][name] = dict(path=str(loaded), sha256=base.sha(loaded))
        report['fixture_helpers'] = {Path(m.__file__).name: base.sha(Path(m.__file__))
                                    for m in (base, producer_fixture, flat_fixture, pool_fixture)}
        live_rows(modules, report, folder, path)
        live_rows(modules, report, folder, path, waves=True)
        synthetic_rows(modules, report, folder, path)
        report.update(status='VERIFIED', stage='complete')
    except BaseException:
        report.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        try:
            if addon is not None:
                addon.unregister()
        finally:
            base.write_report(path, report)
    print('FSPT-DEVICE-CONSUMER-PASS', path, flush=True)


if __name__ == '__main__':
    main()
