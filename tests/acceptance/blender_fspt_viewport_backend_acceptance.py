"""Real CUDA broker/worker lifecycle acceptance; no GUI or Final-handoff claim."""
from __future__ import annotations

import importlib
import gc
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def check_frame(frame, expected, device, key):
    """Shared grader: reject wrong source/device, missing or constant output."""
    assert frame is not None, 'missing completed image'
    assert frame.device == str(device), 'workload device differs from requested CUDA device'
    assert frame.key == key, 'completed image belongs to a different captured request'
    assert frame.samples == 1, 'unexpected accumulation sample count'
    expected = np.asarray(expected)
    assert (np.isfinite(expected).all()
            and np.unique(expected[..., :3].reshape(-1, 3), axis=0).shape[0] > 1), 'oracle is trivial or invalid'
    assert np.isfinite(frame.pixels).all(), 'image contains nonfinite values'
    np.testing.assert_array_equal(frame.pixels, expected)


def until(operation, accept, message, seconds=30):
    deadline, polls = time.monotonic() + seconds, 0
    while time.monotonic() < deadline:
        polls += 1
        value = operation()
        if accept(value):
            return value, polls
        time.sleep(0.001)
    raise TimeoutError(message)


def worker_collect(backend, wp, device):
    result = []
    def work():
        try:
            with wp.ScopedDevice(device):
                result.append((threading.get_ident(), backend.collect(wait=True)))
        except BaseException as exc:
            result.append((threading.get_ident(), exc))
    thread = threading.Thread(target=work, name='FSPT acceptance collector', daemon=True)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive() and len(result) == 1, 'worker collection deadline exceeded'
    ident, value = result[0]
    assert ident != threading.get_ident(), 'collection did not run on another thread'
    if isinstance(value, BaseException):
        raise RuntimeError('native worker collection failed') from value
    return ident, value


def run_broker(modules, helpers, report, folder, path):
    base, flat_fixture, gate_type = helpers.base, helpers.flat, helpers.gate
    wp = modules('runtime.warp_runtime')._import_warp()
    device = wp.get_device(base.script_device())
    assert device.is_cuda, 'native broker acceptance requires CUDA'
    contracts, scene_api = modules('fspt.contracts'), modules('fspt.device_scene')
    producer, pool_api = modules('sim.device_surface'), modules('fspt.surface_pool')
    route_api = modules('fspt.viewport_backend')
    pool = pool_api.SurfacePool(device, byte_limit=32 * 1024 * 1024)
    budget = scene_api.GeometryArrayBudget(pool, byte_limit=128 * 1024 * 1024)
    factories, gates, stream_receipts = [], [], []
    stream = wp.get_stream(device)
    materials = tuple(contracts.Material(kind='emission', emission=c) for c in (
        (0.8, 0.1, 0.15), (0.1, 0.2, 0.9), (0.1, 0.8, 0.2)))
    world = (0.01, 0.02, 0.03)
    left = contracts.SceneSnapshot([[-.95, -.95, .3], [-.15, -.95, .3], [-.55, -.1, .3]],
                                   [[0, 1, 2]], [0], materials, world, 1)
    right = contracts.SceneSnapshot([[.15, .2, .2], [.95, .2, .2], [.55, .95, .2]],
                                    [[0, 1, 2]], [2], materials, world, 1)
    camera = contracts.CameraSnapshot(np.eye(4))
    width, height, seed = 48, 40, 91
    owner = object()
    oracle = flat_fixture.oracle_kernel(wp)

    def evaluated_factory(dev):
        assert budget.reserved_bytes == 0 and budget.generation_count == 0, 'early evaluated allocation'
        factories.append(('EVALUATED', threading.get_ident()))
        return modules('fspt.async_backend').AsyncCudaBackend(str(dev))

    def device_factory(dev, **options):
        factories.append(('GPU', threading.get_ident()))
        return modules('fspt.device_backend').DeviceAsyncCudaBackend(str(dev), **options)

    broker = route_api.ViewportBackendBroker(device, budget, stop_production=pool.invalidate,
        evaluated_factory=evaluated_factory, device_factory=device_factory)
    first, second = broker.new_backend(), broker.new_backend()

    def publish(generation, shift):
        positions = np.asarray([[-.4 + shift, -.4, 0], [.4 + shift, -.4, 0],
                                [.4 + shift, .4, 0], [-.4 + shift, .4, 0]], np.float32)
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]], np.int32)
        stamp = producer.DeviceSurfaceStamp(owner, (generation,), 'sparse', generation,
                                            generation, float(generation), False, 0, 0, ())
        key = pool_api.SurfaceKey(7, (9, 'broker verifier'), 1, 'LIVE', 'GPU_FINAL',
                                 generation, stamp, generation, generation, generation, float(generation))
        assert pool.select(key)
        pool.poll()
        def capture(destination, expected):
            return destination._capture(device, expected, lambda: producer.SurfaceDeviceArrays(
                wp.array(positions, dtype=wp.vec3, device=device),
                wp.array(triangles.ravel(), dtype=wp.int32, device=device)),
                producer_stream=stream, known_counts=(len(positions), len(triangles)))
        assert budget.submit_surface(key, len(positions), len(triangles), capture, producer_stream=stream)
        wp.synchronize_stream(stream)  # verifier-only source preparation
        pool.poll()
        reference = scene_api.DeviceGeometryReference(pool, key, len(positions), len(triangles), 1)
        request = scene_api.DeviceSceneRequest((left, reference, right), materials, world, generation)
        flat = contracts.SceneSnapshot(np.concatenate((left.vertices, positions, right.vertices)),
            np.concatenate((left.triangles, triangles + 3, right.triangles + 7)),
            [0, 1, 1, 2], materials, world, generation)
        expected = flat_fixture.flat_pixels(wp, oracle, device, flat, camera, width, height, seed)
        return request, flat, expected

    def submit(view, request):
        submitted = view.submit(request, camera, width, height, samples=1, max_bounces=1, seed=seed)
        stream = getattr(view._delegate, '_render_stream', None)
        receipt = getattr(stream, 'retirement', None)
        if receipt is not None and not any(old is receipt for old in stream_receipts):
            stream_receipts.append(receipt)
        return submitted

    def verify(view, request, expected):
        assert submit(view, request)
        frame, polls = until(view.collect, lambda value: value is not None, 'image did not complete')
        check_frame(frame, expected, device, contracts.frame_key(request, camera, width, height, 1, seed))
        assert frame.scene_frame == request.frame and frame.geometry_digest == request.geometry_digest
        return frame, polls

    try:
        base.checkpoint(path, report, 'worker-predecessor')
        request_a, _flat_a, expected_a = publish(1, -.1)
        frame_a, _ = verify(first, request_a, expected_a)
        old = first._delegate._device_current
        request_b, flat_b, expected_b = publish(2, .15)
        assert not np.array_equal(expected_a, expected_b), 'fixture cannot distinguish stale imagery'
        assert submit(first, request_b)
        charged = budget.reserved_bytes
        ident, frame_b = worker_collect(first, wp, device)
        check_frame(frame_b, expected_b, device, contracts.frame_key(request_b, camera, width, height, 1, seed))
        assert frame_b.scene_frame == request_b.frame and frame_b.geometry_digest == request_b.geometry_digest
        assert old in first._delegate._device_retired and old.lease._slot is not None
        assert old.arrays is not None and budget.reserved_bytes == charged
        assert not first.pending and broker.arbiter._active is None
        current = first._delegate._device_current
        assert current.arrays['points'].device.is_cuda and current.arrays['indices'].device.is_cuda
        # Verifier-only independent geometry readback, after checked worker completion.
        actual_positions = current.arrays['points'].numpy()
        actual_triangles = current.arrays['indices'].numpy().reshape(-1, 3)
        np.testing.assert_array_equal(actual_positions, flat_b.vertices)
        np.testing.assert_array_equal(actual_triangles, flat_b.triangles)
        np.testing.assert_array_equal(current.arrays['materials'].numpy(), flat_b.material_indices)
        until(broker.pump, lambda _: old.arrays is None, 'main thread failed checked retirement')
        assert budget.generation_count == 1 and budget.reserved_bytes < charged
        second_frame, _ = verify(second, request_b, expected_b)
        assert first._delegate.mesh is not second._delegate.mesh
        assert budget.generation_count == 1  # duplicate aggregate bytes are still charged
        shared_bytes = budget.reserved_bytes

        base.checkpoint(path, report, 'queued-stale-reference')
        stale = broker.new_backend()
        rival = broker.arbiter.try_acquire(object())
        assert rival is not None and not submit(stale, request_b)
        queued_owner = stale.owner
        assert queued_owner in broker.arbiter._viewport_waiters
        pool.invalidate()
        try:
            submit(stale, request_b)
        except scene_api.DeviceSceneUnavailable as exc:
            assert 'retired' in str(exc) or 'ready' in str(exc)
        else:
            raise AssertionError('stale request remained renderable')
        assert queued_owner not in broker.arbiter._viewport_waiters and broker.arbiter._active is rival
        broker.arbiter.release(rival)
        stale.close()

        base.checkpoint(path, report, 'two-facade-transition')
        gate = gate_type(modules, wp, device, second._delegate._render_stream)
        gates.append(gate)
        gate.arm()
        assert submit(second, request_b)  # accepted owned aggregate needs no source reacquire
        broker.request_mode('EVALUATED')
        assert broker.pump() is None and second._delegate is not None and second._delegate.pending
        assert not any(mode == 'EVALUATED' for mode, _ in factories)
        try:
            submit(first, flat_b)
        except scene_api.DeviceSceneUnavailable as exc:
            assert 'transition' in str(exc)
        else:
            raise AssertionError('evaluated allocation escaped pending GPU retirement')
        gate.open()
        until(broker.pump, lambda mode: mode == 'EVALUATED', 'route transition failed to retire readers')
        gate.verify()
        assert budget.reserved_bytes == 0 and budget.generation_count == 0
        assert second.collect() is not None  # pending result was preserved during route teardown
        host_frame, _ = verify(first, flat_b, expected_b)

        base.checkpoint(path, report, 'gpu-reentry-and-poison-recovery')
        broker.request_mode('GPU')
        until(broker.pump, lambda mode: mode == 'GPU', 'evaluated backend did not retire')
        request_c, flat_c, expected_c = publish(3, 0)
        final_gpu, _ = verify(first, request_c, expected_c)
        charged = budget.reserved_bytes
        broker.request_mode('EVALUATED')
        reached = []
        def fail_record(*_):
            reached.append(True)
            raise RuntimeError('verifier injected last-read record failure')
        with patch.object(pool._ops, 'record', fail_record):
            try:
                broker.pump()
            except RuntimeError as exc:
                assert 'injected last-read record failure' in str(exc)
            else:
                raise AssertionError('last-read failure control was not reached')
        assert reached and broker.recovery_required and budget.reserved_bytes == charged
        assert broker.pump() is None and broker.geometry_recovery_ready
        assert broker.recovery_reason and budget.reserved_bytes == charged
        before_factories = len(factories)
        try:
            submit(first, flat_c)
        except scene_api.DeviceSceneUnavailable:
            pass
        else:
            raise AssertionError('poisoned geometry allowed evaluated allocation')
        assert len(factories) == before_factories
        pool.close()  # explicit owned-resource recovery, never the normal pump
        assert budget.reserved_bytes == 0 and budget.generation_count == 0
        assert broker.pump() == 'EVALUATED'
        recovered, _ = verify(first, flat_c, expected_c)
        original = folder / 'broker-images.npz'
        np.savez(original, first=frame_a.pixels, worker=frame_b.pixels, second=second_frame.pixels,
                 evaluated=host_frame.pixels, gpu_reentry=final_gpu.pixels, recovered=recovered.pixels,
                 oracle_b=expected_b, oracle_c=expected_c, positions=actual_positions,
                 triangles=actual_triangles, oracle_positions=flat_b.vertices, oracle_triangles=flat_b.triangles)
        report['broker'] = dict(status='VERIFIED', worker_thread=ident,
            publication_thread=threading.get_ident(), two_facade_shared_bytes=shared_bytes,
            peak_explicit_array_bytes=budget.peak_bytes, peak_generations=budget.peak_generations,
            memory_excludes='BVH internals, shading, image/history, solver and allocator caching',
            actual_factories=factories, queued_stale_rejected=True, poisoned_recovery_checked=True,
            originals=str(original), sha256=base.sha(original))
        assert budget.peak_generations <= 2 and budget.peak_bytes <= budget.byte_limit
    finally:
        for gate in gates:
            gate.open()
        broker.close()
        pool.close()
        for gate in gates:
            gate.stream = None  # release verifier-only owners of retired render streams
        gates.clear()
        gc.collect()
        assert all(receipt.result == 0 for receipt in stream_receipts), 'unchecked native stream destruction'
        if 'broker' in report:
            assert stream_receipts, 'native rendering produced no owned stream receipts'
            report['broker']['render_stream_retirement_results'] = [r.result for r in stream_receipts]


def run_empty_generation(modules, helpers, report):
    wp = modules('runtime.warp_runtime')._import_warp()
    device = wp.get_device(helpers.base.script_device())
    pool_api, producer = modules('fspt.surface_pool'), modules('sim.device_surface')
    pool = pool_api.SurfacePool(device, byte_limit=1024)
    budget = modules('fspt.device_scene').GeometryArrayBudget(pool, byte_limit=2048)
    broker = modules('fspt.viewport_backend').ViewportBackendBroker(device, budget,
                                                                  stop_production=pool.invalidate)
    stream = wp.get_stream(device)
    gate = helpers.gate(modules, wp, device, stream)
    stamp = producer.DeviceSurfaceStamp(object(), (1,), 'sparse', 1, 1, 0.0, False, 0, 0, ())
    key = pool_api.SurfaceKey(8, (9, 'empty broker verifier'), 1, 'LIVE', 'GPU_FINAL',
                             1, stamp, 1, 1, 1, 0.0)
    destinations = []
    try:
        assert pool.select(key)
        def capture(destination, expected):
            destinations.append(destination)
            record = destination._driver.record
            def delayed(warp, dev, source, event):
                record(warp, dev, source, event)
                if event is destination._ready:
                    gate.arm()
            with patch.object(destination._driver, 'record', delayed):
                return destination._capture(device, expected, lambda: producer.SurfaceDeviceArrays(
                    wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)),
                    producer_stream=stream, known_counts=(0, 0))
        assert budget.submit_surface(key, 0, 0, capture, producer_stream=stream)
        pool.poll()
        assert budget.reserved_bytes == 0 and budget.generation_count == 1
        assert destinations[0]._sources is not None
        broker.request_mode('EVALUATED')
        assert broker.pump() is None, 'zero bytes bypassed the pending geometry generation'
        gate.open()
        until(broker.pump, lambda mode: mode == 'EVALUATED', 'empty source did not retire')
        gate.verify()
        assert budget.generation_count == 0 and destinations[0]._sources is None
        report['empty_generation'] = dict(status='VERIFIED', pending_bytes=0,
            pending_generations=1, completion='actual checked CUDA copy fence', final_generations=0)
    finally:
        gate.open()
        broker.close()
        pool.close()


def run_live_capture(modules, helpers, report, folder):
    """Actual Blender RNA plus controlled selection; not the runtime producer hook."""
    import bpy

    wp = modules('runtime.warp_runtime')._import_warp()
    device = wp.get_device(helpers.base.script_device())
    publication = modules('blender.surface_publication')
    bridge = modules('blender.fspt_bridge')
    adapter = modules('blender.fspt_device_scene')
    session = modules('blender.scene_bridge').session()
    pool = modules('fspt.surface_pool').SurfacePool(device, byte_limit=32 * 1024 * 1024)
    budget = modules('fspt.device_scene').GeometryArrayBudget(pool, byte_limit=128 * 1024 * 1024)
    backend = modules('fspt.device_backend').DeviceAsyncCudaBackend(str(device), geometry_budget=budget)
    try:
        scene, solver = helpers.flat.fixture(bpy, helpers.base, helpers.producer, modules, device, session)
        original, asset_key = helpers.flat.advance(bpy, modules, scene, solver, session, 2)
        host_stamp, = original.publications
        asset = next(span for span in original.asset_spans if span.key == asset_key)
        surface = bridge._surface_object(scene)
        graph = bpy.context.evaluated_depsgraph_get()
        assert adapter.surface_capability_reason(scene, graph.view_layer) == ''
        stamp = solver.surface.device_surface_stamp()
        key = modules('fspt.surface_pool').SurfaceKey(
            host_stamp.scene_token, host_stamp.owner_token, host_stamp.epoch, 'LIVE', 'GPU_FINAL',
            1, stamp, host_stamp.source_timeline_frame, host_stamp.state_revision,
            host_stamp.solver_frame, host_stamp.simulation_time)
        assert pool.select(key)
        stream = wp.get_stream(device)
        # Fixture-only packet production after the independent host oracle.
        # This deliberately makes no cadence/shared-extraction performance claim.
        assert budget.submit_surface(key, asset.vertex_count, asset.triangle_count,
            lambda dest, expected: solver.host_frame().copy_final_device_surface(dest, expected_stamp=expected),
            producer_stream=stream)
        wp.synchronize_stream(stream)
        pool.poll()
        with publication.surface_attempt(scene, publication.SurfaceChannel.GPU_FINAL,
            source=publication.SurfaceSource.LIVE, epoch=host_stamp.epoch,
            timeline_frame=host_stamp.timeline_frame, source_timeline_frame=host_stamp.source_timeline_frame,
            solver_frame=host_stamp.solver_frame, state_revision=host_stamp.state_revision,
            simulation_time=host_stamp.simulation_time, state_ready_at=host_stamp.state_ready_at) as attempt:
            attempt.record(vertex_count=asset.vertex_count, triangle_count=asset.triangle_count,
                           visible=True, obj=surface)
            attempt.succeed(synchronized=True, build_key=stamp.build_key)
        gpu_stamp = publication.current_surface_publication(scene, publication.SurfaceChannel.GPU_FINAL)
        selection = SimpleNamespace(publication=gpu_stamp, key=key, pool=pool,
            vertex_count=asset.vertex_count, triangle_count=asset.triangle_count)
        calls = []

        class MeshGuard:
            def __init__(self, mesh):
                self.mesh = mesh
            def __getattr__(self, name):
                if name in {'vertices', 'loop_triangles', 'loops', 'polygons', 'edges'}:
                    calls.append(name)
                    raise AssertionError('fluid bulk mesh array was accessed')
                return getattr(self.mesh, name)

        class ObjectGuard:
            def __init__(self, obj, original=None):
                self.obj, self._original = obj, original
            @property
            def original(self):
                return self if self._original is None else self._original
            @property
            def data(self):
                return MeshGuard(self.obj.data)
            def __getattr__(self, name):
                if name in {'to_mesh', 'evaluated_geometry'}:
                    calls.append(name)
                    raise AssertionError('fluid evaluated geometry was accessed')
                return getattr(self.obj, name)

        guarded = ObjectGuard(surface)
        output_token = publication.id_token(surface)

        class InstanceGuard:
            def __init__(self, instance):
                self.instance = instance
            @property
            def object(self):
                value = self.instance.object
                return (ObjectGuard(value, guarded)
                        if publication.id_token(value.original) == output_token else value)
            def __getattr__(self, name):
                return getattr(self.instance, name)

        class GraphGuard:
            scene = graph.scene
            view_layer = graph.view_layer
            @property
            def object_instances(self):
                # Do not retain Blender's transient depsgraph iterator values.
                for instance in graph.object_instances:
                    yield InstanceGuard(instance)

        # Challenge the guard itself before qualifying its protected capture.
        for forbidden in ('to_mesh', 'evaluated_geometry'):
            try:
                getattr(guarded, forbidden)
            except AssertionError:
                pass
            else:
                raise AssertionError('fluid access guard did not reject its negative control')
        try:
            guarded.data.vertices.foreach_get('co', np.empty(0))
        except AssertionError:
            pass
        else:
            raise AssertionError('fluid bulk-read guard did not reject its negative control')
        assert calls == ['to_mesh', 'evaluated_geometry', 'vertices']
        calls.clear()
        clock = solver.frame, solver.sim_time, solver.liquid_state_revision
        assert hasattr(bridge, '_device_selection'), 'channel-aware bridge is required'
        with patch.object(bridge, '_surface_object', lambda _: guarded), patch.object(
                bridge, '_device_selection', lambda *_: selection):
            captured = adapter.capture_device_scene(GraphGuard(), selection)
        assert not calls, 'fluid mesh guard was reached during protected capture'
        camera = modules('fspt.contracts').CameraSnapshot(
            [[.28, 0, 0, .25], [0, .28, 0, .25], [0, 0, 1, 0], [0, 0, 0, 1]])
        width, height, seed = 80, 64, 91
        expected = helpers.flat.flat_pixels(wp, helpers.flat.oracle_kernel(wp), device,
                                            original, camera, width, height, seed)
        frame, polls = helpers.consumer.guarded_render(backend, captured, camera, width, height, seed)
        check_frame(frame, expected, device, modules('fspt.contracts').frame_key(
            captured, camera, width, height, 1, seed))
        assert clock == (solver.frame, solver.sim_time, solver.liquid_state_revision)
        points, triangles = backend.arrays['points'].numpy(), backend.arrays['indices'].numpy().reshape(-1, 3)
        np.testing.assert_allclose(points, original.vertices, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(triangles, original.triangles)
        np.testing.assert_array_equal(backend.arrays['materials'].numpy(), original.material_indices)
        saved = folder / 'live-adapter.npz'
        np.savez(saved, pixels=frame.pixels, oracle_pixels=expected, points=points, triangles=triangles,
                 oracle_points=original.vertices, oracle_triangles=original.triangles)
        report['live_adapter'] = dict(status='VERIFIED', polls=polls, solver_clock=clock,
            scope='actual Blender scene, injected selection binding; no producer cadence or UI claim',
            evaluated_generation=host_stamp.generation, gpu_publication_generation=gpu_stamp.generation,
            pool_generation=key.generation, fluid_bulk_accesses=calls, guard_negative_controls=3,
            originals=str(saved), sha256=helpers.base.sha(saved))
    finally:
        receipt = getattr(getattr(backend, '_render_stream', None), 'retirement', None)
        backend.close()
        pool.close()
        session._clear_runtime()
        gc.collect()
        if receipt is not None:
            assert receipt.result == 0, 'live adapter render stream destruction was not checked'
            if 'live_adapter' in report:
                report['live_adapter']['render_stream_retirement_result'] = receipt.result


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blender_fspt_acceptance as base
    import blender_fspt_device_consumer_acceptance as consumer
    import blender_fspt_device_surface_acceptance as producer
    import blender_fspt_persistent_assets as flat
    import blender_fspt_surface_pool_acceptance as pool_fixture
    args = base.arguments('artifacts/fspt-p2/viewport-backend.json')
    path = Path(args.json_out).resolve()
    folder = path.with_name(path.stem + '-originals-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='RUNNING', scope='inactive CUDA broker/worker component; no GUI or Final handoff',
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
        for name in ('fspt.viewport_backend', 'fspt.device_backend', 'fspt.device_scene',
                     'fspt.device_scene_kernels', 'fspt.backend', 'fspt.async_backend',
                     'fspt.surface_pool', 'sim.device_surface', 'runtime.cuda_driver',
                     'blender.fspt_device_scene', 'blender.fspt_bridge', 'blender.surface_publication',
                     'sim.surface', 'sim.frame_output'):
            loaded = Path(modules(name).__file__).resolve()
            assert loaded.is_relative_to(package)
            report['loaded_sources'][name] = dict(path=str(loaded), sha256=base.sha(loaded))
        report['fixture_helpers'] = {Path(m.__file__).name: base.sha(Path(m.__file__))
                                    for m in (base, flat, pool_fixture, consumer, producer)}
        helpers = SimpleNamespace(base=base, flat=flat, gate=pool_fixture.StreamGate,
                                  consumer=consumer, producer=producer)
        run_broker(modules, helpers, report, folder, path)
        base.checkpoint(path, report, 'zero-byte-pending-generation')
        run_empty_generation(modules, helpers, report)
        base.checkpoint(path, report, 'actual-blender-fluid-adapter')
        run_live_capture(modules, helpers, report, folder)
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
    print('FSPT-VIEWPORT-BACKEND-PASS', path, flush=True)


if __name__ == '__main__':
    main()
