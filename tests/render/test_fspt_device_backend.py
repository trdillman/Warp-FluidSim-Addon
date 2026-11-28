"""Real Warp CPU preparation and fence-controlled pool ownership; CUDA is separate."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_fspt_device_scene import key

from fluidSim_warp_blender51.fspt.contracts import Material, SceneSnapshot
from fluidSim_warp_blender51.fspt.device_scene import (
    DeviceGeometryReference,
    DeviceSceneRequest,
    DeviceSceneUnavailable,
    GeometryArrayBudget,
)
from fluidSim_warp_blender51.fspt.surface_pool import SurfacePool
from fluidSim_warp_blender51.sim.device_surface import DeviceSurfacePacket


class Ops:
    def producer(self, device, stream):
        assert stream.device == device

    def allocate(self, device, nv, nt):
        return SimpleNamespace(retire_source=lambda: True, drain=lambda: None)

    def validate(self, packet, dest, stamp, nv, nt):
        assert packet._destination is dest

    def reader(self, device, stream):
        assert stream.device == device

    def record(self, device, stream):
        return SimpleNamespace(done=True)

    def completed(self, device, event):
        return event.done

    def sync_reader(self, device, stream):
        pass


@pytest.fixture
def prepared():
    import warp as wp

    from fluidSim_warp_blender51.fspt.device_backend import DeviceAsyncCudaBackend
    device = wp.get_device('cpu')
    pool = SurfacePool(device, byte_limit=1024 * 1024, ops=Ops())
    budget = GeometryArrayBudget(pool, byte_limit=2 * 1024 * 1024)
    backend = DeviceAsyncCudaBackend.__new__(DeviceAsyncCudaBackend)
    backend.wp, backend.device = wp, device
    backend._initialize_resources()
    backend._initialize_device_resources(budget)
    backend._prepare_stream = SimpleNamespace(device=device)
    yield backend, pool, budget
    backend._forget_scene()
    backend._pending_resources.clear()
    backend._release_retired_geometry()
    pool.close()


def publish(backend, pool, budget, vertices, triangles, generation):
    wp = backend.wp
    stamp = key(generation)
    assert pool.select(stamp)
    pool.poll()
    points = wp.array(np.asarray(vertices, np.float32).reshape(-1, 3), dtype=wp.vec3, device=pool.device)
    faces = wp.array(np.asarray(triangles, np.int32).ravel(), dtype=wp.int32, device=pool.device)
    def capture(dest, build):
        return DeviceSurfacePacket(build, pool.device, points, faces, len(vertices), len(triangles),
                                   None, None, dest)
    assert budget.submit_surface(stamp, len(vertices), len(triangles), capture,
                                 producer_stream=backend._prepare_stream)
    pool.poll()
    return DeviceGeometryReference(pool, stamp, len(vertices), len(triangles), 0)


def scene(ref):
    mats = (Material(),)
    a = SceneSnapshot(np.eye(3) - 2, [[0, 1, 2]], [0], mats, (0, 0, 0), 1)
    b = SceneSnapshot(np.eye(3) + 2, [[0, 2, 1]], [0], mats, (0, 0, 0), 1)
    return DeviceSceneRequest((a, ref, b), mats, (0.1, 0.2, 0.3), 1)


def test_ordered_aggregate_matches_independent_flat_arrays_without_device_readback(prepared, monkeypatch):
    backend, pool, budget = prepared
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    request = scene(publish(backend, pool, budget, verts, faces, 1))
    with monkeypatch.context() as patch:
        patch.setattr(backend.wp.array, 'numpy', lambda self: pytest.fail('geometry readback'))
        backend._set_scene(request)
    expected_v = np.vstack([request.parts[0].vertices, verts, request.parts[2].vertices])
    expected_t = np.vstack([request.parts[0].triangles, faces + 3, request.parts[2].triangles + 7])
    np.testing.assert_array_equal(backend.arrays['points'].numpy(), expected_v)
    np.testing.assert_array_equal(backend.arrays['indices'].numpy(), expected_t.ravel())
    assert int(backend.arrays['validation'].numpy()[0]) == 0
    assert budget.reserved_bytes == pool.reserved_bytes + expected_v.nbytes + expected_t.nbytes + 4
    assert len(pool._slots[0].readers) == 1 and pool._slots[0].readers[0]._slot is not None
    mesh = backend.mesh
    backend._set_scene(replace(request, frame=2))
    assert backend.mesh is mesh


def test_retired_aggregate_holds_second_slot_until_checked_job_retirement(prepared):
    backend, pool, budget = prepared
    verts, faces = np.eye(3), [[0, 1, 2]]
    backend._set_scene(scene(publish(backend, pool, budget, verts, faces, 1)))
    first = backend._device_current
    backend._set_scene(scene(publish(backend, pool, budget, verts + 1, faces, 2)))
    assert first in backend._device_retired
    assert pool.occupied == 2 and len(backend._device_retired) == 1
    third = key(3)
    assert pool.select(third)
    pool.poll()
    assert pool.occupied == 2
    called = []
    assert not budget.submit_surface(third, 3, 1, lambda *args: called.append(1),
                                     producer_stream=backend._prepare_stream)
    assert not called
    before = budget.reserved_bytes
    backend._pending_resources.clear()
    backend._release_retired_geometry()
    pool.poll()
    assert pool.occupied == 1 and budget.reserved_bytes < before
    assert backend._device_current.lease._slot is not None


def test_stale_reference_does_no_upload_and_releases_reserved_bytes(prepared):
    backend, pool, budget = prepared
    request = scene(publish(backend, pool, budget, np.eye(3), [[0, 1, 2]], 1))
    assert pool.select(key(2))
    before = budget.reserved_bytes
    with pytest.raises(DeviceSceneUnavailable, match='retired'):
        backend._set_scene(request)
    assert backend.mesh is None and budget.reserved_bytes == before
    assert not backend._device_retired


def test_failed_build_keeps_candidate_owners_until_retirement_and_old_scene_valid(prepared, monkeypatch):
    backend, pool, budget = prepared
    first = scene(publish(backend, pool, budget, np.eye(3), [[0, 1, 2]], 1))
    backend._set_scene(first)
    accepted, mesh = backend._device_current, backend.mesh
    second = scene(publish(backend, pool, budget, np.eye(3) + 1, [[0, 1, 2]], 2))
    with monkeypatch.context() as patch:
        patch.setattr(backend, '_build_mesh', lambda *args: (_ for _ in ()).throw(RuntimeError('build fault')))
        with pytest.raises(RuntimeError, match='build fault'):
            backend._set_scene(second)
    assert backend.mesh is mesh and backend._device_current is accepted
    assert backend.scene_digest == first.digest and len(backend._device_retired) == 1
    assert all(slot.readers and slot.readers[0]._slot is not None for slot in pool._slots)


@pytest.mark.parametrize('bad_vertices,bad_faces', [
    ([[float('nan'), 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
    (np.eye(3), [[0, -1, 99]]),
])
def test_invalid_device_data_is_flagged_and_cannot_enter_unsafe_bvh(prepared, bad_vertices, bad_faces):
    backend, pool, budget = prepared
    backend._set_scene(scene(publish(backend, pool, budget, bad_vertices, bad_faces, 1)))
    assert int(backend.arrays['validation'].numpy()[0]) != 0
    assert np.isfinite(backend.arrays['points'].numpy()).all()
    indices = backend.arrays['indices'].numpy()
    assert indices.min() >= 0 and indices.max() < len(backend.arrays['points'])
    backend._device_current.validation_host.assign(backend.arrays['validation'].numpy())
    with pytest.raises(RuntimeError, match='invalid device geometry'):
        backend._validate_completed_geometry()


def test_actual_kernel_images_preserve_flat_ties_and_global_materials(prepared):
    from test_fspt_persistent_assets import load_verifier

    from fluidSim_warp_blender51.fspt import contracts
    from fluidSim_warp_blender51.fspt.kernels import trace_paths
    backend, pool, budget = prepared
    wp = backend.wp
    oracle = load_verifier()
    oracle_kernel = oracle.oracle_kernel(wp)
    width, height, seed = 32, 24, 2
    palette = set()
    for generation, (label, flat, camera) in enumerate(oracle.tie_scenes(contracts), 1):
        ref = publish(backend, pool, budget, flat.vertices[:4], flat.triangles[:2], generation)
        tail = SceneSnapshot(flat.vertices[4:], flat.triangles[2:] - 4, flat.material_indices[2:],
                             flat.materials, flat.world, flat.frame)
        request = DeviceSceneRequest((ref, tail), flat.materials, flat.world, flat.frame)
        backend._set_scene(request)
        accumulation = wp.zeros(width * height, dtype=wp.vec3, device='cpu')
        output = wp.empty(width * height, dtype=wp.vec4, device='cpu')
        arrays = backend.arrays
        wp.launch(trace_paths, dim=width * height, device='cpu', inputs=[
            backend.mesh.id, request.triangle_count, arrays['materials'], arrays['kinds'],
            arrays['colors'], arrays['emissions'], arrays['iors'], wp.vec3(*request.world),
            wp.mat44(*camera.inverse_view_projection.ravel().tolist()), width, height, 0, 1, 1,
            seed, accumulation, output])
        actual = output.numpy().reshape(height, width, 4)
        expected = oracle.flat_pixels(wp, oracle_kernel, 'cpu', flat, camera, width, height, seed)
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        palette.update(map(tuple, np.unique(expected.reshape(-1, 4), axis=0)))
        backend._pending_resources.clear()
        backend._release_retired_geometry()
        pool.poll()
    assert budget.peak_generations <= 2
    assert len(palette) > 1


def test_readiness_offer_is_metadata_only_and_requires_current_ready_healthy_slot(prepared):
    backend, pool, budget = prepared
    ref = publish(backend, pool, budget, np.eye(3), [[0, 1, 2]], 1)
    slot = pool._slots[0]
    slot.ready = False
    assert budget.surface_ready(ref.key) is None
    slot.ready = True
    assert budget.surface_ready(ref.key) == (3, 1)
    assert not slot.readers
    pool.poisoned = True
    assert budget.surface_ready(ref.key) is None
    pool.poisoned = False
    assert pool.select(key(2))
    assert budget.surface_ready(ref.key) is None


def test_retired_aggregate_stays_charged_until_pool_checks_its_delayed_fence(prepared, monkeypatch):
    import gc
    import weakref

    from fluidSim_warp_blender51.fspt.device_scene import GeometryBudgetExceeded
    backend, pool, budget = prepared
    first = scene(publish(backend, pool, budget, np.eye(3), [[0, 1, 2]], 1))
    backend._set_scene(first)
    old = backend._device_current
    old_mesh, old_points = weakref.ref(old.mesh), weakref.ref(old.arrays['points'])
    changed = replace(first.parts[0], vertices=first.parts[0].vertices + 0.25)
    second = replace(first, parts=(changed, *first.parts[1:]))
    backend._set_scene(second)
    before = budget.reserved_bytes
    fence = SimpleNamespace(done=False)
    monkeypatch.setattr(pool._ops, 'record', lambda *args: fence)
    backend._pending_resources.clear()  # CPU preparation has completed here.
    backend._release_retired_geometry()
    pool.poll()
    gc.collect()
    assert pool.occupied == 1 and budget.generation_count == 2
    assert budget.reserved_bytes == before and old_mesh() is not None and old_points() is not None
    newer = replace(changed, vertices=changed.vertices + 0.25)
    third = replace(first, parts=(newer, *first.parts[1:]))
    with pytest.raises(GeometryBudgetExceeded, match='two'):
        backend._set_scene(third)
    fence.done = True
    pool.poll()
    gc.collect()
    assert old_mesh() is None and old_points() is None
    assert old.arrays is None and old.mesh is None
    assert budget.reserved_bytes < before and budget.generation_count == 1
    backend._set_scene(third)
