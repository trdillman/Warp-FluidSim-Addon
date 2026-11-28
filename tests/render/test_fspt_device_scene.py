"""Device scene requests preserve ordering without materializing fluid geometry."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt import contracts
from fluidSim_warp_blender51.fspt.surface_pool import SurfaceKey
from fluidSim_warp_blender51.sim.device_surface import DeviceSurfaceStamp


def api():
    from fluidSim_warp_blender51.fspt import device_scene
    return device_scene


def key(generation=1):
    return SurfaceKey(3, (1, 'surface'), 1, 'LIVE', 'GPU_FINAL', generation,
                      DeviceSurfaceStamp(object(), (4,), 'sparse', 1, 1, 0.0, False, 0, 0, ()),
                      1, 4, 1, 0.0)


def host(offset=0):
    return contracts.SceneSnapshot(np.eye(3) + offset, [[0, 1, 2]], [0],
                                   (contracts.Material(),), (0, 0, 0), 1)


def test_request_is_metadata_only_and_preserves_order_and_material_mapping():
    m = api()
    pool = SimpleNamespace(device='cuda:0')
    fluid = m.DeviceGeometryReference(pool, key(), 4, 2, 0)
    before, after = host(), host(3)
    request = m.DeviceSceneRequest((before, fluid, after), before.materials, (0.1, 0.2, 0.3), 7)
    assert request.vertex_count == 10 and request.triangle_count == 4
    assert [(p.vertex_offset, p.index_offset) for p in request.layout] == [(0, 0), (3, 3), (7, 9)]
    np.testing.assert_array_equal(request.material_indices, [0, 0, 0, 0])
    assert request.parts[1] is fluid and not hasattr(request, 'vertices')
    assert request.geometry_digest.startswith('gpu-identity:')
    with pytest.raises(ValueError):
        request.material_indices.setflags(write=True)
    assert replace(request, parts=(after, fluid, before)).geometry_digest != request.geometry_digest
    assert replace(request, frame=8).geometry_digest == request.geometry_digest
    assert replace(request, frame=8).digest != request.digest


def test_request_rejects_invalid_metadata_before_any_pool_access():
    m = api()
    pool = SimpleNamespace(device='cuda:0')
    stamp = key()
    for nv, nt, mi in [(-1, 1, 0), (0, 1, 0), (3, -1, 0), (3, 1, -1), (2**31, 0, 0)]:
        with pytest.raises((ValueError, OverflowError)):
            m.DeviceGeometryReference(pool, stamp, nv, nt, mi)
    f = m.DeviceGeometryReference(pool, stamp, 3, 1, 1)
    with pytest.raises(ValueError, match='material'):
        m.DeviceSceneRequest((f,), (contracts.Material(),), (0, 0, 0), 1)
    f = replace(f, material_index=0)
    with pytest.raises(ValueError, match='one'):
        m.DeviceSceneRequest((f, f), (contracts.Material(),), (0, 0, 0), 1)
    with pytest.raises(ValueError, match='one'):
        m.DeviceSceneRequest((host(),), (contracts.Material(),), (0, 0, 0), 1)


def test_empty_device_generation_is_authoritative_and_identity_changes():
    m = api()
    pool = SimpleNamespace(device='cuda:0')
    f = m.DeviceGeometryReference(pool, key(), 0, 0, 0)
    request = m.DeviceSceneRequest((f,), (contracts.Material(),), (0, 0, 0), 1)
    assert request.vertex_count == request.triangle_count == 0
    next_f = replace(f, key=replace(f.key, generation=2, state_revision=5))
    assert replace(request, parts=(next_f,)).geometry_digest != request.geometry_digest
    # Hashing cannot deepcopy/repr the opaque producer owner into a new identity.
    assert replace(request).geometry_digest == request.geometry_digest


def test_shared_budget_denies_before_packet_capture_and_counts_all_aggregate_duplicates():
    m = api()
    calls = []
    pool = SimpleNamespace(reserved_bytes=48, submit=lambda *a, **kw: calls.append(a) or True)
    budget = m.GeometryArrayBudget(pool, byte_limit=200)
    first = budget.reserve(52)
    second = budget.reserve(52)
    assert budget.reserved_bytes == 152
    assert not budget.submit_surface(key(), 3, 2, lambda *a: None, producer_stream=object())
    assert not calls
    with pytest.raises(m.GeometryBudgetExceeded):
        budget.reserve(49)
    first.release()
    assert budget.reserved_bytes == 100
    assert budget.submit_surface(key(), 3, 2, lambda *a: None, producer_stream=object())
    second.release()
    assert budget.reserved_bytes == 48
    first.release()
    assert budget.reserved_bytes == 48


def test_device_request_does_not_allocate_face_arrays_before_renderer_admission(monkeypatch):
    m = api()
    pool = SimpleNamespace(device='cuda:0')
    ref = m.DeviceGeometryReference(pool, key(), 3, 10_000_000, 0)
    with monkeypatch.context() as patch:
        patch.setattr(np, 'full', lambda *a, **k: pytest.fail('face allocation before admission'))
        request = m.DeviceSceneRequest((ref,), (contracts.Material(),), (0, 0, 0), 1)
    assert request.triangle_count == 10_000_000


def test_shared_budget_counts_static_scene_generations_even_when_packet_is_same():
    m = api()
    stamp = key()
    pool = SimpleNamespace(reserved_bytes=48, _slots=[SimpleNamespace(key=stamp)],
                           submit=lambda *a, **kw: pytest.fail('third generation capture'))
    budget = m.GeometryArrayBudget(pool, byte_limit=10000)
    first = budget.reserve(52, generation=(stamp, 'static-A'))
    second = budget.reserve(52, generation=(stamp, 'static-B'))
    with pytest.raises(m.GeometryBudgetExceeded, match='two'):
        budget.reserve(52, generation=(stamp, 'static-C'))
    assert not budget.submit_surface(replace(stamp, generation=2), 3, 1,
                                     lambda *a: None, producer_stream=object())
    second.release()
    third = budget.reserve(52, generation=(stamp, 'static-C'))
    first.release()
    third.release()
