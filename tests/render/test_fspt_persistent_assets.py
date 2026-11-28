"""Persistent geometry decisions and failure recovery on real Warp CPU resources."""
from dataclasses import replace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt.backend import CudaBackend
from fluidSim_warp_blender51.fspt.contracts import AssetSpan, Material, SceneSnapshot


def scene(order=(1, 2), *, offset=0, flip=False):
    vertices, triangles, spans = [], [], []
    for key in order:
        points = np.eye(3, dtype=np.float32) + (offset if key == 2 else 0)
        vertices.extend(points)
        triangles.append(np.array([0, 2, 1] if key == 2 and flip else [0, 1, 2]) + len(vertices) - 3)
        spans.append(AssetSpan((key,), 3, 1))
    return SceneSnapshot(vertices, triangles, [0] * len(triangles), (Material(),), (0, 0, 0),
                         1, asset_spans=tuple(spans))


@pytest.fixture
def backend():
    wp = pytest.importorskip("warp")
    obj = CudaBackend.__new__(CudaBackend)
    obj.wp, obj.device = wp, "cpu"
    obj._initialize_resources()
    yield obj
    obj.close()


def test_unchanged_ranges_skip_uploads_and_bvh_is_fresh(backend):
    first = scene()
    backend._set_scene(first)
    points, indices, mesh = backend.arrays["points"], backend.arrays["indices"], backend.mesh
    counts = backend.statistics.copy()
    backend._set_scene(scene(offset=1))
    assert backend.arrays["points"] is points and backend.arrays["indices"] is indices
    assert backend.mesh is not mesh
    assert backend.statistics["position_upload_bytes"] - counts["position_upload_bytes"] == 36
    assert backend.statistics["index_upload_bytes"] == counts["index_upload_bytes"]
    assert backend.statistics["geometry_array_allocations"] == 2
    mesh, counts = backend.mesh, backend.statistics.copy()
    backend._set_scene(scene(offset=1, flip=True))
    assert backend.mesh is not mesh
    assert backend.statistics["index_upload_bytes"] - counts["index_upload_bytes"] == 12
    assert backend.statistics["position_upload_bytes"] == counts["position_upload_bytes"]
    changed = scene((2, 1), offset=1, flip=True)
    backend._set_scene(changed)
    np.testing.assert_array_equal(backend.arrays["points"].numpy(), changed.vertices)
    np.testing.assert_array_equal(backend.arrays["indices"].numpy(), changed.triangles.ravel())
    shrunk = scene((2,), offset=1)
    backend._set_scene(shrunk)
    assert backend.statistics["geometry_array_allocations"] == 2
    np.testing.assert_array_equal(backend.mesh.points.numpy(), shrunk.vertices)
    np.testing.assert_array_equal(backend.mesh.indices.numpy(), shrunk.triangles.ravel())
    backend._set_scene(scene(()))
    assert backend.mesh is None
    backend._set_scene(first)
    np.testing.assert_array_equal(backend.mesh.points.numpy(), first.vertices)
    assert backend.statistics["mesh_builds"] == 6


def test_failed_build_after_real_point_mutation_forces_full_replacement(backend, monkeypatch):
    first = scene()
    backend._set_scene(first)
    old_points = backend.arrays["points"]

    def fail(points, indices):
        np.testing.assert_array_equal(points.numpy(), scene(offset=1).vertices)
        raise RuntimeError("build failed after upload")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "_build_mesh", fail)
        with pytest.raises(RuntimeError, match="build failed after upload"):
            backend._set_scene(scene(offset=1))
    assert backend.scene_digest is None and backend.mesh is None
    backend._set_scene(first)
    assert backend.arrays["points"] is not old_points
    np.testing.assert_array_equal(backend.mesh.points.numpy(), first.vertices)


def test_reject_cross_asset_indices_and_duplicate_identity():
    original = scene()
    with pytest.raises(ValueError, match="asset"):
        replace(original, triangles=[[0, 1, 3], [3, 4, 5]])
    with pytest.raises(ValueError, match="asset"):
        replace(original, asset_spans=(original.asset_spans[0],) * 2)


def test_instance_key_uses_original_parent_and_path_not_transforms():
    from types import SimpleNamespace
    from fluidSim_warp_blender51.blender.fspt_scene import _asset_key

    instance = SimpleNamespace(object=SimpleNamespace(original=12), is_instance=True, parent=37,
                               persistent_id=(1, 21), matrix_world=np.eye(4))
    stable = _asset_key(instance, int)
    instance.matrix_world *= 2
    assert _asset_key(instance, int) == stable
    instance.parent = 38
    assert _asset_key(instance, int) != stable
    instance.parent, instance.persistent_id = 37, (1, 22)
    assert _asset_key(instance, int) != stable
    instance.is_instance = False
    assert _asset_key(instance, int) != stable


def load_verifier():
    import importlib.util
    import sys
    from pathlib import Path

    name = "fspt_flat_reference"
    if name not in sys.modules:
        path = Path(__file__).parents[1] / "acceptance" / "blender_fspt_persistent_assets.py"
        spec = importlib.util.spec_from_file_location(name, path)
        verifier = importlib.util.module_from_spec(spec)
        sys.modules[name] = verifier
        spec.loader.exec_module(verifier)
    return sys.modules[name]


def test_aggregate_pixels_and_wrong_material_control_against_independent_flat_oracle(backend):
    from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot
    from fluidSim_warp_blender51.fspt.kernels import trace_paths

    oracle = load_verifier()
    wp = backend.wp
    kernel = oracle.oracle_kernel(wp)
    backend.kernel = trace_paths
    camera = CameraSnapshot(np.eye(4))
    rear = [[-1, -1, 0.7], [1, -1, 0.7], [1, 1, 0.7], [-1, 1, 0.7]]
    front = [[-0.4, -0.4, 0.1], [0.4, -0.4, 0.1], [0, 0.4, 0.1]]
    mats = tuple(Material(kind="emission", emission=c) for c in [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    s = SceneSnapshot(rear + front, [[0, 1, 2], [0, 2, 3], [4, 5, 6]], [0, 1, 2], mats,
                      (0.2, 0.2, 0.2), 1, asset_spans=(AssetSpan((1,), 4, 2), AssetSpan((2,), 3, 1)))
    moved = s.vertices.copy()
    moved[4:, 0] += 0.3
    moved_scene = replace(s, vertices=moved)
    rewired = replace(moved_scene, triangles=[[0, 1, 2], [0, 1, 2], [4, 5, 6]])
    for case in [s, moved_scene, rewired, replace(s, vertices=[], triangles=[], material_indices=[], asset_spans=())]:
        reference = oracle.flat_pixels(wp, kernel, "cpu", case, camera, 32, 24, 2)
        actual = backend.render(case, camera, 32, 24, max_bounces=1, seed=2)
        np.testing.assert_array_equal(reference, actual.pixels)
    backend._set_scene(s)
    backend.arrays["materials"].assign(np.zeros(3, dtype=np.int32))
    wrong = backend.render(s, camera, 32, 24, max_bounces=1, seed=2)
    reference = oracle.flat_pixels(wp, kernel, "cpu", s, camera, 32, 24, 2)
    assert len(np.unique(reference.reshape(-1, 4), axis=0)) == 3
    assert np.count_nonzero(np.any(wrong.pixels != reference, axis=2)) > 10


def test_unretired_sync_reader_blocks_upload_and_retains_owners(backend, monkeypatch):
    from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot
    from fluidSim_warp_blender51.fspt.kernels import trace_paths

    backend.kernel = trace_paths
    camera = CameraSnapshot(np.eye(4))
    first = scene()
    backend.render(first, camera, 4, 4)
    old = backend.arrays["points"]
    def unavailable(_device):
        raise RuntimeError("reader retirement unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(backend.wp, "synchronize_device", unavailable)
        with pytest.raises(RuntimeError, match="reader retirement unavailable"):
            backend.render(first, camera, 4, 4)
        assert backend._uncertain[backend][1]["points"] is old
        with pytest.raises(RuntimeError, match="reader retirement unavailable"):
            backend.render(scene(offset=1), camera, 4, 4)
        assert backend.statistics["mesh_builds"] == 1
    backend.render(scene(offset=1), camera, 4, 4)
    assert backend not in backend._uncertain and backend.arrays["points"] is not old


@pytest.mark.parametrize("asynchronous", [False, True])
def test_shading_staging_owns_first_upload_across_failed_drain(asynchronous, monkeypatch):
    import gc
    import weakref
    from test_fspt_async_backend import CheckedWarp, Stream

    from fluidSim_warp_blender51.fspt import async_backend
    from fluidSim_warp_blender51.fspt.contracts import CameraSnapshot
    from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter

    wp = CheckedWarp()
    def init(self, device="cuda:0"):
        self._initialize_resources()
        self.wp, self.device, self.kernel = wp, device, object()

    monkeypatch.setattr(CudaBackend, "__init__", init)
    monkeypatch.setattr(async_backend, "get_arbiter", lambda _: DeviceArbiter())
    monkeypatch.setattr(async_backend, "_nonblocking_stream", lambda w, d: Stream(w, "render"))
    backend = async_backend.AsyncCudaBackend() if asynchronous else CudaBackend()
    accepted = scene()
    backend._set_scene(accepted)
    candidate = replace(accepted, materials=(Material(color=(1, 0, 0)),))
    owners, observations = [], []
    allocate = wp.array

    def fail_second(*args, **kwargs):
        if owners:
            raise RuntimeError("injected shading allocation")
        result = allocate(*args, **kwargs)
        owners.append(weakref.ref(result))
        return result

    def fail_drain(_):
        gc.collect()
        observations.append(owners[0]() is not None)
        raise RuntimeError("injected unavailable retirement")

    method = "synchronize_stream" if asynchronous else "synchronize_device"
    monkeypatch.setattr(wp, method, fail_drain, raising=False)
    monkeypatch.setattr(wp, "array", fail_second)
    with pytest.raises(RuntimeError, match="injected"):
        backend.render(candidate, CameraSnapshot(np.eye(4)), 2, 2)
    assert backend.scene_digest == accepted.digest
    # Drop all traceback frames: only explicit transaction ownership is valid.
    error = backend._job.failure if asynchronous else backend._uncertain[backend][0]
    error.__traceback__ = None
    gc.collect()
    assert observations == [True] and owners[0]() is not None
    monkeypatch.setattr(wp, method, lambda _: None)
    backend.close()
    gc.collect()
    assert owners[0]() is None


def test_flattened_tie_parity_and_geometry_fault_controls(backend, monkeypatch):
    from contextlib import nullcontext
    from test_fspt_async_backend import CheckedWarp, Stream
    from fluidSim_warp_blender51.fspt import async_backend, contracts
    from fluidSim_warp_blender51.fspt.kernels import trace_paths
    from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter

    verifier = load_verifier()
    wp = backend.wp
    backend.kernel = trace_paths
    kernel = verifier.oracle_kernel(wp)
    assert len(verifier.verify_ties(backend, contracts, kernel)) == 5
    queues = CheckedWarp()

    class CpuQueues:
        def __getattr__(self, attr):
            return getattr(wp, attr)

        Stream = staticmethod(queues.Stream)
        ScopedStream = staticmethod(lambda *a, **kw: nullcontext())
        synchronize_event = staticmethod(queues.synchronize_event)
        synchronize_stream = staticmethod(queues.synchronize_stream)

        @staticmethod
        def empty(n, *, dtype, device, pinned=False):
            return wp.empty(n, dtype=dtype, device="cpu")

        @staticmethod
        def launch(*args, stream=None, **kwargs):
            return wp.launch(*args, **kwargs)

        @staticmethod
        def copy(*args, stream=None, **kwargs):
            return wp.copy(*args, **kwargs)

    def init(self, device):
        self._initialize_resources()
        self.wp = CpuQueues() if isinstance(self, async_backend.AsyncCudaBackend) else wp
        self.device, self.kernel = "cpu", trace_paths

    monkeypatch.setattr(CudaBackend, "__init__", init)
    monkeypatch.setattr(async_backend, "get_arbiter", lambda _: DeviceArbiter())
    monkeypatch.setattr(async_backend, "_nonblocking_stream", lambda w, d: Stream(queues, "render"))
    original, camera = next(verifier.tie_scenes(contracts))[1:]
    faces = original.triangles.copy()
    faces[0] = faces[1]
    report = {}
    verifier.verify_geometry_faults(lambda _: async_backend, kernel, "cpu", original,
                                    replace(original, triangles=faces), camera, 32, 24, report)
    assert len(report["geometry_write_failure_recovery"]) == 4
    verifier.verify_reused_reader_retirement(lambda _: async_backend, original, camera, 32, 24,
                                            "cpu", report)
    assert len(report["reused_reader_retirement"]) == 6


def test_vertex_only_span_moves_global_indices_and_regrowth_respects_active_ranges(backend):
    # A zero-face contributor still shifts the flattened indices of later meshes.
    verts = [[5, 5, 5], [-1, -1, 0], [1, -1, 0], [0, 1, 0]]
    prefix = AssetSpan((1,), 1, 0)
    active = AssetSpan((2,), 3, 1)
    first = SceneSnapshot(verts, [[1, 2, 3]], [0], (Material(),), (0, 0, 0), 1,
                          asset_spans=(prefix, active))
    backend._set_scene(first)
    points, indices = backend.arrays["position_storage"], backend.arrays["index_storage"]
    dropped = replace(first, vertices=verts[1:], triangles=[[0, 1, 2]], asset_spans=(active,))
    backend._set_scene(dropped)
    assert backend.arrays["position_storage"] is points and backend.arrays["index_storage"] is indices
    np.testing.assert_array_equal(backend.mesh.points.numpy(), dropped.vertices)
    np.testing.assert_array_equal(backend.mesh.indices.numpy(), [0, 1, 2])
    backend._set_scene(first)
    np.testing.assert_array_equal(backend.mesh.points.numpy(), first.vertices)
    np.testing.assert_array_equal(backend.mesh.indices.numpy(), [1, 2, 3])
    more = replace(first, vertices=[[6, 6, 6], *verts], triangles=[[2, 3, 4]],
                   asset_spans=(AssetSpan((1,), 2, 0), active))
    backend._set_scene(more)
    assert backend.arrays["position_storage"] is not points and backend.arrays["index_storage"] is indices
    np.testing.assert_array_equal(backend.mesh.points.numpy(), more.vertices)
    np.testing.assert_array_equal(backend.mesh.indices.numpy(), more.triangles.ravel())
