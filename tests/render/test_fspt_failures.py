"""CUDA boundary faults must not leave owned resources or reusable partial images."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt.backend import CudaBackend
from fluidSim_warp_blender51.fspt.contracts import (
    AccumulationState,
    CameraSnapshot,
    Material,
    SceneSnapshot,
)


class _HostArray:
    """Only the allocation/readback surface needed by the backend lifecycle."""

    def __init__(self, values, warp, device):
        self.values = np.asarray(values)
        self.warp = warp
        self.device = device

    def numpy(self):
        self.warp.raise_at("readback")
        return self.values.copy()


class _FaultingWarp:
    """No kernel execution or runtime import: faults enter at the external boundary."""

    vec3 = staticmethod(lambda *values: values)
    vec4 = staticmethod(lambda *values: values)
    mat44 = staticmethod(lambda *values: values)
    uint64 = staticmethod(int)

    def __init__(self):
        self.failure = None
        self.error = RuntimeError("injected CUDA boundary failure")

    def raise_at(self, boundary):
        if self.failure == boundary:
            raise self.error

    def array(self, values, *, dtype, device):
        self.raise_at("array")
        return _HostArray(np.array(values, copy=True), self, device)

    def zeros(self, count, *, dtype, device):
        return _HostArray(np.zeros((count, 3), dtype=np.float32), self, device)

    def empty(self, count, *, dtype, device):
        # Finite data lets lifecycle tests complete without imitating the path tracer.
        return _HostArray(np.zeros((count, 4), dtype=np.float32), self, device)

    @staticmethod
    def Mesh(*, points, indices):
        return SimpleNamespace(id=1)

    def launch(self, kernel, *, dim, inputs, device):
        self.raise_at("launch")

    def synchronize_device(self, device):
        self.raise_at("synchronize")


@pytest.fixture
def rendered_backend():
    scene = SceneSnapshot(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        triangles=[[0, 1, 2]],
        material_indices=[0],
        materials=(Material(),),
        world=(0.1, 0.2, 0.3),
        frame=1,
    )
    camera = CameraSnapshot(np.eye(4))
    # Skip only CUDA discovery and kernel import; use the real allocation,
    # scene replacement, image identity, render, and close implementations.
    backend = CudaBackend.__new__(CudaBackend)
    backend.wp = _FaultingWarp()
    backend.device = "cuda:0"
    backend.kernel = object()
    backend.history = AccumulationState()
    backend.scene_digest = None
    backend.geometry_digest = backend.shading_digest = None
    backend.statistics = {"geometry_uploads": 0, "mesh_builds": 0,
                          "shading_uploads": 0, "image_allocations": 0,
                          "dispatches": 0, "readbacks": 0}
    backend.mesh = backend.arrays = backend.accumulation = backend.output = None
    backend.closed = False
    frame = backend.render(scene, camera, 2, 1, samples=5, max_bounces=4, seed=17)
    assert frame.samples == 5
    assert backend.mesh is not None
    return backend, scene, camera


def _assert_no_image_history(backend):
    history = backend.history
    assert history is None or (history.key is None and history.samples == 0)


@pytest.mark.parametrize("sync_failure", [False, True], ids=["success", "sync-error"])
def test_close_releases_ownership_even_when_synchronization_fails(rendered_backend, sync_failure):
    backend, scene, camera = rendered_backend
    if sync_failure:
        backend.wp.failure = "synchronize"
        with pytest.raises(RuntimeError) as caught:
            backend.close()
        assert caught.value is backend.wp.error
    else:
        backend.close()

    assert backend.closed
    assert backend.mesh is None
    assert backend.arrays is None
    assert backend.accumulation is None
    assert backend.output is None
    _assert_no_image_history(backend)

    # Teardown remains idempotent even while the device is still failing.
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.render(scene, camera, 2, 1)


@pytest.mark.parametrize("boundary", ["launch", "synchronize", "readback"])
def test_failed_render_discards_partial_image_before_retry(rendered_backend, boundary):
    backend, scene, camera = rendered_backend
    backend.wp.failure = boundary
    with pytest.raises(RuntimeError) as caught:
        backend.render(scene, camera, 2, 1, samples=3, max_bounces=4, seed=17)
    assert caught.value is backend.wp.error

    _assert_no_image_history(backend)
    assert backend.accumulation is None
    assert backend.output is None

    backend.wp.failure = None
    retry = backend.render(scene, camera, 2, 1, samples=2, max_bounces=4, seed=17)
    assert retry.samples == 2, "a failed image must not contribute prior or partial samples"
    assert retry.pixels.shape == (1, 2, 4)


def test_shading_world_and_timeline_changes_reuse_geometry(rendered_backend):
    backend, scene, camera = rendered_backend
    mesh = backend.mesh
    points = backend.arrays["points"]
    indices = backend.arrays["indices"]
    variants = [replace(scene, materials=(Material(color=(1, 0, 0)),)),
                replace(scene, world=(1, 1, 1)), replace(scene, frame=7)]
    for changed in variants:
        backend.render(changed, camera, 2, 1)
        assert backend.mesh is mesh, "radiance-only edits must not rebuild the BVH"
        assert backend.arrays["points"] is points
        assert backend.arrays["indices"] is indices


def test_equal_count_changed_connectivity_replaces_geometry(rendered_backend):
    backend, scene, camera = rendered_backend
    mesh = backend.mesh
    changed = replace(scene, triangles=[[0, 2, 1]])
    backend.render(changed, camera, 2, 1)
    assert backend.mesh is not mesh
    np.testing.assert_array_equal(backend.arrays["indices"].values, [0, 2, 1])


def test_empty_surface_removes_previous_mesh(rendered_backend):
    backend, scene, camera = rendered_backend
    empty = replace(scene, vertices=[], triangles=[], material_indices=[])
    backend.render(empty, camera, 2, 1)
    assert backend.mesh is None


def test_failed_shading_replacement_preserves_previous_scene(rendered_backend):
    backend, scene, camera = rendered_backend
    mesh, arrays, digest = backend.mesh, backend.arrays, backend.scene_digest
    backend.wp.failure = "array"
    with pytest.raises(RuntimeError, match="injected"):
        backend.render(replace(scene, materials=(Material(color=(1, 0, 0)),)), camera, 2, 1)
    assert backend.mesh is mesh
    assert backend.arrays is arrays
    assert backend.scene_digest == digest


def test_completed_frame_uses_submitted_timeline(rendered_backend):
    backend, scene, camera = rendered_backend
    frame = backend.render(replace(scene, frame=17), camera, 2, 1)
    assert getattr(frame, "scene_frame", None) == 17
