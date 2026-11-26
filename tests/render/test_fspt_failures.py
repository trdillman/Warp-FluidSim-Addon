"""CUDA boundary faults must not leave owned resources or reusable partial images."""
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
