"""Cursor-free submit/poll/retire controls at the external CUDA boundaries."""
from contextlib import contextmanager
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
from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter


class Event:
    def __init__(self, warp):
        self.warp = warp
        self.done = False

    @property
    def is_complete(self):
        self.warp.op("poll")
        return self.done


class Stream:
    def __init__(self, warp, name):
        self.warp, self.name = warp, name
        self.is_blocking = name == "prepare"

    def record_event(self):
        self.warp.op(("record", self.name))
        return Event(self.warp)

    def record_checked(self, stream):
        return stream.record_event()

    def wait_checked(self, event):
        self.warp.op(("wait", self.name))

    def completed(self, event):
        return event.is_complete


class Buffer:
    def __init__(self, wp, values, device):
        self.wp, self.values, self.device = wp, values, device
        self.size = len(values)

    def __getitem__(self, key):
        return Buffer(self.wp, self.values[key], self.device)

    def zero_(self):
        self.wp.op("zero")
        self.values.fill(0)

    def numpy(self):
        assert self.device == "cpu", "unexpected implicit device readback"
        self.wp.op("readback")
        return self.values


class CheckedWarp:
    def __init__(self):
        self.ops = []
        self.failure = None
        self.current = None
        self.vec3 = lambda *v: tuple(v)
        self.vec4 = lambda *v: tuple(v)
        self.mat44 = lambda *v: v
        self.uint64 = int

    def op(self, label):
        self.ops.append(label)
        if self.failure == label:
            raise RuntimeError(f"injected {label}")

    def array(self, values, *, dtype, device):
        self.op("upload")
        return Buffer(self, np.array(values, copy=True), device)

    def empty(self, n, *, dtype, device, pinned=False):
        assert device != "cpu" or pinned
        self.op("allocate")
        return Buffer(self, np.full((n, 3 if dtype == self.vec3 else 4), 71, np.float32), device)

    def Mesh(self, **kwargs):
        self.op("mesh")
        return SimpleNamespace(id=3)

    def Stream(self, device):
        self.op("prepare_stream")
        return Stream(self, "prepare")

    @contextmanager
    def ScopedStream(self, stream, *, sync_enter, sync_exit):
        assert not sync_enter and not sync_exit
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    def launch(self, kernel, *, dim, inputs, device, stream):
        assert not stream.is_blocking
        self.op("launch")
        first, samples = inputs[11:13]
        acc, out = inputs[-2:]
        acc.values[:dim] += samples
        out.values[:dim, :3] = acc.values[:dim] / (first + samples)
        out.values[:dim, 3] = 1

    def copy(self, dest, src, *, count, stream):
        assert not stream.is_blocking and dest.device == "cpu"
        self.op("copy")
        dest.values[:count] = src.values[:count]

    def synchronize_stream(self, stream):
        self.op(("drain", stream.name))

    def synchronize_event(self, event):
        self.op("wait_copy")
        event.done = True


@pytest.fixture
def backend_setup(monkeypatch):
    from fluidSim_warp_blender51.fspt import async_backend as module

    arbiter = DeviceArbiter()
    wp = CheckedWarp()

    def init(self, device):
        self.wp, self.device, self.kernel = wp, device, object()
        self.history = AccumulationState()
        self.scene_digest = self.geometry_digest = self.shading_digest = None
        self.mesh = self.arrays = self.accumulation = self.output = None
        self.closed = False
        self.statistics = dict.fromkeys(("geometry_uploads", "mesh_builds", "shading_uploads",
                                         "image_allocations", "dispatches", "readbacks"), 0)

    monkeypatch.setattr(CudaBackend, "__init__", init)
    monkeypatch.setattr(module, "get_arbiter", lambda _: arbiter)
    monkeypatch.setattr(module, "_nonblocking_stream", lambda wp, device: Stream(wp, "render"))
    scene = SceneSnapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0],
                          (Material(),), (0.3, 0.4, 0.1), 8)
    camera = CameraSnapshot(np.eye(4))
    return module.AsyncCudaBackend("cuda:0"), arbiter, wp, scene, camera


def test_busy_submissions_do_no_cuda_work_and_pending_frames_keep_submitted_state(backend_setup):
    backend, arbiter, wp, scene, camera = backend_setup
    rival = object()
    lease = arbiter.try_acquire(rival)
    assert not backend.submit(scene, camera, 3, 2)
    assert not wp.ops
    arbiter.release(lease)
    assert backend.submit(scene, camera, 3, 2, samples=4)
    order = wp.ops.copy()
    assert order.index(("record", "prepare")) < order.index(("wait", "render")) < order.index("launch")
    assert order.index("launch") < order.index(("record", "render")) < order.index("copy")
    assert order[-1] == ("record", "render")
    assert backend.pending and not backend.submit(replace(scene, frame=19), camera, 3, 2)
    assert wp.ops == order
    assert backend.collect() is None and "readback" not in wp.ops
    assert arbiter.try_acquire(rival) is None
    frame = backend.collect(wait=True)
    assert frame.scene_frame == 8 and frame.samples == 4
    assert frame.geometry_digest == scene.geometry_digest and frame.pixels.shape == (2, 3, 4)
    assert np.all(frame.pixels == 1) and not backend.pending
    assert arbiter.try_acquire(rival) is not None
    backend.close()


def test_resize_camera_edit_and_repeated_history_reuse_capacity_with_owned_pixels(backend_setup):
    backend, _arbiter, wp, scene, camera = backend_setup
    assert backend.submit(scene, camera, 4, 3, samples=3)
    first = backend.collect(wait=True)
    first.pixels[:] = 0.42
    assert backend.submit(scene, camera, 4, 3)
    second = backend.collect(wait=True)
    assert second.samples == 4 and np.all(second.pixels == 1)
    assert backend.submit(scene, camera, 2, 3)
    third = backend.collect(wait=True)
    assert third.samples == 1 and np.all(third.pixels == 1)
    assert np.all(first.pixels == np.float32(0.42))
    assert dict(third.statistics)["image_allocations"] == 1
    assert dict(third.statistics)["geometry_uploads"] == 1
    assert wp.ops.count("zero") == 2
    backend.close()


@pytest.mark.parametrize("failure", ["upload", "launch", "copy", "poll", "readback"])
def test_fault_discards_partial_history_drains_owned_work_and_releases_lease(backend_setup, failure):
    backend, _arbiter, wp, scene, camera = backend_setup
    assert backend.submit(scene, camera, 2, 2, samples=3)
    backend.collect(wait=True)
    changed = replace(scene, triangles=[[0, 2, 1]]) if failure == "upload" else scene
    wp.failure = failure
    with pytest.raises(RuntimeError, match="injected"):
        if backend.submit(changed, camera, 2, 2):
            backend.collect(wait=failure == "readback")
    assert any(isinstance(op, tuple) and op[0] == "drain" for op in wp.ops)
    assert not backend.pending and backend.history.key is None
    assert backend.history.samples == 0
    wp.failure = None
    assert backend.submit(changed, camera, 2, 2)
    assert backend.collect(wait=True).samples == 1
    backend.close()


def test_close_drains_pending_and_cancels_queued_demand(backend_setup):
    backend, arbiter, wp, scene, camera = backend_setup
    assert backend.submit(scene, camera, 2, 2)
    assert backend.pending
    backend.close()
    assert not backend.pending and backend.closed
    assert ("drain", "render") in wp.ops and "readback" not in wp.ops
    lease = arbiter.try_acquire(object())
    assert lease is not None
    ops = wp.ops.copy()
    backend.close()
    assert wp.ops == ops


def test_control_callbacks_cannot_collect_a_half_submitted_job(backend_setup):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event as ThreadEvent

    backend, _, wp, scene, camera = backend_setup
    entered, resume = ThreadEvent(), ThreadEvent()
    launch = wp.launch

    def delayed(*args, **kwargs):
        entered.set()
        assert resume.wait(2)
        launch(*args, **kwargs)

    wp.launch = delayed
    with ThreadPoolExecutor(max_workers=1) as executor:
        submitted = executor.submit(backend.submit, scene, camera, 2, 2)
        try:
            assert entered.wait(2)
            assert backend.collect() is None
            assert not backend.submit(scene, camera, 2, 2)
        finally:
            resume.set()
        assert submitted.result(timeout=2)
    assert backend.collect(wait=True).samples == 1
    backend.close()


def test_abandoned_pending_backend_drains_before_owner_resources_die(backend_setup):
    import gc
    import weakref

    backend, _arbiter, wp, scene, camera = backend_setup
    # Keep the fixture's backend and create a disposable view with the same arbiter.
    other = type(backend)("cuda:0")
    assert other.submit(scene, camera, 2, 2)
    gone = weakref.ref(other)
    del other
    gc.collect()
    assert gone() is None and ("drain", "render") in wp.ops
    assert backend.submit(scene, camera, 2, 2)
    backend.close()


def test_uncertain_drain_holds_exact_lease_until_close_can_prove_retirement(backend_setup):
    backend, arbiter, wp, scene, camera = backend_setup
    assert backend.submit(scene, camera, 2, 2)
    sync = wp.synchronize_stream

    def failed(stream):
        raise RuntimeError("injected drain")

    wp.synchronize_stream = failed
    wp.failure = "poll"
    with pytest.raises(RuntimeError, match="drain"):
        backend.collect()
    assert backend.pending and arbiter.try_acquire(object()) is None
    wp.failure = None
    wp.synchronize_stream = sync
    backend.close()
    assert not backend.pending and backend.closed


def test_closing_denied_view_cancels_its_fifo_position(backend_setup):
    backend, arbiter, _, scene, camera = backend_setup
    holder = arbiter.try_acquire(object())
    assert not backend.submit(scene, camera, 2, 2)
    backend.close()
    arbiter.release(holder)
    assert arbiter.try_acquire(object()) is not None


def test_borrowed_cuda_handle_is_checked_and_unregistered_before_native_destroy(monkeypatch):
    import ctypes
    from contextlib import nullcontext
    from unittest.mock import Mock

    from fluidSim_warp_blender51.fspt import async_backend as module

    calls = []
    device = SimpleNamespace(context=11, runtime=SimpleNamespace(core=SimpleNamespace(
        wp_cuda_stream_unregister=lambda context, handle: calls.append(('unregister', handle)))))

    class ExternalStream:
        is_blocking = False

        def __init__(self, device, cuda_stream):
            self.cuda_stream = cuda_stream
            calls.append(("register", cuda_stream))

        def __del__(self):
            calls.append(("unregister", self.cuda_stream))

    def create(out, flags):
        assert flags == 1
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 9876
        calls.append(("create", flags))
        return 0

    driver = SimpleNamespace(
        cuStreamCreate=Mock(side_effect=create),
        cuStreamDestroy_v2=Mock(side_effect=lambda handle: calls.append(("destroy", handle)) or 0),
        cuEventRecord=Mock(return_value=0), cuStreamWaitEvent=Mock(return_value=0),
        cuEventQuery=Mock(return_value=600))
    monkeypatch.setattr(ctypes, "CDLL", lambda _: driver)
    monkeypatch.setattr(ctypes, "WinDLL", lambda _: driver, raising=False)
    from fluidSim_warp_blender51.runtime.cuda_driver import CudaDriver
    monkeypatch.setattr(module, "cuda_driver", CudaDriver)
    wp = SimpleNamespace(Stream=ExternalStream, ScopedDevice=lambda _: nullcontext(),
                         Event=lambda _: SimpleNamespace(cuda_event=23, is_complete=False))
    stream = module._nonblocking_stream(wp, device)
    event = stream.record_checked(stream)
    stream.wait_checked(event)
    assert not stream.completed(event)
    driver.cuEventQuery.return_value = 700
    with pytest.raises(RuntimeError, match="cuEventQuery.*700"):
        stream.completed(event)
    driver.cuEventRecord.return_value = 201
    with pytest.raises(RuntimeError, match="cuEventRecord.*201"):
        stream.record_checked(stream)
    receipt = stream.retirement
    del stream
    assert calls == [("create", 1), ("register", 9876), ("unregister", 9876), ("destroy", 9876)]
    assert receipt.result == 0


def test_real_warp_cpu_kernel_and_image_reset_match_reference(backend_setup):
    """Use the real tracer/meshes/arrays on CPU; only CUDA queue operations are doubled."""
    from contextlib import nullcontext
    real_wp = pytest.importorskip("warp")
    from fluidSim_warp_blender51.fspt.kernels import trace_paths

    backend, _, queues, _scene, camera = backend_setup

    class CpuWarp:
        def __getattr__(self, name):
            return getattr(real_wp, name)

        Stream = staticmethod(queues.Stream)
        op = staticmethod(queues.op)
        ScopedStream = staticmethod(lambda *a, **kw: nullcontext())
        synchronize_event = staticmethod(queues.synchronize_event)
        synchronize_stream = staticmethod(queues.synchronize_stream)

        @staticmethod
        def empty(n, *, dtype, device, pinned=False):
            return real_wp.empty(n, dtype=dtype, device="cpu")

        @staticmethod
        def launch(kernel, *, dim, inputs, device, stream):
            real_wp.launch(kernel, dim=dim, inputs=inputs, device="cpu")

        @staticmethod
        def copy(dest, src, *, count, stream):
            real_wp.copy(dest, src, count=count)

    backend.wp, backend.device, backend.kernel = CpuWarp(), "cpu", trace_paths
    reference = CudaBackend("cuda:0")  # fixture replaces only discovery here
    reference.wp, reference.device, reference.kernel = real_wp, "cpu", trace_paths
    triangle = SceneSnapshot([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], [0],
                             (Material(kind="emission", emission=(1.5, 0.3, 0.1)),),
                             (0.2, 0.5, 0.75), 8)
    revised = replace(triangle, world=(0.9, 0.2, 0.1), frame=9)
    try:
        previous = None
        for scene, w, h, spp in ((triangle, 12, 8, 3), (triangle, 12, 8, 2),
                                 (revised, 6, 4, 1)):
            expected = reference.render(scene, camera, w, h, samples=spp, seed=7)
            assert backend.submit(scene, camera, w, h, samples=spp, seed=7)
            result = backend.collect(wait=True)
            np.testing.assert_array_equal(result.pixels, expected.pixels)
            assert result.key == expected.key and result.samples == expected.samples
            assert np.std(result.pixels[..., :3]) > 0.15
            if previous is not None:
                assert not np.shares_memory(previous, result.pixels)
            previous = result.pixels
        assert backend.statistics["image_allocations"] == 1
        assert backend.statistics["mesh_builds"] == 1
    finally:
        backend.close()
        reference.close()


def test_busy_one_shot_final_withdraws_its_abandoned_priority(backend_setup):
    backend, arbiter, _, scene, camera = backend_setup
    active = arbiter.try_acquire(object())
    with pytest.raises(RuntimeError, match="busy"):
        backend.render(scene, camera, 2, 2)
    arbiter.release(active)
    assert arbiter.try_acquire(object()) is not None, "one-shot final starved later viewport demand"
    backend.close()


def test_failed_drain_retry_preserves_original_partial_preparation_owners(backend_setup):
    import gc
    import weakref

    backend, _, wp, scene, camera = backend_setup
    failure = []

    class Partial:
        pass

    def fail_staging(_):
        partial = Partial()
        failure.append(weakref.ref(partial))
        raise RuntimeError("partial upload")

    backend._set_scene = fail_staging
    sync = wp.synchronize_stream
    wp.synchronize_stream = failing_drain = MockDrainFailure()
    for action in (lambda: backend.submit(scene, camera, 2, 2), backend.close):
        with pytest.raises(RuntimeError, match="unretired"):
            action()
        gc.collect()
        assert failure[0]() is not None
    assert failing_drain.calls == 2
    wp.synchronize_stream = sync
    backend.close()
    gc.collect()
    assert failure[0]() is None


class MockDrainFailure:
    calls = 0

    def __call__(self, _stream):
        self.calls += 1
        raise RuntimeError("unretired")
