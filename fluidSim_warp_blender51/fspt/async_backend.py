"""One admitted CUDA tracing quantum and one pinned-host transfer per backend.

Blender supplies immutable host snapshots at a serialized capture boundary.
Preparing scene assets still uses the reference uploader and Warp software BVH.
The dedicated Warp preparation stream owns their allocation, initialization and
build. A CUDA nonblocking stream waits for those assets, traces and copies to
pinned memory. No render or presentation work is placed on the solver stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import RLock

import numpy as np

from ..runtime.cuda_driver import check_cuda, cuda_driver
from .backend import CudaBackend
from .contracts import AccumulationState, RenderFrame, frame_key
from .scheduler import get_arbiter

BACKEND_ID = "warp-cuda-async-v1"
TRANSPORT_ID = "host-pinned-rgba32f"


class CheckedSubmissionDiscarded(RuntimeError):
    """A failed job was discarded only after checked owned-work retirement."""


@dataclass
class _StreamRetirement:
    result: int | None = None
    attempts: int = 0


# Last-resort strong ownership when Python finalization itself encounters a
# native failure. Explicit retire() retries remove the matching owner only.
_UNRETIRED_STREAMS = set()


class _StreamCreationError(RuntimeError):
    def __init__(self, stream):
        super().__init__('FSPT stream construction cleanup requires checked retirement')
        self.stream = stream


def _nonblocking_stream(wp, device):
    # Warp 1.19's own Stream(device) uses CU_STREAM_DEFAULT. Let its wrapper
    # borrow a nonblocking stream and unregister before native destruction.
    driver = cuda_driver()

    class OwnedStream(wp.Stream):
        def retire(self):
            """Destroy the exact owned handle once, retaining failed attempts."""
            with self._retirement_lock:
                if self._native_handle is None:
                    return
                try:
                    if self.cuda_stream and not self._unregistered:
                        # Warp 1.19's borrowed Stream.__del__ suppresses some
                        # failures and does not clear its handle. Call its
                        # primary unregister operation explicitly and once.
                        device.runtime.core.wp_cuda_stream_unregister(device.context, self.cuda_stream)
                        self._unregistered = True
                    self.retirement.attempts += 1
                    self.retirement.result = driver.destroy(wp, device, self._native_handle)
                    check_cuda(self.retirement.result, 'cuStreamDestroy_v2')
                except BaseException:
                    _UNRETIRED_STREAMS.add(self)
                    raise
                self._native_handle = self.cuda_stream = None
                _UNRETIRED_STREAMS.discard(self)

        def _require_open(self):
            if self._native_handle is None or self._unregistered:
                raise RuntimeError('FSPT CUDA stream is retiring or retired')

        def record_checked(self, stream):
            self._require_open()
            event = wp.Event(device)
            driver.record(wp, device, stream, event)
            return event

        def wait_checked(self, event):
            self._require_open()
            driver.wait(wp, device, self, event)

        def completed(self, event):
            self._require_open()
            return driver.completed(wp, device, event)

        def __del__(self):
            try:
                if hasattr(self, '_native_handle'):
                    self.retire()
            except BaseException:
                # Retain the raw handle and wrapper if destructor fallback
                # cannot prove retirement; never let Warp destroy it again.
                _UNRETIRED_STREAMS.add(self)

    with wp.ScopedDevice(device):
        # Create the Python owner before allocating its native handle, so a
        # failed wrapper allocation cannot orphan an already-created stream.
        stream = OwnedStream.__new__(OwnedStream)
        stream._native_handle = None
        stream._unregistered = False
        stream._retirement_lock = RLock()
        stream.retirement = _StreamRetirement()
        # __new__ on Warp initializes this; explicit initialization also keeps
        # constructor-failure test doubles and partial wrappers well-defined.
        stream.cuda_stream = None
        try:
            stream._native_handle = driver.create_nonblocking(wp, device)
            OwnedStream.__init__(stream, device, cuda_stream=stream._native_handle)
            if stream.is_blocking:
                raise RuntimeError("FSPT CUDA driver stream is blocking")
        except BaseException:
            try:
                stream.retire()
            except BaseException as exc:
                raise _StreamCreationError(stream) from exc
            raise
    return stream


@dataclass(eq=False, slots=True)
class _Owner:
    # If CUDA cannot prove retirement, the arbiter's active owner keeps every
    # allocation alive even if its Blender view is destroyed.
    quarantined: object = None


@dataclass
class _Job:
    lease: object
    scene: object
    camera: object
    key: tuple
    samples: int
    prepared_scene: bool = False
    producer_ready: object = None
    rendered: object = None
    copied: object = None
    failure: BaseException | None = None


class AsyncCudaBackend(CudaBackend):
    """A private accumulation history sharing device-wide fair admission."""

    def __init__(self, device="cuda:0"):
        super().__init__(device)
        self.arbiter = get_arbiter(str(self.device))
        self.owner = _Owner()
        self._state_lock = RLock()
        self._job = None
        self._prepare_stream = None
        self._render_stream = None
        self._pinned = None
        self._capacity = 0
        self._closing = False

    @property
    def pending(self):
        return self._job is not None

    def _ensure_storage(self, count):
        if count <= self._capacity:
            return
        wp = self.wp
        accumulation = self._retain(wp.empty(count, dtype=wp.vec3, device=self.device))
        output = self._retain(wp.empty(count, dtype=wp.vec4, device=self.device))
        pinned = self._retain(wp.empty(count, dtype=wp.vec4, device="cpu", pinned=True))
        self.accumulation, self.output, self._pinned = accumulation, output, pinned
        self._capacity = count
        self.statistics["image_allocations"] += 1

    def submit(self, scene, camera, width, height, *, samples=1, max_bounces=6, seed=0, final=False):
        """Submit one quantum. A denied caller retains only its scheduling intent."""
        if not self._state_lock.acquire(blocking=False):
            return False
        try:
            return self._submit(scene, camera, width, height, samples, max_bounces, seed, final)
        finally:
            self._state_lock.release()

    def _submit(self, scene, camera, width, height, samples, max_bounces, seed, final):
        if self.closed or self._closing:
            raise RuntimeError("FSPT backend is closed")
        if self.pending:
            return False
        width, height, samples = index(width), index(height), index(samples)
        max_bounces, seed = index(max_bounces), index(seed)
        if not 1 <= samples <= 4096:
            raise ValueError("samples per dispatch must be between 1 and 4096")
        key = frame_key(scene, camera, width, height, max_bounces, seed)
        count = width * height
        if count > 16_777_216:
            raise ValueError("FSPT image exceeds its explicit 16M-pixel allocation limit")

        lease = self.arbiter.try_acquire(self.owner, final=final)
        if lease is None:
            return False
        self._job = job = _Job(lease, scene, camera, key, samples)
        wp = self.wp
        try:
            if self._prepare_stream is None:
                self._prepare_stream = wp.Stream(self.device)
            if self._render_stream is None:
                self._render_stream = _nonblocking_stream(wp, self.device)
            # Warp's allocator and BVH implementation rely on blocking-stream
            # lifetime semantics for internal temporaries. Use a *private*
            # blocking producer stream for this work and pass its ready event
            # to the nonblocking tracer. Do not import the solver's queue.
            with wp.ScopedStream(self._prepare_stream, sync_enter=False, sync_exit=False):
                if self.scene_digest != scene.digest:
                    self._set_scene(scene)
                    job.prepared_scene = True
                self._ensure_storage(count)
                self._retain_dispatch(self._pinned)
                if self.history.begin(key):
                    self.accumulation[:count].zero_()
                job.producer_ready = self._render_stream.record_checked(self._prepare_stream)
            stream = self._render_stream
            stream.wait_checked(job.producer_ready)
            arrays = self.arrays
            inverse = wp.mat44(*camera.inverse_view_projection.ravel().tolist())
            wp.launch(self.kernel, dim=count, inputs=[
                self.mesh.id if self.mesh else wp.uint64(0),
                scene.triangle_count if hasattr(scene, "triangle_count") else len(scene.triangles),
                arrays["materials"], arrays["kinds"], arrays["colors"], arrays["emissions"], arrays["iors"],
                wp.vec3(*scene.world), inverse, width, height, self.history.samples,
                samples, max_bounces, seed, self.accumulation, self.output,
            ], device=self.device, stream=stream)
            self.statistics["dispatches"] += 1
            job.rendered = stream.record_checked(stream)
            wp.copy(self._pinned, self.output, count=count, stream=stream)
            self._queue_geometry_validation(stream)
            job.copied = stream.record_checked(stream)
        except BaseException as exc:
            if isinstance(exc, _StreamCreationError):
                self._render_stream = exc.stream
                self._closing = True
            self._abort(exc)
            raise
        return True

    def collect(self, *, wait=False):
        """Return a completed, independently owned image or None while pending."""
        if not self._state_lock.acquire(blocking=wait):
            return None
        try:
            return self._collect(wait)
        finally:
            self._state_lock.release()

    def _collect(self, wait):
        job = self._job
        if job is None:
            return None
        if job.failure is not None:
            self._abort(job.failure)
            raise CheckedSubmissionDiscarded("FSPT discarded a failed submission")
        try:
            if wait:
                self.wp.synchronize_event(job.copied)
            elif not self._render_stream.completed(job.copied):
                return None
            self._validate_completed_geometry()
            _, _, width, height, _, _ = job.key
            pixels = self._pinned.numpy()[:width * height].reshape(height, width, 4).copy()
            if not np.isfinite(pixels).all():
                raise RuntimeError("FSPT CUDA output contains nonfinite values")
            self.statistics["readbacks"] += 1
            self.history.advance(job.samples)
            scene = job.scene
            frame = RenderFrame(
                pixels, job.key, self.history.samples, str(self.device), BACKEND_ID, TRANSPORT_ID,
                scene.frame, scene.publications, scene.geometry_digest, tuple(self.statistics.items()))
        except BaseException as exc:
            self._abort(exc)
            raise
        self._retire()
        return frame

    def _queue_geometry_validation(self, stream):
        """Optional bounded device-geometry status transfer, ordered before copied."""

    def _validate_completed_geometry(self):
        """An optimized geometry consumer may reject completed invalid inputs."""

    def _retire(self):
        job = self._job
        self._pending_resources.clear()
        self.arbiter.release(job.lease)
        job.failure = None
        self.owner.quarantined = None
        self._job = None

    def _drain(self):
        if self._prepare_stream is not None:
            self.wp.synchronize_stream(self._prepare_stream)
        if self._render_stream is not None:
            self.wp.synchronize_stream(self._render_stream)

    def _abort(self, cause):
        # Keep the exact lease, allocations and any partial-build locals if
        # draining itself fails. A later close can safely retry retirement.
        if self._job.failure is None:
            self._job.failure = cause
        self.history = AccumulationState()
        try:
            self._drain()
        except BaseException:
            self.owner.quarantined = self
            raise
        if self._job.prepared_scene:
            # Queued uploads/builds may have failed after Warp returned.
            self._forget_scene()
        self._retire()

    def render(self, scene, camera, width, height, *, samples=1, max_bounces=6, seed=0):
        """Explicit synchronous final wrapper; never wait on another owner."""
        if not self._state_lock.acquire(blocking=False):
            raise RuntimeError("FSPT CUDA device is busy")
        try:
            if not self._submit(scene, camera, width, height, samples, max_bounces, seed, True):
                raise RuntimeError("FSPT CUDA device is busy")
            return self._collect(True)
        finally:
            self.arbiter.withdraw_final(self.owner)
            self._state_lock.release()

    def close(self):
        with self._state_lock:
            if self.closed:
                return
            self._closing = True
            self.arbiter.cancel(self.owner)
            if self.pending:
                self._abort(RuntimeError("FSPT submission cancelled by close"))
            stream = self._render_stream
            if stream is not None:
                retire = getattr(stream, 'retire', None)
                if retire is not None:
                    retire()
                else:
                    retirement = getattr(stream, 'retirement', None)
                    if retirement is not None:
                        if retirement.result is None:
                            raise RuntimeError('FSPT CUDA stream retirement has no checked receipt')
                        check_cuda(retirement.result, 'cuStreamDestroy_v2')
            self._forget_scene()
            self.accumulation = self.output = self._pinned = None
            self._prepare_stream = self._render_stream = None
            self._capacity = 0
            self.history = AccumulationState()
            self.closed = True

    def __del__(self):
        if hasattr(self, "_state_lock") and not self.closed:
            self.close()
