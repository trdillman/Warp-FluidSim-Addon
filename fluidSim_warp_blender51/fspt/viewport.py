"""Bounded viewport state; Blender adapters own capture and draw callbacks.

This module retains owned immutable snapshots and small newest-view requests.
It never retains a depsgraph, accesses bpy, or creates display resources.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from operator import index
from threading import RLock, current_thread, main_thread
from weakref import WeakSet

from .contracts import MAX_BOUNCES, frame_key
from .device_scene import DeviceSceneUnavailable


@dataclass(frozen=True)
class ViewRequest:
    camera: object
    width: int
    height: int
    target_samples: int
    max_bounces: int
    seed: int
    device: str

    def __post_init__(self):
        for name in ("width", "height", "target_samples", "max_bounces", "seed"):
            object.__setattr__(self, name, index(getattr(self, name)))
        if not 1 <= self.width <= 16384 or not 1 <= self.height <= 16384:
            raise ValueError("invalid viewport dimensions")
        if self.width * self.height > 16_777_216:
            raise ValueError("FSPT viewport exceeds the explicit 16M-pixel limit")
        if not 1 <= self.target_samples <= 4096 or not 1 <= self.max_bounces <= MAX_BOUNCES:
            raise ValueError("invalid viewport sample or bounce limit")
        if not str(self.device).startswith("cuda:"):
            raise ValueError("FSPT viewport requires an explicit CUDA device")

    @property
    def radiance_key(self):
        return (self.camera.digest, self.width, self.height, self.max_bounces, self.seed, self.device)


class ViewportCoordinator:
    """Final jobs suspend viewport admission on their selected device only."""

    def __init__(self):
        self.views = WeakSet()
        self.retirement_failures = set()
        self._final_depth = {}
        self._admission_locks = {}

    def _admission_lock(self, device):
        alias = str(device)
        lock = self._admission_locks.get(alias)
        return self._admission_locks.setdefault(alias, RLock()) if lock is None else lock

    def blocked(self, device):
        return self._final_depth.get(str(device), 0) > 0

    @contextmanager
    def admission(self, device):
        """Never block optional work, and exclude a concurrent final claim."""
        lock = self._admission_lock(device)
        acquired = lock.acquire(blocking=False)
        try:
            yield acquired and not self.blocked(device)
        finally:
            if acquired:
                lock.release()

    @contextmanager
    def final(self, device):
        device = str(device)
        lock = self._admission_lock(device)
        # Existing capture/tick callbacks finish before the final claim. Once
        # blocked is set, no callback can slip a submission between polling
        # and this owned-work drain, including Blender's render worker thread.
        with lock:
            self._final_depth[device] = self._final_depth.get(device, 0) + 1
        try:
            with lock:
                for view in tuple(self.views):
                    if view.device == device:
                        view.suspend()
            yield
        finally:
            with lock:
                depth = self._final_depth[device] - 1
                if depth:
                    self._final_depth[device] = depth
                else:
                    del self._final_depth[device]


class ViewportState:
    """One capture, one pending quantum and one independently owned image.

    A denied capture keeps only the newest request and a dirty bit. The caller
    must retry with a fresh depsgraph callback rather than retain producer IDs.
    """

    def __init__(self, coordinator, backend_factory):
        self.coordinator, self.backend_factory = coordinator, backend_factory
        self.coordinator.views.add(self)
        self.request = self.snapshot = self.presentable = None
        self.presented_snapshot = None
        self.backend = None
        self._capture_owner = object()
        self._submitted_snapshot = self._completed = None
        self._scene_revision = self._submitted_revision = 0
        self._revision = 0
        self.dirty, self.closed = True, False
        self.error = ""
        self.statistics = {"captures": 0, "capture_denials": 0, "surface_invalidations": 0,
                           "scene_invalidations": 0,
                           "submitted": 0, "collected": 0, "discarded": 0}

    @property
    def device(self):
        if self.backend is not None:
            return str(self.backend.device)
        return self.request.device if self.request is not None else None

    @property
    def key(self):
        if self.snapshot is None or self.request is None:
            return None
        r = self.request
        return frame_key(self.snapshot, r.camera, r.width, r.height, r.max_bounces, r.seed)

    @property
    def at_goal(self):
        return (not self.dirty and self.presentable is not None and self.presentable.key == self.key
                and self.presentable.samples >= self.request.target_samples)

    @property
    def presentation_key(self):
        if self.presented_snapshot is None or self.request is None:
            return None
        r = self.request
        return frame_key(self.presented_snapshot, r.camera, r.width, r.height, r.max_bounces, r.seed)

    @property
    def needs_tick(self):
        if self.closed or self.request is None:
            return False
        if self.backend is not None and (self.backend.pending or self.device != self.request.device):
            return True
        if self.error:
            return False
        return self._completed is not None or self.dirty or not self.at_goal

    def request_update(self, request):
        """Update cheap metadata only; safe even from the draw callback."""
        if self.closed:
            return
        old = self.request
        self.request = request
        if old is None or old.radiance_key != request.radiance_key:
            self.presentable = self.presented_snapshot = None
            if old is None or old.device != request.device or self.error:
                self.invalidate()
        if self.at_goal and self.backend is not None:
            # A denied refinement may still be queued when a smaller goal
            # makes this view idle. Withdraw that demand before its FIFO entry
            # can stall every other view; active leases remain owned.
            self.backend.arbiter.cancel(self.backend.owner)

    def invalidate(self, *, surface_only=False):
        if not self.closed:
            self._revision += 1
            self.dirty = True
            self.error = ""
            if self.backend is not None:
                # A previously denied trace must not remain ahead of the
                # recapture this dirty view now needs. Cancellation removes
                # only scheduling intent; any active GPU lease stays owned.
                self.backend.arbiter.cancel(self.backend.owner)
            if surface_only:
                self.statistics["surface_invalidations"] += 1
            else:
                self._scene_revision += 1
                self.statistics["scene_invalidations"] += 1
                self.presentable = self.presented_snapshot = None

    def rebind(self):
        self.invalidate()
        self.snapshot = None
        self._completed = None

    def _ensure_backend(self):
        if self.backend is not None and str(self.backend.device) != self.request.device:
            self.backend.arbiter.cancel(self._capture_owner)
            self.backend.close()
            self.backend = None
            self._submitted_snapshot = self._completed = None
            self.snapshot = None
        if self.backend is None:
            backend = self.backend_factory(self.request.device)
            if str(backend.device) != self.request.device:
                backend.close()
                raise RuntimeError("FSPT backend substituted a different CUDA device")
            self.backend = backend
        return self.backend

    def capture(self, extract):
        """Call extract only after a short lease grants optional host work."""
        if self.closed or self.request is None:
            return False
        device = self.request.device
        with self.coordinator.admission(device) as admitted:
            if not admitted or self.request.device != device:
                return False
            try:
                return self._capture(extract)
            except DeviceSceneUnavailable:
                self.defer()
                return False

    def _capture(self, extract):
        if self.closed or self.request is None or self.coordinator.blocked(self.request.device):
            return False
        if self.backend is not None and self.backend.pending:
            return False
        if not self.dirty and self.snapshot is not None:
            return True
        backend = self._ensure_backend()
        lease = backend.arbiter.try_acquire(self._capture_owner)
        if lease is None:
            self.statistics["capture_denials"] += 1
            return False
        revision = self._revision
        try:
            candidate = extract()
            if revision != self._revision:
                return False
            self.snapshot = candidate
            self.dirty = False
            self.error = ""
            self.statistics["captures"] += 1
            return True
        except DeviceSceneUnavailable:
            raise
        except Exception as exc:
            self.fail(exc)
            raise
        finally:
            backend.arbiter.release(lease)

    def _accept(self, frame, eligible):
        self.statistics["collected"] += 1
        submitted = self._submitted_snapshot
        r = self.request
        expected = (frame_key(submitted, r.camera, r.width, r.height, r.max_bounces, r.seed)
                    if submitted is not None else None)
        if (self._submitted_revision != self._scene_revision or frame.device != r.device
                or frame.key != expected or not eligible(frame.publications)):
            self.statistics["discarded"] += 1
            if frame.key == self.key and not self.dirty:
                self.invalidate()
            return
        self.presentable = frame
        self.presented_snapshot = submitted

    def tick(self, eligible, *, presentation_eligible=None):
        """Poll once and submit at most one sample quantum, without a wait."""
        if self.closed:
            return "closed"
        if self.request is None:
            return "idle"
        device = self.request.device
        with self.coordinator.admission(device) as admitted:
            if not admitted:
                return "suspended"
            if self.request.device != device:
                return "capture"
            try:
                return self._tick(eligible, presentation_eligible or eligible)
            except DeviceSceneUnavailable:
                self.defer()
                return "capture"

    def _tick(self, eligible, presentation_eligible):
        if self.closed:
            return "closed"
        if self.request is None:
            return "idle"
        if self.coordinator.blocked(self.request.device):
            return "suspended"
        if self.backend is not None and self.device != self.request.device:
            # A device edit is a lifecycle boundary, outside drawing. The
            # backend retains failed retirement ownership if this drain fails.
            self._ensure_backend()
            return "capture"
        if self.backend is not None and self.backend.pending:
            frame = self.backend.collect()
            if frame is None:
                return "pending"
            self._accept(frame, presentation_eligible)
            self._submitted_snapshot = None
        if self._completed is not None:
            frame, self._completed = self._completed, None
            self._accept(frame, presentation_eligible)
            self._submitted_snapshot = None
        if self.error:
            return "error"
        if self.dirty or self.snapshot is None:
            return "capture"
        if self.at_goal:
            return "idle"
        if not eligible(self.snapshot.publications):
            self.invalidate()
            return "capture"
        r = self.request
        if self.device != r.device:
            return "capture"
        snapshot, revision = self.snapshot, self._scene_revision
        if self.backend.submit(snapshot, r.camera, r.width, r.height, samples=1,
                               max_bounces=r.max_bounces, seed=r.seed):
            self._submitted_snapshot = snapshot
            self._submitted_revision = revision
            self.statistics["submitted"] += 1
            return "submitted"
        return "admission"

    def suspend(self):
        """Final-render boundary: stop queue intents and drain only owned work."""
        backend, submitted = self.backend, self._submitted_snapshot
        if backend is not None:
            backend.arbiter.cancel(self._capture_owner)
            backend.arbiter.cancel(backend.owner)
            if backend.pending:
                completed = backend.collect(wait=True)
                # A device edit can retire this backend while a final job on
                # its old device is entering. Never redirect the drain to the
                # replacement device or publish an obsolete completion there.
                if self.backend is backend and self._submitted_snapshot is submitted:
                    self._completed = completed

    def defer(self):
        """A retired optional source requests recapture without a fatal error."""
        self.dirty, self.error, self.snapshot = True, "", None
        if self.backend is not None:
            self.backend.arbiter.cancel(self._capture_owner)
            self.backend.arbiter.cancel(self.backend.owner)

    def retire_backend(self):
        """Main-thread Final handoff: retire geometry, retain the live request.

        Close must succeed before any owner reference is cleared. A worker may
        suspend/drain a view, but only the publication thread may retire its
        pooled geometry and authorize a separately allocated Final scene.
        """
        if current_thread() is not main_thread():
            raise RuntimeError("FSPT geometry retirement requires the main thread")
        if self.backend is not None:
            self.backend.arbiter.cancel(self._capture_owner)
            try:
                self.backend.close()
            except BaseException:
                self.coordinator.retirement_failures.add(self)
                raise
            self.backend = None
        self.snapshot = self.presentable = self.presented_snapshot = None
        self._submitted_snapshot = self._completed = None
        self.coordinator.retirement_failures.discard(self)
        if self.closed:
            self.coordinator.views.discard(self)
            self.request = None
        self.invalidate()

    def fail(self, exc):
        self.error = f"{type(exc).__name__}: {exc}"
        self.dirty = False
        self.snapshot = self.presentable = self.presented_snapshot = None
        if self.backend is not None:
            self.backend.arbiter.cancel(self._capture_owner)
            self.backend.arbiter.cancel(self.backend.owner)

    def close(self):
        if self.closed and self.backend is None:
            return
        self.closed = True
        if self.backend is not None:
            self.backend.arbiter.cancel(self._capture_owner)
            try:
                self.backend.close()
            except BaseException:
                self.coordinator.retirement_failures.add(self)
                raise
            self.backend = None
        self.coordinator.views.discard(self)
        self.coordinator.retirement_failures.discard(self)
        self.request = self.snapshot = self.presentable = self.presented_snapshot = None
        self._submitted_snapshot = self._completed = None
