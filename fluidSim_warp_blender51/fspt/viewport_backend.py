"""Inactive device-wide viewport route owner with checked mode transitions.

The runtime owns one broker per CUDA device. Evaluated and GPU route geometry
are never retained together: a switch polls jobs and the shared geometry ledger
before allowing the other backend kind to allocate. Drawing never uses this API.
"""
from __future__ import annotations

from threading import RLock, current_thread

from .contracts import SceneSnapshot
from .device_scene import DeviceSceneRequest, DeviceSceneUnavailable
from .scheduler import get_arbiter


class RouteTransitionPending(DeviceSceneUnavailable):
    """The requested route is waiting for checked retirement; recapture later."""


class ViewportBackendBroker:
    """One publication-thread owner for all optional viewport routes on a device."""

    def __init__(self, device, geometry_budget, *, stop_production,
                 evaluated_factory=None, device_factory=None, mode='GPU'):
        if mode not in {'GPU', 'EVALUATED'}:
            raise ValueError('unknown viewport route')
        if str(device) != str(geometry_budget.pool.device):
            raise ValueError('viewport broker and surface pool devices differ')
        self.device = geometry_budget.pool.device
        self.geometry_budget = geometry_budget
        self.arbiter = get_arbiter(str(self.device))
        self._stop_production = stop_production
        self._evaluated_factory = evaluated_factory
        self._device_factory = device_factory
        self._thread = current_thread()
        self._mode = self._target = mode
        self._views = set()
        self.closed = False

    def _check_thread(self):
        if current_thread() is not self._thread:
            raise RuntimeError('viewport route control requires its publication thread')

    @property
    def ready_mode(self):
        return None if self.closed else self._mode

    @property
    def recovery_required(self):
        return bool(self.geometry_budget.pool.poisoned)

    @property
    def geometry_recovery_ready(self):
        """An explicit runtime pool close is safe only after all readers surrender."""
        return (self._mode is None and
                not any(view._delegate is not None for view in self._views))

    @property
    def recovery_reason(self):
        return ('GPU geometry is quarantined; checked pool close is required'
                if self.recovery_required and (self.geometry_budget.reserved_bytes or
                                               self.geometry_budget.generation_count) else '')

    def new_backend(self):
        self._check_thread()
        if self.closed:
            raise RuntimeError('viewport route broker is closed')
        backend = _ViewportBackend(self)
        self._views.add(backend)
        return backend

    def request_mode(self, mode):
        self._check_thread()
        if mode not in {'GPU', 'EVALUATED'}:
            raise ValueError('unknown viewport route')
        if self.closed:
            raise RuntimeError('viewport route broker is closed')
        if mode == self._target:
            return
        self._target, self._mode = mode, None
        for view in tuple(self._views):
            view._cancel()
        # Retire eligibility before any old backend can relinquish ownership.
        # The producer callback must invalidate only, without waiting/closing.
        self._stop_production()

    def _factory(self, mode):
        if mode == 'GPU':
            if self._device_factory is None:
                from .device_backend import DeviceAsyncCudaBackend
                self._device_factory = DeviceAsyncCudaBackend
            return self._device_factory(self.device, geometry_budget=self.geometry_budget)
        if self._evaluated_factory is None:
            from .async_backend import AsyncCudaBackend
            self._evaluated_factory = AsyncCudaBackend
        return self._evaluated_factory(self.device)

    def pump(self):
        """Poll once, never synchronize a job or release an unproven allocation."""
        self._check_thread()
        waiting = False
        transition = self._mode is None
        for view in tuple(self._views):
            if not view._lock.acquire(blocking=False):
                waiting = True
                continue
            try:
                delegate = view._delegate
                retiring = transition or view.closed or self.closed
                if delegate is not None and retiring and delegate.pending:
                    completed = delegate.collect()
                    if completed is not None and not view.closed:
                        view._completed = completed
                    if delegate.pending:
                        waiting = True
                        continue
                if delegate is not None and not delegate.pending:
                    retire = getattr(delegate, 'retire_completed_geometry', None)
                    if retire is not None:
                        retire()
                    if retiring:
                        delegate.close()  # no pending job: this does not drain tracing
                        view._delegate = None
                if view.closed and view._delegate is None:
                    view._completed = None
                    self._views.discard(view)
            finally:
                view._lock.release()
        self.geometry_budget.pool.poll()
        if transition and not waiting and not any(v._delegate is not None for v in self._views):
            # Closed device backends can still be retained by last-read fences.
            # No evaluated constructor is permitted until the actual ledger is empty.
            if (self.geometry_budget.reserved_bytes == 0
                    and self.geometry_budget.generation_count == 0):
                self._mode = self._target
        return self.ready_mode

    def close(self):
        """Explicit lifecycle drain; the runtime closes the pool after this owner."""
        self._check_thread()
        if not self.closed:
            self._stop_production()
            self.closed = True
        for view in tuple(self._views):
            view.close()
        self.pump()


class _ViewportBackend:
    """Per-view history facade; strong broker ownership survives worker closure."""

    def __init__(self, broker):
        self.broker = broker
        self.device, self.arbiter = broker.device, broker.arbiter
        self._owner = object()
        self._delegate = self._completed = None
        self._lock = RLock()
        self.closed = False

    @property
    def owner(self):
        return self._delegate.owner if self._delegate is not None else self._owner

    @property
    def pending(self):
        return self._completed is not None or (self._delegate is not None and self._delegate.pending)

    def _cancel(self):
        self.arbiter.cancel(self._owner)
        if self._delegate is not None:
            self.arbiter.cancel(self._delegate.owner)

    def submit(self, scene, camera, width, height, **options):
        self.broker._check_thread()
        if self.closed:
            raise RuntimeError('viewport backend is closed')
        if isinstance(scene, DeviceSceneRequest):
            requested = 'GPU'
        elif isinstance(scene, SceneSnapshot):
            requested = 'EVALUATED'
        else:
            raise TypeError('expected an owned viewport scene request')
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self.pending:
                return False
            if requested == 'EVALUATED' and self.broker._target != requested:
                self.broker.request_mode(requested)
            if self.broker.ready_mode is None:
                self._cancel()
                raise RouteTransitionPending('viewport route transition requires checked retirement')
            if self.broker.ready_mode != requested:
                self._cancel()
                raise RouteTransitionPending(f'{requested} viewport route is not currently admitted')
            if requested == 'GPU' and self.broker.recovery_required:
                self._cancel()
                raise RouteTransitionPending(self.broker.recovery_reason or
                                             'GPU geometry pool is retired after quarantine')
            if self._delegate is None:
                candidate = self.broker._factory(requested)
                if str(candidate.device) != str(self.device) or candidate.arbiter is not self.arbiter:
                    candidate.close()
                    raise RuntimeError('viewport route factory changed device or arbiter')
                self._delegate = candidate
            try:
                return self._delegate.submit(scene, camera, width, height, **options)
            except DeviceSceneUnavailable:
                self._cancel()
                raise
        finally:
            self._lock.release()

    def collect(self, *, wait=False):
        if not self._lock.acquire(blocking=wait):
            return None
        try:
            if self._completed is not None:
                result, self._completed = self._completed, None
                return result
            return self._delegate.collect(wait=wait) if self._delegate is not None else None
        finally:
            self._lock.release()

    def close(self):
        # A render worker can request closure but cannot mutate the main-thread
        # pool. The broker keeps this facade and delegate until its next pump.
        self.closed = True
        self._cancel()
        if current_thread() is not self.broker._thread:
            return
        with self._lock:
            if self._delegate is not None:
                self._delegate.close()
                self._delegate = None
            self._completed = None
        self.broker._views.discard(self)
