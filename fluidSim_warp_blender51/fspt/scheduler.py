"""Device-wide admission of FSPT work. Callers own GPU completion and retirement."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, eq=False, slots=True)
class _Lease:
    owner: object
    final: bool


class DeviceArbiter:
    """Admit one quantum, with FIFO waiters and priority for final rendering.

    Supply a stable logical owner token, such as an object allocated once per
    view. Owners match by identity. Requests carry no renderer or Blender
    payload, and a grant requires a fresh attempt by the owner at the front.
    """

    def __init__(self):
        self._lock = Lock()
        self._active: _Lease | None = None
        self._viewport_waiters: list[object] = []
        self._final_waiters: list[object] = []

    def try_acquire(self, owner: object, *, final: bool = False) -> _Lease | None:
        """Record demand and grant a lease immediately when eligible."""
        final = bool(final)
        with self._lock:
            queue = self._final_waiters if final else self._viewport_waiters
            if not any(waiter is owner for waiter in queue):
                queue.append(owner)
            if self._active is not None or queue[0] is not owner:
                return None
            if not final and self._final_waiters:
                return None
            queue.pop(0)
            self._active = _Lease(owner, final)
            return self._active

    def release(self, lease: _Lease) -> None:
        """Retire the exact current lease after the caller verifies completion."""
        with self._lock:
            if self._active is None or lease is not self._active:
                raise ValueError("Lease is not the current FSPT device lease")
            self._active = None

    def cancel(self, owner: object) -> None:
        """Remove queued demand for this owner while its active lease stays held."""
        with self._lock:
            self._viewport_waiters[:] = [
                waiter for waiter in self._viewport_waiters if waiter is not owner
            ]
            self._final_waiters[:] = [
                waiter for waiter in self._final_waiters if waiter is not owner
            ]

    def withdraw_final(self, owner: object) -> None:
        """Remove this owner's queued final intent and preserve viewport demand."""
        with self._lock:
            self._final_waiters[:] = [
                waiter for waiter in self._final_waiters if waiter is not owner
            ]


_arbiters: dict[str, DeviceArbiter] = {}
_registry_lock = Lock()


def get_arbiter(device: str) -> DeviceArbiter:
    """Share an arbiter for callers supplying the same canonical device alias."""
    alias = str(device)
    with _registry_lock:
        arbiter = _arbiters.get(alias)
        if arbiter is None:
            arbiter = _arbiters[alias] = DeviceArbiter()
        return arbiter
