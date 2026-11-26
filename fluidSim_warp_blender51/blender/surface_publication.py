"""Scene-scoped identity and readiness of completed FluidSim surfaces.

This contract stores facts only. It neither reads geometry nor owns a solver,
and GPU consumers still need their own device packet and lifetime contract.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import count
from time import monotonic
from types import MethodType
from weakref import WeakMethod, ref

from .scene_access import active_domain_object, scene_identity


class SurfaceChannel(StrEnum):
    EVALUATED_MESH = "EVALUATED_MESH"
    GPU_FINAL = "GPU_FINAL"


class SurfaceSource(StrEnum):
    LIVE = "LIVE"
    CACHE = "CACHE"


class SurfaceStatus(StrEnum):
    READY = "READY"
    EMPTY = "EMPTY"
    UNCHANGED = "UNCHANGED"
    PENDING = "PENDING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SurfacePublication:
    """Geometry source and host timing captured by one publication attempt.

    captured_at marks adapter entry. published_at marks completion of geometry
    publication and the enclosing attempt, including a sync-pending outcome.
    source_timeline_frame is the actual admitted, solved or validated frame.
    """

    scene_token: int
    owner_token: tuple[int | None, str]
    epoch: int
    channel: SurfaceChannel
    generation: int
    source: SurfaceSource
    timeline_frame: int
    solver_frame: int | None
    state_revision: int | None
    simulation_time: float | None
    cache_frame: int | None
    cache_identity: tuple[str, str] | None
    build_key: tuple | None
    object_token: int | None
    mesh_token: int | None
    vertex_count: int
    triangle_count: int
    visible: bool
    published_at: float
    captured_at: float = 0.0
    source_timeline_frame: int | None = None

    @property
    def status(self) -> SurfaceStatus:
        return SurfaceStatus.READY if self.triangle_count else SurfaceStatus.EMPTY


@dataclass(frozen=True)
class ChannelState:
    attempt: int = 0
    status: SurfaceStatus = SurfaceStatus.PENDING
    reason: str = ""
    committed: SurfacePublication | None = None
    pending: SurfacePublication | None = None
    quarantined: bool = False

    @property
    def eligible(self) -> SurfacePublication | None:
        if self.quarantined or self.pending is not None or self.status == SurfaceStatus.PENDING:
            return None
        return self.committed


@dataclass
class _Channel:
    source: SurfaceSource
    owner_token: tuple[int | None, str]
    epoch: int
    cache_root: str | None
    state: ChannelState


_epochs = count(1)
_attempts = count(1)
_generations = {channel: count(1) for channel in SurfaceChannel}
_channels: dict[tuple[int, SurfaceChannel], _Channel] = {}
_active_attempt = ContextVar("surface_publication_attempt", default=None)
_subscribers = {}
_unchecked_output = object()
_log = logging.getLogger(__name__)


def id_token(datablock) -> int | None:
    """Use a Blender session UID where available, including across wrappers."""
    if datablock is None:
        return None
    try:
        original = getattr(datablock, "original", None)
        if original is not None:
            datablock = original
        uid = int(getattr(datablock, "session_uid", 0) or 0)
        return uid or scene_identity(datablock)
    except ReferenceError:
        return None


def surface_owner_token(scene) -> tuple[int | None, str]:
    collection = getattr(getattr(scene, "fluidsim", None), "output_collection_name", "")
    return id_token(active_domain_object(scene)), str(collection or "FluidSim Output")


def new_surface_epoch() -> int:
    return next(_epochs)


def get_surface_state(scene, channel=SurfaceChannel.EVALUATED_MESH) -> ChannelState:
    slot = _channels.get((id_token(scene), channel))
    if slot is None or slot.owner_token != surface_owner_token(scene):
        return ChannelState()
    return slot.state


def current_surface_publication(
    scene, channel=SurfaceChannel.EVALUATED_MESH,
) -> SurfacePublication | None:
    return get_surface_state(scene, channel).eligible


def subscribe_surfaces(scene, callback):
    """Observe completed identity changes, with a weak scene-scoped callback."""
    scene_token = id_token(scene)

    def discard(listener):
        listeners = _subscribers.get(scene_token)
        if listeners is not None:
            listeners[:] = [candidate for candidate in listeners if candidate is not listener]
            if not listeners:
                del _subscribers[scene_token]

    weak_callback = WeakMethod if isinstance(callback, MethodType) else ref
    listener = weak_callback(callback, discard)
    _subscribers.setdefault(scene_token, []).append(listener)
    return lambda: discard(listener)


def _notify(key, state):
    scene_token, channel = key
    for listener in tuple(_subscribers.get(scene_token, ())):
        callback = listener()
        if callback is not None:
            try:
                callback(channel, state)
            except Exception:
                _log.exception("Surface publication observer failed for %s", channel)


def invalidate_surfaces(scene=None, *, source=None) -> None:
    """Retire scene/channel eligibility before an owner or Blender ID goes away."""
    scene_token = id_token(scene)
    keys = [
        key for key, slot in _channels.items()
        if (scene is None or key[0] == scene_token) and (source is None or slot.source == source)
    ]
    _retire_channels(keys)


def cache_root_token(cache_path) -> str:
    """Resolve aliases for an owner-driven filesystem notification."""
    return os.path.normcase(os.path.realpath(cache_path))


def invalidate_cache_surfaces(cache_path, *, first_frame=None) -> None:
    """Retire publications whose verified cache frames an owner replaced."""
    root = cache_root_token(cache_path)
    keys = []
    for key, slot in tuple(_channels.items()):
        if slot.source != SurfaceSource.CACHE or slot.cache_root is None:
            continue
        frames = tuple(
            stamp.cache_frame for stamp in (slot.state.pending, slot.state.committed)
            if stamp is not None and stamp.cache_frame is not None
        )
        if first_frame is not None and frames and max(frames) < first_frame:
            continue
        if cache_root_token(slot.cache_root) == root:
            keys.append(key)
    _retire_channels(keys)


def _retire_channels(keys):
    for key in keys:
        del _channels[key]
    for key in keys:
        _notify(key, ChannelState(reason="Surface output retired"))


def _matches_output(publication, obj) -> bool:
    return (publication.object_token == id_token(obj)
            and publication.mesh_token == id_token(getattr(obj, "data", None)))


class _Attempt:
    def __init__(self, key, slot, publication):
        self.key = key
        self.slot = slot
        self.publication = publication
        self.previous = slot.state
        self.number = next(_attempts)
        self.mutated = False
        self.candidate = None
        self.finished = False
        slot.state = replace(self.previous, attempt=self.number, status=SurfaceStatus.PENDING,
                             reason="Surface publication in progress")

    def mark_mutation(self) -> None:
        """Call before any operation that can alter persistent surface geometry."""
        self.mutated = True

    def record(self, *, vertex_count, triangle_count, visible, obj=None):
        self.candidate = replace(
            self.publication, object_token=id_token(obj),
            mesh_token=id_token(getattr(obj, "data", None)),
            vertex_count=int(vertex_count), triangle_count=int(triangle_count),
            visible=bool(visible),
        )

    def _finish(self, status, *, committed, pending=None, reason="", quarantined=False):
        if self.finished:
            return
        self.finished = True
        if _channels.get(self.key) is self.slot and self.slot.state.attempt == self.number:
            self.slot.state = ChannelState(
                self.number, status, reason, committed, pending, quarantined)
            _notify(self.key, self.slot.state)

    def succeed(self, *, synchronized, warning="", build_key=None,
                retained_obj=_unchecked_output):
        old = self.previous
        if self.publication.source == SurfaceSource.CACHE and self.publication.cache_identity is None:
            self._finish(SurfaceStatus.ERROR, committed=None,
                         reason="Cache surface has no validated frame receipt")
            return
        if self.candidate is None and old.quarantined:
            self._finish(SurfaceStatus.ERROR, committed=old.committed, quarantined=True,
                         reason=old.reason)
            return

        candidate = old.pending
        retained = candidate or old.committed
        if self.candidate is not None:
            candidate = replace(
                self.candidate, build_key=build_key,
                generation=next(_generations[self.publication.channel]),
                published_at=monotonic(),
            )
        elif (retained is not None and retained_obj is not _unchecked_output
              and not _matches_output(retained, retained_obj)):
            self._finish(SurfaceStatus.ERROR, committed=old.committed, quarantined=True,
                         reason="Surface output was replaced before synchronization")
            return
        if not synchronized:
            self._finish(SurfaceStatus.PENDING, committed=old.committed,
                         pending=candidate or old.committed, reason=warning,
                         quarantined=old.quarantined)
        elif candidate is not None:
            self._finish(candidate.status, committed=candidate)
        else:
            self._finish(SurfaceStatus.UNCHANGED, committed=old.committed)

    def fail(self, error):
        old = self.previous
        self._finish(
            SurfaceStatus.ERROR, committed=old.committed,
            pending=None if self.mutated else old.pending,
            reason=f"{type(error).__name__}: {error}",
            quarantined=old.quarantined or self.mutated,
        )


@contextmanager
def surface_attempt(scene, channel=SurfaceChannel.EVALUATED_MESH, *, source,
                    epoch=None, timeline_frame=None, solver_frame=None,
                    state_revision=None, simulation_time=None, cache_frame=None,
                    cache_identity=None, source_timeline_frame=None):
    """Stage one surface and keep the previous committed descriptor until success."""
    owner = surface_owner_token(scene)
    key = id_token(scene), channel
    cache_root = cache_identity[0] if cache_identity is not None else None
    slot = _channels.get(key)
    if (slot is None or slot.source != source or slot.owner_token != owner
            or (epoch is not None and slot.epoch != epoch) or slot.cache_root != cache_root):
        initial = ChannelState(quarantined=slot.state.quarantined if slot else False)
        slot = _channels[key] = _Channel(
            source, owner, new_surface_epoch() if epoch is None else epoch, cache_root, initial)
    publication = SurfacePublication(
        key[0], owner, slot.epoch, channel, 0, source,
        int(getattr(scene, "frame_current", 0) if timeline_frame is None else timeline_frame),
        solver_frame, state_revision, simulation_time, cache_frame, cache_identity,
        None, None, None, 0, 0, False, 0.0, monotonic(), source_timeline_frame,
    )
    attempt = _Attempt(key, slot, publication)
    token = _active_attempt.set(attempt)
    try:
        yield attempt
    except BaseException as exc:
        attempt.fail(exc)
        raise
    finally:
        if not attempt.finished:
            attempt.fail(RuntimeError("Surface publication ended without a result"))
        _active_attempt.reset(token)


def active_surface_attempt(scene, channel=SurfaceChannel.EVALUATED_MESH):
    attempt = _active_attempt.get()
    if attempt is not None and attempt.key == (id_token(scene), channel):
        return attempt
    return None


def acknowledge_surface_sync(scene, obj) -> None:
    """Expose the exact pending mesh after a successful sync-only retry."""
    key = id_token(scene), SurfaceChannel.EVALUATED_MESH
    slot = _channels.get(key)
    if slot is None:
        return
    state = slot.state
    pending = state.pending
    if pending is None or state.quarantined or active_surface_attempt(scene) is not None:
        return
    if pending.owner_token == surface_owner_token(scene) and _matches_output(pending, obj):
        slot.state = ChannelState(state.attempt, pending.status, committed=pending)
        _notify(key, slot.state)
