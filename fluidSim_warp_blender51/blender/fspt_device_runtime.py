"""Main-thread coordinator for experimental FSPT device surface publication."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from threading import Lock, current_thread, main_thread
from time import monotonic
from weakref import ref

from . import surface_publication as surfaces
from .scene_access import scene_identity

_ENV = "FLUIDSIM_FSPT_DEVICE_SURFACE"
# This development ceiling bounds explicitly tracked geometry arrays only.
_GEOMETRY_BUDGET_BYTES = 256 * 1024 * 1024
_LOG = logging.getLogger(__name__)
_SCENES = {}
_DEVICES = {}
_DEVICE_ERRORS = {}
_HOST_EVENTS = {}
_EVENT_LOCK = Lock()


@dataclass
class _Demand:
    view: ref
    device: str
    channels: frozenset
    reason: str | None = None


@dataclass
class _SceneDemand:
    callback: object
    unsubscribe: object
    views: dict = field(default_factory=dict)
    fallback: dict = field(default_factory=dict)
    event_serial: int = 0


@dataclass(frozen=True)
class _Ticket:
    device: str
    key: object
    source: object
    captured_at: float


@dataclass(frozen=True)
class DeviceSurfaceSelection:
    publication: surfaces.SurfacePublication
    pool: object = field(repr=False)
    key: object
    vertex_count: int
    triangle_count: int


@dataclass
class DeviceResources:
    device: object
    pool: object
    budget: object
    bridge: object
    broker: object
    subscriptions: dict = field(default_factory=dict, repr=False)
    ticket: _Ticket | None = None
    host_publication: surfaces.SurfacePublication | None = None
    selected: DeviceSurfaceSelection | None = None
    committed: bool = False
    recovery_attempted: bool = False
    pump_blocked: bool = False
    pump_recovery_pending: bool = False
    closing: bool = False


def enabled():
    return os.environ.get(_ENV) == "1"


def _on_main():
    return current_thread() is main_thread()


def _final_blocked(alias):
    from ..fspt.final_handoff import MAIN_FINAL_HANDOFF

    return MAIN_FINAL_HANDOFF.blocked(alias)


def _active_views(*, token=None, device=None):
    for uid, scene in _SCENES.items():
        if token is not None and uid != token:
            continue
        for demand in scene.views.values():
            if (device is None or demand.device == device) and demand.view() is not None:
                yield uid, scene, demand


def has_demand(scene):
    """Check weak scene/view demand without importing Warp or touching a pool."""
    if not _on_main() or not enabled():
        return False
    token = surfaces.id_token(scene)
    return token is not None and next(_active_views(token=token), None) is not None


def has_resources(device=None):
    """Read registry metadata even when the opt-in is off or owners are closing."""
    return bool(_DEVICES) if device is None else str(device) in _DEVICES


def resource_error(device):
    """Read retained fault diagnostics without touching native owners."""
    alias = str(device)
    resource = _DEVICES.get(alias)
    reason = _DEVICE_ERRORS.get(alias, "")
    if not reason and resource is not None and (resource.closing or not enabled()):
        return "FSPT device geometry is still retiring"
    return reason


def _drop_offer(resource, token=None):
    ticket = resource.ticket
    if ticket is not None and (token is None or ticket.source.scene_token == token):
        resource.ticket = resource.host_publication = resource.selected = None
        resource.committed = False
        try:
            resource.bridge.invalidate(ticket.source.scene_token)
        finally:
            surfaces.invalidate_surface_channel(ticket.source.scene_token, surfaces.SurfaceChannel.GPU_FINAL)


def _retire_device(alias, resource):
    resource.closing = True
    try:
        _drop_offer(resource)
        resource.broker.close()
        resource.bridge.close()
        if resource.budget.reserved_bytes or resource.budget.generation_count:
            raise RuntimeError("Explicit FSPT geometry remains after owner retirement")
    except Exception as exc:
        # The pool/broker remain registered while their owners are uncertain.
        _DEVICE_ERRORS[alias] = f"{type(exc).__name__}: {exc}"
        return False
    _DEVICES.pop(alias, None)
    _DEVICE_ERRORS.pop(alias, None)
    return True


def _drop_scene(token):
    scene = _SCENES.pop(token, None)
    if scene is not None:
        scene.unsubscribe()
    for resource in tuple(_DEVICES.values()):
        subscription = resource.subscriptions.pop(token, None)
        if subscription is not None:
            subscription[1]()
        _drop_offer(resource, token)


def _retire_scene_offers(token, reason):
    for resource in tuple(_DEVICES.values()):
        _drop_offer(resource, token)
    record = _SCENES.get(token)
    if record is not None:
        aliases = {demand.device for _, _, demand in _active_views(token=token)}
        for alias in aliases:
            _set_fallback(record, alias, None, reason)


def _host_state_changed(token, channel, state):
    if channel != surfaces.SurfaceChannel.EVALUATED_MESH:
        return
    try:
        if state.reason == "Surface output retired":
            _retire_scene_offers(token, "waiting for a new committed native surface")
            return
        if any(stamp is not None and stamp.source == surfaces.SurfaceSource.CACHE
               for stamp in (state.committed, state.pending)):
            _retire_scene_offers(token, "cache playback requires evaluated mesh")
            return
        for resource in tuple(_DEVICES.values()):
            ticket = resource.ticket
            if ticket is None or ticket.source.scene_token != token:
                continue
            host = resource.host_publication
            current = state.eligible or state.pending
            if (state.status == surfaces.SurfaceStatus.ERROR or state.quarantined
                    or current is None or not _same_source(current, ticket)
                    or (host is not None and host != current)):
                _drop_offer(resource, token)
    except Exception:
        _LOG.exception("Could not retire optional FSPT device publication")


def subscribe(scene, view, *, device, required_channels=()):
    """Register view lifetime without keeping a Blender scene or view alive."""
    if not _on_main() or not enabled():
        return lambda: None
    token = surfaces.id_token(scene)
    if token is None:
        return lambda: None
    alias = str(device)
    record = _SCENES.get(token)
    if record is None:
        def host_changed(channel, state):
            if channel != surfaces.SurfaceChannel.EVALUATED_MESH:
                return
            # Blender can notify synchronously on the Final worker. Queue only
            # immutable host metadata there; never touch a pool or Blender ID.
            with _EVENT_LOCK:
                record.event_serial += 1
                if not _on_main():
                    _HOST_EVENTS[token] = (record, record.event_serial, channel, state)
                    return
                _HOST_EVENTS.pop(token, None)
            _host_state_changed(token, channel, state)

        record = _SCENES[token] = _SceneDemand(
            host_changed, surfaces.subscribe_surfaces(scene, host_changed))
    entry = object()
    record.views[entry] = _Demand(ref(view), alias, frozenset(required_channels))

    def withdraw():
        if not _on_main() or _SCENES.get(token) is not record:
            return
        record.views.pop(entry, None)
        if next(_active_views(token=token), None) is None:
            _drop_scene(token)
        elif next(_active_views(token=token, device=alias), None) is None:
            resource = _DEVICES.get(alias)
            if resource is not None:
                subscription = resource.subscriptions.pop(token, None)
                if subscription is not None:
                    subscription[1]()
                _drop_offer(resource, token)
        resource = _DEVICES.get(alias)
        if resource is not None and not _final_blocked(alias):
            if next(_active_views(device=alias), None) is None:
                _retire_device(alias, resource)
            else:
                _request_mode(alias, resource)

    return withdraw


def set_capability(scene, view, device, *, reason=""):
    """Update an active view's capability without resetting its source latch."""
    if not _on_main() or not enabled():
        return
    token, alias = surfaces.id_token(scene), str(device)
    if token is None:
        return
    changed = False
    for _uid, record, demand in _active_views(token=token, device=alias):
        if demand.view() is view and demand.reason != reason:
            # A genuine capability transition can reconsider a fallback.
            record.fallback.pop(alias, None)
            demand.reason = reason
            changed = True
    resource = _DEVICES.get(alias)
    if changed and resource is not None:
        try:
            _request_mode(alias, resource)
        except Exception as exc:
            fail_optional(alias, exc)


def _required_mode(alias):
    if alias in _DEVICE_ERRORS:
        return "EVALUATED"
    for _uid, scene, demand in _active_views(device=alias):
        if demand.reason or alias in scene.fallback:
            return "EVALUATED"
    return "GPU"


def _request_mode(alias, resource):
    if resource.closing or _final_blocked(alias):
        return
    mode = _required_mode(alias)
    if mode == "EVALUATED":
        _drop_offer(resource)
    resource.broker.request_mode(mode)


def _new_device(alias):
    # Keep runtime imports off entirely on the default/no-demand route.
    from ..fspt.device_scene import GeometryArrayBudget
    from ..fspt.surface_pool import SurfacePool
    from ..fspt.viewport_backend import ViewportBackendBroker
    from ..runtime.warp_runtime import _import_warp
    from .fspt_device_bridge import DeviceSurfaceBridge

    wp = _import_warp()
    if wp.__version__ != "1.19.0":
        raise ValueError("FSPT device publication requires the verified Warp 1.19.0 sparse surface route")
    device = wp.get_device(alias)
    if not device.is_cuda or device.alias != alias or device.is_capturing:
        raise ValueError("FSPT device publication needs a matching CUDA device outside graph capture")
    pool = SurfacePool(device, byte_limit=_GEOMETRY_BUDGET_BYTES)
    try:
        budget = GeometryArrayBudget(pool, byte_limit=_GEOMETRY_BUDGET_BYTES)
        bridge = DeviceSurfaceBridge(budget)

        def stop_production():
            _drop_offer(resource)
            bridge.invalidate()

        broker = ViewportBackendBroker(device, budget, stop_production=stop_production,
                                       mode=_required_mode(alias))
        resource = DeviceResources(device, pool, budget, bridge, broker)
        return resource
    except Exception:
        pool.close()
        raise


def device_resources(device):
    """Return this device's owner, creating it only for active opt-in demand."""
    if not _on_main() or not enabled():
        return None
    alias = str(device)
    if next(_active_views(device=alias), None) is None or _final_blocked(alias):
        return None
    resource = _DEVICES.get(alias)
    if resource is not None and resource.closing:
        if not _retire_device(alias, resource):
            return None
        resource = None
    if resource is None and alias not in _DEVICE_ERRORS:
        try:
            resource = _DEVICES[alias] = _new_device(alias)
        except Exception as exc:
            _DEVICE_ERRORS[alias] = f"{type(exc).__name__}: {exc}"
    return resource


def _set_fallback(scene, alias, signature, reason):
    scene.fallback[alias] = (signature, reason)
    resource = _DEVICES.get(alias)
    if resource is not None:
        _request_mode(alias, resource)


def _device_fault(alias, error):
    _DEVICE_ERRORS[alias] = error
    resource = _DEVICES.get(alias)
    if resource is not None:
        try:
            _request_mode(alias, resource)
        except Exception:
            _LOG.exception("Failed to withdraw an unsafe FSPT device selection")


def fail_optional(device, exc):
    """Quarantine optional orchestration failures without interrupting host output."""
    if _on_main() and enabled():
        _device_fault(str(device), f"{type(exc).__name__}: {exc}")


def fallback_reason(scene, device):
    """Return capability or producer fallback metadata without GPU operations."""
    if not _on_main() or not enabled():
        return ""
    alias, token = str(device), surfaces.id_token(scene)
    if token is None:
        return ""
    if alias in _DEVICE_ERRORS:
        return _DEVICE_ERRORS[alias]
    for _uid, record, demand in _active_views(token=token, device=alias):
        if demand.reason:
            return demand.reason
        if alias in record.fallback:
            return record.fallback[alias][1]
    for _uid, record, demand in _active_views(device=alias):
        if demand.reason or alias in record.fallback:
            return demand.reason or record.fallback[alias][1]
    return ""


def _set_bridge_demand(resource, token, active):
    channels = frozenset().union(*(demand.channels for _, _, demand in active))
    existing = resource.subscriptions.get(token)
    if existing is not None:
        if existing[0] == channels:
            return
        existing[1]()
        _drop_offer(resource, token)
    resource.subscriptions[token] = (
        channels, resource.bridge.subscribe(token, required_channels=channels))


def prepare_live(scene, sess, host, completion, *, publish_surface, host_required):
    """Optionally prepare before native host readback, on the publication thread."""
    if not has_demand(scene) or not publish_surface or completion is None:
        return None
    alias, token = str(sess.solver.device), surfaces.id_token(scene)
    if _final_blocked(alias):
        return None
    active = tuple(_active_views(token=token, device=alias))
    if not active or not any(demand.reason == "" for _, _, demand in active):
        return None
    scene_record = active[0][1]
    captured_at = monotonic()
    try:
        if (not sess.owns_scene(scene) or completion.scene_token != scene_identity(scene)
                or completion.owner_token != surfaces.surface_owner_token(scene)
                or completion.source_timeline_frame is None):
            return None
        if alias in _DEVICE_ERRORS:
            return None
        from ..sim.device_surface import DeviceSurfaceUnavailable
        from .fspt_device_bridge import LiveSurfaceSource

        source = LiveSurfaceSource(token, completion.owner_token, completion.epoch,
                                  completion.source_timeline_frame, completion.state_revision,
                                  completion.solver_frame, completion.simulation_time, completion.ready_at)
        pipeline = sess.solver.surface
        stamp = None
        try:
            stamp = pipeline.device_surface_stamp()
        except DeviceSurfaceUnavailable as exc:
            unavailable = str(exc)
        signature = (source, stamp, host_required)
        old_fallback = scene_record.fallback.get(alias)
        if old_fallback is not None and old_fallback[0] == signature:
            return None
        scene_record.fallback.pop(alias, None)
        if not host_required:
            _set_fallback(scene_record, alias, signature, "exact native host publication is required")
            return None
        if stamp is None:
            _set_fallback(scene_record, alias, signature, unavailable)
            return None
        if stamp.backend != "sparse":
            _set_fallback(scene_record, alias, signature, f"{stamp.backend} requires evaluated mesh")
            return None
        resource = _DEVICES.get(alias)
        ticket = resource.ticket if resource is not None else None
        if ticket is not None and ticket.source == source and ticket.key.build_stamp == stamp:
            return ticket
        if pipeline.mesh_cache is not None and pipeline.mesh_key == stamp.build_key:
            _set_fallback(scene_record, alias, signature, "mandatory host surface was already materialized")
            return None
        resource = device_resources(alias)
        if resource is None:
            return None
        _request_mode(alias, resource)
        if resource.broker.ready_mode != "GPU":
            return None
        stream = resource.device.stream
        if resource.device.is_capturing or pipeline._surface_source_stream is not stream:
            _set_fallback(scene_record, alias, signature, "current device surface source stream is unavailable")
            return None
        _set_bridge_demand(resource, token, active)
        _drop_offer(resource)
        offer = resource.bridge.prepare(host, source, producer_stream=stream, host_required=True)
        if offer.status == "PREPARED":
            ticket = _Ticket(alias, offer.key, source, captured_at)
            resource.ticket = ticket
            return ticket
        if offer.status in ("FAILED", "UNSUPPORTED", "UNAVAILABLE"):
            if offer.status == "FAILED":
                _device_fault(alias, offer.reason)
            else:
                _set_fallback(scene_record, alias, signature, offer.reason)
        return None
    except Exception as exc:
        _device_fault(alias, f"{type(exc).__name__}: {exc}")
        return None


def _same_source(stamp, ticket):
    source = ticket.source
    return (stamp.source == surfaces.SurfaceSource.LIVE
            and (stamp.scene_token, stamp.owner_token, stamp.epoch, stamp.source_timeline_frame,
                 stamp.solver_frame, stamp.state_revision, stamp.simulation_time, stamp.state_ready_at)
            == (source.scene_token, source.owner_token, source.epoch, source.source_timeline_frame,
                source.solver_frame, source.state_revision, source.simulation_time, source.state_ready_at)
            and stamp.timeline_frame == source.source_timeline_frame
            and stamp.build_key == ticket.key.build_stamp.build_key)


def _matching_host(scene, ticket, *, allow_pending=False):
    """Verify the exact current host receipt and its live output IDs."""
    from . import output_publication as output

    state = surfaces.get_surface_state(scene)
    eligible = state.eligible
    stamp = eligible or (state.pending if allow_pending else None)
    if (state.status == surfaces.SurfaceStatus.ERROR or state.quarantined
            or stamp is None or not _same_source(stamp, ticket)):
        return None
    obj = output.output_object(scene, output.SURFACE_NAME, ensure=False)
    if (obj is None or surfaces.id_token(obj) != stamp.object_token
            or surfaces.id_token(obj.data) != stamp.mesh_token):
        return None
    return stamp, obj, eligible is not None


def cancel_live(ticket):
    if not _on_main() or ticket is None:
        return
    resource = _DEVICES.get(ticket.device)
    if resource is not None and resource.ticket is ticket:
        try:
            _drop_offer(resource)
        except Exception as exc:
            _device_fault(ticket.device, f"{type(exc).__name__}: {exc}")


def finish_live(scene, ticket):
    """Accept only after the enclosing evaluated publication attempt has exited."""
    if (not _on_main() or ticket is None or _final_blocked(ticket.device)
            or surfaces.active_surface_attempt(scene) is not None):
        return
    resource = _DEVICES.get(ticket.device)
    if resource is None or resource.ticket is not ticket:
        return
    try:
        host = _matching_host(scene, ticket, allow_pending=True)
        if host is None:
            cancel_live(ticket)
            return
        resource.host_publication = host[0]
        if host[2] and not resource.committed:
            resource.committed = resource.bridge.commit(ticket.key)
            if not resource.committed:
                cancel_live(ticket)
    except Exception as exc:
        _device_fault(ticket.device, f"{type(exc).__name__}: {exc}")


def selection(scene, device):
    """Read-only eligibility. Stale, superseded, or uncommitted hosts fail closed."""
    if not has_demand(scene) or surfaces.active_surface_attempt(scene) is not None:
        return None
    alias = str(device)
    if _final_blocked(alias):
        return None
    resource = _DEVICES.get(alias)
    if resource is None or resource.closing or resource.selected is None:
        return None
    selected = resource.selected
    if (resource.broker.ready_mode != "GPU"
            or next(_active_views(token=surfaces.id_token(scene), device=str(device)), None) is None
            or surfaces.current_surface_publication(scene, surfaces.SurfaceChannel.GPU_FINAL)
            != selected.publication):
        return None
    try:
        host = _matching_host(scene, resource.ticket)
        if host is not None and host[0] == resource.host_publication:
            counts = resource.budget.surface_ready(selected.key)
            if counts == (selected.vertex_count, selected.triangle_count):
                return selected
    except Exception:
        pass
    return None


def _recover_failed_pool(alias, resource):
    broker = resource.broker
    if (resource.recovery_attempted or not broker.recovery_required
            or not broker.geometry_recovery_ready):
        return
    resource.recovery_attempted = True
    try:
        # Only an explicit owner drain can retire a poisoned pool. It remains
        # latched out of the GPU route even if evaluated rendering can resume.
        resource.bridge.close()
        if resource.budget.reserved_bytes or resource.budget.generation_count:
            raise RuntimeError("Explicit FSPT geometry remains after failed-pool retirement")
        resource.pump_blocked = resource.pump_recovery_pending = False
    except Exception as exc:
        resource.pump_blocked = True
        _DEVICE_ERRORS[alias] = f"Device retirement requires explicit shutdown: {type(exc).__name__}: {exc}"


def poll(scene, device):
    """Main-thread ready/commit join. Publish GPU_FINAL at most once per key."""
    if not has_demand(scene) or surfaces.active_surface_attempt(scene) is not None:
        return None
    alias, token = str(device), surfaces.id_token(scene)
    if (next(_active_views(token=token, device=alias), None) is None
            or _final_blocked(alias)):
        return None
    resource = _DEVICES.get(alias)
    if resource is None or resource.closing:
        return None
    if resource.pool.poisoned:
        if alias not in _DEVICE_ERRORS:
            _device_fault(alias, "FSPT device surface pool is quarantined")
        _recover_failed_pool(alias, resource)
        return None
    if resource.broker.ready_mode != "GPU":
        return None
    ticket = resource.ticket
    if ticket is None or ticket.source.scene_token != token or resource.host_publication is None:
        return None
    try:
        host = _matching_host(scene, ticket, allow_pending=True)
        if host is None or host[0] != resource.host_publication:
            cancel_live(ticket)
            return None
        if not host[2]:
            return None
        if not resource.committed:
            resource.committed = resource.bridge.commit(ticket.key)
            if not resource.committed:
                cancel_live(ticket)
                return None
        if resource.selected is not None:
            return selection(scene, alias)
        ready = resource.bridge.poll(token)
        if ready.status == "FAILED":
            _device_fault(alias, ready.reason)
            return None
        if ready.status != "READY" or ready.key != ticket.key or ready.source != ticket.source:
            return None
        stamp, obj, _eligible = host
        if (ready.vertex_count, ready.triangle_count) != (stamp.vertex_count, stamp.triangle_count):
            _device_fault(alias, "GPU and native surface counts disagree")
            return None
        channel = surfaces.SurfaceChannel.GPU_FINAL
        with surfaces.surface_attempt(
            scene, channel, source=surfaces.SurfaceSource.LIVE, epoch=stamp.epoch,
            timeline_frame=stamp.timeline_frame, source_timeline_frame=stamp.source_timeline_frame,
            solver_frame=stamp.solver_frame, state_revision=stamp.state_revision,
            simulation_time=stamp.simulation_time, state_ready_at=stamp.state_ready_at,
            captured_at=ticket.captured_at,
        ) as attempt:
            attempt.record(vertex_count=ready.vertex_count, triangle_count=ready.triangle_count,
                           visible=stamp.visible, obj=obj)
            attempt.succeed(synchronized=True, build_key=stamp.build_key)
        publication = surfaces.current_surface_publication(scene, channel)
        if resource.ticket is not ticket:
            return None
        if publication is None:
            raise RuntimeError("checked device surface publication lost its channel")
        resource.selected = DeviceSurfaceSelection(
            publication, resource.pool, ticket.key, ready.vertex_count, ready.triangle_count)
        return selection(scene, alias)
    except Exception as exc:
        _device_fault(alias, f"{type(exc).__name__}: {exc}")
        return None


def invalidate(scene=None, *, clear_demand=False, reason=""):
    """Retire GPU selection without touching the evaluated host channel."""
    if not _on_main() or not _SCENES:
        return
    token = surfaces.id_token(scene) if scene is not None else None
    if scene is not None and token is None:
        return
    tokens = tuple(_SCENES) if scene is None else (token,)
    for token in tokens:
        try:
            if clear_demand:
                _drop_scene(token)
            elif reason:
                _retire_scene_offers(token, reason)
            else:
                for resource in tuple(_DEVICES.values()):
                    _drop_offer(resource, token)
        except Exception:
            _LOG.exception("Failed to retire optional FSPT device publication")


def pump():
    """Service existing owners and replay worker notifications on the main thread."""
    if not _on_main() or not (_SCENES or _DEVICES or _HOST_EVENTS):
        return False
    with _EVENT_LOCK:
        events = tuple(_HOST_EVENTS.items())
        _HOST_EVENTS.clear()
    for token, (record, serial, channel, state) in events:
        if _SCENES.get(token) is record and record.event_serial == serial:
            _host_state_changed(token, channel, state)
    for token, record in tuple(_SCENES.items()):
        for entry, demand in tuple(record.views.items()):
            if demand.view() is None:
                record.views.pop(entry, None)
        if not record.views:
            try:
                _drop_scene(token)
            except Exception:
                _LOG.exception("Failed to retire collected FSPT view demand")
    pending = False
    for alias, resource in tuple(_DEVICES.items()):
        if resource.closing or _final_blocked(alias):
            continue
        if not enabled() or next(_active_views(device=alias), None) is None:
            _retire_device(alias, resource)
            continue
        recovering = resource.pump_blocked
        if recovering:
            if not resource.pump_recovery_pending:
                continue
            # A last-read record fault can leave a surrendered lease attached
            # to its delegate. Permit exactly one follow-up owner pass, before
            # checked pool close; never turn an arbitrary pump fault into a
            # recurring timer retry. Final suppression above preserves this
            # allowance without executing or consuming it.
            resource.pump_recovery_pending = False
        try:
            resource.broker.pump()
            if recovering and getattr(resource.broker, 'retirement_deferred', False):
                # No destructive owner pass started. Contention and an active
                # peer must not consume the one checked recovery allowance.
                resource.pump_recovery_pending = True
                pending = True
                continue
            if resource.pool.poisoned and alias not in _DEVICE_ERRORS:
                _device_fault(alias, "FSPT device surface pool is quarantined")
            if resource.pool.poisoned:
                _recover_failed_pool(alias, resource)
            if recovering and resource.pump_blocked:
                _DEVICE_ERRORS[alias] = (
                    "Device retirement requires explicit shutdown after bounded broker recovery: "
                    + _DEVICE_ERRORS.get(alias, "ownership remains uncertain"))
                continue
            pending |= (resource.broker.ready_mode is None and not
                        (resource.recovery_attempted and resource.pool.poisoned and
                         (resource.budget.reserved_bytes or resource.budget.generation_count)))
        except Exception as exc:
            resource.pump_blocked = True
            resource.pump_recovery_pending = (
                not recovering and resource.pool.poisoned and not resource.recovery_attempted)
            pending |= resource.pump_recovery_pending
            fail_optional(alias, exc)
    return pending


def retire_for_final(device):
    """Explicit main-thread rendezvous. Retire ownership while preserving demand."""
    if not _on_main():
        raise RuntimeError("Final retirement must use the main thread")
    alias = str(device)
    resource = _DEVICES.get(alias)
    if resource is None:
        return {"reserved_bytes": 0, "generation_count": 0}
    if not _retire_device(alias, resource):
        raise RuntimeError(_DEVICE_ERRORS[alias])
    return {"reserved_bytes": resource.budget.reserved_bytes,
            "generation_count": resource.budget.generation_count}


def shutdown():
    """Drain explicit owners; return aliases that still retain uncertain work."""
    if not _on_main():
        return ()
    invalidate(clear_demand=True)
    with _EVENT_LOCK:
        _HOST_EVENTS.clear()
    pending = tuple(alias for alias, resource in tuple(_DEVICES.items())
                    if not _retire_device(alias, resource))
    for alias in tuple(_DEVICE_ERRORS):
        if alias not in pending:
            del _DEVICE_ERRORS[alias]
    return pending
