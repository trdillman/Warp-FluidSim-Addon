"""Preview and cache playback policy for Blender timeline events.

This module owns timeline interpretation, live/cache selection, pre-playback
admission, re-entry and render-restore suppression, and the two persistent
playback handlers.  The Solver Session remains the owner of direct scene
synchronization, stepping, reset, error storage, and teardown.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

_IN_FRAME_HANDLER = False
_SUPPRESS_FRAME_HANDLERS = 0
_RENDER_RESTORE_FRAME = None
_IN_MODE_TRANSITION = False
_CACHE_CURSORS = {}

_ACTIVE_BAKE_STATUSES = frozenset({
    "PREFLIGHT",
    "INITIALIZING",
    "RUNNING",
    "CANCELLING",
})


@dataclass(frozen=True)
class ActivitySnapshot(Mapping):
    """One activity model compatible with both Properties UI and legacy callers."""

    state: str
    label: str
    detail: str
    icon: str
    busy: bool = False
    progress: float = 0.0
    id: str = "IDLE"
    owns_scene: bool = False
    bake_status: str = "IDLE"

    def _mapping(self):
        return {
            "id": self.id,
            "label": self.label,
            "icon": self.icon,
            "owns_scene": self.owns_scene,
            "bake_status": self.bake_status,
            "state": self.state,
            "detail": self.detail,
            "busy": self.busy,
            "progress": self.progress,
        }

    def __getitem__(self, key):
        return self._mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping())

    def __len__(self) -> int:
        return len(self._mapping())


def activity_snapshot(
        scene, *, owns_scene=None,
        is_animation_playing: bool = False) -> ActivitySnapshot:
    """Return the single public FluidSim activity state for ``scene``.

    Raw live/cache flags remain transport details.  The snapshot is deliberately
    side-effect free so Scene Properties, the viewport launcher, reports, and
    tests consume the same precedence rules.
    """
    fs = getattr(scene, "fluidsim", None) if scene is not None else None
    if fs is None:
        return ActivitySnapshot(
            "IDLE", "Idle", "", "PAUSE", id="IDLE",
            owns_scene=False, bake_status="IDLE")
    if owns_scene is None:
        try:
            owns_scene = bool(_bridge().session().owns_scene(scene))
        except Exception:
            owns_scene = False
    from . import production_bake

    bake_status = str(getattr(fs, "bake_status", "IDLE") or "IDLE")
    controller = production_bake.active_bake(scene)
    if controller is not None and production_bake.bake_is_active(scene):
        status = controller.status
        if status is production_bake.BakeStatus.CANCELLING:
            return ActivitySnapshot(
                "CANCELLING", "Cancelling Bake", str(controller.message),
                "CANCEL", True, float(controller.progress),
                "PRODUCTION_BAKE", bool(owns_scene), status.value)
        label = "Baking" if status is production_bake.BakeStatus.RUNNING else "Preparing Bake"
        return ActivitySnapshot(
            "BAKING", label, str(controller.message), "RENDER_ANIMATION",
            True, float(controller.progress), "PRODUCTION_BAKE",
            bool(owns_scene), status.value)
    if bake_status in _ACTIVE_BAKE_STATUSES:
        return ActivitySnapshot(
            "CANCELLING" if bake_status == "CANCELLING" else "BAKING",
            "Cancelling Bake" if bake_status == "CANCELLING" else "Production Bake",
            str(getattr(fs, "bake_message", "")),
            "CANCEL" if bake_status == "CANCELLING" else "RENDER_ANIMATION",
            True, float(getattr(fs, "bake_progress", 0.0)),
            "PRODUCTION_BAKE", bool(owns_scene), bake_status)
    if bool(getattr(fs, "cache_playback_enabled", False)):
        return ActivitySnapshot(
            "CACHE_PLAYBACK",
            "Playing Cache" if is_animation_playing else "Cache Playback Ready",
            str(getattr(fs, "status", "")), "PLAY", bool(is_animation_playing),
            0.0, "CACHE_PLAYBACK", bool(owns_scene), bake_status)
    if bool(getattr(fs, "live", False)):
        return ActivitySnapshot(
            "LIVE_PLAYING" if is_animation_playing else "LIVE_READY",
            "Playing Live" if is_animation_playing else "Live Ready",
            str(getattr(fs, "status", "")), "PLAY", bool(is_animation_playing),
            0.0, "LIVE_PREVIEW", bool(owns_scene), bake_status)
    detail = str(getattr(fs, "bake_message", ""))
    progress = float(getattr(fs, "bake_progress", 0.0))
    if bake_status == "COMPLETE":
        return ActivitySnapshot(
            "COMPLETE", "Bake Complete", detail, "CHECKMARK", False, progress,
            "IDLE", bool(owns_scene), bake_status)
    if bake_status == "FAILED":
        return ActivitySnapshot(
            "FAILED", "Bake Failed", detail, "ERROR", False, progress,
            "IDLE", bool(owns_scene), bake_status)
    if bake_status == "CANCELLED":
        return ActivitySnapshot(
            "IDLE", "Bake Cancelled", detail, "CANCEL", False, progress,
            "IDLE", bool(owns_scene), bake_status)
    return ActivitySnapshot(
        "IDLE", "Idle", str(getattr(fs, "status", "")), "PAUSE",
        id="IDLE", owns_scene=bool(owns_scene), bake_status=bake_status)


_CACHE_SCENE_KEY_MESSAGE = (
    "Cache frame {frame} was baked from different scene settings "
    "(cache {baked}, scene {current}). Re-bake the cache, or restore the "
    "settings it was baked with."
)


def _bridge():
    # Lazy at the Blender seam: scene_bridge owns the singleton Solver Session
    # and imports this module only from adapter entry points.
    from . import scene_bridge

    return scene_bridge


def in_frame_handler() -> bool:
    """Whether the playback callback currently owns Blender evaluation."""
    return _IN_FRAME_HANDLER


@contextmanager
def suppress_frame_handlers():
    """Suppress nested timeline delivery while an owned sequential runner acts."""
    global _SUPPRESS_FRAME_HANDLERS
    _SUPPRESS_FRAME_HANDLERS += 1
    try:
        yield
    finally:
        _SUPPRESS_FRAME_HANDLERS -= 1


@contextmanager
def suppress_mode_transitions():
    """Suppress live/cache-playback mode side effects during a bulk field write.

    Migration copies persisted settings with ``setattr``, which fires the RNA
    update callbacks in real Blender. Copying ``live`` (or
    ``cache_playback_enabled``) would otherwise route through
    ``set_live_enabled``/``set_cache_playback_enabled`` and spawn a Solver
    Session mid-load — before object transforms are evaluated — binding the
    scene to a bogus unit-cube domain. Migration is state restoration, not a
    user action, so its writes must not drive live transitions; both mode
    functions already early-return while ``_IN_MODE_TRANSITION`` is set.
    """
    global _IN_MODE_TRANSITION
    previous = _IN_MODE_TRANSITION
    _IN_MODE_TRANSITION = True
    try:
        yield
    finally:
        _IN_MODE_TRANSITION = previous


def prepare_live_preview(scene, *, depsgraph=None, allow_restart=False):
    """Make the owned simulation ready without starting GUI playback."""
    from . import pending_settings, production_bake

    sess = _bridge().session()
    if production_bake.bake_is_active(scene):
        raise RuntimeError("Production Bake owns this scene")
    if sess.active and not sess.owns_scene(scene):
        raise RuntimeError("Another scene owns the solver; release it from that scene first")
    previous_cache_mode = bool(getattr(scene.fluidsim, "cache_playback_enabled", False))
    with suppress_mode_transitions():
        scene.fluidsim.cache_playback_enabled = False
    try:
        if not sess.owns_scene(scene):
            pending_settings.apply_changes(
                scene, depsgraph=depsgraph, allow_restart=True, force_restart=True)
            return
        pending_settings.apply_changes(scene, depsgraph=depsgraph, allow_restart=allow_restart)
    except Exception:
        with suppress_mode_transitions():
            scene.fluidsim.cache_playback_enabled = previous_cache_mode
        raise


def pause_preview_for_changes(scene):
    """Stop this scene's advancement after an incompatible authoring edit."""
    fs = scene.fluidsim
    if not fs.live or getattr(fs, "cache_playback_enabled", False):
        return
    fs.live = False
    fs.status = "Preview paused: apply changes or resume to prepare the edited effects"
    manager = getattr(bpy.context, "window_manager", None)
    for window in getattr(manager, "windows", ()):
        if window.scene == scene and window.screen.is_animation_playing:
            try:
                with bpy.context.temp_override(window=window, screen=window.screen):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
            except RuntimeError as exc:
                fs.status = f"Simulation paused; timeline could not pause: {exc}"


def prepare_cache_preview(scene):
    """Validate the requested frame before handing the scene to cache playback."""
    from ..io import cache
    from . import production_bake

    sess = _bridge().session()
    if production_bake.bake_is_active(scene):
        raise RuntimeError("Production Bake owns this scene")
    if sess.active and not sess.owns_scene(scene):
        raise RuntimeError("Another scene owns the solver; release it from that scene first")
    base = bpy.path.abspath(scene.fluidsim.cache_path)
    frame = int(scene.frame_current)
    if getattr(scene.fluidsim, "cache_use_custom_range", False):
        start, end = cache.resolve_frame_range(
            scene, use_custom=True, frame_start=scene.fluidsim.cache_frame_start,
            frame_end=scene.fluidsim.cache_frame_end)
        if not start <= frame <= end:
            raise FileNotFoundError(f"Cache frame {frame} is outside playback range {start}-{end}")
    data = cache.read_frame(base, frame)
    if data is None:
        raise FileNotFoundError(f"Cache frame {frame} is missing or corrupt in {base}")
    _check_cache_scene_key(scene, base, data)
    if not set_cache_playback_enabled(scene, True):
        raise RuntimeError(scene.fluidsim.status)
    with suppress_mode_transitions():
        scene.fluidsim.cache_playback_enabled = True
    play_cache_frame(sess, scene, frame)


def set_live_enabled(scene, enabled: bool):
    """Apply the characterized live enable/disable transition."""
    global _IN_MODE_TRANSITION
    if _IN_MODE_TRANSITION:
        return True
    if _IN_FRAME_HANDLER:
        # fs.live writes from the handler invoke the RNA callback inline.  The
        # outer handler owns teardown and must not allocate or touch IDs again.
        return
    sess = _bridge().session()
    fs = scene.fluidsim
    if enabled:
        from . import production_bake

        if production_bake.bake_is_active(scene):
            _IN_MODE_TRANSITION = True
            try:
                fs.live = False
            finally:
                _IN_MODE_TRANSITION = False
            fs.status = "Production Bake is active - Live Playback is unavailable"
            return False
    if enabled:
        if getattr(fs, "cache_playback_enabled", False):
            fs.status = "Cache playback enabled - press play"
        elif not sess.owns_scene(scene):
            sess.initialize(scene)
            fs.status = "Live simulation enabled - press play"
        else:
            fs.status = "Live simulation enabled - press play"
    else:
        fs.status = "Live simulation disabled"
    return True


def set_cache_playback_enabled(scene, enabled: bool):
    """Apply the characterized mutually-exclusive cache playback transition."""
    global _IN_MODE_TRANSITION
    if _IN_MODE_TRANSITION:
        return True
    sess = _bridge().session()
    fs = scene.fluidsim
    if enabled:
        from . import production_bake

        if production_bake.bake_is_active(scene):
            _IN_MODE_TRANSITION = True
            try:
                fs.cache_playback_enabled = False
            finally:
                _IN_MODE_TRANSITION = False
            fs.status = "Production Bake is active - Cache Playback is unavailable"
            return False
        if sess.active and not sess.owns_scene(scene):
            _IN_MODE_TRANSITION = True
            try:
                fs.cache_playback_enabled = False
            finally:
                _IN_MODE_TRANSITION = False
            fs.status = (
                "Another scene owns the live solver - release or switch that "
                "scene before enabling Cache Playback")
            return False
    if enabled:
        sess.release(scene)
        _CACHE_CURSORS.pop(_scene_identity(scene), None)
        if getattr(fs, "live", False):
            fs.status = "Cache playback enabled - press play"
    else:
        _CACHE_CURSORS.pop(_scene_identity(scene), None)
        if getattr(fs, "live", False):
            set_live_enabled(scene, True)
    return True


def _scene_identity(scene):
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
            if value:
                return value
        except (ReferenceError, TypeError, ValueError):
            pass
    return id(scene)


def clear_cache_cursors():
    """Clear all non-authoritative cache playback positions."""
    _CACHE_CURSORS.clear()


def _check_cache_scene_key(scene, base, frame_data):
    """Refuse cache data stamped for materially different scene settings."""
    import numpy as np

    from ..io import cache
    from . import cache_identity

    frame = int(frame_data.get("frame", -1))
    metadata = cache.read_sequence_metadata(base) or {}
    metadata_key = str(metadata.get("scene_key", ""))
    frame_key = (str(np.asarray(frame_data["scene_key"]).item())
                 if "scene_key" in frame_data else "")
    if not (metadata_key or frame_key):
        return
    if int(metadata.get("format_version", 1)) >= 2:
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except (AttributeError, RuntimeError):
            depsgraph = None
        current = cache_identity.signature_bundle(scene, depsgraph)["scene_key"]
    else:
        current = cache_identity.scene_key(scene)
    for candidate in (metadata_key, frame_key):
        if candidate and candidate != current:
            raise ValueError(_CACHE_SCENE_KEY_MESSAGE.format(
                frame=frame, baked=candidate[:12], current=current[:12]))


def play_cache_frame(sess, scene, frame=None):
    """Publish one cached frame without initializing or stepping a solver."""
    import numpy as np

    from ..io import cache
    from . import output_publication

    fs = scene.fluidsim
    frame = int(scene.frame_current if frame is None else frame)
    if getattr(fs, "cache_use_custom_range", False):
        start, end = cache.resolve_frame_range(
            scene,
            use_custom=True,
            frame_start=fs.cache_frame_start,
            frame_end=fs.cache_frame_end,
        )
        if frame < start or frame > end:
            raise FileNotFoundError(
                f"Cache frame {frame} is outside playback range {start}-{end}")
    base = bpy.path.abspath(fs.cache_path)
    frame_data = cache.read_frame(base, frame)
    if frame_data is None:
        raise FileNotFoundError(
            f"Cache frame {frame} is missing or corrupt in {base}")
    _check_cache_scene_key(scene, base, frame_data)
    publication = output_publication.update_outputs_from_cache_frame(
        scene, frame_data
    )
    fs.stat_particles = len(frame_data.get("positions", ()))
    fs.stat_whitewater = len(frame_data.get("whitewater", ()))

    def scalar(key, default=0):
        value = frame_data.get(key, default)
        return value.item() if hasattr(value, "item") else value

    fs.stat_whitewater_required = int(scalar("whitewater_required"))
    fs.stat_whitewater_admitted = int(scalar("whitewater_admitted"))
    fs.stat_whitewater_dropped = int(scalar("whitewater_dropped"))
    fs.stat_whitewater_policy = str(
        scalar("whitewater_policy", "UNAVAILABLE_OLD_CACHE"))
    fs.stat_whitewater_status = str(
        scalar("whitewater_status", "UNAVAILABLE_OLD_CACHE"))
    type_values = frame_data.get("whitewater_type")
    if type_values is None:
        fs.stat_whitewater_foam = 0
        fs.stat_whitewater_bubbles = 0
        fs.stat_whitewater_spray = 0
        fs.stat_whitewater_invalid_types = fs.stat_whitewater
    else:
        values = np.asarray(type_values, dtype=np.int32)
        fs.stat_whitewater_foam = int(np.count_nonzero(values == 0))
        fs.stat_whitewater_bubbles = int(np.count_nonzero(values == 1))
        fs.stat_whitewater_spray = int(np.count_nonzero(values == 2))
        fs.stat_whitewater_invalid_types = int(np.count_nonzero(
            (values < 0) | (values > 2)))
    fs.stat_voxels = 0
    fs.stat_step_ms = 0.0
    fs.stat_substeps = 0
    fs.stat_cfl_clamped = False
    fs.stat_capillary_dt = 0.0
    fs.stat_capillary_substeps_required = 1
    fs.stat_capillary_clamped = False
    fs.stat_substep_limiter = "CACHE_PLAYBACK"
    fs.stat_surface_tension_active_voxels = 0
    fs.stat_surface_tension_max_curvature = 0.0
    fs.stat_surface_tension_max_acceleration = 0.0
    fs.stat_surface_tension_temporary_bytes = 0
    fs.stat_surface_tension_submission_ms = 0.0
    fs.stat_surface_tension_status = "CACHE_PLAYBACK"
    fs.stat_surface_tension_message = ""
    fs.status = f"Cache frame {frame}: {fs.stat_particles:,} particles"
    warning = str(getattr(publication, "warning", "") or "")
    if warning:
        sess.set_error(scene, warning)
        fs.status = f"{fs.status} — {warning}"
    else:
        sess.clear_error(scene)
    return frame_data


def ensure_session_before_frames(scene, reason):
    """Admit the solver before playback/render frame evaluation when needed."""
    fs = getattr(scene, "fluidsim", None) if scene is not None else None
    if fs is None or not getattr(fs, "live", False):
        return
    if getattr(fs, "cache_playback_enabled", False):
        return
    bridge = _bridge()
    sess = bridge.session()
    if sess.owns_scene(scene):
        return
    try:
        sess.initialize(scene)
    except Exception as exc:  # noqa: BLE001 - never break playback or render
        sess.set_error(scene, repr(exc))
        fs.status = f"Solver initialization error: {exc}"
        print(f"FluidSim {reason} initialization failed: {exc!r}")


def handle_frame_change(sess, scene, depsgraph):
    """Interpret one Blender timeline event using the frozen preview policy."""
    fs = getattr(scene, "fluidsim", None)
    if fs is None or not fs.live:
        return
    frame = scene.frame_current
    if getattr(fs, "cache_playback_enabled", False):
        scene_id = _scene_identity(scene)
        if _CACHE_CURSORS.get(scene_id) == frame:
            return
        try:
            play_cache_frame(sess, scene, frame)
            # Do not consume a timeline frame until publication succeeds. A
            # repaired/replaced cache can then be retried at the same frame.
            _CACHE_CURSORS[scene_id] = frame
        except Exception as exc:  # noqa: BLE001 - never break playback
            sess.set_error(scene, repr(exc))
            fs.status = f"Cache playback error: {exc}"
            print(f"fluidSim cache playback failed: {exc!r}")
        return
    if frame == sess.last_frame:
        return
    if not sess.owns_scene(scene):
        try:
            sess.initialize(scene, depsgraph=depsgraph)
        except Exception as exc:  # noqa: BLE001 - never break playback
            sess.set_error(scene, repr(exc))
            fs.status = f"Solver initialization error: {exc}"
            print(f"FluidSim playback initialization failed: {exc!r}")
            return

    bridge = _bridge()
    if frame <= sess.start_frame or frame < sess.last_frame:
        try:
            from . import pending_settings

            if not pending_settings.apply_pending_on_rewind(
                    scene, sess, depsgraph=depsgraph):
                sess.reset(scene, depsgraph=depsgraph)
        except Exception as exc:  # noqa: BLE001 - retain pending on failure
            sess.set_error(scene, repr(exc))
            fs.live = False
            fs.status = f"Pending settings apply failed; playback stopped: {exc}"
        sess.last_frame = frame
        return

    gap = (frame - sess.last_frame
           if sess.last_frame >= sess.start_frame else 1)
    bound = max(1, int(getattr(fs, "max_frame_catchup", 8)))
    if gap > bound:
        sess.last_frame = frame
        message = (
            f"Timeline jumped {gap} frames, past the {bound}-frame recovery "
            f"limit. The solver would be out of sync with the timeline. "
            f"Rewind to restart, or increase Maximum Playback Recovery "
            f"Frames under Advanced Playback.")
        sess.set_error(scene, message)
        fs.status = f"Solver desync: {message}"
        return

    previous_last_frame = sess.last_frame
    sess.last_frame = frame
    try:
        for offset in range(gap - 1, -1, -1):
            sess.sync_scene(scene, depsgraph)
            bridge.step_frame(
                scene, sess, refresh_pending=False,
                timeline_frame=frame - offset,
                publish_outputs=(offset == 0))
    except Exception as exc:  # noqa: BLE001 - never break playback
        from ..sim.fields import TopologyCapacityError

        if isinstance(exc, TopologyCapacityError):
            sess.last_frame = previous_last_frame
            sess.set_error(scene, repr(exc))
            fs.live = False
            fs.status = (
                "Topology capacity blocked; Live paused, solver preserved "
                f"(need {int(exc.required):,}, capacity {int(exc.capacity):,})")
            print(f"fluidSim topology step blocked: {exc!r}")
            return
        sess.set_error(scene, repr(exc))
        fs.live = False
        fs.status = f"Solver error: {exc}"
        sess.force_release(scene=scene, preserve_error=True)
        print(f"fluidSim step failed: {exc!r}")


def arm_render_frame_restore(frame):
    """Ignore exactly one Blender bookkeeping restore after a render job."""
    global _RENDER_RESTORE_FRAME
    _RENDER_RESTORE_FRAME = int(frame)


@persistent
def _on_playback_pre(scene, *_args):
    ensure_session_before_frames(scene, "playback")


@persistent
def _on_frame_change(scene, depsgraph):
    global _IN_FRAME_HANDLER
    global _RENDER_RESTORE_FRAME
    restore_frame, _RENDER_RESTORE_FRAME = _RENDER_RESTORE_FRAME, None
    if (restore_frame is not None and scene is not None
            and scene.frame_current == restore_frame):
        if getattr(getattr(scene, "fluidsim", None),
                   "cache_playback_enabled", False):
            _CACHE_CURSORS[_scene_identity(scene)] = restore_frame
        else:
            sess = _bridge().session()
            if sess.owns_scene(scene):
                sess.last_frame = restore_frame
        return
    if _IN_FRAME_HANDLER or _SUPPRESS_FRAME_HANDLERS:
        return
    _IN_FRAME_HANDLER = True
    try:
        handle_frame_change(_bridge().session(), scene, depsgraph)
    finally:
        _IN_FRAME_HANDLER = False


def register_handlers():
    """Deduplicate and install the two handlers exclusively owned here."""
    _remove_handlers()
    bpy.app.handlers.frame_change_post.append(_on_frame_change)
    if hasattr(bpy.app.handlers, "animation_playback_pre"):
        bpy.app.handlers.animation_playback_pre.append(_on_playback_pre)


def _remove_handlers():
    for name in ("frame_change_post", "animation_playback_pre"):
        handlers = getattr(bpy.app.handlers, name, None)
        if handlers is None:
            continue
        for fn in list(handlers):
            if getattr(fn, "__module__", "") == __name__:
                handlers.remove(fn)


def unregister_handlers():
    """Remove playback handlers before releasing their Solver Session."""
    _remove_handlers()
    clear_cache_cursors()
    _bridge().session().force_release()
