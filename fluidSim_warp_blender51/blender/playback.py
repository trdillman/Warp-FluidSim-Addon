"""Preview and cache playback policy for Blender timeline events.

This module owns timeline interpretation, live/cache selection, pre-playback
admission, re-entry and render-restore suppression, and the two persistent
playback handlers.  The Solver Session remains the owner of direct scene
synchronization, stepping, reset, error storage, and teardown.
"""

from __future__ import annotations

from contextlib import contextmanager

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

_IN_FRAME_HANDLER = False
_SUPPRESS_FRAME_HANDLERS = 0
_RENDER_RESTORE_FRAME = None
_IN_MODE_TRANSITION = False

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
    if enabled:
        sess.release(scene)
        if getattr(fs, "live", False):
            fs.status = "Cache playback enabled - press play"
    elif getattr(fs, "live", False):
        set_live_enabled(scene, True)
    return True


def _check_cache_scene_key(scene, base, frame_data):
    """Refuse cache data stamped for materially different scene settings."""
    import numpy as np

    from ..io import cache

    frame = int(frame_data.get("frame", -1))
    metadata = cache.read_sequence_metadata(base) or {}
    metadata_key = str(metadata.get("scene_key", ""))
    frame_key = (str(np.asarray(frame_data["scene_key"]).item())
                 if "scene_key" in frame_data else "")
    if not (metadata_key or frame_key):
        return
    current = (
        cache.signature_bundle(scene)["scene_key"]
        if metadata.get("format_version") == 2
        else cache.scene_key(scene)
    )
    for candidate in (metadata_key, frame_key):
        if candidate and candidate != current:
            raise ValueError(_CACHE_SCENE_KEY_MESSAGE.format(
                frame=frame, baked=candidate[:12], current=current[:12]))


def play_cache_frame(sess, scene, frame=None):
    """Publish one cached frame without initializing or stepping a solver."""
    from . import output_publication
    from ..io import cache

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
    output_publication.update_outputs_from_cache_frame(scene, frame_data)
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
    fs.stat_voxels = 0
    fs.stat_step_ms = 0.0
    fs.stat_substeps = 0
    fs.stat_cfl_clamped = False
    fs.status = f"Cache frame {frame}: {fs.stat_particles:,} particles"
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
    if sess.active:
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
    if frame == sess.last_frame:
        return
    if getattr(fs, "cache_playback_enabled", False):
        sess.last_frame = frame
        try:
            play_cache_frame(sess, scene, frame)
        except Exception as exc:  # noqa: BLE001 - never break playback
            sess.set_error(scene, repr(exc))
            fs.status = f"Cache playback error: {exc}"
            print(f"fluidSim cache playback failed: {exc!r}")
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
        sess.release(preserve_error=True)
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
        _bridge().session().last_frame = restore_frame
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
    _bridge().session().release()
