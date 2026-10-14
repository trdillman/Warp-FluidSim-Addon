"""Deferred kernel prewarm: a one-shot ``bpy.app.timers`` callback.

Registered as the last step of the registration table, so it exists only once
every other step succeeded and is the first thing torn down. Failing to
schedule or cancel it must never fail enabling or disabling the add-on, so both
directions swallow errors.
"""

from __future__ import annotations

_FIRST_INTERVAL = 0.5


def _background_prewarm():
    try:
        from .warp_runtime import prewarm_simulation_kernels, probe_runtime

        # Without CUDA (playback-only) there is nothing to compile; the probe's
        # own Warp import is the only one attempted.
        if probe_runtime().available:
            prewarm_simulation_kernels()
    except Exception:  # noqa: BLE001
        pass
    return None


def _timers():
    try:
        import bpy
    except Exception:  # noqa: BLE001
        return None
    return getattr(getattr(bpy, "app", None), "timers", None)


def register_timer():
    try:
        timers = _timers()
        if timers is not None and hasattr(timers, "register"):
            if not getattr(timers, "is_registered", lambda _: False)(_background_prewarm):
                timers.register(_background_prewarm, first_interval=_FIRST_INTERVAL)
    except Exception:  # noqa: BLE001
        pass


def unregister_timer():
    try:
        timers = _timers()
        if timers is not None and hasattr(timers, "unregister"):
            if getattr(timers, "is_registered", lambda _: False)(_background_prewarm):
                timers.unregister(_background_prewarm)
    except Exception:  # noqa: BLE001
        pass
