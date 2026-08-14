"""Render output integration.

Simulation outputs are real datablocks, so they render directly. This module
keeps render visibility coherent and steps the solver during animation
renders (render frame-change fires the same handler as playback when the
simulation is live).
"""

from __future__ import annotations

import bpy

from . import output_publication

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)


def ensure_render_visibility(scene):
    """Delegate output visibility; hide authored helper objects for renders."""
    output_publication.ensure_render_visibility(scene)
    for obj in scene.objects:
        pg = getattr(obj, "fluidsim", None)
        if pg is not None and pg.role in {"SOURCE", "SINK", "FORCE"}:
            obj.hide_render = True


def configure_cycles_points(scene):
    """Delegate point-output render preparation to its persistent owner."""
    output_publication.configure_cycles_points(scene)


# Frame the render job started from. Blender restores it after the job, and
# that restore must not reach the solver -- see _on_render_end.
_PRE_RENDER_FRAME = None


@persistent
def _on_render_init(scene, *_args):
    # Must be @persistent: Blender frees non-persistent app handlers on file
    # load, so without it render visibility silently stops being enforced
    # after the user opens any .blend while the add-on stays enabled.
    #
    # This module owns render_init outright -- unregister_handlers() below drops
    # every add-on callback on that list -- so the solver hoist rides along here
    # rather than registering a second handler that this module would strip.
    from . import playback

    global _PRE_RENDER_FRAME
    _PRE_RENDER_FRAME = scene.frame_current
    playback.ensure_session_before_frames(scene, "render")
    ensure_render_visibility(scene)


@persistent
def _on_render_end(_scene, *_args):
    # Arm here rather than in _on_render_init: during the job itself the render
    # walks real frames, and one of them can equal the pre-render frame. Only
    # the restore that lands after the job is Blender's bookkeeping.
    from . import playback

    if _PRE_RENDER_FRAME is not None:
        playback.arm_render_frame_restore(_PRE_RENDER_FRAME)


_HANDLER_LISTS = ("render_init", "render_complete", "render_cancel")


def register_handlers():
    unregister_handlers()
    bpy.app.handlers.render_init.append(_on_render_init)
    # Both endings restore the frame, so both have to arm.
    for name in ("render_complete", "render_cancel"):
        if hasattr(bpy.app.handlers, name):
            getattr(bpy.app.handlers, name).append(_on_render_end)


def _remove_all(handler_list, function):
    """Remove every registration of one function without touching other add-ons."""
    while function in handler_list:
        handler_list.remove(function)


def unregister_handlers():
    render_init = getattr(bpy.app.handlers, "render_init", None)
    if render_init is not None:
        _remove_all(render_init, _on_render_init)
    for name in ("render_complete", "render_cancel"):
        handlers = getattr(bpy.app.handlers, name, None)
        if handlers is not None:
            _remove_all(handlers, _on_render_end)
