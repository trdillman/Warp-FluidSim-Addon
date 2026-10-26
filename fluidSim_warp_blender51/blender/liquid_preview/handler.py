"""Liquid Preview lifecycle: one draw handler while the mode is on, none otherwise.

The handler draws in ``POST_VIEW`` so the bound depth buffer is the scene's:
solids in front of the liquid occlude it without a CPU depth read. It never
touches ``bpy.data`` or the solver; its only input is the record Output
Publication hands over. A failure inside the draw (including a shader that
does not compile on this backend) is logged once, recorded as the session
failure, and from then on the handler draws nothing. The PointCloud output is
never hidden, so the viewport keeps showing the particles.
"""

from __future__ import annotations

import sys

from .. import output_publication

TOGGLE_OPERATOR = "fluidsim.toggle_liquid_preview"
TOGGLE_KEY = "L"

_DRAW_HANDLE = None
_KEYMAP_ITEMS: list = []
_RENDERER = None
_FAILURE: str | None = None
bpy = sys.modules.get("bpy")


def _require_bpy():
    global bpy
    if bpy is None:
        try:
            import bpy as blender_api
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Liquid Preview requires Blender's Python runtime") from exc
        bpy = blender_api
    return bpy


def _persistent(function):
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    return getattr(handlers, "persistent", lambda fn: fn)(function)


def failure() -> str | None:
    """The session failure that disabled drawing, or ``None``."""
    return _FAILURE


def is_draw_registered() -> bool:
    return _DRAW_HANDLE is not None


def _preview_wanted() -> bool:
    try:
        scenes = list(bpy.data.scenes)
    except (AttributeError, TypeError):
        return False  # restricted data during add-on registration
    for scene in scenes:
        fs = getattr(scene, "fluidsim", None)
        if fs is not None and bool(getattr(fs, "liquid_preview_enabled", False)):
            return True
    return False


def _add_draw_handler():
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), "WINDOW", "POST_VIEW")


def _remove_draw_handler():
    global _DRAW_HANDLE, _RENDERER
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_DRAW_HANDLE, "WINDOW")
        except (ValueError, RuntimeError, AttributeError):
            pass
        _DRAW_HANDLE = None
    if _RENDERER is not None:
        try:
            _RENDERER.free()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        _RENDERER = None


def sync_registration():
    """Register the draw handler iff any scene has Liquid Preview on."""
    _require_bpy()
    if _preview_wanted():
        _add_draw_handler()
    else:
        _remove_draw_handler()


def tag_redraw():
    """Ask every 3D viewport to redraw (no-op without a window manager)."""
    try:
        windows = list(bpy.context.window_manager.windows)
    except (AttributeError, TypeError):
        return
    for window in windows:
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()):
            if getattr(area, "type", "") == "VIEW_3D":
                area.tag_redraw()


def _fail(exc):
    global _FAILURE
    if _FAILURE is None:
        _FAILURE = f"{type(exc).__name__}: {exc}"
        print(f"FluidSim: Liquid Preview disabled for this session "
              f"({_FAILURE}); the particle output stays visible.")


def _draw_frame(context):
    global _RENDERER
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    fs = scene.fluidsim
    if not fs.liquid_preview_enabled:
        return
    record = output_publication.latest_published_points()
    if record is None:
        return
    pointer = getattr(scene, "as_pointer", None)
    if (record.scene_key is not None and callable(pointer)
            and record.scene_key != pointer()):
        return
    region_data = getattr(context, "region_data", None)
    if region_data is None:
        return
    import gpu

    from .passes import LiquidPreviewRenderer, preview_settings

    if _RENDERER is None:
        _RENDERER = LiquidPreviewRenderer()
    _RENDERER.draw(
        record,
        view_matrix=region_data.view_matrix,
        projection_matrix=region_data.window_matrix,
        viewport=tuple(gpu.state.viewport_get()),
        settings=preview_settings(fs),
    )


def _draw_callback():
    """Draw-handler entry point: never raises."""
    if _FAILURE is not None:
        return
    try:
        _draw_frame(bpy.context)
    except Exception as exc:  # noqa: BLE001 - a draw handler must not raise
        _fail(exc)


@_persistent
def _on_load_post(*_args):
    sync_registration()


def _register_keymap():
    try:
        keyconfig = bpy.context.window_manager.keyconfigs.addon
    except AttributeError:
        keyconfig = None
    if keyconfig is None:
        return  # background mode has no add-on keyconfig
    keymap = keyconfig.keymaps.new(name="3D View", space_type="VIEW_3D")
    item = keymap.keymap_items.new(TOGGLE_OPERATOR, TOGGLE_KEY, "PRESS")
    _KEYMAP_ITEMS.append((keymap, item))


def _unregister_keymap():
    while _KEYMAP_ITEMS:
        keymap, item = _KEYMAP_ITEMS.pop()
        try:
            keymap.keymap_items.remove(item)
        except (ValueError, RuntimeError, ReferenceError):
            pass


def register():
    global _FAILURE
    from ...unregister_guard import handler_registration

    _require_bpy()
    previous_failure = _FAILURE
    previous_handle = _DRAW_HANDLE
    keymap_start = len(_KEYMAP_ITEMS)
    _FAILURE = None
    try:
        _register_keymap()
        with handler_registration(((bpy.app.handlers.load_post, _on_load_post),)):
            sync_registration()
    except Exception as exc:
        if previous_handle is None and _DRAW_HANDLE is not None:
            _remove_draw_handler()
        while len(_KEYMAP_ITEMS) > keymap_start:
            keymap, item = _KEYMAP_ITEMS.pop()
            try:
                keymap.keymap_items.remove(item)
            except Exception as undo_exc:
                exc.add_note(f"FluidSim preview keymap rollback: {undo_exc!r}")
                print(f"FluidSim: could not roll back preview keymap: {undo_exc!r}")
        _FAILURE = previous_failure
        raise


def unregister():
    global _FAILURE
    _require_bpy()
    load_post = bpy.app.handlers.load_post
    while _on_load_post in load_post:
        load_post.remove(_on_load_post)
    _unregister_keymap()
    _remove_draw_handler()
    _FAILURE = None
