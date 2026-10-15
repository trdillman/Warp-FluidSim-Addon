"""The Deck's event layer: a pass-through modal that takes clicks only on the bar.

``fluidsim.deck_modal`` runs once per window. Every event passes through to
Blender except a left click (press, release, double click) over the bar card:
over a button it runs that button's :class:`controls.Control`; elsewhere on the
card it is swallowed so the click does not select or deselect in the viewport.
Esc is never consumed, so the Deck cannot take Esc away from a running bake or
from any Blender tool; a bake is cancelled with the bar's Cancel Bake button.

The modal keeps no settings state: only which bar item is under the pointer, for
the hover highlight. A generation number retires modals left from an earlier
``register()`` (add-on reload) or from before a file load (the load-post handler
starts a new generation, whether or not Blender called ``cancel()``). A timer
started by ``register()`` starts the modal in each window, and again after a file
load drops it; ``--background`` has no windows and starts none.
"""

from __future__ import annotations

import bpy

from . import bar, controls

ENSURE_SECONDS = 1.0
_GENERATION = 0
_RUNNING: set = set()  # window pointers with a live modal of this generation
_HOVER: dict = {"region": None, "item": None}


def new_generation() -> int:
    """Retire every running Deck modal; the next timer tick starts fresh ones."""
    global _GENERATION
    _GENERATION += 1
    _RUNNING.clear()
    _HOVER.update(region=None, item=None)
    return _GENERATION


def hover_item(context):
    """The bar item under the pointer in ``context.region``, or ``None``."""
    region = getattr(context, "region", None)
    if region is None or _HOVER["region"] is None:
        return None
    try:
        same = region.as_pointer() == _HOVER["region"]
    except (AttributeError, ReferenceError):
        return None
    return _HOVER["item"] if same else None


def _window_key(window):
    try:
        return window.as_pointer()
    except (AttributeError, ReferenceError):
        return id(window)


def region_under(window, x, y):
    """``(area, region)`` of the 3D View main region under window pixel (x, y)."""
    screen = getattr(window, "screen", None)
    for area in getattr(screen, "areas", ()):
        if area.type != "VIEW_3D":
            continue
        hit = None
        for region in area.regions:
            inside = (region.x <= x < region.x + region.width
                      and region.y <= y < region.y + region.height)
            if not inside:
                continue
            if region.type != "WINDOW":
                return None, None  # a toolbar, sidebar or header overlaps the point
            hit = region
        if hit is not None:
            return area, hit
    return None, None


class FLUIDSIM_OT_deck_modal(bpy.types.Operator):
    """Route clicks on the Liquid Deck bar; every other event passes through."""

    bl_idname = "fluidsim.deck_modal"
    bl_label = "FluidSim Liquid Deck Events"
    bl_description = (
        "Let the Liquid Deck bar take clicks on its buttons. Started automatically; "
        "every other event reaches Blender unchanged")
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        window = getattr(context, "window", None)
        if window is None:
            return {"CANCELLED"}
        key = _window_key(window)
        if key in _RUNNING:
            return {"CANCELLED"}
        self.generation = _GENERATION
        self.window_key = key
        _RUNNING.add(key)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL", "PASS_THROUGH"}

    def _finish(self):
        if getattr(self, "generation", None) == _GENERATION:
            _RUNNING.discard(getattr(self, "window_key", None))

    def _set_hover(self, region, item):
        key = region.as_pointer() if region is not None else None
        if (_HOVER["region"], _HOVER["item"]) != (key, item):
            _HOVER.update(region=key, item=item)
            if region is not None:
                region.tag_redraw()

    def modal(self, context, event):
        if self.generation != _GENERATION:
            return {"FINISHED", "PASS_THROUGH"}
        if event.type not in {"MOUSEMOVE", "LEFTMOUSE"}:
            return {"PASS_THROUGH"}  # Esc included: the Deck never consumes it
        window = getattr(context, "window", None)
        if window is None:
            return {"PASS_THROUGH"}
        # Event and region coordinates are both window-relative.
        area, region = region_under(window, event.mouse_x, event.mouse_y)
        if region is None:
            self._set_hover(None, None)
            return {"PASS_THROUGH"}
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        with context.temp_override(window=window, area=area, region=region):
            layout = bar.current_layout(bpy.context)
            over, item = bar.hit_test(layout, x, y)
            self._set_hover(region, item)
            if event.type == "MOUSEMOVE" or not over:
                return {"PASS_THROUGH"}
            if item is not None and event.value == "PRESS":
                if controls.control_enabled(bpy.context, item.control):
                    controls.run_control(bpy.context, item.control)
                region.tag_redraw()
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._finish()


CLASSES = (FLUIDSIM_OT_deck_modal,)


def _ensure_running():
    """Timer: start the Deck modal in every window that lacks one."""
    if getattr(bpy.app, "background", False):
        return None
    try:
        from . import deck_mode

        if deck_mode(bpy.context) == "OFF":
            return ENSURE_SECONDS
        for window in bpy.context.window_manager.windows:
            if _window_key(window) in _RUNNING:
                continue
            with bpy.context.temp_override(window=window):
                bpy.ops.fluidsim.deck_modal("INVOKE_DEFAULT")
    except Exception as exc:  # noqa: BLE001 - a timer must never raise
        print(f"FluidSim Liquid Deck: could not start the event layer: {exc!r}")
    return ENSURE_SECONDS


def start(registry) -> None:
    """Begin a new generation and, with a UI, keep the modal running in each window."""
    new_generation()
    if not getattr(bpy.app, "background", False):
        registry.add_timer("deck_modal", _ensure_running, 0.1)
