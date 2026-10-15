"""Radial wheel: six slots per workflow stage, the stage shown and cycled at the hub.

``fluidsim.radial_wheel`` (Alt+Right Mouse in the 3D View) opens a ring around
the pointer. Each slot is a :class:`controls.Control` from
:data:`controls.STAGE_SLOTS`, so a slot calls the operator or toggles the
property the panels draw. Clicking the hub cycles ``scene.fluidsim.deck_stage``,
the same stage the viewport header shows. Releasing the mouse button over a
slot runs it (press-drag-release), or click a slot after release. A click
outside the ring, or Esc, closes the wheel.
"""

from __future__ import annotations

import math

import bpy

from . import _draw, bar, controls, tokens

STAGE_SLOTS = controls.STAGE_SLOTS
HUB_RADIUS = 30.0
RING_RADIUS = 92.0
SLOT_RADIUS = 26.0


def slot_centre(index: int, count: int = 6):
    """Offset of slot ``index`` from the hub: slot 0 at the top, then clockwise."""
    angle = math.pi / 2.0 - 2.0 * math.pi * index / count
    return RING_RADIUS * math.cos(angle), RING_RADIUS * math.sin(angle)


def slot_at(dx: float, dy: float, scale: float = 1.0, count: int = 6):
    """``"HUB"``, a slot index, or ``None`` for a pointer offset from the hub."""
    distance = math.hypot(dx, dy)
    if distance <= HUB_RADIUS * scale:
        return "HUB"
    if distance > (RING_RADIUS + SLOT_RADIUS * 1.6) * scale:
        return None
    angle = (math.pi / 2.0 - math.atan2(dy, dx)) % (2.0 * math.pi)
    return int((angle + math.pi / count) // (2.0 * math.pi / count)) % count


def current_stage(scene) -> str:
    stage = str(getattr(getattr(scene, "fluidsim", None), "deck_stage", "SETUP"))
    return stage if stage in STAGE_SLOTS else controls.STAGES[0]


def _draw_wheel(operator):
    context = bpy.context
    try:
        if getattr(context, "region", None) != operator.region:
            return
        palette = tokens.current_tokens(context)
        scale = bar._ui_scale(context)
        cx, cy = operator.centre
        stage = current_stage(context.scene)
        size = 11.0 * scale
        _draw.circle(cx, cy, (RING_RADIUS + SLOT_RADIUS + 8.0) * scale,
                     tokens.rgba(palette.surface, tokens.CARD_ALPHA * 0.85), 64)
        for index, control in enumerate(STAGE_SLOTS[stage]):
            ox, oy = slot_centre(index)
            x, y = cx + ox * scale, cy + oy * scale
            enabled = controls.control_enabled(context, control)
            on = controls.control_active(context, control)
            fill = palette.accent if (index == operator.hover or on) and enabled else palette.text_dim
            _draw.circle(x, y, SLOT_RADIUS * scale,
                         tokens.rgba(fill, 0.35 if enabled else 0.12))
            label = control.label
            width = _draw.text_width(label, size * 0.85)
            _draw.text(x - width / 2.0, y - size / 3.0, label, size * 0.85,
                       tokens.rgba(palette.text if enabled else palette.text_dim))
        hub = palette.accent if operator.hover == "HUB" else palette.text_dim
        _draw.circle(cx, cy, HUB_RADIUS * scale, tokens.rgba(hub, 0.4))
        label = controls.STAGE_LABELS[stage]
        width = _draw.text_width(label, size)
        _draw.text(cx - width / 2.0, cy - size / 3.0, label, size, tokens.rgba(palette.text))
    except Exception as exc:  # noqa: BLE001 - a draw callback must never raise
        print(f"FluidSim Liquid Deck: wheel draw failed: {exc!r}")


class FLUIDSIM_OT_radial_wheel(bpy.types.Operator):
    """Open the FluidSim radial wheel at the pointer."""

    bl_idname = "fluidsim.radial_wheel"
    bl_label = "FluidSim Radial Wheel"
    bl_description = (
        "Open a radial wheel with six actions for the current workflow stage; click "
        "the hub to change the stage. Click outside to close")

    @classmethod
    def poll(cls, context):
        from . import deck_visible

        return (getattr(getattr(context, "area", None), "type", "") == "VIEW_3D"
                and deck_visible(context))

    def invoke(self, context, event):
        from .registry import REGISTRY

        self.centre = (float(event.mouse_region_x), float(event.mouse_region_y))
        self.hover = None
        self.region = context.region
        self.opened_with = event.type
        # Replacing the handler retires a wheel still open in another window.
        self.handle = REGISTRY.add_draw_handler("wheel", bpy.types.SpaceView3D, _draw_wheel,
                                                (self,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _close(self, context):
        from .registry import REGISTRY

        if REGISTRY.draw_handle("wheel") is getattr(self, "handle", None):
            try:
                REGISTRY.remove_draw_handler("wheel")
            except Exception as exc:  # noqa: BLE001
                print(f"FluidSim Liquid Deck: could not remove the wheel draw handler: {exc!r}")
        if getattr(context, "area", None) is not None:
            context.area.tag_redraw()

    def _pick(self, context, target):
        scene = context.scene
        if target == "HUB":
            scene.fluidsim.deck_stage = controls.next_stage(current_stage(scene))
            return None
        control = STAGE_SLOTS[current_stage(scene)][target]
        self._close(context)
        controls.run_control(context, control)
        return {"FINISHED"}

    def modal(self, context, event):
        from .registry import REGISTRY

        area = getattr(context, "area", None)
        if area is None or REGISTRY.draw_handle("wheel") is not self.handle:
            # The area closed, or a newer wheel took the handler: finish, keep the event.
            self._close(context)
            return {"CANCELLED", "PASS_THROUGH"}
        area.tag_redraw()
        scale = bar._ui_scale(context)
        target = slot_at(event.mouse_region_x - self.centre[0],
                         event.mouse_region_y - self.centre[1], scale)
        self.hover = target
        if event.type == "MOUSEMOVE":
            return {"RUNNING_MODAL"}
        if event.type == "ESC" and event.value == "PRESS":
            self._close(context)
            return {"CANCELLED"}
        if event.type == self.opened_with and event.value == "RELEASE":
            # Press-drag-release onto a slot runs it; releasing elsewhere keeps the wheel.
            if isinstance(target, int):
                return self._pick(context, target)
            return {"RUNNING_MODAL"}
        if event.type in {"LEFTMOUSE", "RIGHTMOUSE"} and event.value == "PRESS":
            if target is None:
                self._close(context)
                return {"CANCELLED"}
            return self._pick(context, target) or {"RUNNING_MODAL"}
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._close(context)


CLASSES = (FLUIDSIM_OT_radial_wheel,)
