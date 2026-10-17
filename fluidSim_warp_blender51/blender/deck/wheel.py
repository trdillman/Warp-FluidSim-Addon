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


def pointer_in_region(event, region):
    """Region-space pointer; window-relative so it is right when invoked from another handler."""
    try:
        return float(event.mouse_x - region.x), float(event.mouse_y - region.y)
    except (AttributeError, TypeError):
        return float(event.mouse_region_x), float(event.mouse_region_y)


def clamp_centre(x, y, width, height, scale=1.0):
    """Keep the whole wheel inside the region and above the Deck bar (it opens from there too)."""
    extent = (RING_RADIUS + SLOT_RADIUS + 8.0) * scale
    floor = extent + (bar.BOTTOM + bar.HEIGHT + bar.GAP) * scale
    cx = min(max(x, extent), max(extent, float(width) - extent))
    cy = min(max(y, floor), max(floor, float(height) - extent))
    return cx, cy


def slot_tip(stage, hover, statuses):
    """``(title, body, note)`` for the hovered hub or slot; empty strings when none."""
    if hover == "HUB":
        following = controls.STAGE_LABELS[controls.next_stage(stage)]
        return (f"Stage: {controls.STAGE_LABELS[stage]}",
                f"Click to show the {following} actions", "")
    if not isinstance(hover, int):
        return "", "", ""
    control = STAGE_SLOTS[stage][hover]
    ready, reason = statuses[hover]
    return control.label, control.tooltip, "" if ready else f"Unavailable: {reason}"


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
        statuses = [controls.control_status(context, c) for c in STAGE_SLOTS[stage]]
        for index, control in enumerate(STAGE_SLOTS[stage]):
            ox, oy = slot_centre(index)
            x, y = cx + ox * scale, cy + oy * scale
            enabled = statuses[index][0]
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
        title, body, note = slot_tip(stage, operator.hover, statuses)
        if title:
            # Above the wheel when it fits, else below it.
            extent = (RING_RADIUS + SLOT_RADIUS + 8.0) * scale
            height = bar.tip_card_height(scale, size, note)
            y0 = cy + extent + bar.GAP * scale
            if y0 + height > getattr(context.region, "height", y0 + height):
                y0 = cy - extent - bar.GAP * scale - height
            bar.draw_tip_card(context, cx, y0, title, body, palette, scale, size, note)
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

        self.region = context.region
        pointer = pointer_in_region(event, self.region)
        self.centre = clamp_centre(*pointer, getattr(self.region, "width", 0),
                                   getattr(self.region, "height", 0), bar._ui_scale(context))
        self.hover = None
        # A wheel moved off the pointer (opened from the bar) has no press-drag-release pick.
        self.opened_with = event.type if self.centre == pointer else ""
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
        ready, reason = controls.control_status(context, control)
        if not ready:
            # Keep the wheel open and say why, rather than closing and doing nothing.
            self.report({"WARNING"}, f"{control.label}: {reason}")
            return None
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
        x, y = pointer_in_region(event, self.region)
        target = slot_at(x - self.centre[0], y - self.centre[1], scale)
        self.hover = target
        if event.type == "MOUSEMOVE":
            return {"RUNNING_MODAL"}
        if event.type == "ESC" and event.value == "PRESS":
            self._close(context)
            return {"CANCELLED"}
        if event.type == self.opened_with and event.value == "RELEASE":
            # Press-drag-release onto a slot runs it; releasing elsewhere keeps the wheel.
            if isinstance(target, int):
                return self._pick(context, target) or {"RUNNING_MODAL"}
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
