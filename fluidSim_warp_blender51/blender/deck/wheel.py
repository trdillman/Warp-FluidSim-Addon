"""Radial wheel: six slots per workflow stage, the stage shown and cycled at the hub.

``fluidsim.radial_wheel`` (Alt+Right Mouse in the 3D View) opens a ring around
the pointer. Each slot is a :class:`controls.Control` from
:data:`controls.STAGE_SLOTS`, so a slot calls the operator or toggles the
property the panels draw. Previous/Next buttons, arrow keys and scrolling change
``scene.fluidsim.deck_stage``, the same stage the viewport header shows. Clicking
the hub also cycles forward. Releasing the mouse button over a
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
NAV_BOTTOM = -170.0
NAV_TOP = -144.0
FOOTER_EXTENT = 192.0
PAN_THRESHOLD = 40.0


def navigation_rects(scale=1.0):
    """Button bounds relative to the hub, shared by drawing and pointer routing."""
    return {name: tuple(value * scale for value in rect) for name, rect in (
        ("PREVIOUS", (-126.0, NAV_BOTTOM, -30.0, NAV_TOP)),
        ("NEXT", (30.0, NAV_BOTTOM, 126.0, NAV_TOP)),
    )}


def adjacent_stage(stage, step):
    return controls.STAGES[(controls.STAGES.index(stage) + step) % len(controls.STAGES)]


def slot_centre(index: int, count: int = 6):
    """Offset of slot ``index`` from the hub: slot 0 at the top, then clockwise."""
    angle = math.pi / 2.0 - 2.0 * math.pi * index / count
    return RING_RADIUS * math.cos(angle), RING_RADIUS * math.sin(angle)


def slot_at(dx: float, dy: float, scale: float = 1.0, count: int = 6):
    """Navigation button, ``"HUB"``, slot index, or ``None`` at a hub offset."""
    for name, (x0, y0, x1, y1) in navigation_rects(scale).items():
        if x0 <= dx <= x1 and y0 <= dy <= y1:
            return name
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
    floor = (FOOTER_EXTENT + bar.BOTTOM + bar.HEIGHT + bar.GAP) * scale
    cx = min(max(x, extent), max(extent, float(width) - extent))
    cy = min(max(y, floor), max(floor, float(height) - extent))
    return cx, cy


def tooltip_y(cy, region_height, scale, height):
    """Fit above the ring or below the entire footer; omit when neither fits."""
    gap = bar.GAP * scale
    above = cy + (RING_RADIUS + SLOT_RADIUS + 8.0) * scale + gap
    if above + height <= region_height:
        return above
    below = cy - FOOTER_EXTENT * scale - gap - height
    floor = (bar.BOTTOM + bar.HEIGHT + bar.GAP) * scale
    return below if below >= floor else None


def stage_controls(context, stage):
    """The stage's six slots as they stand now (Play and Bake follow the state)."""
    return tuple(controls.resolve(context, c) for c in STAGE_SLOTS[stage])


def slot_tip(stage, hover, statuses, slots=None):
    """``(title, body, note)`` for the hovered hub or slot; empty strings when none."""
    if hover in {"PREVIOUS", "NEXT"}:
        destination = controls.STAGE_LABELS[adjacent_stage(stage, -1 if hover == "PREVIOUS" else 1)]
        return ("Previous page" if hover == "PREVIOUS" else "Next page",
                f"Show the {destination} actions", "Use arrow keys or scroll to change pages")
    if hover == "HUB":
        following = controls.STAGE_LABELS[controls.next_stage(stage)]
        return (f"Stage: {controls.STAGE_LABELS[stage]}",
                f"Click to show the {following} actions", "")
    if not isinstance(hover, int):
        return "", "", ""
    control = (slots or STAGE_SLOTS[stage])[hover]
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
        operator.painted = None
        palette = tokens.current_tokens(context)
        scale = bar._ui_scale(context)
        cx, cy = operator.centre
        stage = current_stage(context.scene)
        size = 11.0 * scale
        _draw.circle(cx, cy, (RING_RADIUS + SLOT_RADIUS + 8.0) * scale,
                     tokens.rgba(palette.surface, tokens.CARD_ALPHA * 0.85), 64)
        slots = stage_controls(context, stage)
        statuses = [controls.control_status(context, c) for c in slots]
        for index, control in enumerate(slots):
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
        for target, (x0, y0, x1, y1) in navigation_rects(scale).items():
            fill = palette.accent if operator.hover == target else palette.surface
            _draw.rounded_rect(cx + x0, cy + y0, cx + x1, cy + y1, 5 * scale,
                               tokens.rgba(fill, tokens.CARD_ALPHA))
            label = "< Previous" if target == "PREVIOUS" else "Next >"
            width = _draw.text_width(label, size)
            _draw.text(cx + (x0 + x1 - width) / 2, cy + (y0 + y1) / 2 - size / 3,
                       label, size, tokens.rgba(palette.text))
        label = f"{controls.STAGES.index(stage) + 1}/{len(controls.STAGES)}"
        width = _draw.text_width(label, size)
        _draw.text(cx - width / 2, cy + (NAV_BOTTOM + NAV_TOP) / 2 * scale - size / 3,
                   label, size, tokens.rgba(palette.text))
        label = "Arrow keys / scroll: change page"
        width = _draw.text_width(label, size * 0.85)
        _draw.text(cx - width / 2, cy - 186 * scale, label, size * 0.85,
                   tokens.rgba(palette.text_dim))
        title, body, note = slot_tip(stage, operator.hover, statuses, slots)
        if title:
            height = bar.tip_card_height(scale, size, note)
            y0 = tooltip_y(cy, float(context.region.height), scale, height)
            if y0 is not None:
                bar.draw_tip_card(context, cx, y0, title, body, palette, scale, size, note)
        # Publish only after the whole paint succeeds. Actions still check current availability.
        operator.painted = (context.scene, context.region.width, context.region.height,
                            scale, operator.centre, slots)
    except Exception as exc:  # noqa: BLE001 - a draw callback must never raise
        print(f"FluidSim Liquid Deck: wheel draw failed: {exc!r}")


class FLUIDSIM_OT_radial_wheel(bpy.types.Operator):
    """Open the FluidSim radial wheel at the pointer."""

    bl_idname = "fluidsim.radial_wheel"
    bl_label = "FluidSim Radial Wheel"
    bl_description = (
        "Open a radial wheel with six actions per page; use Previous/Next, arrow "
        "keys or scrolling to change pages. Click outside or press Esc to close")

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
        self.painted = None
        self.pan_distance = 0.0
        # A wheel moved off the pointer (opened from the bar) has no press-drag-release pick.
        self.opened_with = event.type if self.centre == pointer else ""
        # Replacing the handler retires a wheel still open in another window.
        self.handle = REGISTRY.add_draw_handler("wheel", bpy.types.SpaceView3D, _draw_wheel,
                                                (self,), "WINDOW", "POST_PIXEL")
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _close(self, context):
        self.painted = None
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
        if target in {"HUB", "PREVIOUS", "NEXT"}:
            scene.fluidsim.deck_stage = adjacent_stage(
                current_stage(scene), -1 if target == "PREVIOUS" else 1)
            return None
        painted = getattr(self, "painted", None)
        if (painted is None or painted[0] is not scene
                or context.region is not self.region
                or painted[1:5] != (context.region.width, context.region.height,
                                    bar._ui_scale(context), self.centre)):
            return None
        control = painted[5][target]
        ready, reason = controls.control_status(context, control)
        if not ready:
            # Keep the wheel open and say why, rather than closing and doing nothing.
            self.report({"WARNING"}, f"{control.label}: {reason}")
            return None
        self._close(context)
        controls.run_control(context, control, report=self.report)
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
        if event.type == "MOUSEPAN":
            dx = getattr(event, "mouse_x", 0) - getattr(event, "mouse_prev_x", 0)
            dy = getattr(event, "mouse_y", 0) - getattr(event, "mouse_prev_y", 0)
            if (any(getattr(event, key, False) for key in ("shift", "ctrl", "alt", "oskey"))
                    or abs(dy) <= abs(dx)
                    or not (0 <= x < self.region.width and 0 <= y < self.region.height)):
                self.pan_distance = 0.0
                return {"RUNNING_MODAL"}
            if self.pan_distance * dy < 0:
                self.pan_distance = 0.0
            self.pan_distance += dy
            if abs(self.pan_distance) >= PAN_THRESHOLD * scale:
                self._pick(context, "PREVIOUS" if self.pan_distance > 0 else "NEXT")
                # At most one page per event; do not carry momentum into the next page.
                self.pan_distance = 0.0
            return {"RUNNING_MODAL"}
        if event.value == "PRESS":
            self.pan_distance = 0.0
            if event.type in {"LEFT_ARROW", "PAGE_UP", "WHEELUPMOUSE"}:
                self._pick(context, "PREVIOUS")
                return {"RUNNING_MODAL"}
            if event.type in {"RIGHT_ARROW", "PAGE_DOWN", "WHEELDOWNMOUSE"}:
                self._pick(context, "NEXT")
                return {"RUNNING_MODAL"}
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
