"""The Deck bar: a bottom-anchored card with transport, readout, pending chip and bake ring.

Nothing here holds settings state. Transport buttons are the operators
``panels.draw_transport`` draws (:func:`controls.transport_controls`); the chip
reads ``pending_settings.pending_presentation`` and offers the panel's Apply
(``fluidsim.apply_changes``) and Revert (``fluidsim.revert_pending``); the bake
ring reads ``production_bake.active_bakes`` and the scene's ``bake_progress``.

The only memory is presentation: when a Deck control writes a setting that
applies live, :func:`note_deck_write` timestamps it so the chip can say
"LIVE" for :data:`tokens.LIVE_CHIP_SECONDS`, and a short ms/frame history feeds
the sparkline. A RESET or REINIT chip comes from Pending Settings and stays
until Apply or Revert.

Layout is a pure function of the region size and the state, so the modal hit
test (``modal.py``) and the draw callback always agree on where things are.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from . import controls, tokens

_FLASHES: dict = {}
_SAMPLES: dict = {}
SPARKLINE_SAMPLES = 48
_last_draw_error = ""


def _now() -> float:
    return time.monotonic()


def _scene_key(scene):
    from ..scene_access import scene_identity

    return scene_identity(scene)


def note_deck_write(scene, field_name: str) -> None:
    """Record that a Deck control just wrote ``field_name`` (drives the LIVE chip)."""
    _FLASHES[_scene_key(scene)] = (str(field_name), _now())


def forget_scene_state() -> None:
    """Drop presentation memory (file load, unregister)."""
    _FLASHES.clear()
    _SAMPLES.clear()


@dataclass(frozen=True)
class Chip:
    kind: str  # "NONE" | "LIVE" | "RESET_PENDING" | "REINIT_PENDING"
    text: str
    actions: tuple = ()  # (operator id, label) pairs
    detail: str = ""


def chip_state(scene, now: float | None = None) -> Chip:
    """The pending chip for ``scene``: Pending Settings first, then a recent live write."""
    from ..pending_settings import pending_presentation

    pending = pending_presentation(scene)
    action = str(pending.get("action", "NONE"))
    if action != "NONE":
        return Chip(
            kind=action,
            text=str(pending.get("summary", "")) or action.replace("_", " ").title(),
            actions=tuple((c.target, c.label) for c in controls.PENDING_ACTIONS),
            detail="\n".join(pending.get("reasons", ())),
        )
    flash = _FLASHES.get(_scene_key(scene))
    now = _now() if now is None else now
    if flash is not None and 0.0 <= now - flash[1] <= tokens.LIVE_CHIP_SECONDS:
        return Chip(kind="LIVE", text=f"LIVE · {flash[0]}", detail="Applied live")
    return Chip(kind="NONE", text="")


def bake_fraction(scene):
    """Completed fraction of this scene's running Production Bake, or ``None``."""
    from .. import production_bake

    for baking_scene, _controller in production_bake.active_bakes():
        if baking_scene == scene:
            try:
                return max(0.0, min(1.0, float(scene.fluidsim.bake_progress)))
            except (AttributeError, TypeError, ValueError):
                return 0.0
    return None


@dataclass(frozen=True)
class Readout:
    frame: int
    particles: int
    substeps: int
    step_ms: float
    samples: tuple = ()


def sample_readout(scene) -> Readout:
    """Session stats the solver already writes (``scene_bridge``), plus the ms history."""
    fs = scene.fluidsim
    frame = int(getattr(scene, "frame_current", 0))
    step_ms = float(getattr(fs, "stat_step_ms", 0.0) or 0.0)
    history = _SAMPLES.setdefault(_scene_key(scene), [None, deque(maxlen=SPARKLINE_SAMPLES)])
    if history[0] != frame and step_ms > 0.0:
        history[0] = frame
        history[1].append(step_ms)
    return Readout(frame=frame, particles=int(getattr(fs, "stat_particles", 0) or 0),
                   substeps=int(getattr(fs, "stat_substeps", 0) or 0), step_ms=step_ms,
                   samples=tuple(history[1]))



# Layout -----------------------------------------------------------------------

PAD = 8.0
GAP = 6.0
HEIGHT = 44.0
BOTTOM = 14.0
BUTTON_HEIGHT = 30.0
ICON = 14.0  # icon box inside a button
ICON_ONLY = 32.0  # compact button width
SEPARATOR = 1.0
READOUT = 268.0
READOUT_COMPACT = 160.0
SPARKLINE = 72.0
CHIP_ACTION = 56.0
CHAR = 6.5  # estimated glyph width at the bar text size, for layout only

# Hover tips for the bar parts that are not buttons: (title, body).
INFO_TIPS = {
    "readout": ("Live stats",
                "Current frame, live particles, solver substeps and milliseconds per frame"),
    "sparkline": ("Step time", "Milliseconds per frame over the last 48 frames; lower is faster"),
    "chip": ("Pending changes", "Edits that need a reset or re-initialize wait here: "
             "Apply runs it, Revert puts the old values back. Live edits flash green"),
}


@dataclass(frozen=True)
class Item:
    rect: tuple  # (x0, y0, x1, y1), region pixels
    role: str  # "button" | "chip_action" | "readout" | "sparkline" | "chip" | "separator"
    control: controls.Control | None = None
    label: str = ""
    tip: tuple = ()  # (title, body) shown on hover



@dataclass(frozen=True)
class BarLayout:
    card: tuple
    items: tuple = field(default_factory=tuple)
    compact: bool = False

    def hits(self):
        """Items the pointer can hover (buttons and the informational parts)."""
        return tuple(item for item in self.items if item.control is not None or item.tip)


def button_width(control, compact: bool) -> float:
    if compact:
        return ICON_ONLY
    return PAD + ICON + GAP + CHAR * len(control.label) + PAD


def _parts(transport, chip: Chip, tools, compact: bool):
    parts = [("button", c, button_width(c, compact)) for c in transport]
    parts.append(("separator", None, SEPARATOR))
    parts.append(("readout", None, READOUT_COMPACT if compact else READOUT))
    parts.append(("sparkline", None, SPARKLINE))
    parts.append(("separator", None, SEPARATOR))
    parts.append(("chip", None, max(80.0, CHAR * len(chip.text or "No pending changes")
                                     + 2 * PAD + 12.0)))
    parts.extend(("chip_action", next(c for c in controls.PENDING_ACTIONS
                                      if c.target == target), CHIP_ACTION)
                 for target, _label in chip.actions)
    parts.append(("separator", None, SEPARATOR))
    parts.extend(("button", c, button_width(c, compact)) for c in tools)
    return parts


def _width(parts, s):
    return sum(w for _r, _c, w in parts) * s + GAP * s * (len(parts) - 1) + 2 * PAD * s


def _tip(role, control, chip):
    if control is not None:
        return (control.label, control.tooltip)
    if role == "chip" and chip.detail:
        return (chip.text, chip.detail)
    return INFO_TIPS.get(role, ())


def bar_layout(region_width: float, scale: float, transport, chip: Chip,
               bake=None, tools=None) -> BarLayout:
    """Place the bar's parts left to right, centred at the bottom of the region.

    Buttons carry an icon and their full label; when that does not fit the region
    the bar falls back to icon-only buttons (the hover tip still names each one).
    """
    s = max(0.5, float(scale))
    tools = controls.TOOL_CONTROLS if tools is None else tuple(tools)
    compact = False
    parts = _parts(transport, chip, tools, compact)
    if _width(parts, s) > float(region_width) - 2 * GAP * s:
        compact = True
        parts = _parts(transport, chip, tools, compact)
    width = _width(parts, s)
    x = max(0.0, (float(region_width) - width) / 2.0)
    y0, y1 = BOTTOM * s, (BOTTOM + HEIGHT) * s
    card = (x, y0, x + width, y1)
    x += PAD * s
    inset = (HEIGHT - BUTTON_HEIGHT) / 2.0 * s
    items = []
    for role, control, w in parts:
        rect = (x, y0 + inset, x + w * s, y1 - inset)
        items.append(Item(rect, role, control, control.label if control else "",
                          _tip(role, control, chip)))
        x += (w + GAP) * s
    return BarLayout(card=card, items=tuple(items), compact=compact)


def _ui_scale(context) -> float:
    try:
        return float(context.preferences.system.ui_scale)
    except (AttributeError, TypeError, ValueError):
        return 1.0


def current_layout(context):
    """The bar layout for this viewport region, or ``None`` when the Deck is hidden."""
    from . import deck_visible
    from ..ui_common import activity_ui_state

    scene = getattr(context, "scene", None)
    region = getattr(context, "region", None)
    if scene is None or region is None or not deck_visible(context, scene):
        return None
    state = activity_ui_state(scene)
    screen = getattr(context, "screen", None)
    playing = bool(getattr(screen, "is_animation_playing", False))
    transport = controls.transport_controls(str(state.get("id", "IDLE")), playing,
                                            bool(state.get("owns_scene")))
    return bar_layout(region.width, _ui_scale(context), transport, chip_state(scene),
                      bake_fraction(scene))


def hit_test(layout: BarLayout | None, x: float, y: float):
    """``(over_deck, item)`` for a region-space point; ``item.control`` may be ``None``."""
    if layout is None:
        return False, None
    for item in layout.hits():
        x0, y0, x1, y1 = item.rect
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True, item
    x0, y0, x1, y1 = layout.card
    return (x0 <= x <= x1 and y0 <= y <= y1), None


# Drawing ----------------------------------------------------------------------

def _chip_colour(palette, kind):
    return {"LIVE": palette.live, "RESET_PENDING": palette.reset,
            "REINIT_PENDING": palette.reinit}.get(kind, palette.text_dim)


def _draw_icon(icon, cx, cy, r, colour, accent, fraction=None):
    """A small vector icon for ``icon`` (the control's Blender icon name)."""
    from . import _draw

    if icon == "PLAY":
        _draw.triangle((cx - 0.6 * r, cy - 0.8 * r), (cx - 0.6 * r, cy + 0.8 * r),
                       (cx + 0.8 * r, cy), colour)
    elif icon == "PAUSE":
        _draw.rect(cx - 0.7 * r, cy - 0.8 * r, cx - 0.2 * r, cy + 0.8 * r, colour)
        _draw.rect(cx + 0.2 * r, cy - 0.8 * r, cx + 0.7 * r, cy + 0.8 * r, colour)
    elif icon == "CANCEL":
        _draw.rect(cx - 0.65 * r, cy - 0.65 * r, cx + 0.65 * r, cy + 0.65 * r, colour)
    elif icon == "FRAME_NEXT":
        _draw.triangle((cx - 0.8 * r, cy - 0.8 * r), (cx - 0.8 * r, cy + 0.8 * r),
                       (cx + 0.4 * r, cy), colour)
        _draw.rect(cx + 0.45 * r, cy - 0.8 * r, cx + 0.8 * r, cy + 0.8 * r, colour)
    elif icon == "FILE_REFRESH":
        _draw.ring(cx, cy, 0.7 * r, 0.28 * r, 0.78, colour)
        _draw.triangle((cx - 0.15 * r, cy + 0.35 * r), (cx - 0.15 * r, cy + 1.05 * r),
                       (cx - 0.65 * r, cy + 0.7 * r), colour)
    elif icon == "SHADING_RENDERED":
        _draw.ring(cx, cy, 0.8 * r, 0.22 * r, 1.0, colour)
        _draw.circle(cx - 0.15 * r, cy + 0.15 * r, 0.45 * r, colour, 20)
    elif icon == "VIEWZOOM":
        _draw.ring(cx - 0.15 * r, cy + 0.15 * r, 0.55 * r, 0.25 * r, 1.0, colour)
        _draw.polyline([(cx + 0.25 * r, cy - 0.25 * r), (cx + 0.85 * r, cy - 0.85 * r)],
                       colour, 0.3 * r)
    elif icon == "PROP_CON":
        _draw.ring(cx, cy, 0.8 * r, 0.22 * r, 1.0, colour)
        _draw.circle(cx, cy, 0.3 * r, colour, 16)
    elif icon == "RENDER_ANIMATION":
        if fraction is None:
            _draw.circle(cx, cy, 0.55 * r, colour, 20)
            _draw.ring(cx, cy, 0.85 * r, 0.2 * r, 1.0, colour)
        else:
            _draw.ring(cx, cy, 0.8 * r, 0.3 * r, 1.0, tokens.rgba(colour[:3], 0.3))
            _draw.ring(cx, cy, 0.8 * r, 0.3 * r, fraction, accent)
    else:
        _draw.circle(cx, cy, 0.5 * r, colour, 16)


def _draw_tip(context, layout, item, palette, s, size):
    """A two-line tip card above the hovered item, kept inside the region."""
    from . import _draw

    if not item.tip:
        return
    title, body = item.tip
    pad = PAD * s
    width = max(_draw.text_width(title, size), _draw.text_width(body, size * 0.9)) + 2 * pad
    height = size * 2.0 + size * 0.9 + 1.5 * pad
    region_width = float(getattr(getattr(context, "region", None), "width", width))
    x0 = (item.rect[0] + item.rect[2] - width) / 2.0
    x0 = max(GAP * s, min(x0, region_width - width - GAP * s))
    y0 = layout.card[3] + GAP * s
    radius = tokens.RADIUS_SCALE[1] * s
    _draw.rounded_rect(x0, y0, x0 + width, y0 + height, radius,
                       tokens.rgba(palette.surface, tokens.CARD_ALPHA))
    _draw.text(x0 + pad, y0 + height - pad - size * 0.8, title, size, tokens.rgba(palette.text))
    _draw.text(x0 + pad, y0 + pad, body, size * 0.9, tokens.rgba(palette.text_dim))


def _paint(context, layout, palette, hover):
    from . import _draw

    scene = context.scene
    s = _ui_scale(context)
    size = 11.0 * s
    radius_chip, radius_control, radius_card = (r * s for r in tokens.RADIUS_SCALE)
    _draw.rounded_rect(*layout.card, radius_card, tokens.rgba(palette.surface, tokens.CARD_ALPHA))
    readout = sample_readout(scene)
    chip = chip_state(scene)
    bake = bake_fraction(scene)
    for item in layout.items:
        x0, y0, x1, y1 = item.rect
        mid_y = (y0 + y1) / 2.0 - size / 3.0
        if item.role in {"button", "chip_action"}:
            control = item.control
            enabled = controls.control_enabled(context, control)
            on = control in {controls.PAUSE} or controls.control_active(context, control)
            hovered = hover is not None and hover.rect == item.rect and enabled
            fill = palette.accent if (hovered or on) else palette.text_dim
            _draw.rounded_rect(x0, y0, x1, y1, radius_control,
                               tokens.rgba(fill, (0.40 if hovered else 0.30) if enabled else 0.10))
            colour = tokens.rgba(palette.text if enabled else palette.text_dim)
            if item.role == "chip_action":
                width = _draw.text_width(control.label, size)
                _draw.text((x0 + x1 - width) / 2.0, mid_y, control.label, size, colour)
                continue
            fraction = bake if control is controls.BAKE and bake is not None else None
            r = ICON / 2.0 * s
            if layout.compact:
                _draw_icon(control.icon, (x0 + x1) / 2.0, (y0 + y1) / 2.0, r, colour,
                           tokens.rgba(palette.accent), fraction)
            else:
                _draw_icon(control.icon, x0 + PAD * s + r, (y0 + y1) / 2.0, r, colour,
                           tokens.rgba(palette.accent), fraction)
                label = f"{round(fraction * 100)}%" if fraction is not None else control.label
                _draw.text(x0 + (PAD + ICON + GAP) * s, mid_y, label, size, colour)
        elif item.role == "separator":
            _draw.rect(x0, y0 + 2 * s, x1, y1 - 2 * s, tokens.rgba(palette.text_dim, 0.35))
        elif item.role == "readout":
            if layout.compact:
                line = (f"f{readout.frame}  {readout.particles:,} pts  "
                        f"{readout.substeps} sub  {readout.step_ms:.1f} ms")
            else:
                line = (f"frame {readout.frame}   {readout.particles:,} particles   "
                        f"{readout.substeps} substeps   {readout.step_ms:.1f} ms")
            _draw.text(x0, mid_y, line, size * 0.9, tokens.rgba(palette.text))
        elif item.role == "sparkline":
            _draw.rect(x0, y0, x1, y0 + 1.0 * s, tokens.rgba(palette.text_dim, 0.35))
            if len(readout.samples) > 1:
                peak = max(readout.samples) or 1.0
                step = (x1 - x0) / (len(readout.samples) - 1)
                points = [(x0 + i * step, y0 + (y1 - y0) * (value / peak))
                          for i, value in enumerate(readout.samples)]
                _draw.polyline(points, tokens.rgba(palette.accent), 1.5 * s)
        elif item.role == "chip":
            colour = _chip_colour(palette, chip.kind)
            _draw.rounded_rect(x0, y0, x1, y1, radius_chip,
                               tokens.rgba(colour, 0.10 if chip.kind == "NONE" else 0.22))
            _draw.circle(x0 + PAD * s + 3 * s, (y0 + y1) / 2.0, 3.5 * s, tokens.rgba(colour), 12)
            _draw.text(x0 + PAD * s + 12 * s, mid_y, chip.text or "No pending changes", size,
                       tokens.rgba(colour))
    from .registry import REGISTRY

    # The palette and wheel draw above the bar; a stale hover tip would cover them.
    if hover is not None and not any(REGISTRY.draw_handle(k) for k in ("palette", "wheel")):
        _draw_tip(context, layout, hover, palette, s, size)

def draw_deck():
    """``POST_PIXEL`` callback for every 3D View: the bar, then gizmo labels."""
    global _last_draw_error
    import bpy

    context = bpy.context
    try:
        layout = current_layout(context)
        if layout is None:
            return
        palette = tokens.current_tokens(context)
        from . import gizmos, modal

        _paint(context, layout, palette, modal.hover_item(context))
        gizmos.draw_overlay(context, palette)
    except Exception as exc:  # noqa: BLE001 - a draw callback must never raise
        message = repr(exc)
        if message != _last_draw_error:
            _last_draw_error = message
            print(f"FluidSim Liquid Deck: draw failed: {message}")
