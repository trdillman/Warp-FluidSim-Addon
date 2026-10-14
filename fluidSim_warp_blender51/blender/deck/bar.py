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

BUTTON = 32.0
GAP = 6.0
PAD = 8.0
HEIGHT = 44.0
BOTTOM = 14.0
READOUT = 150.0
SPARKLINE = 72.0
CHIP_ACTION = 56.0
RING = 32.0
CHAR = 6.5  # estimated glyph width at the bar text size, for layout only


@dataclass(frozen=True)
class Item:
    rect: tuple  # (x0, y0, x1, y1), region pixels
    role: str  # "button" | "chip_action" | "readout" | "sparkline" | "chip" | "ring"
    control: controls.Control | None = None
    label: str = ""


@dataclass(frozen=True)
class BarLayout:
    card: tuple
    items: tuple = field(default_factory=tuple)

    def hits(self):
        """Items a click acts on, then the card itself (clicks on it are consumed)."""
        return tuple(item for item in self.items if item.control is not None)


def bar_layout(region_width: float, scale: float, transport, chip: Chip,
               bake, preview_button: bool) -> BarLayout:
    """Place the bar's parts left to right, centred at the bottom of the region."""
    s = max(0.5, float(scale))
    parts = [("button", c, BUTTON) for c in transport]
    parts.append(("readout", None, READOUT))
    parts.append(("sparkline", None, SPARKLINE))
    if chip.kind != "NONE":
        parts.append(("chip", None, max(80.0, CHAR * len(chip.text) + 2 * PAD)))
        parts.extend(("chip_action", next(c for c in controls.PENDING_ACTIONS
                                          if c.target == target), CHIP_ACTION)
                     for target, _label in chip.actions)
    if bake is not None:
        parts.append(("ring", None, RING))
    if preview_button:
        parts.append(("button", controls.LIQUID_PREVIEW, BUTTON))
    width = sum(w for _r, _c, w in parts) * s + GAP * s * (len(parts) - 1) + 2 * PAD * s
    x = max(0.0, (float(region_width) - width) / 2.0)
    y0, y1 = BOTTOM * s, (BOTTOM + HEIGHT) * s
    card = (x, y0, x + width, y1)
    x += PAD * s
    items = []
    for role, control, w in parts:
        inset = (HEIGHT - BUTTON) / 2.0 * s
        rect = (x, y0 + inset, x + w * s, y1 - inset)
        items.append(Item(rect, role, control, control.label if control else ""))
        x += (w + GAP) * s
    return BarLayout(card=card, items=tuple(items))


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
                      bake_fraction(scene), True)


def hit_test(layout: BarLayout | None, x: float, y: float):
    """``(over_deck, item)`` for a region-space point."""
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


def _paint(context, layout, palette, hover):
    from . import _draw

    scene = context.scene
    s = _ui_scale(context)
    size = 11.0 * s
    radius_chip, radius_control, radius_card = (r * s for r in tokens.RADIUS_SCALE)
    _draw.rounded_rect(*layout.card, radius_card, tokens.rgba(palette.surface, tokens.CARD_ALPHA))
    readout = sample_readout(scene)
    chip = chip_state(scene)
    for item in layout.items:
        x0, y0, x1, y1 = item.rect
        mid_y = (y0 + y1) / 2.0 - size / 3.0
        if item.role in {"button", "chip_action"}:
            enabled = controls.control_enabled(context, item.control)
            fill = palette.accent if item == hover and enabled else palette.text_dim
            _draw.rounded_rect(x0, y0, x1, y1, radius_control,
                               tokens.rgba(fill, 0.28 if enabled else 0.10))
            label = item.control.label if item.role == "chip_action" else item.control.label[:1]
            colour = palette.text if enabled else palette.text_dim
            width = _draw.text_width(label, size)
            _draw.text((x0 + x1 - width) / 2.0, mid_y, label, size, tokens.rgba(colour))
        elif item.role == "readout":
            line = (f"f{readout.frame}  {readout.particles:,} pts  "
                    f"{readout.substeps} sub  {readout.step_ms:.1f} ms")
            _draw.text(x0, mid_y, line, size * 0.9, tokens.rgba(palette.text))
        elif item.role == "sparkline" and len(readout.samples) > 1:
            peak = max(readout.samples) or 1.0
            step = (x1 - x0) / (len(readout.samples) - 1)
            points = [(x0 + i * step, y0 + (y1 - y0) * (value / peak))
                      for i, value in enumerate(readout.samples)]
            _draw.polyline(points, tokens.rgba(palette.accent), 1.5 * s)
        elif item.role == "chip":
            colour = _chip_colour(palette, chip.kind)
            _draw.rounded_rect(x0, y0, x1, y1, radius_chip, tokens.rgba(colour, 0.22))
            _draw.text(x0 + PAD * s, mid_y, chip.text, size, tokens.rgba(colour))
        elif item.role == "ring":
            fraction = bake_fraction(scene) or 0.0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            _draw.ring(cx, cy, (x1 - x0) / 2.6, 3.0 * s, 1.0, tokens.rgba(palette.text_dim, 0.3))
            _draw.ring(cx, cy, (x1 - x0) / 2.6, 3.0 * s, fraction, tokens.rgba(palette.accent))


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
