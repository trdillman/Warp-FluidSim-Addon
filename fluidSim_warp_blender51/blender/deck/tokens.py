"""Liquid Deck design tokens: colours from the Blender theme, one radius scale, one timing.

Every Deck surface reads its colours from here, and they come from
``context.preferences.themes[0]`` (the 3D View background, the regular widget
text and the tool widget's selection colour) so the Deck follows the user's
theme instead of the mockup's palette. Each colour is pushed to a minimum WCAG
contrast against the card it sits on, so a hostile theme (text close to the
background) still yields legible text. Pure Python: no ``bpy`` import.
"""

from __future__ import annotations

from dataclasses import dataclass

# WCAG 2.x minimum contrast: body text, and large text / UI components.
TEXT_CONTRAST = 4.5
UI_CONTRAST = 3.0

# One radius scale (pixels at 1x UI scale): chip, control, card.
RADIUS_SCALE = (4.0, 8.0, 14.0)
# One motion timing for every Deck transition (hover, chip fade, wheel open).
MOTION_SECONDS = 0.16
# The LIVE chip clears itself this long after the last Deck write.
LIVE_CHIP_SECONDS = 2.0
# Card opacity over the viewport; no backdrop blur (Python cannot sample it).
CARD_ALPHA = 0.92

# Blender's default dark theme, used when the theme cannot be read.
_REFERENCE_BACKGROUND = (0.239, 0.239, 0.239)
_REFERENCE_TEXT = (0.9, 0.9, 0.9)
_REFERENCE_ACCENT = (0.278, 0.447, 0.702)
# State hues; only their lightness is adjusted to the card.
_LIVE = (0.36, 0.78, 0.47)
_RESET = (0.95, 0.68, 0.25)
_REINIT = (0.92, 0.40, 0.38)


@dataclass(frozen=True)
class DeckTokens:
    """Resolved RGB colours (0..1, display space) for one theme."""

    surface: tuple[float, float, float]
    text: tuple[float, float, float]
    text_dim: tuple[float, float, float]
    accent: tuple[float, float, float]
    live: tuple[float, float, float]
    reset: tuple[float, float, float]
    reinit: tuple[float, float, float]
    dark: bool


def _rgb(value, fallback):
    try:
        channels = tuple(float(value[i]) for i in range(3))
    except (TypeError, IndexError, ValueError, KeyError):
        return fallback
    return tuple(min(1.0, max(0.0, c)) for c in channels)


def _channel(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb) -> float:
    """WCAG relative luminance of a display-space (sRGB) colour."""
    r, g, b = (_channel(float(c)) for c in tuple(rgb)[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b) -> float:
    """WCAG contrast ratio between two colours, 1 (none) to 21 (black on white)."""
    la, lb = relative_luminance(a), relative_luminance(b)
    high, low = max(la, lb), min(la, lb)
    return (high + 0.05) / (low + 0.05)


def mix(a, b, t: float):
    return tuple(float(x) + (float(y) - float(x)) * t for x, y in zip(a[:3], b[:3]))


def ensure_contrast(fg, bg, minimum: float):
    """``fg`` moved toward black or white, as little as needed, to reach ``minimum``.

    The direction is whichever extreme contrasts more with ``bg``; that extreme
    reaches at least 4.58:1 against any background, so the WCAG text minimum
    is always attainable. The hue order of ``fg`` is kept.
    """
    fg = tuple(float(c) for c in tuple(fg)[:3])
    if contrast_ratio(fg, bg) >= minimum:
        return fg
    white, black = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    target = white if contrast_ratio(white, bg) >= contrast_ratio(black, bg) else black
    for step in range(1, 21):
        candidate = mix(fg, target, step / 20.0)
        if contrast_ratio(candidate, bg) >= minimum:
            return candidate
    return target


def _theme_colours(theme):
    try:
        background = theme.view_3d.space.gradients.high_gradient
    except AttributeError:
        background = None
    try:
        text = theme.user_interface.wcol_regular.text
    except AttributeError:
        text = None
    try:
        accent = theme.user_interface.wcol_tool.inner_sel
    except AttributeError:
        accent = None
    return (_rgb(background, _REFERENCE_BACKGROUND), _rgb(text, _REFERENCE_TEXT),
            _rgb(accent, _REFERENCE_ACCENT))


def tokens_from_theme(theme) -> DeckTokens:
    """Resolve the Deck palette for ``theme`` (``None`` gives the reference theme)."""
    background, text, accent = _theme_colours(theme)
    dark = relative_luminance(background) < 0.5
    # The card sits one step further from mid-grey than the viewport behind it.
    surface = mix(background, (0.0, 0.0, 0.0), 0.35) if dark else mix(
        background, (1.0, 1.0, 1.0), 0.55)
    text = ensure_contrast(text, surface, TEXT_CONTRAST)
    return DeckTokens(
        surface=surface,
        text=text,
        text_dim=ensure_contrast(mix(text, surface, 0.35), surface, UI_CONTRAST),
        accent=ensure_contrast(accent, surface, UI_CONTRAST),
        live=ensure_contrast(_LIVE, surface, UI_CONTRAST),
        reset=ensure_contrast(_RESET, surface, UI_CONTRAST),
        reinit=ensure_contrast(_REINIT, surface, UI_CONTRAST),
        dark=dark,
    )


def current_tokens(context=None) -> DeckTokens:
    """Tokens for the active Blender theme, or the reference theme without one."""
    try:
        if context is None:
            import bpy

            context = bpy.context
        theme = context.preferences.themes[0]
    except (AttributeError, IndexError, ImportError, KeyError, TypeError):
        theme = None
    return tokens_from_theme(theme)


def rgba(colour, alpha: float = 1.0):
    return (*tuple(colour)[:3], float(alpha))
