"""Liquid Deck tokens derive from the Blender theme and stay legible on dark and light themes."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

tokens = importlib.import_module("fluidSim_warp_blender51.blender.deck.tokens")


def _theme(background, text, accent):
    widget = SimpleNamespace(text=text, inner_sel=accent + (1.0,), inner=(0.3, 0.3, 0.3, 1.0))
    return SimpleNamespace(
        view_3d=SimpleNamespace(space=SimpleNamespace(
            gradients=SimpleNamespace(high_gradient=background))),
        user_interface=SimpleNamespace(wcol_regular=widget, wcol_tool=widget),
    )


DARK = _theme((0.239, 0.239, 0.239), (0.9, 0.9, 0.9), (0.278, 0.447, 0.702))
LIGHT = _theme((0.78, 0.80, 0.82), (0.1, 0.1, 0.1), (0.278, 0.447, 0.702))
# A hostile theme: text colour nearly equal to the background.
HOSTILE = _theme((0.5, 0.5, 0.5), (0.55, 0.55, 0.55), (0.5, 0.52, 0.5))


@pytest.mark.parametrize("theme", [DARK, LIGHT, HOSTILE], ids=["dark", "light", "hostile"])
def test_theme_tokens_pass_the_contrast_check(theme):
    deck = tokens.tokens_from_theme(theme)
    surface = deck.surface
    assert tokens.contrast_ratio(deck.text, surface) >= tokens.TEXT_CONTRAST
    assert tokens.contrast_ratio(deck.text_dim, surface) >= tokens.UI_CONTRAST
    for colour in (deck.accent, deck.live, deck.reset, deck.reinit):
        assert tokens.contrast_ratio(colour, surface) >= tokens.UI_CONTRAST


def test_dark_and_light_themes_give_a_dark_and_a_light_card():
    dark = tokens.tokens_from_theme(DARK)
    light = tokens.tokens_from_theme(LIGHT)
    assert tokens.relative_luminance(dark.surface) < 0.2
    assert tokens.relative_luminance(light.surface) > 0.5


def test_accent_follows_the_theme_widget_selection_colour():
    deck = tokens.tokens_from_theme(DARK)
    # Same hue family as the theme's selected-widget colour, not the mockup cyan.
    assert deck.accent[2] > deck.accent[0]


def test_one_accent_one_radius_scale_one_timing():
    assert tokens.RADIUS_SCALE == tuple(sorted(tokens.RADIUS_SCALE))
    assert isinstance(tokens.MOTION_SECONDS, float)
    assert tokens.LIVE_CHIP_SECONDS == pytest.approx(2.0)


def test_contrast_helper_matches_wcag_reference_values():
    assert tokens.contrast_ratio((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)) == pytest.approx(21.0)
    assert tokens.contrast_ratio((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) == pytest.approx(1.0)
    fixed = tokens.ensure_contrast((0.55, 0.55, 0.55), (0.5, 0.5, 0.5), 4.5)
    assert tokens.contrast_ratio(fixed, (0.5, 0.5, 0.5)) >= 4.5


def test_a_missing_theme_falls_back_to_the_reference_palette():
    deck = tokens.tokens_from_theme(None)
    assert tokens.contrast_ratio(deck.text, deck.surface) >= tokens.TEXT_CONTRAST
