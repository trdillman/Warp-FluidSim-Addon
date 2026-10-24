"""Deck bar layout: every button says what it is, and the parts the mockup shows are there.

The bar used to draw one letter per button ("P", "R", "S", "L") with no tooltip.
These checks pin the readable layout: full labels beside icons, a hover tip on
every part, the Search / Wheel / Bake tools, and an always-visible pending chip.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck


@pytest.fixture
def deck(monkeypatch):
    install_deck_bpy(monkeypatch, deck_mode="ON")
    return reload_deck()


def _layout(deck, width=1600.0, chip=None, transport=None):
    c = deck.controls
    chip = chip or deck.bar.Chip(kind="NONE", text="")
    transport = transport or c.transport_controls("IDLE", False, True)
    return deck.bar.bar_layout(width, 1.0, transport, chip)


def _buttons(layout):
    return [item for item in layout.items if item.role == "button"]


def test_wide_bar_shows_full_labels_and_the_mockup_tools(deck):
    layout = _layout(deck)
    assert not layout.compact
    labels = [item.label for item in _buttons(layout)]
    assert labels == ["Play", "Step", "Restart", "Liquid Preview", "Search", "Wheel", "Bake"]
    for item in _buttons(layout):
        width = item.rect[2] - item.rect[0]
        assert width >= deck.bar.CHAR * len(item.label), item.label


def test_every_hoverable_part_has_a_plain_language_tip(deck):
    layout = _layout(deck)
    roles = {item.role for item in layout.hits()}
    assert {"button", "readout", "sparkline", "chip"} <= roles
    for item in layout.hits():
        title, body = item.tip
        assert title and body
        if item.control is not None:
            assert item.control.target not in body
            assert "fluidsim." not in body


def test_idle_chip_is_shown_and_pending_chip_adds_apply_and_revert(deck):
    idle = _layout(deck)
    assert [i.role for i in idle.items].count("chip") == 1
    assert not [i for i in idle.items if i.role == "chip_action"]
    pending = _layout(deck, chip=deck.bar.Chip(
        kind="RESET_PENDING", text="Reset pending", actions=(
            ("fluidsim.apply_changes", "Apply"), ("fluidsim.revert_pending", "Revert"))))
    actions = [i.control.target for i in pending.items if i.role == "chip_action"]
    assert actions == ["fluidsim.apply_changes", "fluidsim.revert_pending"]


def test_narrow_region_falls_back_to_icon_buttons_that_fit(deck):
    layout = _layout(deck, width=700.0)
    assert layout.compact
    assert layout.card[2] - layout.card[0] <= 700.0
    assert all(i.rect[2] - i.rect[0] == deck.bar.ICON_ONLY for i in _buttons(layout))
    assert all(i.tip for i in _buttons(layout))  # the hover tip still names each one


def test_hit_test_returns_info_items_but_clicking_them_runs_nothing(deck):
    layout = _layout(deck)
    readout = next(i for i in layout.items if i.role == "readout")
    x, y = (readout.rect[0] + readout.rect[2]) / 2, (readout.rect[1] + readout.rect[3]) / 2
    over, item = deck.bar.hit_test(layout, x, y)
    assert over and item is readout and item.control is None


def test_wheel_opened_from_the_bar_stays_inside_the_region(deck):
    wheel = deck.wheel
    cx, cy = wheel.clamp_centre(900.0, 20.0, 1600.0, 900.0)
    extent = wheel.RING_RADIUS + wheel.SLOT_RADIUS + 8.0
    bar_top = deck.bar.BOTTOM + deck.bar.HEIGHT
    assert cx == 900.0 and cy - extent > bar_top  # whole wheel clears the bar
    event = SimpleNamespace(mouse_x=950, mouse_y=120, mouse_region_x=-1, mouse_region_y=-1)
    assert wheel.pointer_in_region(event, SimpleNamespace(x=50, y=100)) == (900.0, 20.0)
