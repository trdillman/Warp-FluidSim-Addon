"""Both wheel entry points expose reversible page navigation without closing."""
from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck


@pytest.fixture
def deck(monkeypatch):
    install_deck_bpy(monkeypatch, deck_mode="ON")
    return reload_deck()


def event(kind, *, x=500, y=400, value="PRESS"):
    return SimpleNamespace(type=kind, value=value, mouse_region_x=x, mouse_region_y=y)


def open_wheel(deck, entry):
    context = SimpleNamespace(
        scene=SimpleNamespace(fluidsim=SimpleNamespace(deck_stage="SETUP")),
        region=SimpleNamespace(width=1000, height=800),
        area=SimpleNamespace(tag_redraw=lambda: None),
        window_manager=SimpleNamespace(modal_handler_add=lambda _op: None))
    operator = deck.wheel.FLUIDSIM_OT_radial_wheel()
    opening = event(entry, y=20 if entry == "LEFTMOUSE" else 400)
    assert operator.invoke(context, opening) == {"RUNNING_MODAL"}
    return operator, context


@pytest.mark.parametrize("entry", ["LEFTMOUSE", "RIGHTMOUSE"])
@pytest.mark.parametrize("key,step", [
    ("RIGHT_ARROW", 1), ("LEFT_ARROW", -1), ("PAGE_DOWN", 1), ("PAGE_UP", -1),
    ("WHEELDOWNMOUSE", 1), ("WHEELUPMOUSE", -1)])
def test_navigation_visits_every_page_and_wraps_without_closing(deck, entry, key, step):
    operator, context = open_wheel(deck, entry)
    stages = deck.controls.STAGES
    for index in range(1, len(stages) + 1):
        assert operator.modal(context, event(key)) == {"RUNNING_MODAL"}
        assert context.scene.fluidsim.deck_stage == stages[(index * step) % len(stages)]
        assert deck.registry.REGISTRY.has_draw_handler("wheel")
    # A key release must not turn a second page.
    operator.modal(context, event(key, value="RELEASE"))
    assert context.scene.fluidsim.deck_stage == "SETUP"
    assert operator.modal(context, event("ESC")) == {"CANCELLED"}
    assert not deck.registry.REGISTRY.has_draw_handler("wheel")


@pytest.mark.parametrize("entry", ["LEFTMOUSE", "RIGHTMOUSE"])
@pytest.mark.parametrize("target,expected", [("PREVIOUS", "BAKE"), ("NEXT", "SIMULATE")])
@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_visible_navigation_buttons_hit_their_drawn_bounds(deck, monkeypatch, entry,
                                                          target, expected, scale):
    monkeypatch.setattr(deck.bar, "_ui_scale", lambda _context: scale)
    operator, context = open_wheel(deck, entry)
    x0, y0, x1, y1 = deck.wheel.navigation_rects(scale)[target]
    dx, dy = (x0 + x1) / 2, (y0 + y1) / 2
    assert deck.wheel.slot_at(dx, dy, scale) == target
    x, y = operator.centre[0] + dx, operator.centre[1] + dy
    assert y0 + operator.centre[1] >= (deck.bar.BOTTOM + deck.bar.HEIGHT + deck.bar.GAP) * scale
    assert operator.modal(context, event("LEFTMOUSE", x=x, y=y)) == {"RUNNING_MODAL"}
    assert context.scene.fluidsim.deck_stage == expected
    assert deck.registry.REGISTRY.has_draw_handler("wheel")


def test_navigation_tip_explains_the_destination_and_keyboard(deck):
    for target, destination in (("PREVIOUS", "Bake"), ("NEXT", "Simulate")):
        title, body, note = deck.wheel.slot_tip("SETUP", target, [])
        assert destination in body
        assert "arrow" in note.lower() and "scroll" in note.lower()
        assert title
