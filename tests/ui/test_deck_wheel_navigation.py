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
        region=SimpleNamespace(x=0, y=0, width=1000, height=800),
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


@pytest.fixture
def painted_wheel(deck, monkeypatch):
    operator, context = open_wheel(deck, "RIGHTMOUSE")
    palette = SimpleNamespace(surface=(0, 0, 0), accent=(1, 1, 1),
                              text_dim=(1, 1, 1), text=(1, 1, 1))
    monkeypatch.setattr(deck.wheel.bpy, "context", context)
    monkeypatch.setattr(deck.tokens, "current_tokens", lambda _ctx: palette)
    monkeypatch.setattr(deck.tokens, "rgba", lambda *args: (0, 0, 0, 1))
    for name in ("circle", "rounded_rect", "text"):
        monkeypatch.setattr(deck.wheel._draw, name, lambda *args: None)
    monkeypatch.setattr(deck.wheel._draw, "text_width", lambda *args: 10)
    monkeypatch.setattr(deck.controls, "control_status", lambda *args: (True, ""))
    monkeypatch.setattr(deck.controls, "control_active", lambda *args: False)
    calls = []
    monkeypatch.setattr(deck.controls, "run_control", lambda _ctx, c, **kw: calls.append(c))
    operator.report = lambda *args: None
    return deck, operator, context, calls


@pytest.mark.parametrize("entry", ["LEFTMOUSE", "RIGHTMOUSE"])
@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
def test_top_opening_tooltip_clears_footer(painted_wheel, monkeypatch, entry, scale):
    deck, _, context, _ = painted_wheel
    monkeypatch.setattr(deck.bar, "_ui_scale", lambda _ctx: scale)
    context.region.height = 1000 * scale
    operator = deck.wheel.FLUIDSIM_OT_radial_wheel()
    operator.invoke(context, event(entry, y=context.region.height - 1))
    operator.hover = "NEXT"
    cards = []
    monkeypatch.setattr(deck.bar, "draw_tip_card", lambda *args: cards.append(args))
    deck.wheel._draw_wheel(operator)
    assert len(cards) == 1
    height = deck.bar.tip_card_height(scale, 11 * scale, cards[0][-1])
    assert cards[0][2] + height <= operator.centre[1] - deck.wheel.FOOTER_EXTENT * scale
    assert cards[0][2] >= (deck.bar.BOTTOM + deck.bar.HEIGHT + deck.bar.GAP) * scale


@pytest.mark.parametrize("navigation", ["RIGHT_ARROW", "WHEELDOWNMOUSE", "HUB", "NEXT"])
def test_queued_navigation_click_runs_the_painted_control(painted_wheel, navigation):
    deck, operator, context, calls = painted_wheel
    deck.wheel._draw_wheel(operator)
    shown = deck.wheel.stage_controls(context, "SETUP")[1]
    if navigation in {"HUB", "NEXT"}:
        operator._pick(context, navigation)
    else:
        operator.modal(context, event(navigation))
    assert context.scene.fluidsim.deck_stage == "SIMULATE"
    dx, dy = deck.wheel.slot_centre(1)
    operator.modal(context, event("LEFTMOUSE", x=operator.centre[0]+dx,
                                  y=operator.centre[1]+dy))
    assert calls == [shown]


def test_repaint_updates_control_and_failed_paint_invalidates_it(painted_wheel, monkeypatch):
    deck, operator, context, calls = painted_wheel
    deck.wheel._draw_wheel(operator)
    context.scene.fluidsim.deck_stage = "SIMULATE"
    deck.wheel._draw_wheel(operator)
    shown = deck.wheel.stage_controls(context, "SIMULATE")[1]
    operator._pick(context, 1)
    assert calls == [shown]
    calls.clear()
    def fail(*args):
        raise RuntimeError("paint failure")
    monkeypatch.setattr(deck.wheel._draw, "circle", fail)
    deck.wheel._draw_wheel(operator)
    operator._pick(context, 1)
    assert calls == []


def test_no_paint_resize_and_scene_change_cannot_run_slots(painted_wheel):
    deck, operator, context, calls = painted_wheel
    operator._pick(context, 1)
    assert calls == []
    deck.wheel._draw_wheel(operator)
    context.region.width += 1
    operator._pick(context, 1)
    assert calls == []
    deck.wheel._draw_wheel(operator)
    context.scene = SimpleNamespace(fluidsim=SimpleNamespace(deck_stage="SETUP"))
    operator._pick(context, 1)
    assert calls == []


def pan(dy, dx=0, **modifiers):
    return SimpleNamespace(type="MOUSEPAN", value="NOTHING", mouse_x=500,
                           mouse_y=400, mouse_prev_x=500-dx, mouse_prev_y=400-dy,
                           **modifiers)


@pytest.mark.parametrize("dy,expected", [(10, "BAKE"), (-10, "SIMULATE")])
def test_trackpad_accumulates_vertical_nothing_events(deck, dy, expected):
    operator, context = open_wheel(deck, "RIGHTMOUSE")
    for _ in range(3):
        operator.modal(context, pan(dy))
        assert context.scene.fluidsim.deck_stage == "SETUP"
    operator.modal(context, pan(dy))
    assert context.scene.fluidsim.deck_stage == expected
    assert deck.registry.REGISTRY.has_draw_handler("wheel")


@pytest.mark.parametrize("gesture", [pan(0), pan(10, 50), pan(100, ctrl=True),
                                      pan(100, shift=True), pan(100, alt=True)])
def test_trackpad_does_not_page_for_horizontal_or_modified_gestures(deck, gesture):
    operator, context = open_wheel(deck, "RIGHTMOUSE")
    for _ in range(5):
        operator.modal(context, gesture)
    assert context.scene.fluidsim.deck_stage == "SETUP"


def test_trackpad_reversal_and_large_event_have_no_residual_pages(deck):
    operator, context = open_wheel(deck, "RIGHTMOUSE")
    operator.modal(context, pan(30))
    operator.modal(context, pan(-20))
    assert context.scene.fluidsim.deck_stage == "SETUP"
    operator.modal(context, pan(-500))
    assert context.scene.fluidsim.deck_stage == "SIMULATE"
    operator.modal(context, pan(-1))
    assert context.scene.fluidsim.deck_stage == "SIMULATE"


@pytest.mark.parametrize("height", [200, 350])
def test_small_region_omits_tip_instead_of_covering_footer(deck, height):
    assert deck.wheel.tooltip_y(240, height, 1, 60) is None


def test_trackpad_threshold_follows_ui_scale(deck, monkeypatch):
    monkeypatch.setattr(deck.bar, "_ui_scale", lambda _ctx: 2.0)
    operator, context = open_wheel(deck, "RIGHTMOUSE")
    operator.modal(context, pan(40))
    assert context.scene.fluidsim.deck_stage == "SETUP"
    operator.modal(context, pan(40))
    assert context.scene.fluidsim.deck_stage == "BAKE"


def test_release_pick_uses_painted_slot_and_rechecks_availability(painted_wheel, monkeypatch):
    deck, operator, context, calls = painted_wheel
    deck.wheel._draw_wheel(operator)
    operator.modal(context, event("RIGHT_ARROW"))
    monkeypatch.setattr(deck.controls, "control_status", lambda *args: (False, "unavailable now"))
    dx, dy = deck.wheel.slot_centre(1)
    operator.modal(context, event("RIGHTMOUSE", value="RELEASE",
                                  x=operator.centre[0]+dx, y=operator.centre[1]+dy))
    assert calls == []
    assert deck.registry.REGISTRY.has_draw_handler("wheel")
