"""Liquid Deck palette and wheel modals survive a closed area, a second invoke, and undo.

The operators' ``invoke``/``modal``/``cancel`` are called directly against the
recording ``bpy`` double, as Blender's window manager would call them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck


class _Area:
    type = "VIEW_3D"

    def __init__(self):
        self.redraws = 0

    def tag_redraw(self):
        self.redraws += 1


def _context():
    return SimpleNamespace(
        area=_Area(), region=SimpleNamespace(width=1000, height=800), scene=SimpleNamespace(),
        window_manager=SimpleNamespace(modal_handler_add=lambda _op: None))


def _event(kind="MOUSEMOVE", value="PRESS", unicode="", x=500, y=400, **modifiers):
    event = SimpleNamespace(type=kind, value=value, unicode=unicode, mouse_region_x=x,
                            mouse_region_y=y, ctrl=False, shift=False, alt=False, oskey=False)
    for name, on in modifiers.items():
        setattr(event, name, on)
    return event


@pytest.fixture
def deck(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    entry = deck.palette.Entry("operator", "fluidsim.bake_all", "Bake All")
    monkeypatch.setattr(deck.palette, "build_index", lambda _scene: [entry])
    deck.bpy = bpy
    return deck


def _open(deck, kind):
    context = _context()
    if kind == "palette":
        operator = deck.palette.FLUIDSIM_OT_command_palette()
    else:
        operator = deck.wheel.FLUIDSIM_OT_radial_wheel()
    assert operator.invoke(context, _event("RIGHTMOUSE")) == {"RUNNING_MODAL"}
    return operator, context


# Risk 1: the area closes (or the workspace changes) while the modal runs -----------

@pytest.mark.parametrize("kind", ["palette", "wheel"])
def test_modal_finishes_and_removes_its_handler_when_the_area_is_gone(deck, kind):
    operator, context = _open(deck, kind)
    assert deck.registry.REGISTRY.has_draw_handler(kind)
    context.area = None  # Blender clears a modal handler's area when the area closes
    result = operator.modal(context, _event())
    assert "CANCELLED" in result and "PASS_THROUGH" in result
    assert not deck.registry.REGISTRY.has_draw_handler(kind)
    assert deck.bpy.types.SpaceView3D.handlers == []


# Risk 2: a second invoke retires the first modal ----------------------------------

@pytest.mark.parametrize("kind", ["palette", "wheel"])
def test_second_invoke_retires_the_first_modal_and_keeps_one_handler(deck, kind):
    first, first_context = _open(deck, kind)
    second, second_context = _open(deck, kind)
    handlers = deck.bpy.types.SpaceView3D.handlers
    assert len(handlers) == 1

    # The first modal's next event retires it without taking the event or the handler.
    result = first.modal(first_context, _event("A", unicode="a"))
    assert result == {"CANCELLED", "PASS_THROUGH"}
    first.cancel(first_context)
    assert len(handlers) == 1 and deck.registry.REGISTRY.has_draw_handler(kind)

    # The second one still works and closes its own handler.
    assert second.modal(second_context, _event("ESC")) == {"CANCELLED"}
    assert handlers == []
