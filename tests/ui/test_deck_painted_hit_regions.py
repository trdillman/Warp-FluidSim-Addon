"""Deck clicks must target what was painted, even if the LIVE chip expires."""
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck


@pytest.fixture
def routing(monkeypatch):
    bpy = install_deck_bpy(monkeypatch, deck_mode="ON")
    deck = reload_deck()
    # Optional exact-installed-package lane; the default stays repository-owned.
    installed = os.environ.get("FLUIDSIM_DECK_TEST_ROOT")
    if installed:
        deck.__path__ = [str(Path(installed) / "blender" / "deck")]
        for module in (deck.bar, deck.modal):
            assert Path(module.__file__).is_relative_to(Path(installed))
    region = SimpleNamespace(type="WINDOW", x=0, y=0, width=2400, height=900,
                             as_pointer=lambda: 123, tag_redraw=lambda: None)
    area = SimpleNamespace(type="VIEW_3D", regions=(region,), tag_redraw=lambda: None)
    scene = SimpleNamespace(fluidsim=SimpleNamespace(deck_shown=True))
    window = SimpleNamespace(screen=SimpleNamespace(areas=(area,)))
    context = SimpleNamespace(scene=scene, region=region, area=area, window=window,
                              preferences=bpy.context.preferences)

    @contextmanager
    def override(**_kwargs):
        yield

    context.temp_override = override
    bpy.context = context
    monkeypatch.setattr(deck.tokens, "current_tokens", lambda _ctx: None)
    monkeypatch.setattr(deck.bar, "_paint", lambda *_args: None)
    monkeypatch.setattr(deck.gizmos, "draw_overlay", lambda *_args: None)
    monkeypatch.setattr(deck.controls, "control_status", lambda *_args: (True, ""))
    calls = []
    monkeypatch.setattr(deck.controls, "run_control", lambda _ctx, c, **_kw: calls.append(c.target))
    op = deck.modal.FLUIDSIM_OT_deck_modal()
    op.generation = deck.modal._GENERATION
    state = {"layout": None}
    monkeypatch.setattr(deck.bar, "current_layout", lambda _ctx: state["layout"])
    return deck, context, op, state, calls


def layout(deck, chip):
    return deck.bar.bar_layout(2400, 1, (deck.controls.PLAY,), chip)


@pytest.mark.parametrize("target", ["liquid_preview_enabled", "fluidsim.command_palette",
                                   "fluidsim.radial_wheel"])
def test_chip_expiry_does_not_move_click_targets_before_the_next_paint(routing, target):
    deck, context, op, state, calls = routing
    visible = layout(deck, deck.bar.Chip("LIVE", "LIVE · liquid_preview_enabled"))
    state["layout"] = visible
    deck.bar.draw_deck()
    # Time passes without another redraw. The displayed chip and boxes are still
    # the expanded ones, although calculating a fresh layout would shrink them.
    state["layout"] = layout(deck, deck.bar.Chip("NONE", ""))
    button = next(i for i in visible.items if i.control and i.control.target == target)
    _x0, y0, x1, y1 = button.rect
    event = SimpleNamespace(type="LEFTMOUSE", value="PRESS", mouse_x=x1 - 1,
                            mouse_y=(y0 + y1) / 2)
    op.modal(context, event)
    assert calls == [target]


def test_next_successful_paint_updates_click_geometry(routing):
    deck, context, op, state, calls = routing
    state["layout"] = layout(deck, deck.bar.Chip("LIVE", "LIVE · liquid_preview_enabled"))
    deck.bar.draw_deck()
    current = layout(deck, deck.bar.Chip("NONE", ""))
    state["layout"] = current
    deck.bar.draw_deck()
    button = next(i for i in current.items if i.control == deck.controls.SEARCH)
    x0, y0, x1, y1 = button.rect
    op.modal(context, SimpleNamespace(type="LEFTMOUSE", value="PRESS",
                                     mouse_x=(x0+x1)/2, mouse_y=(y0+y1)/2))
    assert calls == [deck.controls.SEARCH.target]


def test_hidden_resized_and_unpainted_regions_cannot_use_stale_hitboxes(routing):
    deck, context, _op, state, _calls = routing
    state["layout"] = layout(deck, deck.bar.Chip("NONE", ""))
    assert deck.bar.event_layout(context) is None
    deck.bar.draw_deck()
    assert deck.bar.event_layout(context) is state["layout"]
    context.region.width += 10
    assert deck.bar.event_layout(context) is None
    context.region.width -= 10
    context.scene.fluidsim.deck_shown = False
    assert deck.bar.event_layout(context) is None
    context.scene.fluidsim.deck_shown = True
    deck.bar.forget_scene_state()
    assert deck.bar.event_layout(context) is None


def test_failed_paint_invalidates_click_geometry(routing, monkeypatch):
    deck, context, _op, state, _calls = routing
    state["layout"] = layout(deck, deck.bar.Chip("NONE", ""))
    deck.bar.draw_deck()
    assert deck.bar.event_layout(context) is not None

    def fail(*_args):
        raise RuntimeError("draw failed")

    monkeypatch.setattr(deck.bar, "_paint", fail)
    deck.bar.draw_deck()
    assert deck.bar.event_layout(context) is None
