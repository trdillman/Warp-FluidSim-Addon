"""Regressions from the UI review: modal cleanup, input limits and stale panel state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from deck_doubles import install_deck_bpy, reload_deck
from settings_doubles import full_settings, split_settings


class _Area:
    type = "VIEW_3D"

    def tag_redraw(self):
        pass


def _context():
    return SimpleNamespace(
        area=_Area(), region=SimpleNamespace(width=1000, height=800), scene=SimpleNamespace(),
        window_manager=SimpleNamespace(modal_handler_add=lambda _op: None))


def _event(kind="MOUSEMOVE", value="PRESS", x=500, y=400):
    return SimpleNamespace(type=kind, value=value, unicode="", mouse_region_x=x,
                           mouse_region_y=y, ctrl=False, shift=False, alt=False, oskey=False)


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
    operator = (deck.palette.FLUIDSIM_OT_command_palette() if kind == "palette"
                else deck.wheel.FLUIDSIM_OT_radial_wheel())
    assert operator.invoke(context, _event("RIGHTMOUSE")) == {"RUNNING_MODAL"}
    return operator, context


@pytest.mark.parametrize("kind", ["palette", "wheel"])
def test_a_modal_that_raises_still_removes_its_draw_handler(deck, kind, monkeypatch):
    """Blender never calls cancel() after a raise, so the modal must clean up itself."""
    operator, context = _open(deck, kind)
    assert deck.registry.REGISTRY.has_draw_handler(kind)

    def boom(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(operator), "_modal", boom)
    with pytest.raises(RuntimeError):
        operator.modal(context, _event("A"))
    assert not deck.registry.REGISTRY.has_draw_handler(kind)
    assert deck.bpy.types.SpaceView3D.handlers == []


def test_the_wheel_treats_a_double_click_like_a_press(deck):
    operator, context = _open(deck, "wheel")
    # A press on empty space closes the wheel; the fast second click of a
    # double click must do the same instead of being swallowed.
    result = operator.modal(context, _event("LEFTMOUSE", value="DOUBLE_CLICK", x=0, y=0))
    assert result == {"CANCELLED"}
    assert not deck.registry.REGISTRY.has_draw_handler("wheel")


def test_the_deck_event_layer_frees_its_window_when_its_modal_raises(deck, monkeypatch):
    modal = deck.modal
    operator = modal.FLUIDSIM_OT_deck_modal()
    operator.generation = modal._GENERATION
    operator.window_key = "window-1"
    modal._RUNNING.add("window-1")

    def boom(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(type(operator), "_modal", boom)
    try:
        with pytest.raises(RuntimeError):
            operator.modal(_context(), _event("MOUSEMOVE"))
        assert "window-1" not in modal._RUNNING  # the ensure timer may restart it
    finally:
        modal._RUNNING.discard("window-1")


def test_palette_out_of_range_numbers_are_a_bad_entry_not_a_crash(deck, monkeypatch):
    import importlib

    fs = full_settings(resolution_divisions=64)
    scene = SimpleNamespace(fluidsim=split_settings(fs), objects=[])
    entries = {entry.target: entry for entry in deck.palette.setting_entries(scene)}
    divisions = entries["resolution_divisions"]

    class _Limited:
        resolution_divisions = 64

        def __setattr__(self, name, value):
            if value > 2**31 - 1:
                raise OverflowError("Python int too large to convert to C int")
            object.__setattr__(self, name, value)

    group = _Limited()
    pending = importlib.import_module("fluidSim_warp_blender51.blender.pending_settings")
    monkeypatch.setattr(pending, "declaring_group", lambda _scene, _name: group)
    with pytest.raises(ValueError):
        deck.palette.commit_inline(scene, divisions, "99999999999")
    assert group.resolution_divisions == 64


def test_loading_a_file_clears_orphaned_palette_and_wheel_handlers(deck):
    for kind in ("palette", "wheel"):
        _open(deck, kind)
        assert deck.registry.REGISTRY.has_draw_handler(kind)
    deck._on_deck_load_post("")
    assert not deck.registry.REGISTRY.has_draw_handler("palette")
    assert not deck.registry.REGISTRY.has_draw_handler("wheel")


def test_generated_source_text_names_are_short_stable_and_distinct():
    import importlib
    import sys
    import types

    bpy = types.ModuleType("bpy")
    props = types.ModuleType("bpy.props")
    props.__getattr__ = lambda _name: (lambda **_kwargs: None)  # any *Property
    bpy.props = props
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(persistent=lambda fn: fn), timers=None)
    bpy.types = SimpleNamespace(
        Operator=object, Panel=object, UIList=object, PropertyGroup=object, ID=object)
    saved = {key: sys.modules.get(key) for key in ("bpy", "bpy.props")}
    sys.modules["bpy"], sys.modules["bpy.props"] = bpy, props
    try:
        workspace = importlib.import_module("fluidSim_warp_blender51.blender.effect_workspace")
        name = workspace.generated_source_text_name
        first = name("effect-123456789abc", "particle_velocity_limit")
        second = name("effect-123456789abc", "particle_velocity_drag")
        assert first != second  # same long prefix, still different texts
        assert first == name("effect-123456789abc", "particle_velocity_limit")  # stable
        assert max(len(first), len(second), len(name("e" * 40, "k" * 60))) <= 63
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sys.modules.pop("fluidSim_warp_blender51.blender.effect_workspace", None)


def _effect_workspace():
    import importlib
    import sys
    import types

    bpy = types.ModuleType("bpy")
    props = types.ModuleType("bpy.props")
    props.__getattr__ = lambda _name: (lambda **_kwargs: None)
    bpy.props = props
    bpy.app = SimpleNamespace(handlers=SimpleNamespace(persistent=lambda fn: fn), timers=None)
    bpy.types = SimpleNamespace(
        Operator=object, Panel=object, UIList=object, PropertyGroup=object, ID=object)
    saved = {key: sys.modules.get(key) for key in ("bpy", "bpy.props")}
    sys.modules["bpy"], sys.modules["bpy.props"] = bpy, props
    try:
        return importlib.import_module("fluidSim_warp_blender51.blender.effect_workspace")
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        sys.modules.pop("fluidSim_warp_blender51.blender.effect_workspace", None)


class _Text(dict):
    def __init__(self, name):
        super().__init__()
        self.name = name


class _Texts:
    def __init__(self):
        self.items = {}

    def get(self, name):
        return self.items.get(name)

    def new(self, name):
        text = _Text(name)
        self.items[name] = text
        return text


def test_generated_text_is_reused_only_by_the_kernel_that_owns_it():
    workspace = _effect_workspace()
    texts = _Texts()
    find = workspace.find_or_create_generated_text

    mine = find(texts, "", "FluidSim Source - k - abc.py", "effect-1\0kernel_a")
    assert find(texts, mine.name, "FluidSim Source - k - abc.py", "effect-1\0kernel_a") is mine

    # A different kernel whose short name collides must not take over the Text.
    other = find(texts, "", "FluidSim Source - k - abc.py", "effect-1\0kernel_b")
    assert other is not mine
    assert mine["fluidsim_generated_key"] == "effect-1\0kernel_a"
