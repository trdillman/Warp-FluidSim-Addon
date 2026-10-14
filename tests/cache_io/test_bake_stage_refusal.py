"""What the Bake stage says when the cache on disk cannot be played or resumed.

Two refusals: a cache baked by FluidSim 2.x (3.0 does not read it), and a
Resume refused because a source or collider animation no longer matches the
frames already baked, which offers "Re-bake from frame N".
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.io.cache_schema import (
    LEGACY_2X_CACHE_MESSAGE,
    manifest_path,
)


class _Layout:
    def __init__(self):
        self.labels: list[tuple[str, str]] = []
        self.operators: list[dict] = []

    def label(self, *, text="", icon="NONE"):
        self.labels.append((text, icon))

    def operator(self, idname, *, text="", icon="NONE"):
        handle = SimpleNamespace()
        self.operators.append(
            {"id": idname, "text": text, "icon": icon, "handle": handle})
        return handle

    def __getattr__(self, name):
        return lambda *args, **kwargs: _Layout()


def _install_bpy_stub(monkeypatch):
    fake = types.ModuleType("bpy")

    class _Base:
        _poll_message = ""

        @classmethod
        def poll_message_set(cls, message):
            cls._poll_message = message

    def _prop(**kwargs):
        return kwargs.get("default")

    fake.types = SimpleNamespace(
        Operator=_Base, Panel=_Base, UIList=_Base, AddonPreferences=_Base,
        PropertyGroup=object, Text=object, Material=object, Image=object,
        Object=type("Object", (), {}), Scene=type("Scene", (), {}),
    )
    fake.props = SimpleNamespace(**{
        name: (lambda **kwargs: _prop(**kwargs))
        for name in (
            "BoolProperty", "BoolVectorProperty", "CollectionProperty", "EnumProperty", "FloatProperty",
            "FloatVectorProperty", "IntProperty", "PointerProperty", "StringProperty")
    })
    fake.path = SimpleNamespace(abspath=lambda value: value)
    fake.context = SimpleNamespace(
        preferences=SimpleNamespace(addons={}), screen=SimpleNamespace(areas=[]))
    fake.utils = SimpleNamespace(
        register_class=lambda _cls: None, unregister_class=lambda _cls: None)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", fake)
    monkeypatch.setitem(sys.modules, "bpy.props", fake.props)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)


def _stage_module(monkeypatch):
    _install_bpy_stub(monkeypatch)
    name = "fluidSim_warp_blender51.blender.ui_stages"
    for loaded in tuple(sys.modules):
        if loaded == name or loaded.startswith(f"{name}."):
            sys.modules.pop(loaded, None)
    return importlib.import_module(name)


def _context(token=21, cache_path="cache"):
    scene = SimpleNamespace(
        as_pointer=lambda: token,
        fluidsim=SimpleNamespace(cache_path=cache_path),
    )
    return SimpleNamespace(scene=scene)


@pytest.fixture(autouse=True)
def _clean_registries():
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    production_bake._ACTIVE_BAKES.clear()
    cache_bake._REBAKE_OFFERS.clear()
    yield
    production_bake._ACTIVE_BAKES.clear()
    cache_bake._REBAKE_OFFERS.clear()


def _cache_2x(directory):
    directory.mkdir(parents=True)
    manifest_path(directory).write_text(
        '{"format_version": 2}', encoding="utf-8")
    return directory


def test_a_2x_cache_shows_the_refusal_message(monkeypatch, tmp_path):
    ui_stages = _stage_module(monkeypatch)
    cache = _cache_2x(tmp_path / "cache")
    ui_scene = sys.modules[ui_stages.draw_cache_bake.__module__]
    monkeypatch.setattr(ui_scene, "_cache_directory", lambda _scene: cache)
    layout = _Layout()

    ui_stages.draw_cache_refusal(layout, _context())

    # textwrap may break "Re-bake" after the hyphen; rejoin that before comparing.
    text = " ".join(line for line, _icon in layout.labels).replace("- ", "-")
    assert text == " ".join(LEGACY_2X_CACHE_MESSAGE.split())
    assert "baked by FluidSim 2.x" in text
    assert "3.0" in text and "2.1.1" in text
    assert layout.labels[0][1] == "ERROR"
    assert layout.operators == []


def test_no_cache_and_a_current_cache_show_no_refusal(monkeypatch, tmp_path):
    ui_stages = _stage_module(monkeypatch)
    ui_scene = sys.modules[ui_stages.draw_cache_bake.__module__]
    for directory in (None, tmp_path / "empty"):
        monkeypatch.setattr(ui_scene, "_cache_directory", lambda _s, d=directory: d)
        layout = _Layout()

        ui_stages.draw_cache_refusal(layout, _context())

        assert layout.labels == []
        assert layout.operators == []


def test_a_changed_animation_offers_a_rebake_from_the_first_changed_frame(
        monkeypatch, tmp_path):
    from fluidSim_warp_blender51.blender import cache_bake

    ui_stages = _stage_module(monkeypatch)
    ui_scene = sys.modules[ui_stages.draw_cache_bake.__module__]
    monkeypatch.setattr(ui_scene, "_cache_directory", lambda _scene: tmp_path)
    context = _context()
    cache_bake._REBAKE_OFFERS[cache_bake.scene_identity(context.scene)] = 17
    layout = _Layout()

    ui_stages.draw_cache_refusal(layout, context)

    assert "from frame 17" in " ".join(line for line, _icon in layout.labels)
    (offer,) = layout.operators
    assert offer["id"] == "fluidsim.bake_resume"
    assert "17" in offer["text"]
    assert offer["handle"].rebake is True
    assert offer["handle"].rebake_frame == 17


def test_nothing_is_drawn_while_a_bake_is_running(monkeypatch, tmp_path):
    from fluidSim_warp_blender51.blender import cache_bake, production_bake

    ui_stages = _stage_module(monkeypatch)
    ui_scene = sys.modules[ui_stages.draw_cache_bake.__module__]
    cache = _cache_2x(tmp_path / "cache")
    monkeypatch.setattr(ui_scene, "_cache_directory", lambda _scene: cache)
    context = _context()
    cache_bake._REBAKE_OFFERS[cache_bake.scene_identity(context.scene)] = 17
    monkeypatch.setattr(production_bake, "bake_is_active", lambda _scene: True)
    layout = _Layout()

    ui_stages.draw_cache_refusal(layout, context)

    assert layout.labels == []
    assert layout.operators == []
