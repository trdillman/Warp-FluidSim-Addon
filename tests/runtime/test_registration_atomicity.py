"""Injected partial enable failures preserve prior Blender registrations."""
from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from test_unregister_loops_guarded import stubbed  # noqa: F401

CLASS_STEPS = {
    "operators": "CLASSES",
    "panels": "CLASSES",
    "preferences": "CLASSES",
    "blender.ui_scene": "SCENE_TAB_PANEL_CLASSES",
    "blender.ui_object": "OBJECT_PANEL_CLASSES",
    "blender.ui_physics": "PHYSICS_PANEL_CLASSES",
    "blender.ui_stages": "STAGE_PANEL_CLASSES",
    "blender.effect_ui": "CLASSES",
    "blender.effect_workspace": "CLASSES",
    "blender.interchange": "CLASSES",
    "blender.demo_gallery": "PANEL_CLASSES",
    "blender.ui_header": "CLASSES",
    "blender.effects": "CLASSES",
    "properties": "CLASSES",
}


@pytest.mark.parametrize("module_name,classes_name", CLASS_STEPS.items())
def test_second_class_failure_only_unwinds_this_calls_first_class(
    stubbed, monkeypatch, module_name, classes_name
):
    module = importlib.import_module(f"fluidSim_warp_blender51.{module_name}")
    classes = tuple(getattr(module, classes_name))
    assert len(classes) >= 2
    prior = classes[1]
    registered = {prior}
    error = RuntimeError("injected second-class failure; pre-existing class")

    def register(cls):
        if cls is prior:
            raise error
        registered.add(cls)

    bpy = sys.modules["bpy"]
    monkeypatch.setattr(bpy.utils, "register_class", register)
    monkeypatch.setattr(bpy.utils, "unregister_class", registered.remove)
    with pytest.raises(RuntimeError) as raised:
        module.register()
    assert raised.value is error
    assert registered == {prior}


def test_property_injection_failure_rolls_back_classes_and_new_pointers(stubbed, monkeypatch):
    module = importlib.import_module("fluidSim_warp_blender51.properties")
    bpy = sys.modules["bpy"]
    registered = []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.setattr(bpy.utils, "unregister_class", registered.remove)
    error = ValueError("injected third pointer construction failure")
    count = 0

    def pointer(**_kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise error
        return object()

    monkeypatch.setattr(module, "PointerProperty", pointer)
    with pytest.raises(ValueError) as raised:
        module.register()
    assert raised.value is error
    assert registered == []
    assert not hasattr(bpy.types.Scene, "fluidsim")
    assert not hasattr(bpy.types.Object, "fluidsim")
    assert not hasattr(bpy.types.Object, "fluidsim_domain")


def test_existing_pointer_is_preserved_before_any_class_registration(stubbed, monkeypatch):
    module = importlib.import_module("fluidSim_warp_blender51.properties")
    bpy = sys.modules["bpy"]
    prior = object()
    monkeypatch.setattr(bpy.types.Object, "fluidsim", prior, raising=False)
    calls = []
    monkeypatch.setattr(bpy.utils, "register_class", calls.append)
    with pytest.raises(RuntimeError, match="property already registered"):
        module.register()
    assert bpy.types.Object.fluidsim is prior
    assert calls == []


@pytest.mark.parametrize("module_name,callback_name", [
    ("blender.ui_header", "draw_header_button"),
    ("blender.menus", "draw_add_menu"),
])
def test_ui_callback_failure_rolls_back_its_classes(stubbed, monkeypatch, module_name, callback_name):
    module = importlib.import_module(f"fluidSim_warp_blender51.{module_name}")
    bpy = sys.modules["bpy"]
    registered = []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.setattr(bpy.utils, "unregister_class", registered.remove)
    error = RuntimeError("injected UI callback failure")
    def fail(_callback):
        raise error
    ui = SimpleNamespace(append=fail)
    if callback_name == "draw_header_button":
        monkeypatch.setattr(module, "_header_type", lambda: ui)
    else:
        monkeypatch.setattr(bpy.types, "VIEW3D_MT_add", ui)
    with pytest.raises(RuntimeError) as raised:
        module.register()
    assert raised.value is error
    assert registered == []


def test_handler_failure_preserves_prior_and_unwinds_only_new(monkeypatch):
    from fluidSim_warp_blender51.unregister_guard import handler_registration
    prior = lambda: None
    added = lambda: None
    first = [prior]
    error = RuntimeError("injected second-handler append failure")
    class Refusing(list):
        def append(self, value):
            raise error
    second = Refusing()
    with pytest.raises(RuntimeError) as raised:
        with handler_registration(((first, prior), (first, added), (second, added))):
            pass
    assert raised.value is error
    assert first == [prior]
    assert second == []


def test_local_followup_failure_keeps_preexisting_handler():
    from fluidSim_warp_blender51.unregister_guard import handler_registration
    prior = lambda: None
    added = lambda: None
    callbacks = [prior]
    error = ValueError("injected source migration failure")
    with pytest.raises(ValueError) as raised:
        with handler_registration(((callbacks, prior), (callbacks, added))):
            raise error
    assert raised.value is error
    assert callbacks == [prior]



def test_class_rollback_failure_reports_it_and_preserves_original_exception(monkeypatch, capsys):
    from fluidSim_warp_blender51.unregister_guard import class_registration
    first, second = type("First", (), {}), type("Second", (), {})
    error = RuntimeError("original second-class error")
    def register(cls):
        if cls is second:
            raise error
    def refuse(_cls):
        raise ValueError("rollback refused")
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace(utils=SimpleNamespace(
        register_class=register, unregister_class=refuse)))
    with pytest.raises(RuntimeError) as raised:
        with class_registration((first, second)):
            pass
    assert raised.value is error
    assert "rollback refused" in capsys.readouterr().out
    assert any("rollback refused" in note for note in error.__notes__)


def test_handler_reload_deduplicates_its_callback_and_preserves_other_owners():
    from fluidSim_warp_blender51.unregister_guard import handler_registration
    def callback():
        pass
    def stale():
        pass
    stale.__name__ = callback.__name__
    def foreign():
        pass
    callbacks = [stale, foreign, callback, callback]
    with handler_registration(((callbacks, callback),)):
        pass
    assert callbacks == [foreign, callback]
