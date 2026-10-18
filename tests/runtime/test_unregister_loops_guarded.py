"""REVIEW-013 (remaining loops): every module's ``unregister()`` finishes past a refusing class.

``operators`` and ``panels`` are covered by ``test_unregister_resilience.py``.
This drives the other unregister loops of the add-on against a stand-in
``bpy.utils.unregister_class`` that raises for chosen classes, and checks the walk
reaches every class and names the failure.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

_PACKAGE = "fluidSim_warp_blender51"
# module -> name of the class tuple its unregister() walks (None: single class)
_LOOPS = {
    "blender.effects": "CLASSES",
    "blender.demo_gallery": "PANEL_CLASSES",
    "preferences": "CLASSES",
    "blender.effect_ui": "CLASSES",
    "blender.effect_workspace": "CLASSES",
    "properties": "CLASSES",
    "blender.ui_header": "CLASSES",
    "blender.ui_physics": "PHYSICS_PANEL_CLASSES",
    "blender.ui_stages": "STAGE_PANEL_CLASSES",
    "blender.ui_scene": "SCENE_TAB_PANEL_CLASSES",
    "blender.ui_object": "OBJECT_PANEL_CLASSES",
    "blender.interchange": "CLASSES",
    "blender.menus": None,
}


class _Refuses:
    """A class Blender will not unregister."""


def _permissive_bpy(unregister_class):
    """A ``bpy`` whose attributes are inert, so add-on modules import without Blender."""

    class _Anything:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Anything()

        def __call__(self, *args, **kwargs):
            return _Anything()

    class _Base:
        @classmethod
        def poll_message_set(cls, _message):
            return None

    def _module(name):
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr: _Anything()
        return module

    bpy = _module("bpy")
    bpy.types = SimpleNamespace()
    for kind in ("Operator", "Panel", "UIList", "AddonPreferences", "PropertyGroup",
                 "Menu", "Header", "Object", "Scene", "Text", "Material", "Image"):
        setattr(bpy.types, kind, type(kind, (_Base,), {}))
    bpy.types.VIEW3D_MT_add = _Anything()
    bpy.props = _module("bpy.props")
    bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=unregister_class,
    )
    return bpy


@pytest.fixture
def stubbed(monkeypatch):
    before = dict(sys.modules)
    calls: list[type] = []
    failures: dict[type, BaseException] = {}

    def unregister_class(cls):
        calls.append(cls)
        if cls in failures:
            raise failures[cls]

    bpy = _permissive_bpy(unregister_class)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", bpy.props)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=lambda v: v))
    for name in tuple(sys.modules):
        if name.startswith(_PACKAGE + ".") or name == _PACKAGE:
            del sys.modules[name]
    try:
        yield SimpleNamespace(calls=calls, failures=failures)
    finally:
        for name in tuple(sys.modules):
            if name not in before and name not in ("bpy", "bpy.props", "mathutils"):
                del sys.modules[name]
        for name, module in before.items():
            if name.startswith(_PACKAGE):
                sys.modules[name] = module


@pytest.mark.parametrize("module_name", sorted(_LOOPS))
@pytest.mark.parametrize(
    "error", [RuntimeError("not registered"), TypeError("unexpected type"), KeyError("missing")],
    ids=lambda error: type(error).__name__,
)
def test_unregister_walks_every_class_past_one_that_raises(
    stubbed, monkeypatch, capsys, module_name, error
):
    module = importlib.import_module(f"{_PACKAGE}.{module_name}")
    attr = _LOOPS[module_name]
    if attr is None:
        refusing = module.FLUIDSIM_OT_add_domain
        expected_all = [refusing]
    else:
        original = tuple(getattr(module, attr))
        classes = list(original)
        classes.insert(len(classes) // 2, _Refuses)
        monkeypatch.setattr(module, attr, tuple(classes))
        refusing = _Refuses
        expected_all = list(reversed(classes))
    stubbed.failures[refusing] = error

    module.unregister()

    assert stubbed.calls[: len(expected_all)] == expected_all
    out = capsys.readouterr().out
    assert refusing.__name__ in out
    assert str(error.args[0]) in out
