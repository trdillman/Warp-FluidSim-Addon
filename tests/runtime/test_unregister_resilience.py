"""REVIEW-013: ``unregister()`` keeps going when one class will not unregister.

``operators.unregister`` walked its classes with no guard, so the first class
that raised left every class after it registered. ``panels.unregister``
guarded each class but only against ``RuntimeError`` and ``ValueError``.
Both now continue past any exception per class and say which class failed.

The tests drive the real ``unregister`` of the real modules against a stand-in
``bpy.utils.unregister_class`` that records every call and raises for chosen
classes, so removing the per-class guard turns them red.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

_PACKAGE = "fluidSim_warp_blender51"
_MODULES = (f"{_PACKAGE}.operators", f"{_PACKAGE}.panels")


class _RefusesToUnregister:
    """A class Blender will not unregister (stand-in for a half-registered one)."""


class _AlsoRefuses:
    """A second failing class, to prove one report does not end the walk."""


def _bpy_stub(unregister_class):
    bpy = types.ModuleType("bpy")

    class _Operator:
        @classmethod
        def poll_message_set(cls, _message):
            return None

        def report(self, *_args, **_kwargs):
            return None

    class _Namespace:
        pass

    def _prop(**kwargs):
        return kwargs.get("default")

    bpy.types = SimpleNamespace(
        Operator=_Operator,
        Panel=_Namespace,
        UIList=_Namespace,
        AddonPreferences=_Namespace,
        PropertyGroup=object,
        Text=object,
        Object=_Namespace,
        Scene=_Namespace,
        Material=object,
        Image=object,
    )
    bpy.props = SimpleNamespace(
        BoolProperty=_prop,
        CollectionProperty=_prop,
        EnumProperty=_prop,
        FloatProperty=_prop,
        FloatVectorProperty=_prop,
        IntProperty=_prop,
        PointerProperty=_prop,
        StringProperty=_prop,
    )
    bpy.path = SimpleNamespace(abspath=lambda value: value)
    bpy.context = SimpleNamespace(screen=SimpleNamespace(areas=[]))
    bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None,
        unregister_class=unregister_class,
    )
    return bpy


@pytest.fixture
def addon_modules(monkeypatch):
    """Import ``operators`` and ``panels`` under a stub ``bpy``; leave no trace.

    The modules (and everything they pull in) are bound to the stub, so the
    teardown puts ``sys.modules`` and the package attributes back exactly as
    they were. Otherwise a later ``from fluidSim_warp_blender51 import panels``
    would hand back the stub-bound copy.
    """
    before = dict(sys.modules)
    calls: list[type] = []
    failures: dict[type, BaseException] = {}

    def unregister_class(cls):
        calls.append(cls)
        if cls in failures:
            raise failures[cls]

    monkeypatch.setitem(sys.modules, "bpy", _bpy_stub(unregister_class))
    monkeypatch.setitem(sys.modules, "bpy.props", sys.modules["bpy"].props)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=lambda v: v))
    for name in tuple(sys.modules):
        if name in _MODULES or name.startswith(f"{_PACKAGE}.operator_domains"):
            del sys.modules[name]
    modules = {name.rsplit(".", 1)[1]: importlib.import_module(name) for name in _MODULES}
    try:
        yield SimpleNamespace(calls=calls, failures=failures, **modules)
    finally:
        for name in tuple(set(sys.modules) | set(before)):
            if sys.modules.get(name) is before.get(name):
                continue
            parent, _, child = name.rpartition(".")
            parent_module = sys.modules.get(parent)
            if name in before:
                sys.modules[name] = before[name]
                if parent_module is not None:
                    setattr(parent_module, child, before[name])
            elif name not in ("bpy", "bpy.props", "mathutils"):  # monkeypatch undoes these
                del sys.modules[name]
                if parent_module is not None and hasattr(parent_module, child):
                    delattr(parent_module, child)


def _with_stand_ins(module, monkeypatch, *, positions):
    """``module.CLASSES`` with a refusing stand-in inserted at each position."""
    original = tuple(module.CLASSES)
    stand_ins = (_RefusesToUnregister, _AlsoRefuses)[: len(positions)]
    classes = list(original)
    for offset, (position, stand_in) in enumerate(zip(positions, stand_ins)):
        classes.insert(min(position + offset, len(classes)), stand_in)
    monkeypatch.setattr(module, "CLASSES", tuple(classes))
    return original, tuple(classes), stand_ins


@pytest.mark.parametrize("module_name", ["operators", "panels"])
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("not registered"),
        ValueError("not registered"),
        TypeError("unexpected type"),
        KeyError("missing"),
        AttributeError("no bl_rna"),
    ],
    ids=lambda error: type(error).__name__,
)
@pytest.mark.parametrize("where", ["first_processed", "middle", "last_processed"])
def test_unregister_walks_every_class_past_one_that_raises(
    addon_modules, monkeypatch, capsys, module_name, error, where
):
    module = getattr(addon_modules, module_name)
    # unregister() walks reversed(CLASSES): the last entry is processed first.
    position = {"first_processed": len(module.CLASSES), "middle": 1, "last_processed": 0}[where]
    original, classes, (stand_in,) = _with_stand_ins(module, monkeypatch, positions=(position,))
    addon_modules.failures[stand_in] = error

    module.unregister()

    # Every class was attempted, in reverse registration order, exactly once.
    assert addon_modules.calls == list(reversed(classes))
    assert set(original) <= set(addon_modules.calls)
    # The failure is reported by class name, not swallowed silently.
    out = capsys.readouterr().out
    assert stand_in.__name__ in out
    assert str(error.args[0]) in out


@pytest.mark.parametrize("module_name", ["operators", "panels"])
def test_unregister_reports_every_failing_class_and_still_finishes(
    addon_modules, monkeypatch, capsys, module_name
):
    module = getattr(addon_modules, module_name)
    original, classes, (first, second) = _with_stand_ins(
        module, monkeypatch, positions=(len(module.CLASSES), 1)
    )
    addon_modules.failures[first] = TypeError("first failure")
    addon_modules.failures[second] = OSError("second failure")

    module.unregister()

    assert addon_modules.calls == list(reversed(classes))
    out = capsys.readouterr().out
    assert first.__name__ in out and "first failure" in out
    assert second.__name__ in out and "second failure" in out


@pytest.mark.parametrize("module_name", ["operators", "panels"])
def test_unregister_is_quiet_when_every_class_unregisters(
    addon_modules, capsys, module_name
):
    module = getattr(addon_modules, module_name)

    module.unregister()

    assert addon_modules.calls == list(reversed(module.CLASSES))
    assert capsys.readouterr().out == ""
