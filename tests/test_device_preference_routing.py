"""Regression tests for the single device-preference lookup.

The preference existed and three places disagreed about it: two open-coded
the ``bpy.context.preferences.addons[...]`` walk with the package name as a
string literal, and ``_probe_warp`` ignored the preference entirely and
probed ``cuda:0``. So the preferences panel could report a checkmark against
a card the solver was never running on. These tests pin the collapse: one
owner, and no caller that re-derives it.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import re
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PACKAGE_ROOT = (pathlib.Path(__file__).resolve().parent.parent
                 / "fluidSim_warp_blender51")


def _install_bpy_stub(monkeypatch, *, prefs):
    fake_bpy = types.ModuleType("bpy")
    fake_props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "StringProperty"):
        setattr(fake_props, name, lambda **kwargs: None)
    fake_bpy.props = fake_props
    fake_bpy.types = SimpleNamespace(Operator=object, AddonPreferences=object)
    fake_bpy.utils = SimpleNamespace(
        register_class=lambda _cls: None, unregister_class=lambda _cls: None)
    addons = {} if prefs is None else {
        "fluidSim_warp_blender51": SimpleNamespace(preferences=prefs)}
    fake_bpy.context = SimpleNamespace(
        preferences=SimpleNamespace(addons=addons))

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_props)
    sys.modules.pop("fluidSim_warp_blender51.preferences", None)
    return importlib.import_module("fluidSim_warp_blender51.preferences")


def test_the_selected_device_is_what_callers_get(monkeypatch):
    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device="cuda:3"))
    assert preferences.configured_device() == "cuda:3"


def test_an_unregistered_addon_falls_back_rather_than_raising(monkeypatch):
    """Reached from operator polls and draw callbacks, where raising is fatal.

    Both the empty-registry and the blank-string cases must yield the default
    rather than an exception or an empty device name.
    """
    preferences = _install_bpy_stub(monkeypatch, prefs=None)
    assert preferences.configured_device() == "cuda:0"

    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device=""))
    assert preferences.configured_device() == "cuda:0"


def test_the_availability_probe_reports_the_device_actually_in_use(monkeypatch):
    """The panel's checkmark is the whole point of the probe.

    Probing cuda:0 while the solver ran on the selected device is how the
    checkmark came to describe a card that was never in use.
    """
    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device="cuda:3"))

    probed = []
    fake_warp = types.ModuleType("warp")
    fake_warp.init = lambda: None
    fake_warp.is_cuda_available = lambda: True
    fake_warp.config = SimpleNamespace(version="1.12.1")

    def _get_device(name):
        probed.append(name)
        return SimpleNamespace(name="RTX 4080 SUPER")

    fake_warp.get_device = _get_device
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setattr(preferences, "_WARP_PROBE_RESULT", None)

    ok, message = preferences._probe_warp(force=True)

    assert probed == ["cuda:3"]
    assert ok and "RTX 4080 SUPER" in message


def test_nothing_else_reaches_into_the_addon_preferences(monkeypatch):
    """A helper the other call sites bypass is how this drifted apart.

    Three copies accumulated because each site could reach the registry
    itself; the source check is what keeps a fourth from appearing.
    """
    lookup = re.compile(r"preferences\.addons\[")
    offenders = sorted(
        str(path.relative_to(_PACKAGE_ROOT))
        for path in _PACKAGE_ROOT.rglob("*.py")
        if path.name != "preferences.py"
        and lookup.search(path.read_text(encoding="utf-8")))

    assert offenders == []


def test_the_owner_does_not_hardcode_the_package_name():
    """Renaming the folder is a supported way to install an add-on."""
    source = (_PACKAGE_ROOT / "preferences.py").read_text(encoding="utf-8")
    body = source.split("def configured_device", 1)[1].split("\ndef ", 1)[0]

    assert "__package__" in body
    assert "fluidSim_warp_blender51" not in body


def test_build_settings_routes_through_the_helper():
    """``SolverSettings.device`` decides which GPU every kernel launches on.

    Running the real ``build_settings`` needs a scene and Warp, so this reads
    the source -- the smallest check that fails if the inline walk returns.
    """
    pytest.importorskip("warp")
    source = (_PACKAGE_ROOT / "blender" / "solver_settings.py").read_text(
        encoding="utf-8")
    body = source.split("def build_settings", 1)[1].split("\ndef ", 1)[0]

    assert "configured_device()" in body
