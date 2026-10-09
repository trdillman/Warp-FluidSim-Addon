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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PACKAGE_ROOT = (pathlib.Path(__file__).resolve().parents[2]
                 / "fluidSim_warp_blender51")


def _install_bpy_stub(monkeypatch, *, prefs):
    fake_bpy = types.ModuleType("bpy")
    fake_props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "EnumProperty", "StringProperty"):
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
    monkeypatch.setattr(
        preferences,
        "probe_runtime",
        lambda: SimpleNamespace(
            cuda_devices=(SimpleNamespace(alias="cuda:3"),)),
    )
    assert preferences.configured_device() == "cuda:3"


def test_an_unregistered_addon_falls_back_rather_than_raising(monkeypatch):
    """Reached from operator polls and draw callbacks, where raising is fatal.

    Both the empty-registry and the blank-string cases must yield the default
    rather than an exception or an empty device name.
    """
    preferences = _install_bpy_stub(monkeypatch, prefs=None)
    monkeypatch.setattr(
        preferences,
        "probe_runtime",
        lambda: SimpleNamespace(
            cuda_devices=(SimpleNamespace(alias="cuda:0"),)),
    )
    assert preferences.configured_device() == "cuda:0"

    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device=""))
    monkeypatch.setattr(
        preferences,
        "probe_runtime",
        lambda: SimpleNamespace(
            cuda_devices=(SimpleNamespace(alias="cuda:0"),)),
    )
    assert preferences.configured_device() == "cuda:0"


def test_preferences_helper_accepts_the_operator_context(monkeypatch):
    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device="cuda:0"))
    custom_prefs = SimpleNamespace(device="cuda:7")
    custom_context = SimpleNamespace(
        preferences=SimpleNamespace(addons={
            "fluidSim_warp_blender51": SimpleNamespace(
                preferences=custom_prefs)
        }))

    assert preferences._preferences(custom_context) is custom_prefs


def test_refresh_operators_tolerate_missing_transient_preferences(monkeypatch):
    """Nonpersistent enables may have no Blender preferences entry."""
    preferences = _install_bpy_stub(monkeypatch, prefs=None)
    refreshed = []

    def fake_refresh(prefs):
        refreshed.append(prefs)
        return None, (True, "runtime ready")

    monkeypatch.setattr(preferences, "_refresh_runtime", fake_refresh)

    for operator_type in (
        preferences.FLUIDSIM_OT_refresh_runtime_probe,
        preferences.FLUIDSIM_OT_redetect_device,
    ):
        operator = operator_type()
        operator.report = lambda *_args: None
        assert operator.execute(preferences.bpy.context) == {"FINISHED"}

    assert refreshed == [None, None]


def test_refresh_runtime_does_not_write_missing_preferences(monkeypatch):
    preferences = _install_bpy_stub(monkeypatch, prefs=None)
    probe = SimpleNamespace(
        available=True,
        cuda_devices=(SimpleNamespace(alias="cuda:0"),))
    monkeypatch.setattr(
        preferences, "probe_runtime", lambda force=False: probe)
    monkeypatch.setattr(
        preferences,
        "runtime_status_for_device",
        lambda device, probe=None: (True, f"ready: {device}"),
    )

    refreshed, (ready, message) = preferences._refresh_runtime(None)

    assert refreshed is probe
    assert ready
    assert message == "ready: cuda:0"


def test_the_runtime_status_reports_the_device_actually_in_use(monkeypatch):
    preferences = _install_bpy_stub(
        monkeypatch, prefs=SimpleNamespace(device="cuda:3"))

    probed = []
    probe = SimpleNamespace(
        available=True,
        cuda_devices=(SimpleNamespace(alias="cuda:3"),))
    monkeypatch.setattr(
        preferences, "probe_runtime", lambda force=False: probe)
    monkeypatch.setattr(
        preferences,
        "runtime_status_for_device",
        lambda device, probe=None: (
            probed.append(device) or True,
            "Warp 1.16.0 - cuda:3: RTX 4080 SUPER",
        ),
    )

    refreshed, (ok, message) = preferences._refresh_runtime(
        preferences._preferences())

    assert probed == ["cuda:3"]
    assert refreshed is probe
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
    owner_source = (_PACKAGE_ROOT / "preferences.py").read_text(encoding="utf-8")
    helper_source = owner_source.split("def _preferences", 1)[1].split(
        "def configured_device", 1)[0]
    assert len(lookup.findall(helper_source)) == 1
    assert len(lookup.findall(owner_source)) == 1


def test_the_owner_does_not_hardcode_the_package_name():
    """Renaming the folder is a supported way to install an add-on."""
    source = (_PACKAGE_ROOT / "preferences.py").read_text(encoding="utf-8")
    start = source.index("def _preferences")
    end = source.index("def configured_device")
    body = source[start:end]

    assert "__package__" in body
    assert "fluidSim_warp_blender51" not in body


def test_command_line_routes_selection_through_the_owner():
    source = (_PACKAGE_ROOT / "blender" / "command_line.py").read_text(
        encoding="utf-8")

    assert "set_configured_device(selected_device)" in source


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
