"""Regression coverage for Blender Warp module discovery."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace


def test_adds_default_blender_user_addons_modules(monkeypatch, tmp_path):
    discovery = importlib.import_module("fluidSim_warp_blender51.warp_discovery")
    candidate = (
        tmp_path
        / "Roaming"
        / "Blender Foundation"
        / "Blender"
        / "5.1"
        / "scripts"
        / "addons"
        / "modules"
    )
    (candidate / "warp").mkdir(parents=True)
    (candidate / "warp" / "__init__.py").write_text("", encoding="utf-8")

    fake_bpy = types.ModuleType("bpy")
    fake_bpy.app = SimpleNamespace(version=(5, 1, 0))
    fake_bpy.utils = SimpleNamespace(script_paths=lambda: [])
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setattr(discovery, "_python_site_candidates", lambda: [])
    monkeypatch.setattr(discovery, "_path_python_candidates", lambda: [])
    monkeypatch.setattr(discovery.importlib, "invalidate_caches", lambda: None)

    def fake_find_spec(name):
        if name == "warp" and str(candidate) in sys.path:
            return SimpleNamespace(origin=str(candidate / "warp" / "__init__.py"))
        return None

    monkeypatch.setattr(discovery.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(candidate)])

    assert discovery.ensure_warp_discoverable() == str(
        candidate / "warp" / "__init__.py"
    )
    assert sys.path[0] == str(candidate)
