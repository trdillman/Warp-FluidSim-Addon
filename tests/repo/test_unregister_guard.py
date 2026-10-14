"""4.0 lane D: every registration site tears down through one guarded loop."""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"


def _guard(monkeypatch, unregister_class):
    bpy = types.ModuleType("bpy")
    bpy.utils = SimpleNamespace(unregister_class=unregister_class)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.delitem(sys.modules, "fluidSim_warp_blender51.unregister_guard",
                        raising=False)
    return importlib.import_module("fluidSim_warp_blender51.unregister_guard")


def test_one_failure_does_not_stop_the_walk_and_is_reported_at_the_end(
        monkeypatch, capsys):
    seen = []

    def unregister_class(cls):
        seen.append(cls.__name__)
        if cls.__name__ == "B":
            raise RuntimeError("still in use")

    guard = _guard(monkeypatch, unregister_class)
    classes = [type(name, (), {}) for name in "ABC"]

    failed = guard.unregister_classes(classes)

    assert seen == ["C", "B", "A"]  # reverse registration order, past the failure
    assert failed == ["B: RuntimeError('still in use')"]
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["FluidSim: could not unregister 1 class(es): "
                   "B: RuntimeError('still in use')"]


def test_a_clean_teardown_prints_nothing(monkeypatch, capsys):
    guard = _guard(monkeypatch, lambda _cls: None)
    assert guard.unregister_classes([type("A", (), {})]) == []
    assert capsys.readouterr().out == ""


def test_no_module_keeps_its_own_unregister_loop():
    ad_hoc = re.compile(r"for \w+ in reversed\([^)]*\):\s*\n\s*try:\s*\n\s*"
                        r"bpy\.utils\.unregister_class\(")
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*.py")
        if "tests" not in path.parts and path.name != "unregister_guard.py"
        and ad_hoc.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
