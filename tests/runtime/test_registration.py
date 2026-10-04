"""One ordered registration table, walked forward and in reverse.

``__init__.register`` and ``__init__.unregister`` used to keep two hand-written
tables (a forward one with rollback, a reverse one with two extra steps) that
had to be edited together; one of the extras named a function
(``blender.effects.unregister_handlers``) that no longer exists and was
silently skipped. ``registration.STEPS`` is now the only list.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from fluidSim_warp_blender51 import registration
from fluidSim_warp_blender51.registration import Step

_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
_FAKE = "fake_registration_pkg"


def _install_fake_package(monkeypatch, names, log, *, fail=None, fail_undo=()):
    """Stand-in step modules that append ``(module, call)`` to ``log``."""
    package = types.ModuleType(_FAKE)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, _FAKE, package)
    for name in names:
        module = types.ModuleType(f"{_FAKE}.{name}")

        def make(label, action, name=name):
            def call():
                log.append((name, label))
                if action == "boom":
                    raise RuntimeError(f"{name} {label} failed")
            return call

        module.on = make("on", "boom" if fail == name else None)
        module.off = make("off", "boom" if name in fail_undo else None)
        monkeypatch.setitem(sys.modules, module.__name__, module)


def _steps(*names):
    return tuple(Step(name, name, "on", "off") for name in names)


def test_register_walks_the_table_in_order(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "abc", log)

    registration.register_all(_FAKE, _steps("a", "b", "c"))

    assert log == [("a", "on"), ("b", "on"), ("c", "on")]


def test_a_failing_step_rolls_back_the_completed_ones_in_reverse(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "abcd", log, fail="c")

    with pytest.raises(RuntimeError, match="c on failed"):
        registration.register_all(_FAKE, _steps("a", "b", "c", "d"))

    # c did not complete, so it is not undone; d was never reached.
    assert log == [("a", "on"), ("b", "on"), ("c", "on"), ("b", "off"), ("a", "off")]


def test_a_failing_rollback_is_reported_and_the_rest_still_run(monkeypatch, capsys):
    log = []
    _install_fake_package(monkeypatch, "abc", log, fail="c", fail_undo={"b"})

    with pytest.raises(RuntimeError, match="c on failed"):
        registration.register_all(_FAKE, _steps("a", "b", "c"))

    assert log[-2:] == [("b", "off"), ("a", "off")]
    assert "rollback of 'b' failed" in capsys.readouterr().out


def test_every_module_is_imported_before_anything_is_registered(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "ab", log)
    steps = _steps("a", "b") + (Step("missing", "missing", "on", "off"),)

    with pytest.raises(ImportError):
        registration.register_all(_FAKE, steps)

    assert log == []


def test_teardown_only_steps_are_not_registered_but_are_torn_down(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "abc", log)
    steps = (
        Step("a", "a", "on", "off"),
        Step("cache", "b", None, "off"),
        Step("c", "c", "on", "off"),
    )

    registration.register_all(_FAKE, steps)
    registration.unregister_all(_FAKE, steps)

    assert log == [
        ("a", "on"), ("c", "on"),
        ("c", "off"), ("b", "off"), ("a", "off"),
    ]


def test_a_failed_register_leaves_no_teardown_only_step_to_roll_back(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "abc", log, fail="c")
    steps = (
        Step("a", "a", "on", "off"),
        Step("cache", "b", None, "off"),
        Step("c", "c", "on", "off"),
    )

    with pytest.raises(RuntimeError):
        registration.register_all(_FAKE, steps)

    assert ("b", "off") not in log and ("a", "off") in log


def test_unregister_skips_what_was_never_loaded_and_continues_past_a_failure(
    monkeypatch, capsys
):
    log = []
    _install_fake_package(monkeypatch, "ac", log, fail_undo={"c"})
    # "b" has no module in sys.modules; "d" has one without the attribute.
    bare = types.ModuleType(f"{_FAKE}.d")
    monkeypatch.setitem(sys.modules, bare.__name__, bare)

    registration.unregister_all(_FAKE, _steps("a", "b", "c", "d"))

    assert log == [("c", "off"), ("a", "off")]
    assert "unregister step 'c' failed" in capsys.readouterr().out
    assert f"{_FAKE}.b" not in sys.modules  # never imported just to tear it down


def test_unregister_is_safe_to_repeat(monkeypatch):
    log = []
    _install_fake_package(monkeypatch, "ab", log)

    registration.unregister_all(_FAKE, _steps("a", "b"))
    registration.unregister_all(_FAKE, _steps("a", "b"))

    assert log == [("b", "off"), ("a", "off")] * 2


# --- the real table ---------------------------------------------------------


def _top_level_functions(module: str) -> set[str]:
    path = _PACKAGE_DIR.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_every_step_names_functions_its_module_defines():
    """A misspelt name would be skipped silently at teardown, not fail loudly."""
    problems = []
    for step in registration.STEPS:
        defined = _top_level_functions(step.module)
        for attribute in (step.register, step.unregister):
            if attribute is not None and attribute not in defined:
                problems.append(f"{step.name}: {step.module}.{attribute}")
    assert problems == []


def test_the_table_is_ordered_so_dependencies_register_first():
    names = [step.name for step in registration.STEPS]
    assert len(names) == len(set(names))
    assert names[0] == "properties"  # every other step reads its property groups
    assert names.index("operators") < names.index("viewport launcher")
    handlers = [n for n in names if n.endswith("handlers")]
    assert max(names.index(n) for n in names if "UI" in n) < min(
        names.index(n) for n in handlers
    )
    assert names[-1] == "viewport overlay"


def test_the_root_entry_points_delegate_to_the_table():
    tree = ast.parse((_PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8"))
    functions = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert set(functions) == {"register", "unregister"}
    assert "registration.register_all(__name__)" in functions["register"]
    assert "registration.unregister_all(__name__)" in functions["unregister"]
