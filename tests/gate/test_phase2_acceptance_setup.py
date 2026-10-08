"""REVIEW-025: importing the Phase 2 source acceptance helper has no side effects.

``tests/acceptance/_phase2_source_acceptance_common.py`` used to edit
``sys.path``, delete every add-on module from ``sys.modules`` and import Blender
and the add-on as a side effect of being imported. Anything that merely
imported it (a test, a linter, a tool listing its ``MARKERS``) lost the add-on
modules it already held. The work now sits behind an explicit ``setup()`` that
each wrapper script calls, so these tests can load the helper in plain pytest
and check the contract without Blender.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "tests" / "acceptance"
SUPPORT = ROOT / "tests" / "support"
COMMON = ACCEPTANCE / "_phase2_source_acceptance_common.py"
ADDON = "fluidSim_warp_blender51"
# Top-level packages the old import-time code pulled in. None may appear as a
# result of importing the helper.
BLENDER_SIDE = {"bpy", "mathutils", "openvdb", ADDON, "_test_device"}


def _load_common():
    """Execute the helper under a private name, as a script would import it."""
    spec = importlib.util.spec_from_file_location("_phase2_common_under_test", COMMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrappers():
    return sorted(
        path for path in ACCEPTANCE.glob("blender_*_acceptance.py")
        if "_phase2_source_acceptance_common" in path.read_text(encoding="utf-8")
    )


def test_importing_the_helper_changes_neither_sys_path_nor_sys_modules(monkeypatch):
    sentinel = types.ModuleType(f"{ADDON}.sentinel_under_test")
    monkeypatch.setitem(sys.modules, sentinel.__name__, sentinel)
    path_before = list(sys.path)
    modules_before = set(sys.modules)

    error = None
    try:
        _load_common()
    except BaseException as exc:  # reported below, after the side-effect checks
        error = exc
    finally:
        path_after = list(sys.path)
        modules_after = dict(sys.modules)
        sys.path[:] = path_before
        for name in set(sys.modules) - modules_before:
            if name.split(".")[0] in BLENDER_SIDE:
                del sys.modules[name]

    assert path_after == path_before, "importing the helper edited sys.path"
    assert sentinel.__name__ in modules_after, (
        "importing the helper purged add-on modules from sys.modules")
    imported = sorted(
        name for name in set(modules_after) - modules_before
        if name.split(".")[0] in BLENDER_SIDE)
    assert imported == [], f"importing the helper imported {imported}"
    assert error is None, f"importing the helper must not need Blender: {error!r}"


def test_purge_removes_only_the_addon_package_and_its_submodules():
    common = _load_common()
    modules = {
        ADDON: 1,
        f"{ADDON}.sim": 2,
        f"{ADDON}.sim.solver": 3,
        f"{ADDON}_sibling": 4,
        "warp": 5,
        f"other.{ADDON}": 6,
    }

    removed = common.purge_addon_modules(modules)

    assert sorted(removed) == [ADDON, f"{ADDON}.sim", f"{ADDON}.sim.solver"]
    assert modules == {f"{ADDON}_sibling": 4, "warp": 5, f"other.{ADDON}": 6}
    assert common.purge_addon_modules(modules) == []


def test_purge_defaults_to_the_live_sys_modules():
    common = _load_common()
    sentinel = types.ModuleType(f"{ADDON}.sentinel_under_test")
    held = {
        name: module for name, module in sys.modules.items()
        if name == ADDON or name.startswith(ADDON + ".")
    }
    sys.modules[sentinel.__name__] = sentinel
    try:
        removed = common.purge_addon_modules()
        assert sentinel.__name__ in removed
        assert sentinel.__name__ not in sys.modules
    finally:
        sys.modules.update(held)  # leave other tests' add-on modules in place
        sys.modules.pop(sentinel.__name__, None)


def test_prepend_paths_puts_support_then_acceptance_then_root_first(monkeypatch):
    common = _load_common()
    monkeypatch.setattr(sys, "path", ["/elsewhere", str(ACCEPTANCE)])

    common._prepend_paths()
    common._prepend_paths()  # idempotent: no duplicates, same order

    assert sys.path == [str(SUPPORT), str(ACCEPTANCE), str(ROOT), "/elsewhere"]


def test_setup_prepares_paths_then_purges_then_imports_once(monkeypatch):
    common = _load_common()
    events = []
    bpy_marker, device_marker = object(), object()
    monkeypatch.setattr(common, "_prepend_paths", lambda: events.append("paths"))
    monkeypatch.setattr(
        common, "purge_addon_modules", lambda: events.append("purge") or [])
    monkeypatch.setattr(
        common, "_import_runtime",
        lambda: events.append("import") or {"bpy": bpy_marker, "script_device": device_marker})

    common.setup()
    common.setup()  # a second call must not purge the modules the first one loaded

    assert events == ["paths", "purge", "import"]
    assert common.bpy is bpy_marker
    assert common.script_device is device_marker


def test_main_refuses_to_run_before_setup():
    common = _load_common()

    with pytest.raises(RuntimeError, match=r"setup\(\)"):
        common.main("point_source")


def test_every_wrapper_script_is_found_and_matches_the_marker_table():
    common = _load_common()

    assert {path.name for path in _wrappers()} == {
        f"blender_{capability}_acceptance.py" for capability in common.MARKERS}


def _runtime_function_and_names():
    """``_import_runtime``'s AST and the names it publishes to module globals."""
    tree = ast.parse(COMMON.read_text(encoding="utf-8"))
    runtime = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_import_runtime")
    returned = next(node for node in ast.walk(runtime) if isinstance(node, ast.Return))
    return tree, runtime, {key.value for key in returned.value.keys}


def test_every_global_the_helpers_read_is_defined_or_published():
    """``import *`` is gone, so a name it used to supply must now be listed.

    A miss would be a ``NameError`` halfway through a Blender run that pytest
    cannot make, so check it statically: every name a helper reads is a local,
    a module-level definition, a builtin or published by ``_import_runtime``.
    """
    tree, _, published = _runtime_function_and_names()
    module_level = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_level.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_level.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            module_level.update(
                n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name))
    infrastructure = {"_import_runtime", "setup", "purge_addon_modules", "_prepend_paths"}
    unresolved = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name in infrastructure:
            continue
        local = set()
        for node in ast.walk(function):
            if isinstance(node, ast.arg):
                local.add(node.arg)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                local.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                local.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                local.update((a.asname or a.name).split(".")[0] for a in node.names)
        for node in ast.walk(function):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id not in local | module_level | published
                    and not hasattr(builtins, node.id)):
                unresolved.setdefault(node.id, set()).add(function.name)

    assert unresolved == {}


def test_names_imported_from_the_addon_resolve_in_the_real_modules():
    """The explicit imports that replaced ``import *`` name things that exist."""
    _, runtime, _ = _runtime_function_and_names()
    checked = 0
    for node in ast.walk(runtime):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module.startswith(f"{ADDON}.sim") or node.module.startswith("tools.gate")):
            continue  # the rest needs a live Blender (``bpy``) to import
        module = importlib.import_module(node.module)
        for alias in node.names:
            assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"
            checked += 1
    assert checked >= 15


@pytest.mark.parametrize("script", _wrappers(), ids=lambda path: path.stem)
def test_each_wrapper_calls_setup_before_main(monkeypatch, script):
    calls = []
    fake = types.ModuleType("_phase2_source_acceptance_common")
    fake.setup = lambda: calls.append(("setup",))
    fake.main = lambda capability: calls.append(("main", capability))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the wrapper prepends to it

    runpy.run_path(str(script), run_name="__main__")

    capability = script.stem[len("blender_"):-len("_acceptance")]
    assert calls == [("setup",), ("main", capability)]
