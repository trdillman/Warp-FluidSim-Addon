from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT / "operators.py"
CAPABILITIES = ROOT / "operator_domains"


def test_operator_facade_contains_registration_not_capability_implementations():
    source = FACADE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert len(source.splitlines()) < 175
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} == {
        "register",
        "unregister",
    }


def test_operator_classes_are_owned_by_named_capability_modules():
    expected_owners = {
        "cache.py",
        "diagnostics.py",
        "export.py",
        "presets.py",
        "rendering.py",
        "session.py",
        "source_roles.py",
    }
    actual_owners = set()
    class_names: list[str] = []

    for path in CAPABILITIES.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owned = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name.startswith("FLUIDSIM_OT_")
        ]
        if owned:
            actual_owners.add(path.name)
            class_names.extend(owned)

    assert actual_owners == expected_owners
    assert len(class_names) == len(set(class_names))
    # The facade discovers operators; naming one there would be a second list.
    assert class_names
    facade = FACADE.read_text(encoding="utf-8")
    assert not [name for name in class_names if name in facade]


def _runtime_pass_throughs(source: str) -> list[str]:
    """Top-level functions whose whole body is ``return _runtime.<same name>(...)``."""
    found = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = [
            stmt for stmt in node.body
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
        ]
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            continue
        call = body[0].value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "_runtime"
            and call.func.attr == node.name
        ):
            found.append(node.name)
    return found


def test_capability_modules_call_the_shared_runtime_directly():
    """A domain module reaches ``common`` as ``_runtime.X(...)``, not through a
    same-named wrapper it re-declares: each wrapper was a shallow module whose
    deletion removed no behavior (and ten copies of it had to be kept in sync)."""
    offenders = {
        path.name: names
        for path in sorted(CAPABILITIES.glob("*.py"))
        if path.name != "common.py"
        for names in [_runtime_pass_throughs(path.read_text(encoding="utf-8"))]
        if names
    }
    assert offenders == {}


def test_the_pass_through_detector_flags_a_wrapper():
    wrapper = "def _bridge():\n    return _runtime._bridge()\n"
    real = "def _bridge():\n    return _runtime._bridge() or None\n"
    other = "def _bridge():\n    return _runtime._pending()\n"
    assert _runtime_pass_throughs(wrapper) == ["_bridge"]
    assert _runtime_pass_throughs(real) == []
    assert _runtime_pass_throughs(other) == []


def test_one_scene_identity_owner():
    """``scene_access.scene_identity`` is the only definition of "which scene"."""
    package = ROOT
    definers = sorted(
        str(path.relative_to(package)).replace("\\", "/")
        for path in package.rglob("*.py")
        if "tests" not in path.parts
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"scene_identity", "scene_key", "_scene_identity_token"}
        and path.name != "cache_identity.py"
    )
    assert definers == ["blender/scene_access.py"]
