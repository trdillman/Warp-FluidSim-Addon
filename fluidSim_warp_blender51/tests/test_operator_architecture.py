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
        "snippets.py",
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
    facade = FACADE.read_text(encoding="utf-8")
    assert all(name in facade for name in class_names)
