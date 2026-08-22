from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_register_has_matching_unregister_path():
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"register", "unregister"}.issubset(names)


def test_no_runtime_subprocess_dependency_probe():
    dependency_surfaces = [
        ROOT / "__init__.py",
        ROOT / "warp_discovery.py",
        *sorted((ROOT / "runtime").rglob("*.py")),
    ]
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in dependency_surfaces
        if "subprocess" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_root_import_does_not_eagerly_import_blender_or_warp():
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    rendered = "\n".join(ast.unparse(node) for node in top_level_imports)
    assert "bpy" not in rendered
    assert "warp" not in rendered


def test_validation_operator_description_does_not_hardcode_stale_test_count():
    source = (ROOT / "operators.py").read_text(encoding="utf-8")
    assert "13-test GPU validation suite" not in source
