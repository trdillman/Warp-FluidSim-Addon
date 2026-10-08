"""The profiling tools call only cache functions that exist.

``tools/profiling/profile_user_path.py`` kept calling ``write_frame_v2_atomic``
after cache v3 replaced it. Nothing imports the tool in the suite and it needs
Blender and a GPU to run, so the break would have shown up only when somebody
profiled a cached bake. This reads the tools' source instead: every attribute a
profiling tool reads from the cache modules must be there.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROFILING = ROOT / "tools" / "profiling"
PACKAGE = "fluidSim_warp_blender51"
# Pure-Python modules that import without Blender.
TRACKED = {
    f"{PACKAGE}.io.cache",
    f"{PACKAGE}.io.cache_schema",
    f"{PACKAGE}.blender.cache_identity",
}


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Local names bound to a tracked module, following the run-dict hand-off."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                dotted = f"{node.module}.{item.name}"
                if dotted in TRACKED:
                    aliases[item.asname or item.name] = dotted
        elif isinstance(node, ast.Import):
            for item in node.names:
                if item.name in TRACKED and item.asname:
                    aliases[item.asname] = item.name
    # profile_user_path.py passes modules between its phases as
    # ``{"cache_io": cache_io, ...}`` and re-binds them with ``x = run["cache_io"]``.
    keyed: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value, ast.Name) and value.id in aliases):
                keyed[key.value] = aliases[value.id]
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.slice, ast.Constant)
                and node.value.slice.value in keyed):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases.setdefault(target.id, keyed[node.value.slice.value])
    return aliases


def _cache_reads(path: Path) -> list[tuple[str, str, int]]:
    """Every ``(module, name, line)`` the file reads from a tracked module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _module_aliases(tree)
    reads: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            reads.append((aliases[node.value.id], node.attr, node.lineno))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in aliases
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            reads.append(
                (aliases[node.args[0].id], node.args[1].value, node.lineno))
        elif (isinstance(node, ast.ImportFrom) and node.module in TRACKED):
            reads.extend((node.module, item.name, node.lineno)
                         for item in node.names)
    return reads


@pytest.mark.parametrize(
    "path", sorted(PROFILING.glob("*.py")), ids=lambda p: p.name)
def test_profiling_tools_read_only_cache_names_that_exist(path: Path):
    missing = []
    for module_name, name, line in _cache_reads(path):
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            missing.append(f"{path.name}:{line} {module_name}.{name}")

    assert missing == []


def test_the_cache_profile_writes_a_v3_frame():
    """The scan above must see the call it exists to protect."""
    reads = {
        (module, name)
        for module, name, _ in _cache_reads(PROFILING / "profile_user_path.py")
    }

    assert (f"{PACKAGE}.io.cache", "write_frame_v3_atomic") in reads
    assert (f"{PACKAGE}.blender.cache_identity", "source_frame_stamp") in reads
    assert (f"{PACKAGE}.io.cache", "write_frame_v2_atomic") not in reads
