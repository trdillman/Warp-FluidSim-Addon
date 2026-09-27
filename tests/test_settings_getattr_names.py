"""Defaulted settings reads must name a field that exists.

``getattr(fs, "name", default)`` on a settings group never fails: a field that
was removed or renamed silently yields the default and the feature quietly
stops working. Every such read in the add-on must name a field declared on a
``FLUIDSIM_PG_*`` group or on ``SolverSettings``.
"""

import ast
import pathlib

PKG = pathlib.Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"

# Variable names that hold a settings group (RNA or SolverSettings).
SETTINGS_NAMES = {"fs", "settings", "scene_fs", "domain_fs", "ds", "dfs",
                  "domain_settings", "sfs", "src_fs"}
# Non-field attributes read defensively on those objects.
NON_FIELD_ATTRIBUTES = {"__dataclass_fields__", "bl_rna"}


def _annotated_fields(path, predicate):
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and predicate(node.name):
            names.update(item.target.id for item in node.body
                         if isinstance(item, ast.AnnAssign)
                         and isinstance(item.target, ast.Name))
    return names


def _declared_fields():
    return (_annotated_fields(PKG / "properties.py",
                              lambda name: name.startswith("FLUIDSIM_PG_"))
            | _annotated_fields(PKG / "sim" / "solver.py",
                                lambda name: name == "SolverSettings")
            | NON_FIELD_ATTRIBUTES)


def _defaulted_settings_reads():
    for path in sorted(PKG.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) == 3):
                continue
            target, name = node.args[0], node.args[1]
            if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                continue
            target_src = ast.unparse(target)
            if (target_src in SETTINGS_NAMES
                    or target_src.endswith((".fluidsim", ".fluidsim_domain"))):
                yield path.relative_to(PKG).as_posix(), node.lineno, name.value


def test_defaulted_settings_reads_name_declared_fields():
    reads = list(_defaulted_settings_reads())
    assert len(reads) > 100, "scanner no longer finds the settings reads"
    declared = _declared_fields()
    stale = [f"{path}:{line} {name}" for path, line, name in reads
             if name not in declared]
    assert stale == []
