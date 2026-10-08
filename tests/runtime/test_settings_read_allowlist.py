"""Settings fields are read directly, never with a default.

Every domain setting is declared by exactly one group, so reading it is plain
attribute access. ``getattr(group, "name", default)`` on a field the group does
not declare is how defects #1, #3 and #5 slipped through: the read came from the
wrong group, silently returned the default, and the solver ran with it. This test
fails on any such read in the two modules that project settings to the solver
and track pending edits, except for the few names on the explicit allowlist in
``blender/solver_settings.py``.
"""

import ast
import pathlib

from fluidSim_warp_blender51.blender import solver_settings

PKG = pathlib.Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
CHECKED_MODULES = ("blender/solver_settings.py", "blender/pending_settings.py")


def _group_fields():
    """``{class name: declared field names}`` for the settings groups, via AST
    so this runs without ``bpy``."""
    tree = ast.parse((PKG / "properties.py").read_text(encoding="utf-8"))
    groups = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("FLUIDSIM_PG_"):
            groups[node.name] = {
                item.target.id for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    return groups


def _settings_fields():
    groups = _group_fields()
    return groups["FLUIDSIM_PG_scene"] | groups["FLUIDSIM_PG_domain"]


def _defaulted_reads(rel):
    """Three-argument ``getattr`` calls whose name is a string literal."""
    tree = ast.parse((PKG / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) == 3
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            yield node.lineno, node.args[1].value


def test_no_defaulted_read_of_a_settings_field():
    fields = _settings_fields()
    offenders = [
        f"{rel}:{line} getattr(..., {name!r}, <default>)"
        for rel in CHECKED_MODULES
        for line, name in _defaulted_reads(rel)
        if name in fields and not solver_settings.is_optional_setting_read(name)
    ]
    assert offenders == [], (
        "read these settings fields directly (group.name) so a read from the "
        "wrong group raises:\n" + "\n".join(offenders)
    )


def test_checker_sees_the_defaulted_reads_it_is_meant_to_catch():
    source = 'getattr(fs, "voxel_size", 0.1)\ngetattr(fs, "voxel_size")\n'
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    three_arg = [c for c in calls if len(c.args) == 3]
    assert len(three_arg) == 1 and three_arg[0].args[1].value in _settings_fields()


def test_allowlist_names_are_not_declared_settings_fields():
    # The exemption exists for names no group declares (legacy carriers) and for
    # the stat_* statistics. A real field added to the allowlist would reopen the
    # hole this test closes.
    fields = _settings_fields()
    assert solver_settings.OPTIONAL_SETTING_READS.isdisjoint(fields)
    assert solver_settings.OPTIONAL_SETTING_READ_PREFIXES == ("stat_",)
    stat_fields = {name for name in fields if name.startswith("stat_")}
    assert stat_fields, "stat_* exemption must still name real scene statistics"
    assert stat_fields <= _group_fields()["FLUIDSIM_PG_scene"]


def test_projection_sources_are_all_domain_fields():
    # apply_scene_settings reads these straight from the domain group; one that
    # the domain class does not declare would raise on a real scene.
    domain = _group_fields()["FLUIDSIM_PG_domain"]
    direct = set(solver_settings.DIRECT_SETTING_FIELDS)
    enum_sources = {source for source, _enum in solver_settings.enum_setting_fields().values()}
    assert direct <= domain, sorted(direct - domain)
    assert enum_sources <= domain, sorted(enum_sources - domain)
