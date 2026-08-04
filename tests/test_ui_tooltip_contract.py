"""Static contract for user-facing Blender tooltip metadata.

This test intentionally avoids importing bpy so it runs in the regular Python
regression lane. Blender resolves ``layout.prop`` tooltips from the
``description=`` metadata on RNA property declarations.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ROOT / "fluidSim_warp_blender51" / "properties.py"
PANELS = ROOT / "fluidSim_warp_blender51" / "panels.py"
OPERATORS = ROOT / "fluidSim_warp_blender51" / "operators.py"
PREFERENCES = ROOT / "fluidSim_warp_blender51" / "preferences.py"
ENUMS = ROOT / "fluidSim_warp_blender51" / "sim" / "enums.py"

PROPERTY_FACTORIES = {
    "BoolProperty",
    "CollectionProperty",
    "EnumProperty",
    "FloatProperty",
    "FloatVectorProperty",
    "IntProperty",
    "PointerProperty",
    "StringProperty",
}

# These words describe an outcome, selection, direction, or trade-off.  The
# contract does not snapshot exact copy, punctuation, or sentence length.
EFFECT_TERMS = {
    "add",
    "affect",
    "allow",
    "apply",
    "bind",
    "build",
    "cap",
    "change",
    "choose",
    "color",
    "disable",
    "display",
    "enable",
    "every",
    "fill",
    "first",
    "hide",
    "higher",
    "ignore",
    "increase",
    "keep",
    "last",
    "kill",
    "limit",
    "load",
    "lower",
    "maximum",
    "minimum",
    "move",
    "prevent",
    "read",
    "remove",
    "replace",
    "run",
    "scale",
    "select",
    "set",
    "show",
    "source",
    "store",
    "use",
    "write",
    "zero",
}

# Operator properties that Blender exposes in a dialog, file selector, or redo
# panel.  Hidden button payloads and list indices are intentionally excluded.
PUBLIC_OPERATOR_FIELDS = {
    ("FLUIDSIM_OT_initialize", "ignore_memory_safeguard"),
    ("FLUIDSIM_OT_step", "frames"),
    ("FLUIDSIM_OT_tag_object", "role"),
    ("FLUIDSIM_OT_run_benchmark", "save_baseline"),
    ("FLUIDSIM_OT_snippet_save_preset", "preset_name"),
    ("FLUIDSIM_OT_snippet_edit_line", "line_text"),
    ("FLUIDSIM_OT_snippet_line_add", "line_text"),
    ("FLUIDSIM_OT_export_frame", "filepath"),
}

# Key high-cost controls need specific trade-offs, not merely an action verb.
SEMANTIC_CONTRACTS = {
    "surface_splat_radius": (
        {"higher", "increase"},
        {"lower", "decrease"},
        {"voxel", "neighborhood"},
        {"gpu", "cost", "work"},
    ),
    "surface_leaf_capacity_scale": (
        {"higher", "increase"},
        {"lower", "decrease"},
        {"leaf", "leaves", "coverage", "cover"},
        {"vram", "memory"},
    ),
    "surface_max_dense_dim": (
        {"higher", "increase"},
        {"lower", "decrease"},
        {"dense", "span", "axis"},
        {"fail", "cap", "vram", "memory"},
    ),
}

REQUIRED_DIRECTIONAL_PHRASES = {
    "limit_percent": (
        "higher values damp excess speed more strongly",
        "lower values retain more",
    ),
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _property_declarations(path: Path):
    for owner in (node for node in _tree(path).body if isinstance(node, ast.ClassDef)):
        for statement in owner.body:
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and isinstance(statement.annotation, ast.Call)
                and isinstance(statement.annotation.func, ast.Name)
                and statement.annotation.func.id in PROPERTY_FACTORIES
            ):
                continue
            yield owner.name, statement.target.id, statement.annotation, statement.lineno


def _literal_keyword(call: ast.Call, keyword: str):
    value = next((item.value for item in call.keywords if item.arg == keyword), None)
    if value is None:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return None


def _assert_actionable(owner: str, field: str, call: ast.Call, line: int):
    description = _literal_keyword(call, "description")
    assert isinstance(description, str) and description.strip(), (
        f"{owner}.{field} at line {line} needs a non-empty description= tooltip"
    )
    normalized = description.lower().replace("-", " ")
    assert any(term in normalized for term in EFFECT_TERMS), (
        f"{owner}.{field} at line {line} must explain what changing the field does; "
        f"current tooltip: {description!r}"
    )


def _literal_panel_fields() -> set[str]:
    fields = set()
    for node in ast.walk(_tree(PANELS)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"prop", "prop_enum"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            continue
        fields.add(node.args[1].value)
    return fields


def test_every_panel_property_has_an_actionable_tooltip():
    visible_fields = _literal_panel_fields()
    declarations = list(_property_declarations(PROPERTIES))
    declared_fields = {field for _owner, field, _call, _line in declarations}

    assert visible_fields <= declared_fields, (
        "Panel fields do not resolve to add-on property declarations: "
        f"{sorted(visible_fields - declared_fields)}"
    )
    for owner, field, call, line in declarations:
        if field in visible_fields:
            _assert_actionable(owner, field, call, line)


def test_high_cost_surface_tooltips_name_directional_tradeoffs():
    declarations = {
        field: call
        for owner, field, call, _line in _property_declarations(PROPERTIES)
        if owner == "FLUIDSIM_PG_settings"
    }
    for field, concept_groups in SEMANTIC_CONTRACTS.items():
        description = _literal_keyword(declarations[field], "description").casefold()
        for alternatives in concept_groups:
            assert any(term in description for term in alternatives), (
                f"FLUIDSIM_PG_settings.{field} tooltip must include one of "
                f"{sorted(alternatives)}; current tooltip: {description!r}"
            )


def test_directional_tooltips_do_not_reverse_solver_behavior():
    declarations = {
        field: call
        for owner, field, call, _line in _property_declarations(PROPERTIES)
        if owner == "FLUIDSIM_PG_settings"
    }
    for field, required_phrases in REQUIRED_DIRECTIONAL_PHRASES.items():
        description = _literal_keyword(declarations[field], "description").casefold()
        for phrase in required_phrases:
            assert phrase in description, (
                f"FLUIDSIM_PG_settings.{field} must state {phrase!r}; "
                f"current tooltip: {description!r}"
            )


def test_public_operator_fields_have_actionable_tooltips():
    declarations = {
        (owner, field): (call, line)
        for owner, field, call, line in _property_declarations(OPERATORS)
    }
    assert PUBLIC_OPERATOR_FIELDS <= declarations.keys()
    for owner, field in sorted(PUBLIC_OPERATOR_FIELDS):
        call, line = declarations[(owner, field)]
        _assert_actionable(owner, field, call, line)


def test_preferences_fields_have_actionable_tooltips():
    for owner, field, call, line in _property_declarations(PREFERENCES):
        if owner == "FLUIDSIM_AP_preferences":
            _assert_actionable(owner, field, call, line)


def _enum_members(tree: ast.Module) -> dict[str, set[str]]:
    members = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        values = {
            statement.targets[0].id
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, int)
        }
        if values:
            members[node.name] = values
    return members


def _default_enum_description_keys(tree: ast.Module) -> dict[str, set[str]]:
    assignment = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_ENUM_DESCRIPTIONS"
            for target in statement.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)
    result = {}
    for key, value in zip(assignment.value.keys, assignment.value.values):
        assert isinstance(key, ast.Name) and isinstance(value, ast.Dict)
        result[key.id] = {
            item.value
            for item in value.keys
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
    return result


def test_ui_int_enums_describe_every_selectable_choice():
    used_without_override = set()
    visible_fields = _literal_panel_fields()
    for _owner, field, call, _line in _property_declarations(PROPERTIES):
        if field not in visible_fields or not (
            isinstance(call.func, ast.Name) and call.func.id == "EnumProperty"
        ):
            continue
        items = next((kw.value for kw in call.keywords if kw.arg == "items"), None)
        if not (
            isinstance(items, ast.Call)
            and isinstance(items.func, ast.Name)
            and items.func.id == "_items"
            and items.args
            and isinstance(items.args[0], ast.Name)
            and len(items.args) == 1
        ):
            continue
        used_without_override.add(items.args[0].id)

    enum_tree = _tree(ENUMS)
    members = _enum_members(enum_tree)
    description_keys = _default_enum_description_keys(enum_tree)
    missing = {
        enum_name: sorted(members[enum_name] - description_keys.get(enum_name, set()))
        for enum_name in used_without_override
        if members[enum_name] - description_keys.get(enum_name, set())
    }
    assert not missing, f"UI enum choices need effect descriptions: {missing}"
