"""Blender RNA acceptance for user-facing FluidSim tooltip metadata."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.repo_addon_loader import enable_repo_addon


PUBLIC_OPERATOR_FIELDS = {
    "FLUIDSIM_OT_initialize": {"ignore_memory_safeguard"},
    "FLUIDSIM_OT_step": {"frames"},
    "FLUIDSIM_OT_tag_object": {"role"},
    "FLUIDSIM_OT_run_benchmark": {"save_baseline"},
    "FLUIDSIM_OT_export_frame": {"filepath"},
}


def panel_fields():
    tree = ast.parse(
        (ROOT / "fluidSim_warp_blender51" / "panels.py").read_text(encoding="utf-8")
    )
    return {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"prop", "prop_enum"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }


def owner_rna(owner):
    if issubclass(owner, bpy.types.Operator):
        module, operator = owner.bl_idname.split(".", 1)
        return getattr(getattr(bpy.ops, module), operator).get_rna_type()
    return owner.bl_rna


def require_tooltip(owner, field):
    prop = owner_rna(owner).properties.get(field)
    assert prop is not None, f"{owner.__name__}.{field} is not registered in Blender RNA"
    assert prop.description.strip(), f"{owner.__name__}.{field} has an empty RNA tooltip"
    if prop.type != "ENUM" or getattr(prop, "is_enum_flag", False):
        return
    for item in prop.enum_items:
        if not item.identifier:
            continue
        description = item.description.strip()
        assert description, (
            f"{owner.__name__}.{field}.{item.identifier} has an empty enum tooltip"
        )
        assert description.casefold() not in {
            item.name.strip().casefold(),
            item.identifier.replace("_", " ").strip().casefold(),
        }, f"{owner.__name__}.{field}.{item.identifier} only repeats its label"


def main():
    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    from fluidSim_warp_blender51 import operators, preferences, properties

    visible = panel_fields()
    checked = set()
    for owner in properties.CLASSES:
        for field in visible:
            if owner.bl_rna.properties.get(field) is None:
                continue
            require_tooltip(owner, field)
            checked.add(field)
    missing = visible - checked
    assert not missing, f"Panel fields did not resolve to registered RNA: {sorted(missing)}"

    for owner_name, fields in PUBLIC_OPERATOR_FIELDS.items():
        owner = getattr(operators, owner_name)
        for field in fields:
            require_tooltip(owner, field)

    for field in ("device", "verbose"):
        require_tooltip(preferences.FLUIDSIM_AP_preferences, field)

    print(
        "TOOLTIP-ACCEPTANCE-PASS "
        f"({len(visible)} panel fields, "
        f"{sum(map(len, PUBLIC_OPERATOR_FIELDS.values()))} operator fields, 2 preferences)"
    )


if __name__ == "__main__":
    main()
