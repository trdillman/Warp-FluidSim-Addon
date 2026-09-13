"""Source contracts for Blender-only demo builders."""

import ast
from pathlib import Path


BUILDER = Path(__file__).parents[1] / "tools" / "build_demo_scenes.py"


def test_adhesive_calender_authors_reachable_selective_whitewater_settings():
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("CALENDER_WW_")
    }
    assert constants == {
        "CALENDER_WW_CHURN_THRESHOLD": 0.005,
        "CALENDER_WW_MIN_SPEED": 0.05,
    }

    builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_adhesive_calender"
    )
    authored = {}
    for node in ast.walk(builder):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "ds" and target.attr.startswith("ww_"):
                authored[target.attr] = node.value

    assert isinstance(authored["ww_churn_threshold"], ast.Name)
    assert authored["ww_churn_threshold"].id == "CALENDER_WW_CHURN_THRESHOLD"
    assert isinstance(authored["ww_min_speed"], ast.Name)
    assert authored["ww_min_speed"].id == "CALENDER_WW_MIN_SPEED"
    assert ast.literal_eval(authored["ww_churn_gain"]) == 1.0
    assert ast.literal_eval(authored["ww_emission_rate"]) == 1.0
