"""Source contracts for Blender-only demo builders."""

import ast
from pathlib import Path

from tools.adhesive_whitewater_calibration import (
    CALENDER_WW_MAX_PEAK_FRACTION,
    CALENDER_WW_MIN_PEAK_FRACTION,
    calender_whitewater_calibration,
    calender_whitewater_population_is_valid,
)


BUILDER = Path(__file__).parents[1] / "tools" / "build_demo_scenes.py"

# The production calender build: 1M particles at the builder's density law.
CALENDER_PRODUCTION_VOXEL_SIZE = 0.08 * (100_000.0 / 1_000_000) ** (1.0 / 3.0)


def test_adhesive_calender_authors_reachable_selective_whitewater_settings():
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))

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

    # The selective filters must come from the voxel-normalized calibration so
    # they keep tracking the build when particle count (hence voxel size)
    # changes; inline magic numbers would silently stop emitting.
    for name in ("ww_churn_threshold", "ww_emission_rate", "ww_min_speed"):
        value = authored[name]
        assert isinstance(value, ast.Attribute), name
        assert isinstance(value.value, ast.Name), name
        assert value.value.id == "whitewater_calibration", name
        assert value.attr == name[len("ww_"):], name
    assert ast.literal_eval(authored["ww_churn_gain"]) == 1.0


def test_calender_whitewater_calibration_stays_selective_but_reachable():
    calibration = calender_whitewater_calibration(CALENDER_PRODUCTION_VOXEL_SIZE)

    # Nonzero filters (selective, not "admit everything") that the calibrated
    # shear cell actually clears: it peaks near 0.021 churn and 0.099 m/s.
    assert 0.0 < calibration.churn_threshold < 0.021
    assert 0.0 < calibration.min_speed < 0.099
    assert calibration.emission_rate > 0.0

    # The admitted population must be visible yet subordinate to the liquid.
    liquid = 1_000_000
    assert not calender_whitewater_population_is_valid(0, liquid)
    assert calender_whitewater_population_is_valid(
        int(liquid * CALENDER_WW_MIN_PEAK_FRACTION), liquid)
    assert calender_whitewater_population_is_valid(
        int(liquid * CALENDER_WW_MAX_PEAK_FRACTION), liquid)
    assert not calender_whitewater_population_is_valid(
        int(liquid * CALENDER_WW_MAX_PEAK_FRACTION) + liquid // 100, liquid)
