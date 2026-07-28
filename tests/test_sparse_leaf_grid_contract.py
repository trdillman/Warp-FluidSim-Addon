"""Regression guard for the live sparse tiled leaf-grid wording contract."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

LIVE_CONTRACT_SURFACES = [
    "PROMPT.md",
    "AGENTS.md",
    "fluidSim_warp_blender51/__init__.py",
    "fluidSim_warp_blender51/sim/fields.py",
    "fluidSim_warp_blender51/sim/validation.py",
    "fluidSim_warp_blender51/docs/user_guide.md",
    "fluidSim_warp_blender51/docs/technical_report.md",
    "tests/test_field_indexing.py",
]


def test_live_contract_surfaces_do_not_promote_hierarchical_tree_storage():
    offenders = []
    pattern = re.compile(r"sparse[- ]octree|octree[- ]equivalent", re.IGNORECASE)
    for relative_path in LIVE_CONTRACT_SURFACES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{relative_path}:{line_number}: {line.strip()}")

    assert offenders == []


def test_validation_contract_names_use_sparse_tiled_leaf_grid():
    from fluidSim_warp_blender51.sim import validation

    assert validation.VALIDATION_TESTS[0] == "sparse tiled leaf-grid coordinate/index roundtrip"
    assert validation._TESTS[0].__name__ == "test_leaf_grid_roundtrip"
    assert validation._TESTS[0].validation_name == validation.VALIDATION_TESTS[0]
