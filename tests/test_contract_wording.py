from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_runtime_contract_uses_sparse_tiled_leaf_grid_wording():
    checked = [
        ROOT / "docs" / "PORTING_BRIEF.md",
        ROOT / "fluidSim_synth" / "validation_plan.py",
        ROOT / "fluidSim_warp_blender51" / "__init__.py",
        ROOT / "fluidSim_warp_blender51" / "docs" / "technical_report.md",
        ROOT / "fluidSim_warp_blender51" / "docs" / "user_guide.md",
        ROOT / "fluidSim_warp_blender51" / "sim" / "validation.py",
    ]

    offenders = []
    for path in checked:
        text = path.read_text(encoding="utf-8").lower()
        if "octree" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
