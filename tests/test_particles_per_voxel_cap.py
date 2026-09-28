"""REVIEW-005: one liquid-occupancy ceiling, not six copies of the literal 64.

Both authored density settings stop at MAX_PARTICLES_PER_VOXEL in RNA, the
host clamps size demand to it, and the emission kernel loops to it. If any
one of them moved alone, a value the UI accepts would be silently truncated
(or over-admitted) somewhere downstream.
"""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

from fluidSim_warp_blender51.sim import constants as C
from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence, SourceLifecycle, SourceOperation,
    SourcePayloadKind, SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_execution import CompiledSourceExecution
from fluidSim_warp_blender51.sim.source_regions import (
    AnalyticRegionPlan, PayloadPlan, SourceKey, SourcePlan,
    normalized_liquid_target,
)

ADDON = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"


def _liquid_plan(ppv):
    return SourcePlan(
        SourceKey("x"), SourceOperation.LIQUID, SourceLifecycle.TOP_UP,
        SourceEvaluationCadence.STATIC,
        AnalyticRegionPlan(shape=SourceShapeShape.SPHERE, position=(1, 1, 1),
                           radius=.4),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES,
                    target_particles_per_voxel=ppv))


def test_host_clamps_stop_at_the_shared_ceiling():
    ceiling = C.MAX_PARTICLES_PER_VOXEL
    assert normalized_liquid_target(_liquid_plan(ceiling)) == ceiling
    assert normalized_liquid_target(_liquid_plan(ceiling * 4)) == ceiling
    legacy = CompiledSourceExecution._legacy_liquid_target
    assert legacy(SimpleNamespace(emit_rate=float(ceiling))) == ceiling
    assert legacy(SimpleNamespace(emit_rate=ceiling * 4.0)) == ceiling


def test_authored_density_settings_are_capped_by_the_constant():
    tree = ast.parse((ADDON / "properties.py").read_text(encoding="utf-8"))
    capped = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in {"emit_rate", "particles_per_voxel"}):
            (limit,) = [k.value for k in node.annotation.keywords if k.arg == "max"]
            capped[node.target.id] = any(
                isinstance(n, ast.Attribute) and n.attr == "MAX_PARTICLES_PER_VOXEL"
                for n in ast.walk(limit))
    assert capped == {"emit_rate": True, "particles_per_voxel": True}


def test_no_literal_occupancy_cap_remains():
    literal = re.compile(r"min\(\s*64\b|range\(\s*64\s*\)|\bmax=64\b")
    offenders = [
        f"{path.relative_to(ADDON)}:{number}"
        for path in sorted(ADDON.rglob("*.py"))
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1)
        if literal.search(line)
    ]
    assert not offenders
    kernels = (ADDON / "sim" / "source_kernels.py").read_text(encoding="utf-8")
    assert "range(C.MAX_PARTICLES_PER_VOXEL)" in kernels
