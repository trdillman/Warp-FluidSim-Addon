"""Nothing outside ``solver.py`` reaches into a FluidSolver's private state.

The sub-solvers (surface extraction, surface tension, validation) are handed the
whole solver, so a private attribute they read is an unwritten part of its
interface. Those few reads are pinned here by name: a new one fails this test
until it is either made a public member or deliberately added to the list.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"

# Solver internals the sub-solvers still read: the voxel count, the point-dimension
# helper, the profiling phase scope, and the host position gather.
ALLOWED_PRIVATE_READS = {
    "sim/surface.py": {"_np_dim", "_nvox", "_phase", "_surface_positions"},
    "sim/surface_tension.py": {"_nvox", "_phase"},
    "sim/validation.py": {"_nvox"},
}


def _is_solver(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "solver"
    return isinstance(node, ast.Attribute) and node.attr in {"solver", "_solver"}


def _private_solver_reads() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).as_posix()
        if relative == "sim/solver.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and _is_solver(node.value)
            ):
                found.setdefault(relative, set()).add(node.attr)
    return found


def test_private_solver_reads_are_exactly_the_pinned_ones():
    assert _private_solver_reads() == ALLOWED_PRIVATE_READS


def test_the_surface_transfer_slot_is_public_on_the_solver():
    source = (PACKAGE / "sim" / "solver.py").read_text(encoding="utf-8")
    assert "self.surface_attribute_transfer = None" in source
    assert "_surface_attribute_transfer" not in source
