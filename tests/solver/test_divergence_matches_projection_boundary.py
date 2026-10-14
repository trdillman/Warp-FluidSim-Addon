"""Divergence must treat a missing low neighbour exactly as projection does.

Regression for review items B14 and B15. ``k_project`` treats a face whose
-axis neighbour is off-grid or unallocated as a solid wall and overwrites it
with zero, but ``k_compute_divergence`` read the stored (nonzero) face
velocity there. Pressure then compensated for a flux that projection deletes,
leaving divergence of order u/h after a converged solve. The same kernel read
``u[-1]`` through an eagerly evaluated ``wp.where`` at positive edges.
"""
from __future__ import annotations

import numpy as np
import pytest

wp = pytest.importorskip("warp")

DEVICE = "cpu"
H = 0.01


def _post_projection_edge_divergence(edge_u):
    from fluidSim_warp_blender51.sim import pressure as P
    from fluidSim_warp_blender51.sim.constants import VOXEL_FLUID
    from fluidSim_warp_blender51.sim.fields import (
        MacVelocityField, ScalarField, SparseGrid,
    )

    grid = SparseGrid(origin=(0.0, 0.0, 0.0), size=(0.08, 0.08, 0.08),
                      voxel_size=H, leaf_capacity=8, device=DEVICE)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.0799, 0.0799, 0.0799))
    grid.finalize_topology()
    n = grid.voxel_capacity
    idx = np.arange(512)
    li, lk = idx & 7, idx >> 6
    vt_np = np.where(lk < 6, VOXEL_FLUID, 0).astype(np.int32)

    vt = ScalarField(grid, dtype=wp.int32)
    vt.data.assign(wp.array(vt_np, dtype=wp.int32, device=DEVICE))
    vel = MacVelocityField(grid)
    vel.zero()
    u = np.zeros(n, np.float32)
    u[(li == 0) & (lk < 6)] = edge_u  # x-min faces: no -x neighbour
    vel.u.assign(wp.array(u, device=DEVICE))
    for weight in (vel.wu, vel.wv, vel.ww):
        weight.fill_(1.0)
    collision_vel = wp.zeros(n, dtype=wp.vec3, device=DEVICE)
    counts = ScalarField(grid, dtype=wp.int32)
    solver = P.PressureSolver(grid, DEVICE)
    solver.build_rhs(vel, vt.data, collision_vel, counts.data)
    solver.build_preconditioner(vt.data)
    solver.solve_jacobi(vt.data, 4000)
    solver.project(vel, vt.data, collision_vel, max_gradient=1.0e6)
    solver.build_rhs(vel, vt.data, collision_vel, counts.data)
    divergence = solver.divergence.numpy()[:512]
    fluid = vt_np != 0
    return float(np.abs(divergence[fluid & (li == 0)]).max())


def test_wall_face_velocity_leaves_no_divergence_after_projection():
    still = _post_projection_edge_divergence(0.0)
    pushed = _post_projection_edge_divergence(1.0)
    # 1/h = 100 per unit face velocity before the fix.
    assert pushed == pytest.approx(still, abs=1e-3)


def test_no_kernel_reads_a_neighbour_through_an_eager_where():
    """``wp.where`` evaluates both operands, so ``wp.where(n >= 0, a[n], x)``
    reads ``a[-1]`` whenever the neighbour is missing."""
    import pathlib
    import re

    package = pathlib.Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "sim"
    pattern = re.compile(r"wp\.where\(\s*(\w+)\s*>=\s*0\s*,\s*\w+\[\s*\1\s*\]")
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(package.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
