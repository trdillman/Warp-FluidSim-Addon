"""4.0 lane D: a deep resting column holds its analytic height.

The 3.0 notes listed "under the default Jacobi preconditioner a deep liquid
column settles about a third lower than it should within 30 frames". 4.0 makes
Multigrid the default (docs/specs/4.0/SPEC-defaults-4.md); this pins that a
2 m column in a closed tank settles within 2 % of its fill height by frame 30
under the default solver settings.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")

H = 1.0 / 32.0
FILL = 2.0
DOMAIN_MAX = (0.25, 0.25, 2.25)
FRAMES = 30


def _lattice(per_axis=2):
    spacing = H / per_axis
    counts = (int(DOMAIN_MAX[0] / spacing), int(DOMAIN_MAX[1] / spacing),
              int(FILL / spacing))
    axes = [(np.arange(n) + 0.5) * spacing for n in counts]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


def _settle(device, **extra):
    points = _lattice()
    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=256, max_particles=len(points) + 4096,
        surface_enabled=False, whitewater_enabled=False, device=device,
        **extra))
    try:
        solver.inject_particles(points)
        for _ in range(FRAMES):
            solver.step_frame()
        height = float(np.percentile(solver.particle_positions()[:, 2], 99.0))
        return height, dict(solver.pressure_solver.last_status)
    finally:
        solver.release()


def test_default_preconditioner_is_multigrid():
    assert SolverSettings().pressure_preconditioner == "MULTIGRID"


def test_two_metre_column_holds_its_height_under_the_default_settings(cuda_device):
    height, status = _settle(cuda_device)
    assert abs(height - FILL) <= 0.02 * FILL, (height, status)
    assert status["reason"] == "converged", status


@pytest.mark.parametrize("preconditioner", ["JACOBI", "MULTIGRID"])
def test_two_metre_column_measurement(cuda_device, preconditioner):
    """Recorded for the 4.0 notes: both preconditioners on the same column."""
    height, status = _settle(cuda_device, pressure_preconditioner=preconditioner)
    print(f"\n{preconditioner}: height={height:.4f} status={status}")
