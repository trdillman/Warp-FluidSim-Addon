"""Device-evaluated viscosity PCG convergence check (CUDA conditional graph).

The shipped loop read the residual back to the host on every check
(``float(self.residual.numpy()[0])``), draining the stream once per check.
These tests pin the two properties that replacement has to hold:

* at the *same* cadence the device-checked loop stops where the host-checked
  loop stops and produces the same velocities (numerics-neutral), and
* a non-converged component solve is reported rather than silently returned.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim import viscosity as V
from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME, VOXEL_FLUID
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid
from fluidSim_warp_blender51.sim.viscosity import ViscositySolver


@pytest.fixture(scope="module")
def cuda_device(cuda_device):
    """Narrow the shared conftest device to one with conditional graphs."""
    if not wp.is_conditional_graph_supported():
        pytest.skip("CUDA conditional graphs not supported")
    return cuda_device


def _one_leaf_grid(device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        voxel_size=0.05,
        leaf_capacity=1,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == 1
    assert grid.voxel_capacity == LEAF_VOLUME
    return grid


def _seed_velocity(vel, n):
    rng = np.random.default_rng(20260902)
    for comp in (vel.u, vel.v, vel.w):
        comp.assign(rng.standard_normal(n).astype(np.float32))
    for weight in (vel.wu, vel.wv, vel.ww):
        weight.fill_(1.0)


def _case(device, *, check_every, conditional, iterations=40,
          tolerance=1.0e-4, viscosity_constant=1.0, dt=0.01):
    grid = _one_leaf_grid(device)
    n = grid.voxel_capacity
    solver = ViscositySolver(grid, device)
    if not conditional:
        solver._conditional_available = False
    vel = MacVelocityField(grid)
    _seed_velocity(vel, n)
    voxel_type = wp.full(n, VOXEL_FLUID, dtype=wp.int32, device=device)
    viscosity_field = wp.full(n, viscosity_constant, dtype=float, device=device)
    status = solver.solve(
        vel, viscosity_field, voxel_type, dt, use_pcg=True,
        iterations=iterations, tolerance=tolerance, check_every=check_every)
    result = {
        "status": status,
        "components": [dict(s) for s in solver.last_component_statuses],
        "mode": solver.last_pcg_execution_mode,
        "u": vel.u.numpy()[:n].copy(),
        "v": vel.v.numpy()[:n].copy(),
        "w": vel.w.numpy()[:n].copy(),
    }
    solver.release()
    return result


def test_device_check_matches_host_check_at_the_shipped_cadence(cuda_device):
    """Numerics-neutrality of moving the check on device.

    Only the *evaluation site* of the criterion changes here, so at the
    shipped cadence of 10 both paths must stop on the same iteration with the
    same verdict. Values are compared with a tolerance rather than bit-exactly
    because the captured loop reduces with an accumulating atomicAdd dot
    product while the host loop uses warp's tree reduction.
    """
    device = _case(cuda_device, check_every=10, conditional=True)
    host = _case(cuda_device, check_every=10, conditional=False)

    assert device["mode"] == "cuda_conditional"
    assert host["mode"] == "host_checked"
    assert [c["iterations"] for c in device["components"]] == [
        c["iterations"] for c in host["components"]]
    assert [c["reason"] for c in device["components"]] == [
        c["reason"] for c in host["components"]]
    for axis in ("u", "v", "w"):
        np.testing.assert_allclose(
            device[axis], host[axis], rtol=2.0e-4, atol=2.0e-6)


def test_per_iteration_cadence_never_runs_longer_than_the_shipped_cadence(
        cuda_device):
    """Checking every iteration can only stop earlier, never later."""
    every_iteration = _case(cuda_device, check_every=1, conditional=True)
    shipped = _case(cuda_device, check_every=10, conditional=True)
    for fine, coarse in zip(every_iteration["components"],
                            shipped["components"]):
        assert fine["iterations"] <= coarse["iterations"]


def test_converged_components_report_a_residual_within_tolerance(cuda_device):
    """A converged verdict has to be backed by the residual it claims."""
    n = LEAF_VOLUME
    tolerance = 1.0e-4
    threshold = tolerance * math.sqrt(n)
    result = _case(cuda_device, check_every=1, conditional=True,
                   tolerance=tolerance)
    converged = [c for c in result["components"] if c["converged"]]
    assert converged, result["components"]
    for component in converged:
        assert component["reason"] == V.VISCOSITY_STATUS_CONVERGED
        assert component["residual"] < threshold


def test_non_converged_component_is_reported_not_silent(cuda_device):
    """An impossible tolerance must surface as max_iterations, not silence."""
    result = _case(cuda_device, check_every=1, conditional=True,
                   iterations=4, tolerance=1.0e-30)
    assert result["status"]["reason"] == V.VISCOSITY_STATUS_MAX_ITERATIONS
    assert result["status"]["converged"] is False
    assert all(c["iterations"] == 4 for c in result["components"])
    assert all(c["reason"] == V.VISCOSITY_STATUS_MAX_ITERATIONS
               for c in result["components"])


def test_empty_grid_reports_skipped(cuda_device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.4, 0.4, 0.4), voxel_size=0.05,
        leaf_capacity=1, device=cuda_device)
    grid.begin_topology()
    assert grid.finalize_topology() == 0
    solver = ViscositySolver(grid, cuda_device)
    vel = MacVelocityField(grid)
    voxel_type = wp.zeros(
        max(1, grid.voxel_capacity), dtype=wp.int32, device=cuda_device)
    viscosity_field = wp.zeros(
        max(1, grid.voxel_capacity), dtype=float, device=cuda_device)
    status = solver.solve(vel, viscosity_field, voxel_type, 0.01)
    assert status["reason"] == V.VISCOSITY_STATUS_SKIPPED
    solver.release()


def test_conditional_graphs_are_reused_across_identical_solves(cuda_device):
    """The capture cost must be paid once per (voxels, dt, component) shape."""
    grid = _one_leaf_grid(cuda_device)
    n = grid.voxel_capacity
    solver = ViscositySolver(grid, cuda_device)
    vel = MacVelocityField(grid)
    _seed_velocity(vel, n)
    voxel_type = wp.full(n, VOXEL_FLUID, dtype=wp.int32, device=cuda_device)
    viscosity_field = wp.full(n, 1.0, dtype=float, device=cuda_device)

    solver.solve(vel, viscosity_field, voxel_type, 0.01, iterations=8,
                 tolerance=1.0e-4, check_every=1)
    after_first = solver._graph_captures
    assert after_first == 3  # one per velocity component
    solver.solve(vel, viscosity_field, voxel_type, 0.01, iterations=8,
                 tolerance=1.0e-4, check_every=1)
    assert solver._graph_captures == after_first
    solver.release()
    assert not solver._pcg_conditional_graphs
