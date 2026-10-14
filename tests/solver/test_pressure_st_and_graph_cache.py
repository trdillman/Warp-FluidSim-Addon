"""Pressure-solver CUDA-graph cache regressions."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME, VOXEL_FLUID  # noqa: E402
from fluidSim_warp_blender51.sim.fields import SparseGrid  # noqa: E402
from fluidSim_warp_blender51.sim.pressure import PressureSolver  # noqa: E402


def _one_leaf_grid(device, mark=True):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        voxel_size=0.05,
        leaf_capacity=1,
        device=device,
    )
    grid.begin_topology()
    if mark:
        grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == (1 if mark else 0)
    assert grid.voxel_capacity == LEAF_VOLUME
    return grid


def _prepared_solver(device):
    grid = _one_leaf_grid(device)
    solver = PressureSolver(grid, device)
    voxel_type = wp.full(
        grid.voxel_capacity, VOXEL_FLUID, dtype=wp.int32, device=device)
    solver.build_preconditioner(voxel_type)
    solver.rhs.fill_(1.0)
    # records the launch commands and captures the first graphs
    solver.solve_pcg(voxel_type, 4, 1.0e-8)
    return solver


def test_pcg_graph_cache_evicts_lru_instead_of_flushing(cuda_device):
    """A cold key must cost one eviction, not the whole captured map.

    Access pattern mirrors real solves: a small hot set of steady-state batch
    sizes plus one fresh tail-iteration key per solve. Wholesale flushing makes
    every hot key re-capture (and re-pay ``wp.synchronize_device``) as soon as
    the map fills.
    """
    solver = _prepared_solver(cuda_device)
    solver._pcg_graphs.clear()
    # even batches only, so start_parity stays 0 and the key is (n, batch, 0)
    solver.rz, solver.rz_new = solver._pcg_rz_a, solver._pcg_rz_b
    work_n = solver.grid.voxel_capacity
    hot = [2, 4, 6, 8, 10, 12]
    solves = 20

    captures = 0
    start = time.perf_counter()
    for i in range(solves):
        for batch in hot + [14 + 2 * i]:
            if (work_n, batch, 0) not in solver._pcg_graphs:
                captures += 1
            solver._launch_pcg_batch(work_n, batch)
    wp.synchronize_device(cuda_device)
    elapsed = time.perf_counter() - start

    distinct_keys = len(hot) + solves
    print(f"\ngraph captures={captures} (distinct keys={distinct_keys}) "
          f"elapsed={elapsed:.3f}s")
    assert captures == distinct_keys, (
        f"expected one capture per distinct key ({distinct_keys}), "
        f"got {captures} in {elapsed:.3f}s")
    # a cold key evicts one entry, so the cache stays full instead of resetting
    assert len(solver._pcg_graphs) == 8


def _count_body_captures(solver):
    """Count conditional-graph captures: each one records the body once."""
    counter = {"captures": 0}
    record = solver._record_pcg_iterations

    def counting_record(*args):
        counter["captures"] += 1
        return record(*args)

    solver._record_pcg_iterations = counting_record
    return counter


def test_pcg_conditional_graph_cache_evicts_lru_instead_of_flushing(cuda_device):
    """REVIEW-015: a cold conditional-graph key flushed all eight captures.

    Same access pattern as the host-batched cache test above: a hot set of
    solve shapes plus one fresh shape per solve. Every flush forced the hot
    shapes to re-capture and re-pay ``wp.synchronize_device``.
    """
    if not wp.is_conditional_graph_supported():
        pytest.skip("conditional CUDA graphs unsupported")
    solver = _prepared_solver(cuda_device)
    assert solver._ensure_pcg_dot()
    solver._pcg_conditional_graphs.clear()
    counter = _count_body_captures(solver)
    work_n = solver.grid.voxel_capacity
    hot = [25, 50, 75, 100, 125, 150]  # max_iterations of the hot shapes
    solves = 12

    for i in range(solves):
        for max_iterations in hot + [175 + 25 * i]:
            solver._get_pcg_conditional_graphs(work_n, 1, max_iterations)

    distinct_keys = len(hot) + solves
    assert counter["captures"] == distinct_keys
    # a cold key evicts one entry, so the cache stays full instead of resetting
    assert len(solver._pcg_conditional_graphs) == solver._PCG_GRAPH_CAPACITY
    solver.release()


def test_pcg_conditional_graph_reads_tolerance_at_launch(cuda_device):
    """REVIEW-015: tol_sq was baked into the captured graph and its cache key.

    tol_sq scales with the active voxel count, which moves on nearly every
    solve, so each solve minted a key that was never reused (dam_break: 39
    captures in 371 solves, one per distinct tol_sq). One capture must serve
    every tolerance and still stop at the tolerance each solve asked for.
    """
    import numpy as np

    if not wp.is_conditional_graph_supported():
        pytest.skip("conditional CUDA graphs unsupported")
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    voxel_type = wp.full(
        grid.voxel_capacity, VOXEL_FLUID, dtype=wp.int32, device=cuda_device)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0], rhs[1] = 1.0, -1.0  # zero mean: solvable under pure Neumann
    solver.build_preconditioner(voxel_type)

    def solve(tolerance):
        wp.copy(solver.rhs, wp.array(rhs, device=cuda_device))
        status = solver.solve_pcg(
            voxel_type, max_iterations=400, tolerance=tolerance, check_every=25)
        assert status["converged"], status
        assert solver.last_pcg_execution_mode == "cuda_conditional"
        assert solver.last_residual_squared < (
            tolerance * tolerance * grid.voxel_capacity)
        return solver.last_iterations

    loose = solve(1.0e-3)
    graphs = dict(solver._pcg_conditional_graphs)
    counter = _count_body_captures(solver)
    tight = solve(1.0e-7)
    assert tight > loose  # the reused graph honored the tighter tolerance
    assert solve(1.0e-3) == loose  # ...and did not keep it afterwards
    assert counter["captures"] == 0
    assert solver._pcg_conditional_graphs == graphs
    solver.release()


def test_native_pcg_capture_failure_is_reported_in_frame_stats(cuda_device):
    """REVIEW-016: a failed graph capture degraded silently to host batches."""
    from types import SimpleNamespace

    from fluidSim_warp_blender51.sim.frame_runtime import FrameStats
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    if not wp.is_conditional_graph_supported():
        pytest.skip("conditional CUDA graphs unsupported")
    solver = _prepared_solver(cuda_device)
    voxel_type = wp.full(
        solver.grid.voxel_capacity, VOXEL_FLUID, dtype=wp.int32,
        device=cuda_device)

    def refuse_capture(*_args):
        raise RuntimeError("capture refused")

    solver._get_pcg_conditional_graphs = refuse_capture
    solver.rhs.fill_(1.0)
    solver.solve_pcg(voxel_type, 25, 1.0e-8)

    assert solver.last_pcg_execution_mode == "host_batched"
    assert solver.last_pcg_native_error == "RuntimeError: capture refused"
    owner = SimpleNamespace(pressure_solver=solver, viscosity_solver=None)
    message = FluidSolver._native_solver_message(owner)
    assert message.startswith("Pressure solve fell back from CUDA graphs")
    assert message.endswith("RuntimeError: capture refused")
    stats = FrameStats(native_solver_message=message)
    assert stats.as_dict()["nativeSolverMessage"] == message
    solver.release()


def test_native_pcg_module_unloads_on_pressure_solver_release(cuda_device):
    from fluidSim_warp_blender51.sim import cuda_reduction

    solver = _prepared_solver(cuda_device)
    assert len(cuda_reduction._MODULE_CACHE.entries) == 1

    solver.release()

    assert cuda_reduction._MODULE_CACHE.entries == {}
    solver.release()
