"""ST-FLIP PCG initialization and CUDA-graph cache regressions."""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME, VOXEL_FLUID
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid
from fluidSim_warp_blender51.sim.pressure import PressureSolver


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


def test_solve_pcg_st_zeroes_working_vectors_before_iterating(cuda_device):
    """solve_pcg_st must not reduce over slots its initializer never wrote.

    The launch dimension comes from the host leaf mirror while the kernels are
    guarded by the device-side ``num_leaves``; when the two disagree the tail
    slots keep the previous frame's values. solve_pcg zeroes the working set to
    neutralize exactly that, and the ST path has to do the same.
    """
    grid = _one_leaf_grid(cuda_device, mark=False)
    solver = PressureSolver(grid, cuda_device)
    vel = MacVelocityField(grid)
    voxel_type = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)

    # leftovers a previous frame's solve would have deposited
    solver.pressure.fill_(7.0)
    solver.r.fill_(3.0)
    solver.z.fill_(5.0)
    solver.p_dir.fill_(4.0)
    solver.ap.fill_(2.0)

    status = solver.solve_pcg_st(voxel_type, 8, 1.0e-6, vel)

    assert status["converged"], status
    assert solver.last_residual == 0.0
    assert float(solver.pressure.numpy()[0]) == 0.0


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


def test_native_pcg_module_unloads_on_pressure_solver_release(cuda_device):
    from fluidSim_warp_blender51.sim import cuda_reduction

    solver = _prepared_solver(cuda_device)
    assert len(cuda_reduction._MODULE_CACHE.entries) == 1

    solver.release()

    assert cuda_reduction._MODULE_CACHE.entries == {}
    solver.release()
