"""End-of-frame whitewater density is persistent state, not scratch.

Regression for review item B13. ``ww_density`` is splatted at frame end and
read by foam ageing during the next frame, after the topology rebuild:
- the rebuild re-ranks leaf slots but did not remap the density, so foam
  read another voxel's density;
- checkpoint restore zeroed it, so a resumed bake aged foam differently from
  an uninterrupted one;
- reset did not clear it.
"""
from __future__ import annotations

import numpy as np
import pytest


def _density_by_voxel(solver):
    wp = pytest.importorskip("warp")
    wp.synchronize_device(solver.device)
    nvox = solver._nvox
    values = solver.ww_density.data.numpy()[:nvox]
    leaves = solver.grid.leaf_coords.numpy()
    local = np.arange(nvox) % 512
    slots = np.arange(nvox) // 512
    ijk = (leaves[slots] * 8
           + np.stack([local & 7, (local >> 3) & 7, local >> 6], axis=1))
    return {tuple(int(c) for c in ijk[i]): float(values[i])
            for i in np.nonzero(values)[0]}


def _solver_with_whitewater(cuda_device):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.6, 1.6, 1.6),
        voxel_size=0.05, leaf_capacity=64, max_particles=8192,
        max_whitewater=1024, whitewater_enabled=True, device=cuda_device)
    solver = FluidSolver(settings)
    rng = np.random.default_rng(4)
    solver.inject_particles(rng.uniform(
        (1.0, 1.0, 1.0), (1.3, 1.3, 1.3), (800, 3)).astype(np.float32))
    count = 64
    positions = rng.uniform((1.05, 1.05, 1.05), (1.25, 1.25, 1.25),
                            (count, 3)).astype(np.float32)
    solver.whitewater.restore_checkpoint_prefix({
        "whitewater_position": positions,
        "whitewater_velocity": np.zeros((count, 3), np.float32),
        "whitewater_age": np.zeros(count, np.float32),
        "whitewater_life": np.full(count, 2.0, np.float32),
        "whitewater_alive": np.ones(count, np.int32),
        "whitewater_type": np.zeros(count, np.int32),
    }, count)
    solver._rebuild_topology()
    solver._finalize_whitewater()
    return solver


def test_density_follows_its_voxels_across_a_leaf_re_rank(cuda_device):
    solver = _solver_with_whitewater(cuda_device)
    try:
        before = _density_by_voxel(solver)
        assert before
        # Liquid near the domain origin adds leaves that rank before the
        # whitewater's leaves, shifting their slots.
        solver.inject_particles(np.full((8, 3), 0.1, dtype=np.float32))
        solver._rebuild_topology()
        assert _density_by_voxel(solver) == before
    finally:
        solver.release()


def test_checkpoint_restore_reproduces_the_frame_end_density(cuda_device):
    solver = _solver_with_whitewater(cuda_device)
    try:
        before = _density_by_voxel(solver)
        state = solver.export_checkpoint_state()
        solver.import_checkpoint_state(state)
        assert _density_by_voxel(solver) == before
    finally:
        solver.release()


def test_reset_clears_the_density(cuda_device):
    solver = _solver_with_whitewater(cuda_device)
    try:
        assert _density_by_voxel(solver)
        solver.reset()
        assert not np.any(solver.ww_density.data.numpy())
    finally:
        solver.release()
