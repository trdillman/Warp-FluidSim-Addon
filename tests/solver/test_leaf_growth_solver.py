"""A frame refused for leaf capacity continues unchanged on a grown solver.

Real GPU solvers: the refusal leaves the live state intact at the start of the
frame (substep 0), so its checkpoint restores into a solver with more leaves and
the same frame then gives the answer a large-enough solver gives directly.
"""

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.fields import TopologyCapacityError
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

warp = pytest.importorskip("warp")

BIG = 64
FRAMES = 12


def _settings(device, capacity):
    return SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.03125,
        leaf_capacity=capacity,
        max_particles=8192,
        max_whitewater=64,
        surface_enabled=False,
        whitewater_enabled=False,
        topology_expand=0,
        device=device,
    )


def _lattice(origin, spacing, counts):
    axes = [origin[i] + spacing * np.arange(counts[i]) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


POINTS = _lattice((0.10, 0.40, 0.30), 0.03, (5, 5, 5))
VELOCITY = (3.0, 0.0, 0.0)


def _start(device, capacity):
    solver = FluidSolver(_settings(device, capacity))
    solver.inject_particles(POINTS, velocity=VELOCITY)
    return solver


def test_refused_frame_continues_on_a_grown_solver(cuda_device):
    probe = _start(cuda_device, BIG)
    counts, positions = [], []
    try:
        for _ in range(FRAMES):
            probe.step_frame()
            counts.append(int(probe.grid.refresh_topology_counts()))
            positions.append(probe.particle_positions().copy())
    finally:
        probe.release()
    growth = [i for i in range(1, FRAMES) if counts[i] > max(counts[:i])]
    assert growth, f"the particles never needed more leaves: {counts}"
    k = growth[0]
    small_capacity = max(counts[:k])

    small = _start(cuda_device, small_capacity)
    grown = FluidSolver(_settings(cuda_device, BIG))
    try:
        for _ in range(k):
            small.step_frame()
        with pytest.raises(TopologyCapacityError) as refused:
            small.step_frame()
        assert refused.value.substep_index == 0
        assert refused.value.required == counts[k]
        assert refused.value.capacity == small_capacity

        # The refusal left the live solver exactly at the start of frame k + 1.
        assert small.frame == k
        grown.import_checkpoint_state(small.export_checkpoint_state())
        grown.step_frame()

        assert grown.frame == k + 1
        np.testing.assert_allclose(
            grown.particle_positions(), positions[k], atol=1e-4)
        assert int(grown.grid.refresh_topology_counts()) == counts[k]
        assert grown.grid.capacity == BIG
    finally:
        grown.release()
        small.release()
