"""Lane Q: a collapsing column must not make narrow band outgrow the full band.

dam_break showed 50k, 237k, then the 600k capacity against a constant 256k:
reseed re-densified every sparse band cell while nothing removed the surplus.
This is the small moving analogue: a deep column that spreads along a floor
until all of it is band.  Narrow band holds fewer particles than the full
band while it has an interior, and never more once it has none.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from _narrow_band_harness import H

from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

pytest.importorskip("warp")

DOMAIN_MAX = (1.0, 0.25, 0.75)
COLUMN = (0.3, 0.25, 0.6)
CAPACITY = 200_000
FRAMES = 36


def _run(device, narrow_band):
    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=512, max_particles=CAPACITY, surface_enabled=False,
        whitewater_enabled=False, narrow_band_enabled=narrow_band,
        narrow_band_voxels=3.0, device=device))
    spacing = H / 2.0
    axes = [(np.arange(int(n / spacing)) + 0.5) * spacing for n in COLUMN]
    grid = np.meshgrid(*axes, indexing="ij")
    solver.inject_particles(
        np.stack([g.ravel() for g in grid], axis=1).astype(np.float32))
    counts = []
    try:
        for _ in range(FRAMES):
            solver.step_frame()
            wp.synchronize()
            counts.append(len(solver.particle_positions()))
        overflow = (solver.narrow_band.reseed_overflow
                    if narrow_band else False)
    finally:
        solver.release()
    return counts, overflow


def test_collapsing_column_never_outgrows_the_full_band(cuda_device):
    full, _ = _run(cuda_device, narrow_band=False)
    band, overflow = _run(cuda_device, narrow_band=True)
    assert not overflow
    assert max(band) <= CAPACITY
    assert band[0] < 0.6 * full[0], "the deep column must start with an interior"
    for frame, (b, f) in enumerate(zip(band, full), start=1):
        assert b <= f, f"frame {frame}: narrow band {b} > full band {f}"
