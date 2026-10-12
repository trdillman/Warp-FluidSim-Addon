"""B37: per-substep topology rebuilds keep the surface-tension leaf halo.

The capillary SDF needs leaves around the liquid wide enough for its splat,
force band and smoothing stencils (``_surface_tension_topology_expand``). The
frame-level rebuild folds that halo into its expansion; when an effect asks
for a topology rebuild every substep, ``step_frame`` passed ``None`` and the
rebuild fell back to the authored ``topology_expand`` alone, truncating the
SDF at the active boundary.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.frame_features import FrameFeatures  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings  # noqa: E402

H = 0.05


def _blob():
    axis = np.arange(0.70, 0.90, H / 2) + H / 4
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([g.ravel() for g in grid], axis=1).astype(np.float32)


@pytest.mark.parametrize("rebuild_each_substep", [False, True])
def test_topology_rebuilds_include_the_capillary_halo(
        cuda_device, monkeypatch, rebuild_each_substep):
    monkeypatch.setattr(FrameFeatures, "rebuild_topology_each_substep",
                        property(lambda self: rebuild_each_substep))
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.6, 1.6, 1.6),
        voxel_size=H,
        leaf_capacity=64,
        max_particles=4096,
        min_substeps=2,
        max_substeps=2,
        topology_expand=0,
        surface_tension_enabled=True,
        whitewater_enabled=False,
        surface_enabled=False,
        device=cuda_device,
    )
    solver = FluidSolver(settings)
    try:
        halo = solver._surface_tension_topology_expand()
        assert halo > settings.topology_expand, "fixture must need a halo"
        solver.inject_particles(_blob())
        expands = []
        original = solver._rebuild_topology

        def spy(expand=None):
            expands.append(expand)
            return original(expand)

        monkeypatch.setattr(solver, "_rebuild_topology", spy)
        solver.step_frame()
        assert len(expands) == (2 if rebuild_each_substep else 1)
        effective = [settings.topology_expand if e is None else e for e in expands]
        assert min(effective) >= halo, effective
    finally:
        solver.release()
