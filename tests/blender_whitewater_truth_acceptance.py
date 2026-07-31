"""Blender/CUDA acceptance for render-independent SDF and honest overflow.

Run with:
  blender --background --factory-startup --python tests/blender_whitewater_truth_acceptance.py
"""

from contextlib import nullcontext
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warp as wp

from fluidSim_warp_blender51.sim.enums import SurfaceUpdateCadence
from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.solver import SolverSettings
from fluidSim_warp_blender51.sim.surface import SurfacePipeline
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem


def _grid(device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0), size=(0.8, 0.8, 0.8),
        voxel_size=0.1, leaf_capacity=4, device=device)
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.79, 0.79, 0.79))
    assert grid.finalize_topology() == 1
    return grid


def _render_disabled_sdf_acceptance(device, grid):
    settings = SolverSettings(
        device=device,
        voxel_size=0.1,
        surface_enabled=False,
        whitewater_enabled=True,
        surface_update_cadence=SurfaceUpdateCadence.MANUAL,
        surface_filter_iterations=0,
        surface_smooth_iterations=0,
    )
    positions = wp.array([(0.2, 0.2, 0.2)], dtype=wp.vec3, device=device)
    count = wp.array([1], dtype=wp.int32, device=device)

    class FakeSolver:
        def _np_dim(self):
            return 1

        def _surface_positions(self):
            return positions

        def _phase(self, _name):
            return nullcontext()

    solver = FakeSolver()
    solver.frame = 0
    solver.sim_time = 0.0
    solver._nvox = grid.voxel_capacity
    solver.particles = type("Particles", (), {"count": count})()
    solver.device = device
    solver.settings = settings
    solver.grid = grid
    surface = SurfacePipeline(solver)
    solver.surface = surface

    assert surface.after_substep() is True
    first = surface.sdf.data.numpy().copy()
    wp.copy(positions, wp.array([(0.6, 0.6, 0.6)], dtype=wp.vec3, device=device))
    solver.sim_time = 0.1
    assert surface.after_substep() is True
    second = surface.sdf.data.numpy().copy()

    assert surface.solver_sdf_key is not None
    assert surface.mesh_cache is None
    assert not np.array_equal(first, second), "moving liquid must refresh solver SDF"


def _capacity_acceptance(device, grid):
    liquid_positions = wp.array(
        [(0.15, 0.35, 0.35), (0.25, 0.35, 0.35),
         (0.35, 0.35, 0.35), (0.45, 0.35, 0.35)],
        dtype=wp.vec3, device=device)
    liquid_velocities = wp.array(
        [(1.0, 0.0, 0.0)] * 4, dtype=wp.vec3, device=device)
    liquid_count = wp.array([4], dtype=wp.int32, device=device)
    churn = wp.ones(grid.voxel_capacity, dtype=float, device=device)
    surface_sdf = wp.zeros(grid.voxel_capacity, dtype=float, device=device)

    outputs = []
    for _ in range(2):
        ww = WhitewaterSystem(capacity=2, device=device, candidate_capacity=4)
        ww.reset_frame_diagnostics()
        ww.emit(
            candidate_count=4,
            liquid_positions=liquid_positions,
            liquid_velocities=liquid_velocities,
            liquid_count=liquid_count,
            churn=churn,
            surface_sdf=surface_sdf,
            leaf_table=grid.leaf_table,
            voxels_dim=grid.voxels_dim,
            leaves_dim=grid.leaves_dim,
            origin=grid.origin,
            voxel_size=grid.voxel_size,
            dt=1.0,
            churn_threshold=0.0,
            emission_rate=100.0,
            surface_band=10.0,
            min_speed=0.0,
            life_min=2.0,
            life_max=1.0,
            seed=123,
        )
        wp.synchronize_device(device)
        diagnostics = ww.refresh_diagnostics()
        outputs.append(ww.position.numpy().copy())
        assert diagnostics["required"] == 4
        assert diagnostics["admitted"] == 2
        assert diagnostics["dropped"] == 2
        assert diagnostics["status"] == "CAPACITY_EXHAUSTED"
        assert np.allclose(ww.life.numpy(), 2.0)

    assert np.array_equal(outputs[0], outputs[1])
    # Ascending liquid tids win, so both admitted particles originate below x=.3.
    assert float(np.max(outputs[0][:, 0])) < 0.31


def main():
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("WHITEWATER-TRUTH-ACCEPTANCE requires CUDA")
    device = "cuda:0"
    grid = _grid(device)
    _render_disabled_sdf_acceptance(device, grid)
    _capacity_acceptance(device, grid)
    print("WHITEWATER-TRUTH-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
