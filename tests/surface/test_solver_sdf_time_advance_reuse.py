"""P11: an unchanged solver SDF is not rebuilt just because time advanced.

The solver SDF cache key holds frame and time as a stand-in for particle
positions. Whitewater builds the SDF after advection; the solver then
advances time (end of substep) and frame (end of frame) without moving a
particle, so the next consumer (surface tension, or frame-end surfacing)
rebuilt an identical SDF. The key is now carried across those advances. The
simulation must be bitwise unchanged while the build count drops.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "solver"))

wp = pytest.importorskip("warp")

import _narrow_band_harness as H  # noqa: E402

from fluidSim_warp_blender51.sim import surface as S  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver  # noqa: E402

FRAMES = 3


def _run(device, monkeypatch, carry):
    builds = []
    original = S.SurfacePipeline.build_sdf

    def counting(self, grid, *args, **kwargs):
        builds.append(grid is self.solver.grid)
        return original(self, grid, *args, **kwargs)

    monkeypatch.setattr(S.SurfacePipeline, "build_sdf", counting)
    if not carry:
        monkeypatch.setattr(S.SurfacePipeline, "advance_solver_sdf_key",
                            lambda self, previous_key: None)
    settings = H.settings(device, narrow_band=False, surface_tension_enabled=True,
                          max_whitewater=50_000, min_substeps=2, max_substeps=2)
    settings.surface_enabled = True
    settings.whitewater_enabled = True
    solver = FluidSolver(settings)
    try:
        solver.inject_particles(H.lattice(0.5))
        for _ in range(FRAMES):
            solver.step_frame()
        n = solver.particles.refresh_host_count()
        m = solver.whitewater.refresh_host_count()
        state = (solver.particles.position.numpy()[:n].copy(),
                 solver.particles.velocity.numpy()[:n].copy(),
                 solver.whitewater.position.numpy()[:m].copy(),
                 solver.surface.sdf.data.numpy()[:solver._nvox].copy())
    finally:
        solver.release()
        monkeypatch.undo()
    return state, builds


def test_carried_solver_sdf_matches_rebuilt_sdf(cuda_device, monkeypatch):
    ref_state, ref_builds = _run(cuda_device, monkeypatch, carry=False)
    state, builds = _run(cuda_device, monkeypatch, carry=True)
    assert all(builds) and all(ref_builds), "fixture surfaces on the solver grid"
    assert len(builds) < len(ref_builds), (len(builds), len(ref_builds))
    for got, want in zip(state, ref_state):
        assert got.shape == want.shape
        assert np.array_equal(got, want)
