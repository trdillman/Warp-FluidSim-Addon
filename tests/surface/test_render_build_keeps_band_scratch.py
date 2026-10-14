"""P11: a render-grid SDF build keeps the capillary propagation scratch.

With surface tension on and a separate render grid (resolution multiplier),
every render build took the "band inactive" branch of ``build_sdf`` and
released the closest-point scratch the solver-grid build needs, so the next
solver build allocated it again: one release/reallocation per frame. The
scratch is fully rewritten by each propagation, so keeping it cannot change
results; this pins both the single allocation and bitwise-equal particles.
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


def _run(device, monkeypatch, fresh_every_build):
    allocations = []
    original = S.SurfacePipeline._ensure_band_buffers

    def counting(self, capacity, device_):
        if fresh_every_build:
            self.release_band_buffers()
        before = getattr(self, "_band_capacity", 0)
        original(self, capacity, device_)
        if getattr(self, "_band_capacity", 0) != before:
            allocations.append(capacity)

    monkeypatch.setattr(S.SurfacePipeline, "_ensure_band_buffers", counting)
    settings = H.settings(device, narrow_band=False, surface_resolution_mode=1,
                          surface_resolution_multiplier=2.0,
                          surface_tension_enabled=True, min_substeps=2, max_substeps=2)
    settings.surface_enabled = True
    solver = FluidSolver(settings)
    try:
        solver.inject_particles(H.lattice(0.5))
        for _ in range(FRAMES):
            solver.step_frame()
        assert solver.surface.grid is not solver.grid, "fixture needs a render grid"
        assert solver.surface.last_sdf_splat is not None
        n = solver.particles.refresh_host_count()
        state = (solver.particles.position.numpy()[:n].copy(),
                 solver.particles.velocity.numpy()[:n].copy())
    finally:
        solver.release()
        monkeypatch.undo()
    return state, allocations


def test_render_builds_do_not_release_the_capillary_scratch(cuda_device, monkeypatch):
    fresh_state, fresh_allocations = _run(cuda_device, monkeypatch, True)
    state, allocations = _run(cuda_device, monkeypatch, False)
    assert len(fresh_allocations) > FRAMES, "fixture must use the propagation scratch"
    assert len(allocations) == 1, f"scratch allocated {len(allocations)} times"
    assert np.array_equal(state[0], fresh_state[0])
    assert np.array_equal(state[1], fresh_state[1])
