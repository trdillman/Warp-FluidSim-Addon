"""P18: the viscosity solve stages dt and the tolerance once, not per component.

``_record_solve_scalars`` used a ``wp.launch`` for each of the three
component solves although ``dt`` and the squared tolerance are the same for
all three. It now launches only when the pair changes. The staged device
values, and so the simulation, must be bitwise unchanged.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

wp = pytest.importorskip("warp")

import _narrow_band_harness as H  # noqa: E402

from fluidSim_warp_blender51.sim import viscosity as V  # noqa: E402
from fluidSim_warp_blender51.sim.solver import FluidSolver  # noqa: E402

FRAMES = 3


def _run(device, monkeypatch, staged_once):
    if not staged_once:
        def always(self, dt, tol_sq):
            wp.launch(V.k_viscosity_set_solve_scalars, dim=1,
                      inputs=[dt, tol_sq, self._solve_dt, self._pcg_tol_sq],
                      device=self.device)
        monkeypatch.setattr(V.ViscositySolver, "_record_solve_scalars", always)
    launches = []
    original_launch = V.wp.launch

    def counting(kernel, *args, **kwargs):
        if kernel is V.k_viscosity_set_solve_scalars:
            launches.append(kwargs.get("inputs", args[1] if len(args) > 1 else None)[:2])
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(V.wp, "launch", counting)
    solver = FluidSolver(H.settings(
        device, narrow_band=False, viscosity_enabled=True, viscosity_constant=0.2,
        min_substeps=2, max_substeps=2))
    try:
        solver.inject_particles(H.lattice(0.4))
        for _ in range(FRAMES):
            solver.step_frame()
        n = solver.particles.refresh_host_count()
        state = (solver.particles.position.numpy()[:n].copy(),
                 solver.particles.velocity.numpy()[:n].copy())
    finally:
        solver.release()
        monkeypatch.undo()
    return state, launches


def test_scalars_staged_once_per_distinct_pair_with_identical_results(
        cuda_device, monkeypatch):
    ref_state, ref_launches = _run(cuda_device, monkeypatch, staged_once=False)
    state, launches = _run(cuda_device, monkeypatch, staged_once=True)
    solves = FRAMES * 2
    assert len(ref_launches) == 3 * solves
    assert 1 <= len(launches) <= solves
    # consecutive launches always carry a new (dt, tolerance) pair
    assert all(a != b for a, b in zip(launches, launches[1:]))
    assert np.array_equal(state[0], ref_state[0])
    assert np.array_equal(state[1], ref_state[1])
