"""4.0 lane R: a Dynamic collider floats by the pressure the fluid puts on it.

docs/specs/4.0/SPEC-rigid-coupling.md acceptance: a density-ratio-0.5 cube in
a still tank settles at 50 % +- 5 % submersion within 48 frames (Multigrid);
ratio 1.5 sinks; ratio 1.0 is neutral within 10 %.

Rotation (and horizontal drift) is locked: a free cube at density ratio 0.5
floats edge-up (a face-up cube is only stable below ~0.21 or above ~0.79), and
this test is about the vertical force balance, not the tipping instability.
"""

from __future__ import annotations

import numpy as np
import pytest

from _rigid_harness import H, box_collider, lattice

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (0.5, 0.5, 0.6)
DEPTH = 0.30
HALF = 0.08          # cube side 0.16 m = 7.7 voxels
FRAMES = 48
LOCKS = (True, True, False, True, True, True)
# The collider_drag default (bluff-body Cd ~ 1). Without drag the heave
# oscillation is damped only by the waves it radiates and is still ringing
# at +-0.06 after 48 frames; the mean is already 0.52.
DRAG = 1.0


def _run(device, ratio, start_submersion, frames=FRAMES):
    import warp as wp
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    side = 2.0 * HALF
    center = (0.25, 0.25, DEPTH - start_submersion * side + HALF)
    lo = tuple(c - HALF for c in center)
    hi = tuple(c + HALF for c in center)
    points = lattice(DOMAIN_MIN, (DOMAIN_MAX[0], DOMAIN_MAX[1], DEPTH),
                     0.5 * H, exclude=(lo, hi))
    solver = FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=1024, max_particles=len(points) + 8192,
        surface_enabled=False, whitewater_enabled=False, device=device))
    try:
        solver.inject_particles(points)
        body = box_collider(
            wp, device, center, (HALF, HALF, HALF), dynamic=True,
            object_uid="cube", density=1000.0 * ratio, drag=DRAG,
            lock_axes=LOCKS)
        trace = []
        for _ in range(frames):
            solver.set_colliders([body])
            solver.step_frame()
            state = solver.rigid_body_states()[0]
            positions = solver.particle_positions()
            far = np.hypot(positions[:, 0] - 0.25, positions[:, 1] - 0.25) > 0.15
            level = float(np.percentile(positions[far, 2], 99.0))
            bottom = state["pos"][2] - HALF
            trace.append((level - bottom) / side)
        return np.asarray(trace), solver.rigid_body_states()[0]
    finally:
        solver.release()


def test_half_density_cube_floats_half_submerged(cuda_device):
    trace, state = _run(cuda_device, 0.5, start_submersion=0.25)
    settled = trace[-8:]
    print("\nratio 0.5 submersion trace:", np.round(trace, 3).tolist())
    assert abs(float(settled.mean()) - 0.5) <= 0.05, trace
    assert float(np.ptp(settled)) <= 0.05, trace
    assert np.allclose(state["quat"], (1.0, 0.0, 0.0, 0.0)), state


def test_dense_cube_sinks(cuda_device):
    trace, state = _run(cuda_device, 1.5, start_submersion=0.25)
    print("\nratio 1.5 submersion trace:", np.round(trace, 3).tolist())
    # Fully under and resting on the domain floor, not through it.
    assert trace[-1] >= 1.0, trace
    bottom = state["pos"][2] - HALF
    assert -1.0e-6 <= bottom <= 0.25 * HALF, state
    assert abs(state["lin_vel"][2]) <= 0.05, state


def test_neutral_cube_stays_put(cuda_device):
    trace, state = _run(cuda_device, 1.0, start_submersion=1.5)
    print("\nratio 1.0 submersion trace:", np.round(trace, 3).tolist())
    start_z = DEPTH - 1.5 * 2.0 * HALF + HALF
    assert abs(state["pos"][2] - start_z) <= 0.1 * 2.0 * HALF, (state, trace)
