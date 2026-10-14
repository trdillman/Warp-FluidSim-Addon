"""4.0 lane R: a bake with a Dynamic collider is bitwise repeatable.

docs/specs/4.0/SPEC-rigid-coupling.md acceptance: two bakes with one Dynamic
body are bitwise identical. Body forces are a segmented per-body gather in a
fixed order (no atomics), so the body state and every particle must match
exactly, frame by frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from _rigid_harness import H, box_collider, lattice

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (0.5, 0.5, 0.5)
FRAMES = 16


def _bake(device):
    import warp as wp
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    # A tilted block dropped into a sloshing tank: rotation, drag and the
    # pressure gather all contribute.
    points = lattice(DOMAIN_MIN, (0.30, 0.5, 0.25), 0.5 * H)
    solver = FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        leaf_capacity=1024, max_particles=len(points) + 8192,
        surface_enabled=False, whitewater_enabled=False, device=device))
    frames = []
    try:
        solver.inject_particles(points)
        body = box_collider(
            wp, device, (0.32, 0.24, 0.33), (0.07, 0.05, 0.04), dynamic=True,
            object_uid="block", density=600.0, drag=1.0,
            initial_velocity=(0.0, 0.2, -0.3))
        for _ in range(FRAMES):
            solver.set_colliders([body])
            solver.step_frame()
            state = solver.rigid_body_states()[0]
            body_row = np.concatenate([np.asarray(state[k], dtype=np.float64)
                                       for k in ("pos", "quat", "lin_vel",
                                                 "ang_vel")])
            frames.append((body_row.tobytes(),
                           solver.particle_positions().tobytes(),
                           solver.particle_velocities().tobytes()))
    finally:
        solver.release()
    return frames


def test_two_bakes_are_bitwise_identical(cuda_device):
    first = _bake(cuda_device)
    second = _bake(cuda_device)
    moved = np.frombuffer(first[-1][0])[:3] - np.frombuffer(first[0][0])[:3]
    assert np.linalg.norm(moved) > 1.0e-3, moved
    for frame, (a, b) in enumerate(zip(first, second), start=1):
        assert a[0] == b[0], f"body state differs at frame {frame}"
        assert a[1] == b[1], f"particle positions differ at frame {frame}"
        assert a[2] == b[2], f"particle velocities differ at frame {frame}"
