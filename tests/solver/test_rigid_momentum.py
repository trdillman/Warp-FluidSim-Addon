"""4.0 lane R: fluid-body exchange conserves total linear momentum.

docs/specs/4.0/SPEC-rigid-coupling.md acceptance: total linear momentum (fluid +
body) is conserved to 2 % between frames 10 and 40.

Deviation from the spec's "dam break into a free crate": under gravity the
floor and the dam's back wall push on the fluid, so total momentum is not
conserved by the physics itself. The test instead throws a slab of liquid at a
free crate in zero gravity, far from every wall, so the only momentum exchange
is fluid <-> body. Drag is off here: ``collider_drag`` acts on the body only
(it has no reaction on the fluid), so it is not a momentum-conserving term.
"""

from __future__ import annotations

import numpy as np
import pytest

from _rigid_harness import H, box_collider, lattice

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (1.5, 0.5, 0.5)
SPACING = 0.5 * H
SLAB_LO = (0.10, 0.10, 0.10)
SLAB_HI = (0.40, 0.40, 0.40)
SPEED = 0.3
CRATE = (0.75, 0.25, 0.25)  # first contact near frame 22
HALF = 0.08
RATIO = 0.5
FRAMES = 40


def test_fluid_plus_body_momentum_is_conserved(cuda_device):
    import warp as wp
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    points = lattice(SLAB_LO, SLAB_HI, SPACING)
    particle_mass = 1000.0 * SPACING ** 3
    solver = FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        gravity=(0.0, 0.0, 0.0), leaf_capacity=2048,
        max_particles=len(points) + 8192, surface_enabled=False,
        whitewater_enabled=False, device=cuda_device))
    try:
        solver.inject_particles(points, velocity=(SPEED, 0.0, 0.0))
        crate = box_collider(
            wp, cuda_device, CRATE, (HALF, HALF, HALF), dynamic=True,
            object_uid="crate", density=1000.0 * RATIO, drag=0.0)
        body_mass = 1000.0 * RATIO * (2.0 * HALF) ** 3
        totals, bodies = {}, {}
        for frame in range(1, FRAMES + 1):
            solver.set_colliders([crate])
            solver.step_frame()
            fluid = particle_mass * solver.particle_velocities().astype(
                np.float64).sum(axis=0)
            body = body_mass * np.asarray(
                solver.rigid_body_states()[0]["lin_vel"], dtype=np.float64)
            totals[frame] = fluid + body
            bodies[frame] = body
    finally:
        solver.release()

    initial = particle_mass * len(points) * SPEED
    print("\nP_x by frame:", [round(float(totals[f][0]) / initial, 4)
                              for f in sorted(totals)])
    print("body P_x by frame:", [round(float(bodies[f][0]) / initial, 4)
                                 for f in sorted(bodies)])
    # The impact falls inside the window: the body must take up momentum
    # there (an inelastic catch gives it at most m_body / m_total = 7 %).
    assert bodies[FRAMES][0] - bodies[10][0] >= 0.03 * initial, bodies
    drift = np.linalg.norm(totals[FRAMES] - totals[10])
    assert drift <= 0.02 * np.linalg.norm(totals[10]), (totals[10], totals[FRAMES])
