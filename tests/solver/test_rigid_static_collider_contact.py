"""A Dynamic body collides with the Static and Animated colliders around it.

Before this a Dynamic collider only touched the domain walls, so it fell
straight through any other collider. ``k_rigid_contact_*`` tests each body
vertex against the non-Dynamic collider meshes at their substep pose.
"""

from __future__ import annotations

import numpy as np
import pytest
from _rigid_harness import H, box_collider

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (0.5, 0.5, 0.6)
HALF = 0.05
SLAB_TOP = 0.20
FRAMES = 36


def _solver(device, gravity=(0.0, 0.0, -9.81)):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    solver = FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        gravity=gravity, leaf_capacity=1024, max_particles=4096,
        surface_enabled=False, whitewater_enabled=False, device=device))
    # A little liquid in a far corner keeps the solver on its normal path.
    corner = np.array([[0.02, 0.02, 0.02], [0.03, 0.02, 0.02]], dtype=np.float32)
    solver.inject_particles(corner)
    return solver


def _body(wp, device, center, **fields):
    return box_collider(
        wp, device, center, (HALF, HALF, HALF), dynamic=True,
        object_uid="body", density=2000.0, **fields)


def _slab(wp, device, **fields):
    return box_collider(
        wp, device, (0.25, 0.25, SLAB_TOP - 0.05), (0.2, 0.2, 0.05), **fields)


def _drop(device, with_slab, slab_fields=None, frames=FRAMES):
    import warp as wp

    solver = _solver(device)
    try:
        body = _body(wp, device, (0.25, 0.25, SLAB_TOP + 0.25))
        trace = []
        for _ in range(frames):
            colliders = [body]
            if with_slab:
                colliders.append(_slab(wp, device, **(slab_fields or {})))
            solver.set_colliders(colliders)
            solver.step_frame()
            state = solver.rigid_body_states()[0]
            trace.append(state)
        return trace
    finally:
        solver.release()


def test_dense_body_rests_on_a_static_collider(cuda_device):
    free = _drop(cuda_device, with_slab=False)
    rested = _drop(cuda_device, with_slab=True)
    # Without the slab the body reaches the domain floor ...
    assert free[-1]["pos"][2] - HALF <= 1.0e-3, free[-1]
    # ... with it the body stops on top, within a voxel, and is at rest.
    bottom = rested[-1]["pos"][2] - HALF
    assert SLAB_TOP - 0.5 * H <= bottom <= SLAB_TOP + 1.5 * H, rested[-1]
    assert abs(rested[-1]["lin_vel"][2]) <= 0.2, rested[-1]
    # A flat landing does not spin the body.
    assert np.allclose(rested[-1]["quat"], (1.0, 0.0, 0.0, 0.0), atol=1.0e-3)


def test_off_centre_landing_tips_the_body(cuda_device):
    import warp as wp

    solver = _solver(cuda_device)
    try:
        # Half of the body overhangs the slab edge, so the contact is off
        # centre and the impulse must rotate it.
        body = _body(wp, cuda_device, (0.25 + 0.2 - 0.02, 0.25, SLAB_TOP + 0.20))
        for _ in range(FRAMES):
            solver.set_colliders([body, _slab(wp, cuda_device)])
            solver.step_frame()
        state = solver.rigid_body_states()[0]
        assert np.linalg.norm(np.asarray(state["quat"]) - [1.0, 0.0, 0.0, 0.0]) > 0.05, state
    finally:
        solver.release()


def test_animated_collider_carries_the_body(cuda_device):
    import warp as wp

    rest = _drop(cuda_device, with_slab=True)[-1]
    solver = _solver(cuda_device)
    try:
        body = _body(wp, cuda_device, (0.25, 0.25, SLAB_TOP + 0.10))
        lift = 0.15  # m/s; the slab stays well inside the domain
        for frame in range(24):
            slab = _slab(wp, cuda_device, linear_velocity=(0.0, 0.0, lift))
            # The scene hands the solver the frame-start pose.
            slab.mesh.points.assign(
                slab.mesh.points.numpy()
                + np.array([0.0, 0.0, lift * frame * (1.0 / 24.0)], dtype=np.float32))
            slab.mesh.refit()
            solver.set_colliders([body, slab])
            solver.step_frame()
        carried = solver.rigid_body_states()[0]
    finally:
        solver.release()
    # The slab rose 0.15 m in 1 s; the body rides it instead of staying put.
    assert carried["pos"][2] > rest["pos"][2] + 0.06, (carried, rest)
