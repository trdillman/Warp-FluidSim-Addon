"""Dynamic bodies collide with each other and with sharp collider points.

``k_rigid_contact_points`` tests every body's points against the other bodies;
``k_rigid_contact_collider_points`` finds collider points (a spike tip) pressed
into a flat body face, which no body point reaches.
"""

from __future__ import annotations

import numpy as np
import pytest
from _rigid_harness import H, box_collider

pytest.importorskip("warp")

HALF = 0.05
TIP = 0.20


def _solver(device, gravity=(0.0, 0.0, -9.81)):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    solver = FluidSolver(SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(0.5, 0.5, 0.6), voxel_size=H,
        gravity=gravity, leaf_capacity=1024, max_particles=4096,
        surface_enabled=False, whitewater_enabled=False, device=device))
    corner = np.array([[0.02, 0.02, 0.02], [0.03, 0.02, 0.02]], dtype=np.float32)
    solver.inject_particles(corner)
    return solver


def _body(wp, device, uid, center, **fields):
    return box_collider(wp, device, center, (HALF, HALF, HALF), dynamic=True,
                        object_uid=uid, density=fields.pop("density", 2000.0),
                        **fields)


def _spikes(wp, device):
    """Four thin closed pyramids, tips at ``TIP``, under one body's footprint."""
    from fluidSim_warp_blender51.sim.solver import ColliderDesc

    base = 0.006
    points, tris = [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x, y = 0.25 + 0.025 * sx, 0.25 + 0.025 * sy
            first = len(points)
            points += [[x - base, y - base, 0.0], [x + base, y - base, 0.0],
                       [x + base, y + base, 0.0], [x - base, y + base, 0.0],
                       [x, y, TIP]]
            a, b, c, d, apex = range(first, first + 5)
            tris += [a, c, b, a, d, c,  # base, facing down
                     a, b, apex, b, c, apex, c, d, apex, d, a, apex]
    points = np.asarray(points, dtype=np.float32)
    mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                   indices=wp.array(np.asarray(tris, dtype=np.int32),
                                    dtype=wp.int32, device=device))
    return ColliderDesc(
        mesh=mesh, center=(0.25, 0.25, 0.1),
        aabb_min=tuple(float(v) for v in points.min(axis=0)),
        aabb_max=tuple(float(v) for v in points.max(axis=0)))


def _run(device, make, frames, gravity=(0.0, 0.0, -9.81)):
    import warp as wp

    solver = _solver(device, gravity)
    try:
        for _ in range(frames):
            solver.set_colliders(make(wp))
            solver.step_frame()
        return {s["object_uid"]: s for s in solver.rigid_body_states()}
    finally:
        solver.release()


def test_body_rests_on_spike_tips_that_touch_no_body_point(cuda_device):
    # The spikes stand inside the bottom face: no body corner is above them.
    states = _run(cuda_device, lambda wp: [
        _body(wp, cuda_device, "body", (0.25, 0.25, TIP + 0.15)),
        _spikes(wp, cuda_device)], frames=36)
    bottom = states["body"]["pos"][2] - HALF
    assert TIP - 1.5 * H <= bottom <= TIP + 1.5 * H, states["body"]
    assert abs(states["body"]["lin_vel"][2]) <= 0.2, states["body"]


def _stack(wp, device):
    return [_body(wp, device, "low", (0.25, 0.25, HALF + 0.01)),
            _body(wp, device, "high", (0.25, 0.25, 3.0 * HALF + 0.12))]


def test_a_body_lands_and_rests_on_another_body(cuda_device):
    states = _run(cuda_device, lambda wp: _stack(wp, cuda_device), frames=36)
    low, high = states["low"], states["high"]
    # The lower body sits on the domain floor ...
    assert low["pos"][2] - HALF <= 1.5 * H, low
    # ... and the upper one on top of it, not through it.
    gap = (high["pos"][2] - HALF) - (low["pos"][2] + HALF)
    assert -1.5 * H <= gap <= 1.5 * H, (low, high)
    assert abs(high["lin_vel"][2]) <= 0.2, high


def test_body_contact_is_deterministic(cuda_device):
    first = _run(cuda_device, lambda wp: _stack(wp, cuda_device), frames=12)
    second = _run(cuda_device, lambda wp: _stack(wp, cuda_device), frames=12)
    for uid in ("low", "high"):
        for key in ("pos", "quat", "lin_vel", "ang_vel"):
            assert first[uid][key] == second[uid][key], (uid, key)


def test_head_on_bodies_stop_each_other_and_keep_momentum(cuda_device):
    speed = 0.5

    def make(wp):
        return [
            _body(wp, cuda_device, "left", (0.15, 0.25, 0.3),
                  initial_velocity=(speed, 0.0, 0.0)),
            _body(wp, cuda_device, "right", (0.35, 0.25, 0.3),
                  initial_velocity=(-speed, 0.0, 0.0)),
        ]

    states = _run(cuda_device, make, frames=12, gravity=(0.0, 0.0, 0.0))
    left, right = states["left"], states["right"]
    # They would have crossed after 0.2 s; they meet and stop instead.
    gap = (right["pos"][0] - HALF) - (left["pos"][0] + HALF)
    assert -1.5 * H <= gap <= 1.5 * H, (left, right)
    total = np.asarray(left["lin_vel"]) + np.asarray(right["lin_vel"])
    assert np.linalg.norm(total) <= 1.0e-3 * speed, (left, right)
    assert abs(left["lin_vel"][0]) <= 0.05 * speed, left
