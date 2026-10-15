"""4.0 lane R: the device integrator matches its host reference.

``k_rigid_integrate`` (the substep loop's integrator, device resident so the
body state never round-trips to the host mid-frame) is a port of
``RigidCoupling._integrate`` followed by ``_contain``. Drive both from the
same crafted impulses and compare: rotation, gyroscopic coupling, axis locks,
the velocity clamp, wall contact and a non-finite impulse row are all covered.
The rasterize parity test checks the body-frame mesh query against the
world-posed mesh it replaces.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from _rigid_harness import H, box_collider

pytest.importorskip("warp")

DOMAIN_MIN = (0.0, 0.0, 0.0)
DOMAIN_MAX = (1.0, 0.5, 0.5)
DT = 1.0 / 96.0


def _solver(wp, device, gravity=(0.0, 0.0, -9.81)):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    return FluidSolver(SolverSettings(
        domain_min=DOMAIN_MIN, domain_max=DOMAIN_MAX, voxel_size=H,
        gravity=gravity, leaf_capacity=1024, max_particles=4096,
        surface_enabled=False, whitewater_enabled=False, device=device))


def _bodies(wp, device):
    return [
        # Spinning, tumbling, near nothing.
        box_collider(wp, device, (0.30, 0.25, 0.25), (0.05, 0.03, 0.04),
                     dynamic=True, object_uid="tumbler", density=600.0),
        # Pushed through the floor: wall contact clips it.
        box_collider(wp, device, (0.60, 0.25, 0.04), (0.04, 0.04, 0.04),
                     dynamic=True, object_uid="floor", density=1500.0),
        # Translation x and rotation z locked.
        box_collider(wp, device, (0.80, 0.25, 0.30), (0.03, 0.05, 0.03),
                     dynamic=True, object_uid="locked", density=900.0,
                     lock_axes=(True, False, False, False, False, True)),
    ]


def _impulses(rng, bodies):
    rows = []
    for body in bodies:
        row = np.zeros(14)
        row[0:3] = rng.normal(scale=0.05, size=3) * body.mass
        row[3:6] = rng.normal(scale=0.002, size=3)
        row[6:9] = rng.normal(scale=0.01, size=3) * body.mass
        row[9:12] = rng.normal(scale=0.0005, size=3)
        row[12] = body.volume * rng.uniform(0.7, 1.3)
        row[13] = body.mass * rng.uniform(0.0, 0.5)
        rows.append(row)
    return np.asarray(rows)


def _copy(body):
    """An independent host copy (the device mesh cannot be deep-copied)."""
    return dataclasses.replace(
        body, local_mesh=None, pos=body.pos.copy(), quat=body.quat.copy(),
        lin_vel=body.lin_vel.copy(), ang_vel=body.ang_vel.copy())


def _host_step(coupling, bodies, rows, solver, dt):
    s = solver.settings
    gravity = np.asarray(s.gravity, dtype=np.float64)
    dv_max = float(solver._cfl_speed_limit(dt))
    for body, row in zip(bodies, rows):
        coupling._integrate(body, row, gravity, dt, dv_max)
        coupling._contain(body, s.domain_min, s.domain_max)


def test_device_integrate_matches_host_reference(cuda_device):
    import warp as wp

    solver = _solver(wp, cuda_device)
    try:
        solver.set_colliders(_bodies(wp, cuda_device))
        rigid = solver.rigid
        assert len(rigid.attached) == 3
        # Non-zero starting spin and velocity, set before the upload.
        for (_, _, body), omega, vel in zip(
                rigid.attached,
                ([3.0, -2.0, 5.0], [0.5, 0.0, 0.0], [1.0, 2.0, 3.0]),
                ([0.1, 0.0, 0.2], [0.0, 0.0, -0.8], [0.3, -0.1, 0.0])):
            body.ang_vel = np.asarray(omega, dtype=np.float64)
            body.lin_vel = np.asarray(vel, dtype=np.float64)
        rigid._dirty = True
        rigid.prepare(solver)
        host = [_copy(body) for _, _, body in rigid.attached]
        rng = np.random.default_rng(7)
        out = rigid._gather_buffers["out"]
        touched = 0
        for substep in range(24):
            rows = _impulses(rng, host)
            if substep == 5:
                rows[1, 4] = np.nan  # a broken gather zeroes the whole row
            out.assign(rows)
            rigid._launch_integrate(solver, DT, out)
            rigid._host_stale = True
            _host_step(rigid, host, rows, solver, DT)
            floor = host[1]
            bottom = floor.pos[2] + (floor.body_points @ floor.rotation().T)[:, 2].min()
            assert bottom >= -1e-9
            touched += abs(bottom) < 1e-9
        rigid.sync_host()
        for (_, _, body), ref in zip(rigid.attached, host):
            np.testing.assert_allclose(body.pos, ref.pos, rtol=0, atol=1e-9)
            np.testing.assert_allclose(body.lin_vel, ref.lin_vel, rtol=0, atol=1e-9)
            np.testing.assert_allclose(body.ang_vel, ref.ang_vel, rtol=0, atol=1e-9)
            np.testing.assert_allclose(body.quat, ref.quat, rtol=0, atol=1e-9)
        tumbler, floor, locked = (body for _, _, body in rigid.attached)
        assert np.linalg.norm(tumbler.quat - [1.0, 0.0, 0.0, 0.0]) > 1e-2
        assert touched  # the floor contact engaged (and matched)
        assert locked.lin_vel[0] == 0.0 and locked.ang_vel[2] == 0.0
    finally:
        solver.release()


def test_device_face_box_tracks_the_body(cuda_device):
    """The face-gather box follows the integrated pose without a host trip."""
    import warp as wp
    from fluidSim_warp_blender51.sim.kernels_rigid import (
        BODY_BOX_DIM, BODY_BOX_LO)

    solver = _solver(wp, cuda_device, gravity=(0.0, 0.0, 0.0))
    try:
        solver.set_colliders(_bodies(wp, cuda_device)[:1])
        rigid = solver.rigid
        rigid.attached[0][2].lin_vel = np.array([4.0, 0.0, 0.0])
        rigid._dirty = True
        rigid.prepare(solver)
        out = rigid._gather_buffers["out"]
        out.zero_()
        for _ in range(10):
            rigid._launch_integrate(solver, DT, out)
        rigid._host_stale = True
        state = rigid._gather_buffers["state"].numpy()[0]
        rigid.sync_host()
        # Re-upload from the synced host state: the host computes the box
        # the same way, so both must agree to the voxel.
        rigid._dirty = True
        rigid.prepare(solver)
        expected = rigid._gather_buffers["state"].numpy()[0]
        np.testing.assert_array_equal(state[BODY_BOX_LO:BODY_BOX_LO + 3],
                                      expected[BODY_BOX_LO:BODY_BOX_LO + 3])
        np.testing.assert_array_equal(state[BODY_BOX_DIM:BODY_BOX_DIM + 3],
                                      expected[BODY_BOX_DIM:BODY_BOX_DIM + 3])
    finally:
        solver.release()


def test_body_frame_rasterize_matches_world_mesh(cuda_device):
    """Rasterizing the body-frame mesh at the pose equals the posed mesh."""
    import warp as wp

    solver = _solver(wp, cuda_device, gravity=(0.0, 0.0, 0.0))
    try:
        from _rigid_harness import lattice

        solver.inject_particles(lattice((0.1, 0.1, 0.1), (0.9, 0.4, 0.4), 0.5 * H))
        col = _bodies(wp, cuda_device)[0]
        solver.set_colliders([col])
        solver.step_frame()  # builds the grid and the collision fields
        rigid = solver.rigid
        rigid.sync_host()
        body = rigid.attached[0][2]
        body.ang_vel = np.array([2.0, 1.0, -3.0])
        body.lin_vel = np.array([0.5, 0.2, -0.1])
        rigid._dirty = True
        rigid.prepare(solver)
        out = rigid._gather_buffers["out"]
        out.zero_()
        for _ in range(12):
            rigid._launch_integrate(solver, DT, out)
        rigid._host_stale = True
        rigid.sync_host()
        assert np.linalg.norm(body.quat - [1.0, 0.0, 0.0, 0.0]) > 1e-2

        def field():
            solver.collision_sdf.data.fill_(1.0e30)
            solver._collision_owner.fill_(-1)
            solver.collision_vel.zero_()
            return solver.collision_sdf.data

        band = solver._collider_band(col)
        field()
        rigid.rasterize(solver, 0, col, 0, band)
        local_sdf = solver.collision_sdf.data.numpy().copy()
        local_vel = solver.collision_vel.numpy().copy()

        # The world-posed mesh the old path refit every substep.
        world = (body.body_points @ body.rotation().T + body.pos).astype(np.float32)
        col.mesh.points.assign(world)
        col.mesh.refit()
        col.linear_velocity = tuple(body.lin_vel)
        col.angular_velocity = tuple(body.ang_vel)
        col.center = tuple(body.pos)
        field()
        solver._rasterize_collider(col, 0)
        world_sdf = solver.collision_sdf.data.numpy()
        world_vel = solver.collision_vel.numpy()

        hit = world_sdf < 1.0e29
        assert hit.sum() > 100
        np.testing.assert_array_equal(hit, local_sdf < 1.0e29)
        np.testing.assert_allclose(local_sdf[hit], world_sdf[hit], atol=1e-5)
        np.testing.assert_allclose(local_vel[hit], world_vel[hit], atol=1e-5)
    finally:
        solver.release()
