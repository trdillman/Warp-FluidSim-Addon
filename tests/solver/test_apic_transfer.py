"""APIC velocity transfer: kernel-level contracts (3.0 lane F).

These tests drive the real transfer kernels on a real sparse grid.  They pin
the decision that the affine matrix is computed from the *actual* trilinear
stencil (B_a * D_a^-1 with D_a built per particle), not from the textbook
quadratic-B-spline constant 4/h^2.

Units: the kernels work in voxel units internally, but ``particles.affine`` is
stored in world units (1/s), so a rigid rotation of rate omega reads
``[[0, -omega, 0], [omega, 0, 0], [0, 0, 0]]`` whatever the voxel size.

Interface these tests freeze (red until the implementation lands):
  * ``ParticleSampleType.APIC == 2``
  * ``solver.particles.affine``: ``wp.array(dtype=wp.mat33)``, row ``a`` of the
    matrix is ``d v_a / d x`` in world units; zero means "no affine motion".
  * ``solver.particles.ensure_affine()``: allocate the channel lazily (zeroed), return it.
  * ``solver._g2p_transfer()``: the velocity-update half of ``_g2p_advect``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.enums import ParticleSampleType
from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

wp = pytest.importorskip("warp")

H = 0.125
OMEGA = 1.7
CENTRE = np.array([0.5, 0.5, 0.5])
ROT = np.array([[0.0, -OMEGA, 0.0], [OMEGA, 0.0, 0.0], [0.0, 0.0, 0.0]])
SHEAR = np.array([[0.3, -0.2, 0.1], [0.5, 0.2, -0.4], [-0.1, 0.3, 0.25]])
OFFSET = np.array([0.2, -0.1, 0.05])

# MAC face offsets in voxels: u at (0, .5, .5), v at (.5, 0, .5), w at (.5, .5, 0).
FACE_OFFSET = (
    np.array([0.0, 0.5, 0.5]),
    np.array([0.5, 0.0, 0.5]),
    np.array([0.5, 0.5, 0.0]),
)


def _apic():  # numeric value is a kernel argument and a cache stamp; never renumbered
    return 2


@wp.kernel
def _k_fill_linear_field(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    h: float,
    a: wp.mat33,
    b: wp.vec3,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
):
    """Write u(x) = a x + b at every face centre of every active voxel."""
    tid = wp.tid()
    if tid >= num_leaves[0] * 512:
        return
    slot = tid // 512
    local = tid - slot * 512
    lc = leaf_coords[slot]
    i = float(lc[0] * 8 + (local & 7))
    j = float(lc[1] * 8 + ((local >> 3) & 7))
    k = float(lc[2] * 8 + ((local >> 6) & 7))
    xu = origin + h * wp.vec3(i, j + 0.5, k + 0.5)
    xv = origin + h * wp.vec3(i + 0.5, j, k + 0.5)
    xw = origin + h * wp.vec3(i + 0.5, j + 0.5, k)
    u[tid] = (a * xu + b)[0]
    v[tid] = (a * xv + b)[1]
    w[tid] = (a * xw + b)[2]
    wu[tid] = 1.0
    wv[tid] = 1.0
    ww[tid] = 1.0


def _settings(device, sample_type, **overrides):
    values = dict(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=H,
        leaf_capacity=64,
        max_particles=8192,
        max_whitewater=64,
        surface_enabled=False,
        whitewater_enabled=False,
        gravity=(0.0, 0.0, 0.0),
        relax_enabled=False,
        sample_type=int(sample_type),
        device=device,
    )
    values.update(overrides)
    return SolverSettings(**values)


def _prepare(device, sample_type, positions, velocities=None):
    """A solver with injected particles and a built, indexed topology."""
    solver = FluidSolver(_settings(device, sample_type))
    positions = np.asarray(positions, dtype=np.float32)
    if velocities is None:
        solver.inject_particles(positions)
    else:
        solver.inject_particles(
            positions, velocities=np.asarray(velocities, dtype=np.float32))
    solver._rebuild_topology()
    solver._clear_grid_fields()
    solver._index_particles()
    return solver


def _fill_linear(solver, a_matrix, b_vector):
    g = solver.grid
    vel = solver.vel
    wp.launch(
        _k_fill_linear_field, dim=solver._nvox,
        inputs=[g.leaf_coords, g.num_leaves, g.origin, g.voxel_size,
                wp.mat33(*[float(x) for x in np.asarray(a_matrix).ravel()]),
                wp.vec3(*[float(x) for x in b_vector])],
        outputs=[vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
        device=solver.device,
    )


def _affine(solver, count=1):
    return solver.particles.affine.numpy()[:count].reshape(count, 3, 3)


def _stencil_reference(position, field):
    """Numpy B_a, D_a (world units) for one particle with a full hat stencil.

    ``field(x)`` returns the analytic velocity at world point ``x``.  Returns
    ``(v_pic, b_rows, d_mats)`` where ``b_rows[a] = sum_i w_i u_i (x_i - x_p)``
    and ``d_mats[a] = sum_i w_i (x_i - x_p)(x_i - x_p)^T``.
    """
    vp = np.asarray(position, dtype=np.float64) / H
    v_pic = np.zeros(3)
    b_rows = np.zeros((3, 3))
    d_mats = np.zeros((3, 3, 3))
    for a in range(3):
        q = vp - FACE_OFFSET[a]
        base = np.floor(q)
        f = q - base
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    wgt = ((f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1])
                           * (f[2] if dz else 1 - f[2]))
                    x_i = (base + np.array([dx, dy, dz]) + FACE_OFFSET[a]) * H
                    rel = x_i - np.asarray(position, dtype=np.float64)
                    ui = field(x_i)[a]
                    v_pic[a] += wgt * ui
                    b_rows[a] += wgt * ui * rel
                    d_mats[a] += wgt * np.outer(rel, rel)
    return v_pic, b_rows, d_mats


def _rotation_field(x):
    return ROT @ (x - CENTRE)


# A particle whose voxel-space position is generic (no f near 0, 1 or 0.5).
GENERIC = np.array([3.31, 3.47, 3.19]) * H


def test_rigid_rotation_is_recovered_as_the_affine_matrix(cuda_device):
    solver = _prepare(cuda_device, _apic(), [GENERIC])
    try:
        _fill_linear(solver, ROT, -ROT @ CENTRE)
        solver._g2p_transfer()
        np.testing.assert_allclose(
            solver.particles.velocity.numpy()[0], _rotation_field(GENERIC),
            atol=1e-5)
        np.testing.assert_allclose(_affine(solver)[0], ROT, atol=1e-5)
    finally:
        solver.release()


def test_four_over_h_squared_constant_does_not_recover_the_rotation():
    """Negative pin: the quadratic-B-spline constant is wrong for hat weights.

    Pure numpy on the same stencil the kernel uses.  The explicit per-particle
    D recovers the rotation; ``B * 4/h^2`` is off by O(omega), which is the
    spurious swirl the textbook constant would inject.
    """
    _, b_rows, d_mats = _stencil_reference(GENERIC, _rotation_field)
    explicit = np.stack([b_rows[a] @ np.linalg.inv(d_mats[a]) for a in range(3)])
    textbook = b_rows * (4.0 / H**2)
    np.testing.assert_allclose(explicit, ROT, atol=1e-5)
    assert np.max(np.abs(textbook - ROT)) > 0.1


@pytest.mark.parametrize("epsilon_world", [0.0, 1.0e-7])
def test_particle_on_a_face_plane_keeps_the_well_posed_gradients(
        cuda_device, epsilon_world):
    """D_a is singular along the face-normal axis when the particle sits on it.

    The per-axis guard must drop only that axis: the other columns are still
    recovered, nothing is NaN, and nothing is amplified.  A particle at
    ``(3, 3, 3) h`` is on the u face plane in x, the v plane in y and the w
    plane in z at once.
    """
    position = np.array([3.0, 3.0, 3.0]) * H + epsilon_world
    # Keep the exact on-plane case on-plane; nudge only the epsilon case.
    position = position.astype(np.float32)
    solver = _prepare(cuda_device, _apic(), [position])
    try:
        _fill_linear(solver, ROT, -ROT @ CENTRE)
        solver._g2p_transfer()
        affine = _affine(solver)[0]
        assert np.all(np.isfinite(affine))
        assert np.max(np.abs(affine)) < 10.0 * OMEGA
        # Rotation has zero diagonal and the off-diagonals sit off the dropped
        # axes, so the guarded result is still the rotation.
        np.testing.assert_allclose(affine, ROT, atol=1e-3)
        np.testing.assert_allclose(
            solver.particles.velocity.numpy()[0],
            _rotation_field(position.astype(np.float64)), atol=1e-4)
    finally:
        solver.release()


def _box_particles(seed=3):
    rng = np.random.default_rng(seed)
    count = 400
    return ((2.0 + 4.0 * rng.random((count, 3))) * H).astype(np.float32)


def _p2g_roundtrip(device, sample_type):
    positions = _box_particles()
    velocities = (positions.astype(np.float64) @ SHEAR.T + OFFSET).astype(np.float32)
    solver = _prepare(device, sample_type, positions, velocities)
    if int(sample_type) == _apic():
        solver.particles.affine.assign(
            np.tile(SHEAR.astype(np.float32).reshape(1, 3, 3),
                    (solver.particles.capacity, 1, 1)))
    solver._p2g()
    return solver, positions


def _grid_face_error(solver):
    """Max |grid - analytic| over faces that received any particle weight."""
    g = solver.grid
    nvox = solver._nvox
    coords = g.leaf_coords.numpy()[:nvox // 512]
    worst = 0.0
    for axis, (field, weight) in enumerate((
            (solver.vel.u, solver.vel.wu), (solver.vel.v, solver.vel.wv),
            (solver.vel.w, solver.vel.ww))):
        data = field.numpy()[:nvox]
        wsum = weight.numpy()[:nvox]
        local = np.arange(512)
        li, lj, lk = local & 7, (local >> 3) & 7, (local >> 6) & 7
        for slot, lc in enumerate(coords):
            ijk = np.stack([lc[0] * 8 + li, lc[1] * 8 + lj, lc[2] * 8 + lk], axis=1)
            x_i = (ijk + FACE_OFFSET[axis]) * H
            expected = (x_i @ SHEAR.T + OFFSET)[:, axis]
            sl = slice(slot * 512, (slot + 1) * 512)
            valid = wsum[sl] > 1e-3
            if np.any(valid):
                worst = max(worst, float(np.max(np.abs(
                    data[sl][valid] - expected[valid]))))
    return worst


def test_p2g_g2p_round_trip_reproduces_a_linear_field_under_apic(cuda_device):
    solver, positions = _p2g_roundtrip(cuda_device, _apic())
    try:
        assert _grid_face_error(solver) < 1e-4
        solver._g2p_transfer()
        interior = np.all((positions > 3.2 * H) & (positions < 4.8 * H), axis=1)
        assert interior.sum() > 5
        n = len(positions)
        expected = positions.astype(np.float64) @ SHEAR.T + OFFSET
        np.testing.assert_allclose(
            solver.particles.velocity.numpy()[:n][interior], expected[interior],
            atol=1e-4)
        np.testing.assert_allclose(
            _affine(solver, n)[interior],
            np.broadcast_to(SHEAR, (int(interior.sum()), 3, 3)), atol=1e-3)
    finally:
        solver.release()


def test_p2g_round_trip_is_not_exact_under_pic(cuda_device):
    solver, _ = _p2g_roundtrip(cuda_device, int(ParticleSampleType.PIC))
    try:
        assert _grid_face_error(solver) > 1e-3
    finally:
        solver.release()


def test_flip_p2g_ignores_the_affine_channel(cuda_device):
    """Non-APIC transfers must not read affine (the term is where()-gated)."""
    positions = _box_particles()
    velocities = (positions.astype(np.float64) @ SHEAR.T + OFFSET).astype(np.float32)
    fields = []
    for poison in (0.0, 5.0):
        solver = _prepare(
            cuda_device, int(ParticleSampleType.FLIP), positions, velocities)
        try:
            solver.particles.ensure_affine().assign(np.full(
                (solver.particles.capacity, 3, 3), poison, dtype=np.float32))
            solver._p2g()
            n = solver._nvox
            fields.append(np.concatenate([
                solver.vel.u.numpy()[:n], solver.vel.v.numpy()[:n],
                solver.vel.w.numpy()[:n]]))
        finally:
            solver.release()
    np.testing.assert_array_equal(fields[0], fields[1])
