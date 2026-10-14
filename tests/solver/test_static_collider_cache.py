"""P08: static colliders are rasterized once per frame, not every substep.

With one moving collider every substep used to reset the collider grid and
re-rasterize every collider, static ones included. The walls plus static
colliders are now snapshotted at the frame's full pass and restored on later
substeps, with only moving colliders re-rasterized. The composition rule of
the sequential pass (smallest distance, earliest collider on a tie, walls
first) must be preserved bit for bit, so this test compares the cached path
against the original sequential rasterization substep by substep.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.solver import (  # noqa: E402
    ColliderDesc,
    FluidSolver,
    SolverSettings,
)

H = 0.05
FRAMES = 3
SUBSTEPS = 3

_FACES = np.array([
    [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
    [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
    [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
], np.int32)


def _box(device, centre, half, **kwargs):
    centre = np.asarray(centre, np.float32)
    half = np.asarray(half, np.float32)
    corners = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)],
                       np.float32) * half + centre
    mesh = wp.Mesh(points=wp.array(corners, dtype=wp.vec3, device=device),
                   indices=wp.array(_FACES.ravel(), dtype=wp.int32, device=device))
    return ColliderDesc(mesh=mesh, center=tuple(centre), aabb_min=tuple(centre - half),
                        aabb_max=tuple(centre + half), **kwargs)


def _colliders(device):
    paddle = dict(centre=(0.8, 0.8, 0.5), half=(0.3, 0.05, 0.2))
    return [
        _box(device, (0.8, 0.8, 0.1), (0.75, 0.75, 0.08), friction=0.2, restitution=0.1),
        _box(device, **paddle, friction=0.5, restitution=0.3,
             linear_velocity=(0.5, 0.0, 0.0), angular_velocity=(0.0, 0.0, 4.0)),
        # Same frame-start geometry as the paddle, listed after it: an exact
        # distance tie the paddle (earlier) must win on the first substep.
        _box(device, **paddle, friction=0.9, restitution=0.0),
    ]


def _run(device, cached):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.6, 1.6, 1.2), voxel_size=H,
        leaf_capacity=128, max_particles=40_000, min_substeps=SUBSTEPS,
        max_substeps=SUBSTEPS, whitewater_enabled=False, surface_enabled=False,
        device=device)
    solver = FluidSolver(settings)
    axis = np.arange(0.3, 1.3, H / 2) + H / 4
    zs = np.arange(0.25, 0.7, H / 2) + H / 4
    grid = np.meshgrid(axis, axis, zs, indexing="ij")
    solver.inject_particles(np.stack([g.ravel() for g in grid], 1).astype(np.float32))
    snapshots = []
    reuses = []
    original = solver._apply_collisions

    def sequential(reuse_static=False):
        # The pre-P08 pass: reset (done by the caller) then every collider in
        # list order with the strict-< kernel.
        for col in solver.colliders:
            if col.mesh is not None:
                solver._rasterize_collider(col)

    def spy(reuse_static=False):
        reuses.append(reuse_static)
        if cached:
            original(reuse_static=reuse_static)
        else:
            sequential()
        nv = solver._nvox
        snapshots.append(tuple(np.array(a.numpy()[:nv]) for a in (
            solver.collision_sdf.data, solver.collision_vel,
            solver.collision_friction, solver.collision_restitution)))

    solver._apply_collisions = spy
    if not cached:
        solver._static_collision_cache_usable = lambda: False
    solver.set_colliders(_colliders(device))
    for _ in range(FRAMES):
        solver.step_frame()
        solver.set_colliders(solver.colliders)
    n = solver.particles.refresh_host_count()
    state = (solver.particles.position.numpy()[:n].copy(),
             solver.particles.velocity.numpy()[:n].copy())
    solver.release()
    return snapshots, state, reuses


def test_cached_static_colliders_match_sequential_rasterization(cuda_device):
    ref_snaps, ref_state, _ = _run(cuda_device, cached=False)
    snaps, state, reuses = _run(cuda_device, cached=True)
    # substeps after the first restore the static snapshot
    assert reuses == [False, True, True] * FRAMES
    assert len(snaps) == len(ref_snaps) == FRAMES * SUBSTEPS
    for k, (got, want) in enumerate(zip(snaps, ref_snaps)):
        for name, a, b in zip(("sdf", "vel", "friction", "restitution"), got, want):
            assert a.shape == b.shape
            assert np.array_equal(a, b), f"substep {k}: collision {name} differs"
    # the tie at the first substep went to the paddle (friction 0.5), as before
    assert np.any(ref_snaps[0][2] == np.float32(0.5))
    assert np.array_equal(state[0], ref_state[0])
    assert np.array_equal(state[1], ref_state[1])
