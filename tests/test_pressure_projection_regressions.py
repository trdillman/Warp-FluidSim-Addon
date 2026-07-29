"""Focused pressure projection safety regressions."""

import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import (
    LEAF_VOLUME,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_NONE,
)
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid
from fluidSim_warp_blender51.sim.kernels_fluid import (
    k_apply_collider_material_to_grid,
    resolve_collision,
    resolve_collision_segment,
)
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import (
    ColliderDesc,
    FluidSolver,
    SolverSettings,
    k_sanitize_mac_velocity,
    k_sanitize_particles,
)


@wp.kernel
def _resolve_material_test(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
):
    tid = wp.tid()
    p, v = resolve_collision(
        positions[tid], velocities[tid], collision_sdf, collision_vel,
        collision_friction, collision_restitution,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size)
    positions[tid] = p
    velocities[tid] = v


@wp.kernel
def _resolve_segment_test(
    previous_positions: wp.array(dtype=wp.vec3),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
):
    tid = wp.tid()
    p, v = resolve_collision_segment(
        previous_positions[tid], positions[tid], velocities[tid],
        collision_sdf, collision_vel, collision_friction,
        collision_restitution, leaf_table, voxels_dim, leaves_dim,
        origin, voxel_size, 0.1)
    positions[tid] = p
    velocities[tid] = v


def _box_collider(center, half_size, friction, restitution, device):
    cx, cy, cz = center
    sx, sy, sz = half_size
    verts = np.asarray([
        [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz],
        [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz],
        [cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz],
        [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz],
    ], dtype=np.float32)
    tris = np.asarray([
        0,2,1, 0,3,2, 4,5,6, 4,6,7,
        0,1,5, 0,5,4, 1,2,6, 1,6,5,
        2,3,7, 2,7,6, 3,0,4, 3,4,7,
    ], dtype=np.int32)
    return ColliderDesc(
        mesh=wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=device),
            indices=wp.array(tris, dtype=wp.int32, device=device)),
        center=center,
        aabb_min=tuple(verts.min(axis=0)),
        aabb_max=tuple(verts.max(axis=0)),
        friction=friction,
        restitution=restitution,
    )


@pytest.fixture(scope="module")
def cuda_device():
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


def _one_leaf_grid(device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        voxel_size=0.05,
        leaf_capacity=1,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == 1
    assert grid.voxel_capacity == LEAF_VOLUME
    return grid


def test_collision_grid_keeps_material_of_each_collider(cuda_device):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.2, 0.4, 0.4),
        voxel_size=0.05, leaf_capacity=8, max_particles=8,
        max_whitewater=8, device=cuda_device,
    )
    solver = FluidSolver(settings)
    solver.set_colliders([
        _box_collider((0.25, 0.2, 0.2), (0.12, 0.12, 0.12),
                      0.1, 0.2, cuda_device),
        _box_collider((0.85, 0.2, 0.2), (0.12, 0.12, 0.12),
                      0.8, 0.9, cuda_device),
    ])

    solver._rebuild_topology()
    solver._clear_grid_fields()
    solver._apply_collisions()
    nvox = solver._nvox
    sdf = solver.collision_sdf.data.numpy()[:nvox]
    friction = solver.collision_friction.numpy()[:nvox]
    restitution = solver.collision_restitution.numpy()[:nvox]
    inside = sdf < 0.0

    assert inside.any()
    assert np.any(np.isclose(friction[inside], 0.1))
    assert np.any(np.isclose(friction[inside], 0.8))
    assert np.any(np.isclose(restitution[inside], 0.2))
    assert np.any(np.isclose(restitution[inside], 0.9))


def test_collision_sdf_covers_deep_resolved_collider_interior(cuda_device):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.2, 1.2, 1.2),
        voxel_size=0.05, leaf_capacity=27, max_particles=8,
        max_whitewater=8, device=cuda_device,
    )
    solver = FluidSolver(settings)
    solver.set_colliders([
        _box_collider((0.6, 0.6, 0.6), (0.55, 0.55, 0.55),
                      0.0, 0.0, cuda_device),
    ])
    solver._rebuild_topology()
    solver._clear_grid_fields()
    solver._apply_collisions()
    ijk = np.floor((np.asarray((0.6, 0.6, 0.6)) -
                    np.asarray(solver.grid.origin)) / solver.grid.voxel_size).astype(int)
    leaf = ijk // 8
    dense = int(leaf[0] + solver.grid.leaves_dim[0] *
                (leaf[1] + solver.grid.leaves_dim[1] * leaf[2]))
    slot = int(solver.grid.leaf_table.numpy()[dense])
    local = ijk & 7
    center = slot * LEAF_VOLUME + int(local[0] + 8 * local[1] + 64 * local[2])

    assert center >= 0
    assert solver.collision_sdf.data.numpy()[center] < 0.0


def test_particles_use_the_material_of_the_collider_they_hit(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    friction = np.empty(LEAF_VOLUME, dtype=np.float32)
    restitution = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        local = flat % LEAF_VOLUME
        i = local & 7
        k = local >> 6
        sdf[flat] = (k + 0.5) * 0.05 - 0.2
        friction[flat] = 0.1 if i < 4 else 0.8
        restitution[flat] = 0.2 if i < 4 else 0.9
    collision_sdf = wp.array(sdf, dtype=float, device=cuda_device)
    collision_vel = wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device)
    collision_friction = wp.array(friction, dtype=float, device=cuda_device)
    collision_restitution = wp.array(
        restitution, dtype=float, device=cuda_device)
    positions = wp.array(
        [(0.125, 0.125, 0.15), (0.275, 0.125, 0.15)],
        dtype=wp.vec3, device=cuda_device)
    velocities = wp.array(
        [(1.0, 0.0, -1.0), (1.0, 0.0, -1.0)],
        dtype=wp.vec3, device=cuda_device)
    wp.launch(
        _resolve_material_test, dim=2,
        inputs=[positions, velocities, collision_sdf,
                collision_vel, collision_friction,
                collision_restitution, g.leaf_table, g.voxels_dim,
                g.leaves_dim, g.origin, g.voxel_size],
        device=cuda_device)
    result = velocities.numpy()

    assert result[0, 0] > result[1, 0]
    assert result[0, 2] < result[1, 2]


def test_particle_material_response_is_continuous_across_voxels(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    friction = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        local = flat % LEAF_VOLUME
        i = local & 7
        k = local >> 6
        sdf[flat] = (k + 0.5) * g.voxel_size - 0.2
        friction[flat] = 0.1 if i < 4 else 0.9
    positions = wp.array(
        [(0.2249, 0.125, 0.15), (0.2251, 0.125, 0.15)],
        dtype=wp.vec3, device=cuda_device)
    velocities = wp.array(
        [(1.0, 0.0, -1.0), (1.0, 0.0, -1.0)],
        dtype=wp.vec3, device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    wp.launch(
        _resolve_material_test, dim=2,
        inputs=[positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                wp.array(friction, dtype=float, device=cuda_device), zero,
                g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size], device=cuda_device)

    tangential = velocities.numpy()[:, 0]
    assert abs(float(tangential[0] - tangential[1])) < 0.1


def test_particle_segment_collision_stops_at_clearance(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        k = (flat % LEAF_VOLUME) >> 6
        sdf[flat] = (k + 0.5) * g.voxel_size - 0.2
    previous = wp.array([(0.125, 0.125, 0.275)], dtype=wp.vec3,
                        device=cuda_device)
    positions = wp.array([(0.125, 0.125, 0.125)], dtype=wp.vec3,
                         device=cuda_device)
    velocities = wp.array([(0.0, 0.0, -3.0)], dtype=wp.vec3,
                          device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    wp.launch(
        _resolve_segment_test, dim=1,
        inputs=[previous, positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                zero, zero, g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size], device=cuda_device)

    assert positions.numpy()[0, 2] >= 0.202
    assert velocities.numpy()[0, 2] == pytest.approx(0.0, abs=1.0e-5)


def test_particle_segment_collision_uses_remaining_timestep(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        k = (flat % LEAF_VOLUME) >> 6
        sdf[flat] = (k + 0.5) * g.voxel_size - 0.2
    previous = wp.array([(0.1, 0.125, 0.275)], dtype=wp.vec3,
                        device=cuda_device)
    positions = wp.array([(0.3, 0.125, 0.125)], dtype=wp.vec3,
                         device=cuda_device)
    velocities = wp.array([(2.0, 0.0, -1.5)], dtype=wp.vec3,
                          device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)

    wp.launch(
        _resolve_segment_test, dim=1,
        inputs=[previous, positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                zero, zero, g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size], device=cuda_device)

    assert positions.numpy()[0, 0] > 0.27


def test_particle_segment_collision_catches_through_crossing(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        k = (flat % LEAF_VOLUME) >> 6
        z = (k + 0.5) * g.voxel_size
        sdf[flat] = abs(z - 0.2) - 0.03
    previous = wp.array([(0.125, 0.125, 0.275)], dtype=wp.vec3,
                        device=cuda_device)
    positions = wp.array([(0.125, 0.125, 0.125)], dtype=wp.vec3,
                         device=cuda_device)
    velocities = wp.array([(0.0, 0.0, -3.0)], dtype=wp.vec3,
                          device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    wp.launch(
        _resolve_segment_test, dim=1,
        inputs=[previous, positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                zero, zero, g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size], device=cuda_device)

    assert positions.numpy()[0, 2] > 0.225
    assert velocities.numpy()[0, 2] == pytest.approx(0.0, abs=1.0e-5)


def test_particle_projection_rechecks_non_unit_sdf(cuda_device):
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        k = (flat % LEAF_VOLUME) >> 6
        sdf[flat] = 0.5 * ((k + 0.5) * g.voxel_size - 0.2)
    positions = wp.array([(0.125, 0.125, 0.15)], dtype=wp.vec3,
                         device=cuda_device)
    velocities = wp.array([(0.0, 0.0, -1.0)], dtype=wp.vec3,
                          device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    wp.launch(
        _resolve_segment_test, dim=1,
        inputs=[positions, positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                zero, zero, g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size], device=cuda_device)

    assert positions.numpy()[0, 2] >= 0.202


def test_sloped_boundary_uses_material_from_inside_collider(cuda_device):
    """Trilinear SDF contact may occur while the floored voxel is outside."""
    g = _one_leaf_grid(cuda_device)
    sdf = np.empty(LEAF_VOLUME, dtype=np.float32)
    friction = np.zeros(LEAF_VOLUME, dtype=np.float32)
    restitution = np.zeros(LEAF_VOLUME, dtype=np.float32)
    for flat in range(LEAF_VOLUME):
        local = flat % LEAF_VOLUME
        i = local & 7
        j = (local >> 3) & 7
        sdf[flat] = 0.1 - 0.3 * (i + j - 4)
        # Rasterization stores collider material throughout its query band,
        # including the positive-SDF samples used for interpolation.
        friction[flat] = 0.8
        restitution[flat] = 0.9
    positions = wp.array([(0.1375, 0.1375, 0.125)], dtype=wp.vec3,
                         device=cuda_device)
    velocities = wp.array([(1.0, 1.0, 0.0)], dtype=wp.vec3,
                          device=cuda_device)
    wp.launch(
        _resolve_material_test, dim=1,
        inputs=[positions, velocities,
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                wp.array(friction, dtype=float, device=cuda_device),
                wp.array(restitution, dtype=float, device=cuda_device),
                g.leaf_table, g.voxels_dim, g.leaves_dim,
                g.origin, g.voxel_size],
        device=cuda_device)

    result = velocities.numpy()[0]
    assert result[0] < -0.5 and result[1] < -0.5


def test_grid_boundary_applies_collider_friction_and_restitution(cuda_device):
    g = _one_leaf_grid(cuda_device)
    fluid = 3 + 3 * 8 + 3 * 64
    solid = fluid + 1
    voxel_type = np.zeros(LEAF_VOLUME, dtype=np.int32)
    voxel_type[fluid] = VOXEL_FLUID
    voxel_type[solid] = VOXEL_COLLISION
    sdf = np.ones(LEAF_VOLUME, dtype=np.float32)
    sdf[solid] = -1.0
    friction = np.zeros(LEAF_VOLUME, dtype=np.float32)
    restitution = np.zeros(LEAF_VOLUME, dtype=np.float32)
    friction[solid] = 0.75
    restitution[solid] = 0.9
    friction[fluid] = 0.75
    restitution[fluid] = 0.9
    sdf[fluid] = 0.02
    u = np.zeros(LEAF_VOLUME, dtype=np.float32)
    v = np.zeros(LEAF_VOLUME, dtype=np.float32)
    w = np.zeros(LEAF_VOLUME, dtype=np.float32)
    # MAC u is stored on the lower-X face. For a solid at fluid + X, the
    # contact face belongs to the solid voxel, not the fluid voxel.
    u[solid] = 1.0
    v[fluid] = 2.0
    u_dev = wp.array(u, dtype=float, device=cuda_device)
    v_dev = wp.array(v, dtype=float, device=cuda_device)
    w_dev = wp.array(w, dtype=float, device=cuda_device)
    wp.launch(
        k_apply_collider_material_to_grid, dim=LEAF_VOLUME,
        inputs=[wp.array(voxel_type, dtype=wp.int32, device=cuda_device),
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                wp.array(friction, dtype=float, device=cuda_device),
                wp.array(restitution, dtype=float, device=cuda_device),
                g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                g.num_leaves, g.voxel_size,
                u_dev, v_dev, w_dev, u_dev, v_dev, w_dev],
        device=cuda_device)

    assert u_dev.numpy()[solid] == pytest.approx(-0.9, abs=1.0e-5)
    assert v_dev.numpy()[fluid] == pytest.approx(2.0, abs=1.0e-5)

    separating = np.zeros(LEAF_VOLUME, dtype=np.float32)
    separating[solid] = -1.0
    separating_u = wp.array(separating, dtype=float, device=cuda_device)
    tangential = np.zeros(LEAF_VOLUME, dtype=np.float32)
    tangential[fluid] = 2.0
    tangential_v = wp.array(tangential, dtype=float, device=cuda_device)
    zero = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    wp.launch(
        k_apply_collider_material_to_grid, dim=LEAF_VOLUME,
        inputs=[wp.array(voxel_type, dtype=wp.int32, device=cuda_device),
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                wp.array(friction, dtype=float, device=cuda_device), zero,
                g.leaf_table, g.leaf_coords,
                g.voxels_dim, g.leaves_dim, g.num_leaves, g.voxel_size,
                separating_u, tangential_v, w_dev,
                separating_u, tangential_v, w_dev],
        device=cuda_device)
    assert separating_u.numpy()[solid] == pytest.approx(-1.0, abs=1.0e-5)
    assert tangential_v.numpy()[fluid] == pytest.approx(2.0, abs=1.0e-5)


def test_grid_friction_does_not_damp_fluid_without_solid_neighbor(cuda_device):
    g = _one_leaf_grid(cuda_device)
    fluid = 3 + 3 * 8 + 3 * 64
    voxel_type = np.zeros(LEAF_VOLUME, dtype=np.int32)
    voxel_type[fluid] = VOXEL_FLUID
    sdf = np.ones(LEAF_VOLUME, dtype=np.float32)
    sdf[fluid - 1:fluid + 2] = (0.05, 0.075, 0.1)
    friction = np.zeros(LEAF_VOLUME, dtype=np.float32)
    friction[fluid] = 0.75
    u = wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device)
    v = np.zeros(LEAF_VOLUME, dtype=np.float32)
    v[fluid] = 2.0
    v_dev = wp.array(v, dtype=float, device=cuda_device)

    wp.launch(
        k_apply_collider_material_to_grid, dim=LEAF_VOLUME,
        inputs=[wp.array(voxel_type, dtype=wp.int32, device=cuda_device),
                wp.array(sdf, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=wp.vec3, device=cuda_device),
                wp.array(friction, dtype=float, device=cuda_device),
                wp.zeros(LEAF_VOLUME, dtype=float, device=cuda_device),
                g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                g.num_leaves, g.voxel_size,
                u, v_dev, u, u, v_dev, u],
        device=cuda_device)

    assert v_dev.numpy()[fluid] == pytest.approx(2.0)


def _one_active_leaf_grid(device, leaf_capacity):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(3.2, 3.2, 3.2),
        voxel_size=0.05,
        leaf_capacity=leaf_capacity,
        device=device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))
    assert grid.finalize_topology() == 1
    assert grid.active_voxel_count == LEAF_VOLUME
    return grid


def _copy_float(dst, values, device):
    src = wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)
    wp.copy(dst, src)


def _copy_int(dst, values, device):
    src = wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)
    wp.copy(dst, src)


def _pressure_phase_elapsed(device, leaf_capacity, repeats=12):
    grid = _one_active_leaf_grid(device, leaf_capacity)
    solver = PressureSolver(grid, device)
    vel = MacVelocityField(grid)
    vel.zero()

    voxel_type = np.zeros(grid.voxel_capacity, dtype=np.int32)
    voxel_type[:LEAF_VOLUME] = VOXEL_FLUID
    voxel_type_wp = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=device)
    collision_vel = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=device)
    voxel_counts = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=device)
    _copy_int(voxel_type_wp, voxel_type, device)

    def run_once():
        solver.build_rhs(vel, voxel_type_wp, collision_vel, voxel_counts)
        solver.build_preconditioner(voxel_type_wp)
        solver.coarse_initial_guess(voxel_type_wp, iterations=2)

    run_once()
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
    wp.synchronize_device(device)
    return time.perf_counter() - start


def test_pressure_threshold_ignores_and_sanitizes_non_finite_pressure(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)

    pressure = np.zeros(grid.voxel_capacity, dtype=np.float32)
    pressure[:7] = [math.nan, math.inf, -math.inf, 0.01, 0.2, -1.0, 2.0]
    _copy_float(solver.pressure, pressure, cuda_device)

    solver.apply_threshold(0.25)

    out = solver.pressure.numpy()
    assert np.all(np.isfinite(out[:7]))
    assert np.allclose(out[:5], 0.0)
    assert out[5] == pytest.approx(-1.0)
    assert out[6] == pytest.approx(2.0)


def test_pressure_project_does_not_spread_non_finite_pressure_to_mac_velocity(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    vel = MacVelocityField(grid)
    vel.zero()

    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    pressure = np.zeros(grid.voxel_capacity, dtype=np.float32)
    pressure[:4] = [math.nan, math.inf, -math.inf, 1.0]

    voxel_type_wp = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    collision_vel = wp.zeros(grid.voxel_capacity, dtype=wp.vec3, device=cuda_device)
    _copy_int(voxel_type_wp, voxel_type, cuda_device)
    _copy_float(solver.pressure, pressure, cuda_device)

    solver.project(vel, voxel_type_wp, collision_vel, max_gradient=2.0)

    for component in (vel.u, vel.v, vel.w):
        assert np.all(np.isfinite(component.numpy()))
        assert np.max(np.abs(component.numpy())) <= 2.0


def test_pcg_reports_failure_status_when_system_cannot_converge(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)

    voxel_type = np.full(grid.voxel_capacity, VOXEL_COLLISION, dtype=np.int32)
    voxel_type_wp = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    _copy_int(voxel_type_wp, voxel_type, cuda_device)
    _copy_float(solver.rhs, np.ones(grid.voxel_capacity, dtype=np.float32), cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=3,
        tolerance=1.0e-12,
        check_every=1,
    )

    assert status["converged"] is False
    assert status["project_velocity"] is False
    assert status["iterations"] == solver.last_iterations
    assert status["reason"] in {
        "max_iterations",
        "non_finite_residual",
        "singular_system",
        "stalled",
    }
    assert math.isfinite(float(status["residual"]))
    assert solver.should_project_pressure() is False


def test_pcg_keeps_finite_best_effort_solution_projectable(cuda_device):
    """A finite iteration-limited solve must reduce divergence, not fail open.

    The old path zeroed pressure for every non-converged status. That discarded
    useful work and made project_frame advect the completely unprojected MAC
    field. Singular/non-finite systems remain a separate fail-closed case.
    """
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    voxel_type = np.zeros(grid.voxel_capacity, dtype=np.int32)
    voxel_type[:128] = VOXEL_FLUID
    voxel_type_wp = wp.zeros(
        grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    _copy_int(voxel_type_wp, voxel_type, cuda_device)
    rng = np.random.default_rng(42)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[:128] = rng.normal(size=128).astype(np.float32)
    rhs[:128] -= np.float32(np.mean(rhs[:128], dtype=np.float64))
    _copy_float(solver.rhs, rhs, cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    seeded_guess = wp.full(
        grid.voxel_capacity, 0.25, dtype=float, device=cuda_device)
    solver.coarse_initial_guess = lambda _voxel_type: wp.copy(
        solver.x0, seeded_guess)
    status = solver.solve_pcg(
        voxel_type_wp, max_iterations=0, tolerance=1.0e-20,
        use_coarse_guess=True, check_every=1)

    assert status["converged"] is False, status
    assert status["reason"] in {"max_iterations", "stalled"}
    assert math.isfinite(status["residual"])
    assert status["project_velocity"] is True
    assert solver.should_project_pressure() is True
    assert np.any(np.abs(solver.pressure.numpy()) > 0.0)


def test_sanitize_mac_velocity_zeroes_non_finite_faces_and_clamps_speed(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    vel = MacVelocityField(grid)
    vel.zero()

    u = np.zeros(grid.voxel_capacity, dtype=np.float32)
    u[:5] = [math.nan, math.inf, 3.5, -3.5, math.nan]
    weights = np.ones(grid.voxel_capacity, dtype=np.float32)
    weights[4] = 0.0
    _copy_float(vel.u, u, cuda_device)
    _copy_float(vel.wu, weights, cuda_device)

    wp.launch(
        k_sanitize_mac_velocity,
        dim=grid.voxel_capacity,
        inputs=[
            vel.u,
            vel.v,
            vel.w,
            vel.wu,
            vel.wv,
            vel.ww,
            grid.num_leaves,
            2.0,
        ],
        device=cuda_device,
    )

    out_u = vel.u.numpy()
    out_wu = vel.wu.numpy()
    assert np.all(np.isfinite(out_u[:5]))
    assert out_u[0] == pytest.approx(0.0)
    assert out_u[1] == pytest.approx(0.0)
    assert out_u[2] == pytest.approx(2.0)
    assert out_u[3] == pytest.approx(-2.0)
    assert out_u[4] == pytest.approx(0.0)
    assert out_wu[0] == pytest.approx(1.0e-6)
    assert out_wu[1] == pytest.approx(1.0e-6)
    assert out_wu[4] == pytest.approx(0.0)


def test_sanitize_particles_kills_non_finite_state_and_clamps_finite_speed(cuda_device):
    positions = wp.array(
        [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(math.nan, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        ],
        dtype=wp.vec3,
        device=cuda_device,
    )
    velocities = wp.array(
        [
            wp.vec3(3.0, 4.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(math.inf, 0.0, 0.0),
            wp.vec3(9.0, 0.0, 0.0),
        ],
        dtype=wp.vec3,
        device=cuda_device,
    )
    alive = wp.array([1, 1, 1, 0], dtype=wp.int32, device=cuda_device)
    count = wp.array([4], dtype=wp.int32, device=cuda_device)

    wp.launch(
        k_sanitize_particles,
        dim=4,
        inputs=[
            positions,
            velocities,
            alive,
            count,
            1.0,
            wp.vec3(-1.0, -1.0, -1.0),
            wp.vec3(1.0, 1.0, 1.0),
        ],
        device=cuda_device,
    )

    assert alive.numpy().tolist() == [1, 0, 0, 0]
    out_vel = velocities.numpy()
    assert np.linalg.norm(out_vel[0]) == pytest.approx(1.0)
    assert np.all(np.isfinite(out_vel[0]))


def test_sparse_grid_finalize_topology_can_defer_host_readback(cuda_device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        voxel_size=0.05,
        leaf_capacity=1,
        device=cuda_device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.399, 0.399, 0.399))

    assert grid.finalize_topology(sync=False) == 0
    assert grid.active_leaf_count == 0
    assert grid.refresh_topology_counts() == 1
    assert grid.active_leaf_count == 1


def test_sparse_grid_deferred_topology_preserves_overflow_error(cuda_device):
    grid = SparseGrid(
        origin=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        voxel_size=0.1,
        leaf_capacity=1,
        device=cuda_device,
    )
    grid.begin_topology()
    grid.mark_world_box((0.0, 0.0, 0.0), (0.999, 0.999, 0.999))

    grid.finalize_topology(sync=False)
    with pytest.raises(RuntimeError, match="leaf capacity exceeded"):
        grid.refresh_topology_counts()


@pytest.mark.skipif(
    not wp.is_conditional_graph_supported(), reason="CUDA conditional graph nodes unavailable")
def test_pcg_persistent_inner_product_is_conditional_body_safe(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    count = grid.voxel_capacity * 64
    a_np = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    b_np = np.linspace(0.25, 1.25, count, dtype=np.float32)
    a = wp.array(a_np, dtype=float, device=cuda_device)
    b = wp.array(b_np, dtype=float, device=cuda_device)

    expected = wp.zeros(1, dtype=float, device=cuda_device)
    wp.utils.array_inner(a, b, out=expected, count=count)
    expected_value = float(expected.numpy()[0])
    for _ in range(3):
        solver._persistent_inner_product(a, b, solver.p_ap, count)
        assert float(solver.p_ap.numpy()[0]) == pytest.approx(
            expected_value, rel=2.0e-5, abs=1.0e-3)

    solver.p_ap.zero_()
    with wp.ScopedCapture(device=cuda_device) as body_capture:
        solver._persistent_inner_product(a, b, solver.p_ap, count)
    condition = wp.zeros(1, dtype=wp.int32, device=cuda_device)
    with wp.ScopedCapture(device=cuda_device):
        wp.capture_while(condition=condition, while_body=body_capture.graph)


@pytest.mark.skipif(
    not wp.is_conditional_graph_supported(), reason="CUDA conditional graph nodes unavailable")
def test_pcg_zero_rhs_classifies_without_host_batched_fallback(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    voxel_type_wp = wp.array(voxel_type, dtype=wp.int32, device=cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )

    assert status["converged"] is True
    assert status["iterations"] == 0
    assert status["residual"] == 0.0
    assert solver.last_pcg_execution_mode == "cuda_conditional"

    solver._pcg_dot = None
    solver._pcg_native_available = False
    solver._pcg_conditional_graphs.clear()
    fallback = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )
    assert fallback["converged"] is True
    assert fallback["iterations"] == 0
    assert solver.last_pcg_execution_mode == "host_batched"


@pytest.mark.skipif(
    not wp.is_conditional_graph_supported(), reason="CUDA conditional graph nodes unavailable")
def test_pcg_non_finite_initial_residual_classifies_on_device(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    voxel_type_wp = wp.array(voxel_type, dtype=wp.int32, device=cuda_device)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0] = np.nan
    _copy_float(solver.rhs, rhs, cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )

    assert status["converged"] is False
    assert status["iterations"] == 0
    assert status["reason"] == "non_finite_residual"
    assert solver.last_pcg_execution_mode == "cuda_conditional"


def test_pcg_converges_with_batched_device_resident_checkpoints(cuda_device):
    """Convergence must still be detected with checkpoint-only readbacks."""
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)

    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    voxel_type_wp = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    _copy_int(voxel_type_wp, voxel_type, cuda_device)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0] = 1.0
    rhs[1] = -1.0  # zero-mean RHS: solvable under the pure-Neumann stencil
    _copy_float(solver.rhs, rhs, cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )

    assert status["converged"] is True
    assert status["project_velocity"] is True
    assert solver.should_project_pressure() is True
    assert solver.last_residual_squared < 1.0e-6 * 1.0e-6 * grid.voxel_capacity
    assert np.all(np.isfinite(solver.pressure.numpy()))
    assert solver.last_pcg_execution_mode == "cuda_conditional"
    assert solver._pcg_conditional_graphs

    commands = (
        solver._pcg_apply_cmd,
        solver._pcg_update_cmd,
        solver._pcg_update_p_cmd,
        solver._pcg_alpha_cmds,
        solver._pcg_beta_cmds,
        solver._pcg_pack_cmds,
    )
    graphs = dict(solver._pcg_conditional_graphs)
    assert graphs
    assert len(graphs) <= 8
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )

    assert status["converged"] is True
    assert commands == (
        solver._pcg_apply_cmd,
        solver._pcg_update_cmd,
        solver._pcg_update_p_cmd,
        solver._pcg_alpha_cmds,
        solver._pcg_beta_cmds,
        solver._pcg_pack_cmds,
    )
    assert solver._pcg_conditional_graphs == graphs
    assert np.all(np.isfinite(solver.pressure.numpy()))

    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=25,
        tolerance=1.0e-3,
        check_every=25,
    )
    assert status["converged"] is True
    assert solver.last_iterations == 25

    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=24,
    )
    assert status["converged"] is True
    assert solver.last_iterations % 2 == 0
    assert solver.last_pcg_execution_mode == "host_batched"

    solver._pcg_dot = None
    solver._pcg_native_available = False
    solver._pcg_graphs.clear()
    solver._pcg_conditional_graphs.clear()
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )
    assert status["converged"] is True
    assert solver.last_pcg_execution_mode == "host_batched"
    solver._pcg_native_available = None

    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=0,
        tolerance=1.0e-6,
        check_every=25,
    )
    assert status["reason"] == "max_iterations"
    assert solver.last_iterations == 0
    assert solver.last_pcg_execution_mode == "host_batched"

    replacement = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    _copy_int(replacement, voxel_type, cuda_device)
    previous_apply_cmd = solver._pcg_apply_cmd
    status = solver.solve_pcg(
        replacement,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )
    assert status["converged"] is True
    assert solver._pcg_apply_cmd is not previous_apply_cmd
    assert solver._pcg_conditional_graphs
    assert not any(
        graph in graphs.values()
        for graph in solver._pcg_conditional_graphs.values()
    )


@pytest.mark.parametrize("check_every", [24, 25])
@pytest.mark.parametrize("use_coarse_guess", [False, True])
def test_pcg_overwrites_stale_active_work_buffers(
    cuda_device, use_coarse_guess, check_every
):
    """Finite and non-finite stale work data must not affect a new solve."""
    grid = _one_leaf_grid(cuda_device)
    clean = PressureSolver(grid, cuda_device)
    stale = PressureSolver(grid, cuda_device)
    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    voxel_type_wp = wp.array(voxel_type, dtype=wp.int32, device=cuda_device)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0] = 1.0
    rhs[1] = -1.0

    for solver in (clean, stale):
        _copy_float(solver.rhs, rhs, cuda_device)
        solver.build_preconditioner(voxel_type_wp)
    poison = np.full(grid.voxel_capacity, np.nan, dtype=np.float32)
    for work_array in (stale.pressure, stale.r, stale.z, stale.p_dir, stale.ap):
        _copy_float(work_array, poison, cuda_device)

    kwargs = {
        "max_iterations": 400,
        "tolerance": 1.0e-6,
        "use_coarse_guess": use_coarse_guess,
        "check_every": check_every,
    }
    clean_status = clean.solve_pcg(voxel_type_wp, **kwargs)
    stale_status = stale.solve_pcg(voxel_type_wp, **kwargs)

    assert stale_status["converged"] == clean_status["converged"]
    assert stale_status["reason"] == clean_status["reason"]
    assert stale_status["project_velocity"] == clean_status["project_velocity"]
    assert stale.last_iterations == clean.last_iterations
    assert stale.last_residual_squared == pytest.approx(
        clean.last_residual_squared, rel=2.0e-5, abs=1.0e-12
    )
    assert np.allclose(
        stale.pressure.numpy(),
        clean.pressure.numpy(),
        rtol=2.0e-5,
        atol=1.0e-6,
    )
    assert np.all(np.isfinite(stale.pressure.numpy()))


def test_pcg_clears_pressure_when_a_fluid_voxel_becomes_air(cuda_device):
    grid = _one_leaf_grid(cuda_device)
    solver = PressureSolver(grid, cuda_device)
    voxel_type = np.full(grid.voxel_capacity, VOXEL_FLUID, dtype=np.int32)
    voxel_type_wp = wp.array(voxel_type, dtype=wp.int32, device=cuda_device)
    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0] = 1.0
    rhs[1] = -1.0
    _copy_float(solver.rhs, rhs, cuda_device)
    solver.build_preconditioner(voxel_type_wp)
    first_status = solver.solve_pcg(
        voxel_type_wp, max_iterations=400, tolerance=1.0e-6, check_every=25
    )
    assert first_status["converged"] is True

    first_pressure = solver.pressure.numpy()
    air_index = int(np.argmax(np.abs(first_pressure)))
    assert abs(float(first_pressure[air_index])) > 1.0e-6
    voxel_type[air_index] = VOXEL_NONE
    _copy_int(voxel_type_wp, voxel_type, cuda_device)
    fluid_indices = np.flatnonzero(voxel_type == VOXEL_FLUID)[:2]
    rhs.fill(0.0)
    rhs[fluid_indices[0]] = 1.0
    rhs[fluid_indices[1]] = -1.0
    _copy_float(solver.rhs, rhs, cuda_device)
    solver.build_preconditioner(voxel_type_wp)
    second_status = solver.solve_pcg(
        voxel_type_wp, max_iterations=400, tolerance=1.0e-6, check_every=25
    )

    assert second_status["converged"] is True
    assert solver.pressure.numpy()[air_index] == 0.0


def test_pressure_pcg_reductions_ignore_inactive_allocation_tail(cuda_device):
    """Inactive allocation slots must not affect active pressure PCG phases."""
    grid = _one_active_leaf_grid(cuda_device, leaf_capacity=512)
    solver = PressureSolver(grid, cuda_device)

    voxel_type = np.zeros(grid.voxel_capacity, dtype=np.int32)
    voxel_type[:grid.active_voxel_count] = VOXEL_FLUID
    voxel_type_wp = wp.zeros(grid.voxel_capacity, dtype=wp.int32, device=cuda_device)
    _copy_int(voxel_type_wp, voxel_type, cuda_device)

    inactive_tail = np.full(grid.voxel_capacity, 1.0e12, dtype=np.float32)
    inactive_tail[:grid.active_voxel_count] = 0.0
    for arr in (solver.r, solver.z, solver.p_dir, solver.ap, solver.x0):
        _copy_float(arr, inactive_tail, cuda_device)

    rhs = np.zeros(grid.voxel_capacity, dtype=np.float32)
    rhs[0] = 1.0
    rhs[1] = -1.0
    _copy_float(solver.rhs, rhs, cuda_device)

    solver.build_preconditioner(voxel_type_wp)
    status = solver.solve_pcg(
        voxel_type_wp,
        max_iterations=400,
        tolerance=1.0e-6,
        check_every=25,
    )

    assert status["converged"] is True
    assert solver.last_residual_squared < 1.0e-6 * 1.0e-6 * grid.active_voxel_count


def test_pressure_grid_phases_scale_with_active_leaves_not_allocation(cuda_device):
    """Fixed active pressure work must not grow with unused leaf allocation."""
    low = _pressure_phase_elapsed(cuda_device, leaf_capacity=1)
    high = _pressure_phase_elapsed(cuda_device, leaf_capacity=512)

    # The high-capacity case allocates 512x more leaf slots but keeps one
    # active leaf. A broad threshold keeps the check stable under CUDA timing
    # noise while still failing capacity-wide launches by a large margin.
    assert high <= max(low * 4.0, low + 0.010), (
        f"pressure phases grew disproportionately at fixed active work: "
        f"low={low:.6f}s high={high:.6f}s ratio={high / max(low, 1.0e-9):.2f}"
    )


def test_whitewater_compact_is_correct_without_host_count_refresh(cuda_device):
    """compact() must preserve live particles with device-only bookkeeping."""
    from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem

    ww = WhitewaterSystem(capacity=64, device=cuda_device)
    n = 10
    pos = np.zeros((64, 3), dtype=np.float32)
    pos[:n, 0] = np.arange(n, dtype=np.float32)
    alive = np.zeros(64, dtype=np.int32)
    alive[:n] = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    src_pos = wp.array(pos, dtype=wp.vec3, device=cuda_device)
    wp.copy(ww.position, src_pos)
    _copy_int(ww.alive, alive, cuda_device)
    _copy_int(ww.count, np.array([n], dtype=np.int32), cuda_device)

    ww.compact()

    live = ww.refresh_host_count()
    assert live == int(alive.sum()) == 6
    got = ww.position.numpy()[:live, 0]
    assert np.allclose(got, [0.0, 2.0, 3.0, 6.0, 7.0, 9.0])
    alive_out = ww.alive.numpy()
    assert np.all(alive_out[:live] == 1)
    assert np.all(alive_out[live:] == 0)


def test_particle_count_refresh_clamps_device_overflow(cuda_device):
    """Overflow races must not leave count[0] beyond allocated particles."""
    from fluidSim_warp_blender51.sim.particles import ParticleSystem

    psys = ParticleSystem(capacity=8, device=cuda_device)
    _copy_int(psys.count, np.array([37], dtype=np.int32), cuda_device)

    assert psys.refresh_host_count() == 8
    assert int(psys.count.numpy()[0]) == 8


def test_compaction_clamps_overflowed_counts_before_indexing(cuda_device):
    """Compaction must not read alive[count-1] after saturated emission."""
    from fluidSim_warp_blender51.sim.particles import ParticleSystem
    from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem

    psys = ParticleSystem(capacity=8, device=cuda_device)
    ww = WhitewaterSystem(capacity=8, device=cuda_device)
    alive = np.ones(8, dtype=np.int32)

    _copy_int(psys.alive, alive, cuda_device)
    _copy_int(psys.count, np.array([37], dtype=np.int32), cuda_device)
    psys.compact()
    assert psys.refresh_host_count() == 8

    _copy_int(ww.alive, alive, cuda_device)
    _copy_int(ww.count, np.array([37], dtype=np.int32), cuda_device)
    ww.compact()
    assert ww.refresh_host_count() == 8


def test_liquid_particle_compact_count_hint_ignores_inactive_tail(cuda_device):
    """Liquid compaction must scan/scatter only the solver-provided live prefix."""
    from fluidSim_warp_blender51.sim.particles import ParticleSystem

    psys = ParticleSystem(capacity=16, device=cuda_device)
    pos = np.zeros((16, 3), dtype=np.float32)
    pos[:, 0] = np.arange(16, dtype=np.float32)
    alive = np.ones(16, dtype=np.int32)
    alive[:8] = [1, 0, 1, 0, 1, 1, 0, 0]

    wp.copy(psys.position, wp.array(pos, dtype=wp.vec3, device=cuda_device))
    _copy_int(psys.alive, alive, cuda_device)
    _copy_int(psys.count, np.array([16], dtype=np.int32), cuda_device)

    psys.compact(count_hint=8)

    live = psys.refresh_host_count()
    assert live == 4
    assert np.allclose(psys.position.numpy()[:live, 0], [0.0, 2.0, 4.0, 5.0])


def test_compute_substeps_skips_cfl_readback_when_substeps_are_fixed(cuda_device):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(0.4, 0.4, 0.4),
        voxel_size=0.1,
        leaf_capacity=1,
        max_particles=8,
        max_whitewater=8,
        min_substeps=3,
        max_substeps=3,
        device=cuda_device,
    )
    solver = FluidSolver(settings)
    solver.particles.host_count = 1

    class _NoCflReduction:
        def zero_(self):
            raise AssertionError("fixed substeps should not reduce max speed")

        def numpy(self):
            raise AssertionError("fixed substeps should not read max speed")

    solver.max_speed_dev = _NoCflReduction()
    # Nothing was computed, so nothing was clamped: a fixed-substep scene must
    # not report a CFL violation it never tested for.
    assert solver._compute_substeps(1.0 / 24.0) == (3, False)


def test_compute_substeps_still_adapts_to_high_particle_speed(cuda_device):
    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(0.4, 0.4, 0.4),
        voxel_size=0.1,
        leaf_capacity=1,
        max_particles=8,
        max_whitewater=8,
        min_substeps=1,
        max_substeps=4,
        gravity=(0.0, 0.0, 0.0),
        device=cuda_device,
    )
    solver = FluidSolver(settings)
    solver.inject_particles([(0.1, 0.1, 0.1)], velocity=(100.0, 0.0, 0.0))

    # One particle at 100 m/s across a 0.1 m voxel demands far more than four
    # substeps, so this frame integrates at a timestep that violates CFL. The
    # second value is that fact; discarding it is what made the resulting
    # artefact unexplainable.
    assert solver._compute_substeps(1.0 / 24.0) == (4, True)


def test_project_frame_owns_algorithm_selection_and_projection_guard(monkeypatch):
    solver = object.__new__(PressureSolver)
    calls = []

    def record(name, status=None):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            if status is not None:
                solver.last_status = status
                solver.last_pressure_valid = status["project_velocity"]
                return status
        return call

    valid = {
        "converged": True,
        "iterations": 4,
        "reason": "jacobi",
        "residual": -1.0,
        "project_velocity": True,
    }
    invalid = {
        "converged": False,
        "iterations": 2,
        "reason": "non_finite_residual",
        "residual": math.inf,
        "project_velocity": False,
    }
    for name in (
            "build_rhs", "build_preconditioner", "build_preconditioner_st",
            "solve_pcg", "solve_pcg_st", "apply_threshold", "project",
            "project_st"):
        monkeypatch.setattr(solver, name, record(name))
    monkeypatch.setattr(solver, "solve_jacobi", record("solve_jacobi", valid))
    monkeypatch.setattr(
        solver, "solve_jacobi_st", record("solve_jacobi_st", valid))

    settings = SolverSettings(
        pressure_solver=0,
        pressure_iterations=4,
        pressure_threshold=0.25,
        spacetime_enabled=False,
    )
    result = solver.project_frame(
        "velocity", "voxel_type", "collision_velocity", "voxel_counts",
        settings, max_gradient=3.0)

    assert result is valid
    assert [call[0] for call in calls] == [
        "build_rhs", "build_preconditioner", "solve_jacobi",
        "apply_threshold", "project",
    ]
    assert calls[-1][2] == {"max_gradient": 3.0}

    calls.clear()
    monkeypatch.setattr(solver, "solve_pcg_st", record("solve_pcg_st", invalid))
    settings.pressure_solver = 1
    settings.spacetime_enabled = True
    settings.spacetime_variable_projection = True
    result = solver.project_frame(
        "velocity", "voxel_type", "collision_velocity", "voxel_counts",
        settings, max_gradient=2.0, reference_mass=7.0)

    assert result is invalid
    assert [call[0] for call in calls] == [
        "build_rhs", "build_preconditioner_st", "solve_pcg_st",
    ]
    assert calls[1][2]["reference_mass"] == 7.0
