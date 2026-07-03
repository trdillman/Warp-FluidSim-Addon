"""Focused pressure projection safety regressions."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import (
    LEAF_VOLUME,
    VOXEL_COLLISION,
    VOXEL_FLUID,
)
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import (
    k_sanitize_mac_velocity,
    k_sanitize_particles,
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


def _copy_float(dst, values, device):
    src = wp.array(np.asarray(values, dtype=np.float32), dtype=float, device=device)
    wp.copy(dst, src)


def _copy_int(dst, values, device):
    src = wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)
    wp.copy(dst, src)


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
