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
)
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid
from fluidSim_warp_blender51.sim.pressure import PressureSolver
from fluidSim_warp_blender51.sim.solver import (
    FluidSolver,
    SolverSettings,
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
    assert solver._compute_substeps(1.0 / 24.0) == 3


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

    assert solver._compute_substeps(1.0 / 24.0) == 4
