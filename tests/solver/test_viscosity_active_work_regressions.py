"""Focused viscosity active-work regressions."""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME, VOXEL_FLUID  # noqa: E402
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid  # noqa: E402
from fluidSim_warp_blender51.sim.viscosity import ViscositySolver  # noqa: E402


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


def _viscosity_inputs(grid, device):
    vel = MacVelocityField(grid)
    n = grid.voxel_capacity
    active = grid.active_voxel_count
    ramp = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    sentinel = np.full(n - active, 7.0, dtype=np.float32)
    values = ramp
    values[active:] = sentinel
    weights = np.zeros(n, dtype=np.float32)
    weights[:active] = 1.0
    viscosity = np.zeros(n, dtype=np.float32)
    viscosity[:active] = 0.25
    voxel_type = np.zeros(n, dtype=np.int32)
    voxel_type[:active] = VOXEL_FLUID

    for component in (vel.u, vel.v, vel.w):
        _copy_float(component, values, device)
    for component in (vel.wu, vel.wv, vel.ww):
        _copy_float(component, weights, device)
    viscosity_wp = wp.zeros(n, dtype=float, device=device)
    voxel_type_wp = wp.zeros(n, dtype=wp.int32, device=device)
    _copy_float(viscosity_wp, viscosity, device)
    _copy_int(voxel_type_wp, voxel_type, device)
    return vel, viscosity_wp, voxel_type_wp, sentinel


def _viscosity_phase_elapsed(device, leaf_capacity, repeats=8):
    grid = _one_active_leaf_grid(device, leaf_capacity)
    solver = ViscositySolver(grid, device)
    vel, viscosity, voxel_type, _ = _viscosity_inputs(grid, device)

    def run_once():
        solver.solve(
            vel,
            viscosity,
            voxel_type,
            dt=1.0 / 120.0,
            use_pcg=True,
            iterations=8,
            tolerance=1.0e-5,
        )

    run_once()
    wp.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
    wp.synchronize_device(device)
    return time.perf_counter() - start


def test_viscosity_smooth_preserves_inactive_tail(cuda_device):
    grid = _one_active_leaf_grid(cuda_device, leaf_capacity=16)
    solver = ViscositySolver(grid, cuda_device)
    vel, viscosity, voxel_type, sentinel = _viscosity_inputs(grid, cuda_device)

    solver.smooth(vel, viscosity, voxel_type, dt=1.0 / 120.0, iterations=1)

    assert np.allclose(vel.u.numpy()[grid.active_voxel_count:], sentinel)
    assert np.allclose(vel.v.numpy()[grid.active_voxel_count:], sentinel)
    assert np.allclose(vel.w.numpy()[grid.active_voxel_count:], sentinel)


def test_viscosity_solve_preserves_inactive_tail(cuda_device):
    grid = _one_active_leaf_grid(cuda_device, leaf_capacity=16)
    solver = ViscositySolver(grid, cuda_device)
    vel, viscosity, voxel_type, sentinel = _viscosity_inputs(grid, cuda_device)

    solver.solve(
        vel,
        viscosity,
        voxel_type,
        dt=1.0 / 120.0,
        use_pcg=True,
        iterations=4,
        tolerance=1.0e-5,
    )

    assert np.allclose(vel.u.numpy()[grid.active_voxel_count:], sentinel)
    assert np.allclose(vel.v.numpy()[grid.active_voxel_count:], sentinel)
    assert np.allclose(vel.w.numpy()[grid.active_voxel_count:], sentinel)


def test_viscosity_fixed_active_work_uses_active_bounds(cuda_device, monkeypatch):
    """Host launches, reductions, and copies must use active voxel bounds."""
    import fluidSim_warp_blender51.sim.viscosity as viscosity_mod

    grid = _one_active_leaf_grid(cuda_device, leaf_capacity=16)
    solver = ViscositySolver(grid, cuda_device)
    vel, viscosity, voxel_type, _ = _viscosity_inputs(grid, cuda_device)
    active = grid.active_voxel_count

    launch_dims = []
    reduction_counts = []
    copy_counts = []
    tracked_copy_dests = {id(vel.u), id(vel.v), id(vel.w), id(solver.rz)}

    def fake_launch(*args, **kwargs):
        launch_dims.append(kwargs["dim"])

    def fake_inner(a, b, out=None, count=None, axis=None):
        reduction_counts.append(count)
        if out is not None:
            return out
        return 0.0

    # The device-checked PCG path reduces with the native accumulating dot
    # product instead of ``array_inner`` (an allocating reduction cannot be
    # captured into a conditional graph), so the bound has to be asserted on
    # both reduction APIs.
    from fluidSim_warp_blender51.sim import cuda_reduction

    real_dot_launch = cuda_reduction.CudaDotProduct.launch

    def fake_dot_launch(self, a, b, output, count):
        reduction_counts.append(count)
        return real_dot_launch(self, a, b, output, count)

    monkeypatch.setattr(
        cuda_reduction.CudaDotProduct, "launch", fake_dot_launch)

    # The captured reductions read the active count on device and launch over
    # its bucket, which for one full leaf is the leaf itself.
    real_dot_launch_active = cuda_reduction.CudaDotProduct.launch_active

    def fake_dot_launch_active(self, a, b, output, num_leaves, dim, *args):
        reduction_counts.append(dim)
        return real_dot_launch_active(self, a, b, output, num_leaves, dim, *args)

    monkeypatch.setattr(
        cuda_reduction.CudaDotProduct, "launch_active", fake_dot_launch_active)

    def fake_copy(dest, src, dest_offset=0, src_offset=0, count=0, stream=None):
        if id(dest) in tracked_copy_dests:
            copy_counts.append(count)

    monkeypatch.setattr(viscosity_mod.wp, "launch", fake_launch)
    monkeypatch.setattr(viscosity_mod.wp.utils, "array_inner", fake_inner)
    monkeypatch.setattr(viscosity_mod.wp, "copy", fake_copy)

    solver.smooth(vel, viscosity, voxel_type, dt=1.0 / 120.0, iterations=1)
    solver.solve(
        vel,
        viscosity,
        voxel_type,
        dt=1.0 / 120.0,
        use_pcg=True,
        iterations=1,
        tolerance=1.0e-5,
    )

    assert grid.voxel_capacity > active
    assert max(launch_dims) <= active
    assert reduction_counts
    assert all(count == active for count in reduction_counts)
    assert copy_counts
    assert all(count in {1, active} for count in copy_counts)


def test_viscosity_grid_phases_scale_with_active_leaves_not_allocation(cuda_device):
    """Fixed active viscosity work must not grow with unused leaf allocation."""
    low = _viscosity_phase_elapsed(cuda_device, leaf_capacity=1)
    high = _viscosity_phase_elapsed(cuda_device, leaf_capacity=512)

    assert high <= max(low * 4.0, low + 0.015), (
        f"viscosity phases grew disproportionately at fixed active work: "
        f"low={low:.6f}s high={high:.6f}s ratio={high / max(low, 1.0e-9):.2f}"
    )
