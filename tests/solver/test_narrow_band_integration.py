"""Lane J integration: bake producer, checkpoint and off-path invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _narrow_band_harness import lattice, run_tank, settings

from fluidSim_warp_blender51.blender.production_bake import _liquid_sdf_arrays
from fluidSim_warp_blender51.io import cache
from fluidSim_warp_blender51.io.checkpoint import capture_solver, restore_solver
from fluidSim_warp_blender51.sim.solver import (
    _CHECKPOINT_FIELD_NAMES, FluidSolver,
)


class _StandInBand:
    def leaf_phi(self):
        coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
        phi = np.full((2, 512), -0.1, dtype=np.float32)
        return coords, phi


def _stand_in_solver(*, band, overflow=False):
    return SimpleNamespace(
        narrow_band=band, frame=3, grid=SimpleNamespace(voxel_size=0.03125, origin=(-1.25, 0.5, -0.75)),
        stats=SimpleNamespace(narrow_band_reseed_overflow=overflow))


def test_producer_returns_none_when_the_mode_is_off():
    assert _liquid_sdf_arrays(_stand_in_solver(band=None), False) is None


def test_producer_arrays_pass_the_cache_validator():
    arrays = _liquid_sdf_arrays(_stand_in_solver(band=_StandInBand()), True)
    assert set(arrays) == set(cache.LIQUID_SDF_KEYS) | {"liquid_sdf_origin"}
    np.testing.assert_array_equal(arrays["liquid_sdf_origin"],
                                  np.array([-1.25, 0.5, -0.75], dtype=np.float32))
    cache.validate_liquid_sdf(arrays, enabled=True)
    assert arrays["liquid_sdf_sparse"].shape == (2, 512)
    assert float(arrays["liquid_sdf_voxel_size"]) == pytest.approx(0.03125)


def test_producer_without_band_state_is_an_error_when_enabled():
    with pytest.raises(RuntimeError):
        _liquid_sdf_arrays(_stand_in_solver(band=None), True)


def test_producer_logs_reseed_overflow(capsys):
    _liquid_sdf_arrays(
        _stand_in_solver(band=_StandInBand(), overflow=True), True)
    assert "reseed" in capsys.readouterr().out


def test_off_path_checkpoint_has_no_narrow_band_state(cuda_device):
    solver = FluidSolver(settings(cuda_device, narrow_band=False))
    try:
        solver.inject_particles(lattice(0.1))
        solver.step_frame()
        assert solver._checkpoint_field_names() is _CHECKPOINT_FIELD_NAMES
        arrays = solver.export_checkpoint_state()["arrays"]
        assert not any(name.startswith("field_narrow_band") for name in arrays)
        assert "narrow_band_seed_state" not in arrays
    finally:
        solver.release()


def test_mode_on_checkpoint_round_trips(cuda_device):
    solver = run_tank(cuda_device, narrow_band=True, frames=4)
    try:
        payload = capture_solver(
            solver, settings_signature="s", source_signature="q")
        assert payload.arrays["narrow_band_seed_state"].dtype == np.uint32
        assert "field_narrow_band_phi" in payload.arrays
        phi = solver.narrow_band.phi_numpy().copy()
        epoch = solver.narrow_band.reseed_epoch
        positions = solver.particle_positions().copy()
        solver.step_frame()
        solver.step_frame()
        restore_solver(solver, payload)
        np.testing.assert_array_equal(solver.narrow_band.phi_numpy(), phi)
        assert solver.narrow_band.reseed_epoch == epoch
        np.testing.assert_array_equal(solver.particle_positions(), positions)
        # Stepping on from the restored state matches an uninterrupted run.
        solver.step_frame()
        resumed = solver.particle_positions().copy()
    finally:
        solver.release()
    straight = run_tank(cuda_device, narrow_band=True, frames=5)
    try:
        np.testing.assert_array_equal(straight.particle_positions(), resumed)
    finally:
        straight.release()


def test_memory_estimate_grows_only_when_the_mode_is_on():
    from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
    base = {"domain_min": (0, 0, 0), "domain_max": (1, 1, 1),
            "voxel_size": 0.05, "leaf_capacity": 64, "max_particles": 1000}
    off = estimate_solver_bytes(base)
    assert estimate_solver_bytes({**base, "narrow_band_enabled": False}) == off
    assert estimate_solver_bytes({**base, "narrow_band_enabled": True}) > off


def test_eikonal_depth_matches_analytic_distances():
    from fluidSim_warp_blender51.sim.kernels_narrow_band import eikonal_depth
    h = 0.5
    big = 100.0
    # One neighbour layer: depth is the neighbour plus one cell.
    assert eikonal_depth(0.0, big, big, h) == pytest.approx(0.5)
    # Two equal neighbours (a corner): h / sqrt(2) above them.
    assert eikonal_depth(0.0, 0.0, big, h) == pytest.approx(
        0.5 * (0.0 + 0.0 + np.sqrt(2 * h * h)))
    # Three equal neighbours: h / sqrt(3) above them.
    assert eikonal_depth(0.0, 0.0, 0.0, h) == pytest.approx(h / np.sqrt(3.0))


def test_credit_then_release_leaves_voxel_counts_unchanged(cuda_device):
    solver = run_tank(cuda_device, narrow_band=True, frames=3)
    try:
        band = solver.narrow_band
        before = solver.voxel_counts.data.numpy().copy()
        band.credit_interior_counts()
        credited = solver.voxel_counts.data.numpy().copy()
        assert (credited > before).any()
        band.release_interior_counts()
        np.testing.assert_array_equal(solver.voxel_counts.data.numpy(), before)
    finally:
        solver.release()
