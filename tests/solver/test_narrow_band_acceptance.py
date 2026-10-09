"""Lane J acceptance: narrow-band particles.

The stop-2 kernels do not exist yet, so every test is a strict xfail that
fails today on the missing ``solver.narrow_band`` API.  Stop 2 deletes the
markers; a test that starts passing early is an XPASS and fails the suite, so
nothing here can go green by accident.
"""

from __future__ import annotations

import numpy as np
import pytest
from _narrow_band_harness import (
    BAND_VOXELS, H, MAX_PARTICLES, SETTLE_FRAMES, lattice, pressure_profile,
    run_tank, surface_height,
)

pytest.importorskip("warp")

STOP_2 = pytest.mark.xfail(strict=True, reason="lane J stop 2")


def _band(solver):
    """The stop-2 state object; AttributeError until the kernels land."""
    return solver.narrow_band


@STOP_2
def test_column_at_rest_matches_full_band_surface_and_hydrostatic_pressure(
        cuda_device):
    full = run_tank(cuda_device, narrow_band=False)
    try:
        full_height = surface_height(full)
        z_full, p_full = pressure_profile(full)
    finally:
        full.release()
    banded = run_tank(cuda_device, narrow_band=True)
    try:
        _band(banded)
        assert abs(surface_height(banded) - full_height) <= 0.1 * H
        z, p = pressure_profile(banded)
        # Hydrostatic: pressure rises linearly with depth below the surface.
        # Compare the interior against the line fitted to the full-band run.
        slope, intercept = np.polyfit(z_full, p_full, 1)
        interior = (z > 3 * H) & (z < full_height - (BAND_VOXELS + 1) * H)
        expected = slope * z[interior] + intercept
        np.testing.assert_allclose(p[interior], expected, rtol=0.01)
    finally:
        banded.release()


@STOP_2
def test_reseeding_is_bitwise_deterministic(cuda_device):
    runs = []
    for _ in range(2):
        solver = run_tank(cuda_device, narrow_band=True, frames=8)
        try:
            runs.append((
                solver.particle_positions().copy(),
                _band(solver).phi_numpy().copy()))
        finally:
            solver.release()
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


@STOP_2
def test_particle_count_never_exceeds_capacity_and_overflow_is_reported(
        cuda_device):
    # A deliberately tight cap: the reseed band alone cannot fit.
    cap = len(lattice()) // 8
    solver = run_tank(cuda_device, narrow_band=True, frames=0, max_particles=cap)
    try:
        band = _band(solver)
        for _ in range(6):
            solver.step_frame()
            assert solver.particles.refresh_host_count() <= cap
        assert band.reseed_overflow is True
    finally:
        solver.release()


@STOP_2
def test_deep_tank_uses_at_least_three_times_fewer_particles(cuda_device):
    full = run_tank(cuda_device, narrow_band=False, frames=SETTLE_FRAMES)
    try:
        full_count = full.particles.refresh_host_count()
    finally:
        full.release()
    banded = run_tank(cuda_device, narrow_band=True, frames=SETTLE_FRAMES)
    try:
        _band(banded)
        assert banded.particles.refresh_host_count() * 3 <= full_count
        # The liquid itself must not shrink with the particle count.
        assert banded.stats.liquid_voxels > 0
    finally:
        banded.release()


@STOP_2
def test_solver_sdf_is_negative_at_the_tank_centre(cuda_device):
    solver = run_tank(cuda_device, narrow_band=True, frames=SETTLE_FRAMES)
    try:
        phi = _band(solver).phi_numpy()
        assert phi.min() < 0.0
        # Union SDF (min of particle splat and phi) as the solver consumers see it.
        sdf = _band(solver).solver_sdf_at(
            np.array([[0.25, 0.25, 0.375]], dtype=np.float32))
        assert float(sdf[0]) < 0.0
    finally:
        solver.release()


def test_acceptance_harness_is_the_documented_tank():
    """Not xfail: pins the geometry the strict tests above are written for."""
    points = lattice()
    assert len(points) == 32 * 32 * 48
    assert points[:, 2].max() < 0.75
    assert MAX_PARTICLES >= len(points)


def test_full_band_baseline_measurements_are_sane(cuda_device):
    """Not xfail: the measurements the strict column test compares against."""
    solver = run_tank(cuda_device, narrow_band=False, frames=SETTLE_FRAMES)
    try:
        # The baseline column is still compressing at this frame count (about
        # 0.56 at frame 30); the strict tests compare against it, not 0.75.
        assert 0.4 < surface_height(solver) < 0.75
        z, p = pressure_profile(solver)
        assert p.size > 8
        slope = np.polyfit(z, p, 1)[0]
        assert slope < 0.0, "pressure must fall with height"
    finally:
        solver.release()
