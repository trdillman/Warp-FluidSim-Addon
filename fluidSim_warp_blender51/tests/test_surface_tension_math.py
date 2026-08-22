from __future__ import annotations

import math

import pytest
from fluidsim_testpkg.sim.surface_tension_math import (
    capillary_dt,
    compact_cosine_delta,
    normal_support_voxels,
    rayleigh_lamb_frequency,
    required_capillary_substeps,
    required_sdf_splat_radius_voxels,
    required_topology_expand_leaves,
)


def test_zero_coefficient_has_no_capillary_limit():
    assert capillary_dt(
        density=1000.0, voxel_size=0.05, coefficient=0.0, safety=0.5
    ) == float("inf")


def test_capillary_dt_decreases_with_resolution_and_tension():
    coarse = capillary_dt(1000.0, 0.1, 0.072, 0.5)
    fine = capillary_dt(1000.0, 0.05, 0.072, 0.5)
    strong = capillary_dt(1000.0, 0.1, 0.144, 0.5)
    assert fine < coarse
    assert strong < coarse


def test_capillary_dt_uses_documented_brackbill_form():
    result = capillary_dt(1000.0, 0.05, 0.072, 0.5)
    expected = 0.5 * math.sqrt(1000.0 * 0.05**3 / (4.0 * math.pi * 0.072))
    assert result == pytest.approx(expected, rel=1.0e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"density": 0.0}, "density"),
        ({"voxel_size": 0.0}, "voxel_size"),
        ({"coefficient": -0.1}, "coefficient"),
        ({"safety": 0.0}, "safety"),
        ({"safety": 1.1}, "safety"),
    ],
)
def test_capillary_dt_rejects_invalid_physical_inputs(kwargs, message):
    values = dict(density=1000.0, voxel_size=0.05, coefficient=0.072, safety=0.5)
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        capillary_dt(**values)


def test_required_substeps_handles_finite_and_disabled_limits():
    assert required_capillary_substeps(0.0, 0.025) == 1
    assert required_capillary_substeps(1.0 / 24.0, float("inf")) == 1
    assert required_capillary_substeps(0.1, 0.025) == 4
    assert required_capillary_substeps(0.1, 0.024999) == 5


def test_required_substeps_rejects_negative_frame_time():
    with pytest.raises(ValueError, match="non-negative"):
        required_capillary_substeps(-0.1, 0.025)


def test_compact_delta_is_symmetric_normalized_and_zero_at_band_edge():
    width = 0.15
    assert compact_cosine_delta(0.0, width) == pytest.approx(1.0 / width)
    assert compact_cosine_delta(0.04, width) == pytest.approx(
        compact_cosine_delta(-0.04, width)
    )
    assert compact_cosine_delta(width, width) == 0.0
    assert compact_cosine_delta(width * 2.0, width) == 0.0


def test_droplet_frequency_has_required_surface_tension_trends():
    baseline = rayleigh_lamb_frequency(1000.0, 0.1, 0.072)
    stronger = rayleigh_lamb_frequency(1000.0, 0.1, 0.144)
    smaller = rayleigh_lamb_frequency(1000.0, 0.05, 0.072)
    assert stronger > baseline
    assert smaller > baseline


def test_sdf_support_covers_force_curvature_and_smoothing_stencils():
    assert normal_support_voxels(1.5, 0) == 2.5
    assert normal_support_voxels(1.5, 2) == 4.5
    assert required_sdf_splat_radius_voxels(0.6, 1.5, 1) == 6
    assert required_sdf_splat_radius_voxels(0.6, 1.5, 1, 2) == 8


def test_sdf_support_rejects_invalid_authored_values():
    with pytest.raises(ValueError, match="force_band_voxels"):
        normal_support_voxels(0.0, 1)
    with pytest.raises(ValueError, match="smoothing_iterations"):
        normal_support_voxels(1.5, -1)
    with pytest.raises(ValueError, match="particle_radius_voxels"):
        required_sdf_splat_radius_voxels(-0.1, 1.5, 1)
    with pytest.raises(ValueError, match="sdf_filter_iterations"):
        required_sdf_splat_radius_voxels(0.6, 1.5, 1, -1)


def test_topology_halo_covers_the_computed_sdf_radius():
    assert required_topology_expand_leaves(0, 8) == 0
    assert required_topology_expand_leaves(8, 8) == 1
    assert required_topology_expand_leaves(9, 8) == 2
    with pytest.raises(ValueError, match="leaf_dim"):
        required_topology_expand_leaves(8, 0)


def test_droplet_frequency_rejects_non_integer_modes():
    for mode in (True, 2.5):
        try:
            rayleigh_lamb_frequency(1000.0, 0.1, 0.072, mode=mode)
        except ValueError as exc:
            assert "integer" in str(exc)
        else:
            raise AssertionError(f"invalid mode {mode!r} was accepted")
