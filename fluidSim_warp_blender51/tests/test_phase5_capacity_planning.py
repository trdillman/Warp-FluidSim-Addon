from __future__ import annotations

from fluidsim_testpkg.sim import capacity_planner
from fluidsim_testpkg.sim.memory_safety import (
    LEAF_VOLUME,
    SURFACE_TENSION_BYTES_PER_VOXEL,
    SURFACE_TENSION_PROPAGATION_BYTES_PER_VOXEL,
    estimate_solver_bytes,
    recommend_safe_settings,
)


def _settings(**overrides):
    values = {
        "domain_min": (0.0, 0.0, 0.0),
        "domain_max": (3.2, 3.2, 3.2),
        "voxel_size": 0.1,
        "leaf_capacity": 64,
        "max_particles": 0,
        "max_whitewater": 0,
        "particles_per_voxel": 8,
        "topology_expand": 0,
        "particle_radius_scale": 1.2,
        "surface_splat_radius": 2,
        "surface_enabled": False,
        "surface_filter_iterations": 0,
        "surface_dilate_iterations": 0,
        "surface_erode_iterations": 0,
        "surface_smooth_iterations": 2,
        "surface_tension_enabled": False,
        "surface_tension_coefficient": 0.072,
        "surface_tension_band_voxels": 1.5,
        "surface_tension_smoothing_iterations": 1,
    }
    values.update(overrides)
    return values


def test_memory_estimate_includes_optional_capillary_scratch():
    disabled = _settings()
    enabled = _settings(surface_tension_enabled=True)
    difference = estimate_solver_bytes(enabled) - estimate_solver_bytes(disabled)
    # Enabling surface tension allocates both the capillary scratch and the
    # SDF band-propagation buffers (_band_closest + _band_points{,_tmp}), so
    # the delta is the sum of the two per-voxel terms, not just the first.
    expected = (
        enabled["leaf_capacity"] * LEAF_VOLUME
        * (SURFACE_TENSION_BYTES_PER_VOXEL + SURFACE_TENSION_PROPAGATION_BYTES_PER_VOXEL)
    )
    assert difference == expected


def test_capacity_leaf_demand_uses_capillary_topology_halo():
    coords = ((1, 1, 1),)
    disabled = capacity_planner._expanded_leaf_coords(_settings(), coords)
    enabled_settings = _settings(surface_tension_enabled=True)
    enabled = capacity_planner._expanded_leaf_coords(enabled_settings, coords)
    assert disabled == coords
    assert capacity_planner._effective_topology_expand(enabled_settings) == 1
    assert len(enabled) == 27


def test_capacity_signature_tracks_capillary_support_inputs():
    baseline = capacity_planner._settings_dict(_settings())
    wider = capacity_planner._settings_dict(_settings(
        surface_tension_enabled=True,
        surface_tension_band_voxels=3.0,
    ))
    assert baseline["surface_tension_enabled"] is False
    assert wider["surface_tension_band_voxels"] == 3.0
    assert baseline != wider


def test_safe_recommendations_keep_capillary_memory_in_candidate_estimates():
    settings = _settings(
        domain_max=(12.8, 12.8, 12.8),
        leaf_capacity=2048,
        max_particles=100_000,
        max_whitewater=50_000,
        surface_tension_enabled=True,
    )
    budget = 640 * 1024**2
    assert estimate_solver_bytes(settings) > budget
    changes = recommend_safe_settings(settings, budget)
    assert changes
    candidate = dict(settings)
    candidate.update(changes)
    assert estimate_solver_bytes(candidate) <= budget


def test_capacity_halo_respects_a_larger_authored_sdf_splat_radius():
    settings = _settings(
        surface_tension_enabled=True,
        surface_splat_radius=16,
    )
    assert capacity_planner._effective_topology_expand(settings) == 2
