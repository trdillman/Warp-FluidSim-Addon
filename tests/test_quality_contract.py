"""Pure and caller-parity tests for the authoritative quality resolver."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.io.cache import settings_record
from fluidSim_warp_blender51.sim.capacity_planner import plan_initialization
from fluidSim_warp_blender51.sim.fields import SparseGrid
from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
from fluidSim_warp_blender51.sim.quality_contract import (
    AuthoredQualitySettings,
    MAX_LIQUID_SAMPLING,
    MIN_VOXEL_SIZE,
    QUALITY_SCALED_FIELDS,
    SourceSamplingMode,
    SourceSamplingSettings,
    quality_tier_fidelity,
    resolve_quality,
    resolve_source_sampling,
    scale_quality_value,
)
from fluidSim_warp_blender51.sim.surface import SurfacePipeline


BOUNDS = ((-0.2, 0.1, 0.0), (1.01, 2.02, 0.81))


def test_resolution_uses_ceil_then_fixed_eight_cell_alignment():
    quality = resolve_quality(AuthoredQualitySettings(voxel_size=0.1), BOUNDS)
    assert quality.unpadded_voxel_dimensions == (13, 20, 9)
    assert quality.padded_voxel_dimensions == (16, 24, 16)
    assert quality.leaf_dimensions == (2, 3, 2)
    assert quality.total_domain_leaf_cells == 12


@pytest.mark.parametrize(
    ("mode", "multiplier", "custom", "expected"),
    [
        ("SIMULATION", 2.0, 0.025, 0.1),
        ("MULTIPLIER", 2.0, 0.025, 0.05),
        ("CUSTOM", 2.0, 0.025, 0.025),
    ],
)
def test_quality_accepts_blender_surface_resolution_identifiers(
        mode, multiplier, custom, expected):
    authored = SimpleNamespace(
        voxel_size=0.1,
        particles_per_voxel=8.0,
        surface_resolution_mode=mode,
        surface_resolution_multiplier=multiplier,
        surface_voxel_size=custom,
    )
    assert resolve_quality(authored, BOUNDS).surface_voxel_size == pytest.approx(expected)


def test_sparse_grid_consumes_exact_resolved_dimensions():
    quality = resolve_quality(AuthoredQualitySettings(voxel_size=0.1), BOUNDS)
    size = tuple(BOUNDS[1][i] - BOUNDS[0][i] for i in range(3))
    grid = SparseGrid(BOUNDS[0], size, 0.1, 99, "cpu")
    assert tuple(int(grid.voxels_dim[i]) for i in range(3)) == quality.padded_voxel_dimensions
    assert tuple(int(grid.leaves_dim[i]) for i in range(3)) == quality.leaf_dimensions
    assert grid.num_leaf_cells == quality.total_domain_leaf_cells


def test_sampling_spacing_and_dimensions_are_monotonic():
    coarse = resolve_quality(
        AuthoredQualitySettings(voxel_size=0.2, liquid_sampling=4), BOUNDS)
    fine = resolve_quality(
        AuthoredQualitySettings(voxel_size=0.1, liquid_sampling=4), BOUNDS)
    dense = resolve_quality(
        AuthoredQualitySettings(voxel_size=0.1, liquid_sampling=32), BOUNDS)
    assert all(fine.unpadded_voxel_dimensions[i] >= coarse.unpadded_voxel_dimensions[i]
               for i in range(3))
    assert fine.nominal_particle_spacing < coarse.nominal_particle_spacing
    assert dense.nominal_particle_spacing < fine.nominal_particle_spacing
    assert dense.rest_particle_count == 32


def test_surface_voxel_size_is_shared_by_surface_and_memory_paths():
    settings = SimpleNamespace(
        domain_min=BOUNDS[0], domain_max=BOUNDS[1], voxel_size=0.1,
        particles_per_voxel=8, surface_resolution_mode=1,
        surface_resolution_multiplier=2.0, surface_voxel_size=0.0,
        leaf_capacity=64, max_particles=1024, max_whitewater=0,
        surface_enabled=True, surface_leaf_capacity_scale=2.0,
        surface_max_dense_dim=128)
    quality = resolve_quality(settings, BOUNDS)
    assert quality.surface_voxel_size == pytest.approx(0.05)
    assert SurfacePipeline.voxel_size_for(settings) == pytest.approx(
        quality.surface_voxel_size)
    # The estimator must accept exactly the same authored object and remain
    # monotonic when only the resolved surface voxel is refined.
    base = estimate_solver_bytes(settings)
    settings.surface_resolution_multiplier = 4.0
    assert estimate_solver_bytes(settings) >= base


def test_source_sampling_modes_share_one_zero_to_64_contract():
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=0.1, liquid_sampling=12), BOUNDS)
    inherited = resolve_source_sampling(quality, SourceSamplingSettings())
    multiplied = resolve_source_sampling(
        quality, SourceSamplingSettings(SourceSamplingMode.MULTIPLIER, 2.0, 0.0))
    legacy_zero = resolve_source_sampling(
        quality, SourceSamplingSettings(
            SourceSamplingMode.LEGACY_ABSOLUTE, 1.0, 0.0))
    capped = resolve_source_sampling(
        quality, SourceSamplingSettings(
            SourceSamplingMode.LEGACY_ABSOLUTE, 1.0, 100.0))
    assert inherited.target_per_voxel == 12
    assert multiplied.target_per_voxel == 24
    assert legacy_zero.target_per_voxel == 0
    assert capped.target_per_voxel == 64


def test_quality_tier_fidelity_maps_tiers_and_defaults_to_preview():
    assert quality_tier_fidelity("DRAFT") == 0.5
    assert quality_tier_fidelity("PREVIEW") == 1.0
    assert quality_tier_fidelity("FINAL") == 2.0
    assert quality_tier_fidelity("CUSTOM") == 1.0
    # Case/whitespace tolerant, and unknown/missing values fall back to PREVIEW.
    assert quality_tier_fidelity(" final ") == 2.0
    assert quality_tier_fidelity("nonsense") == 1.0
    assert quality_tier_fidelity(None) == 1.0


def test_quality_scaled_fields_are_exactly_the_authored_knobs():
    assert QUALITY_SCALED_FIELDS == frozenset({
        "voxel_size", "min_substeps", "max_substeps",
        "pressure_iterations", "particles_per_voxel",
    })


def test_preview_and_custom_tiers_are_identity_on_every_knob():
    for field in QUALITY_SCALED_FIELDS:
        for tier in ("PREVIEW", "CUSTOM"):
            fidelity = quality_tier_fidelity(tier)
            assert scale_quality_value(field, 12, fidelity) == 12
            assert scale_quality_value(field, 0.1, fidelity) == 0.1


def test_final_refines_and_draft_coarsens_each_knob():
    final = quality_tier_fidelity("FINAL")   # 2.0
    draft = quality_tier_fidelity("DRAFT")   # 0.5
    # Voxel size scales inversely: finer grid at FINAL, coarser at DRAFT.
    assert scale_quality_value("voxel_size", 0.1, final) == pytest.approx(0.05)
    assert scale_quality_value("voxel_size", 0.1, draft) == pytest.approx(0.2)
    # Iteration knobs scale directly and round to a floored-at-1 integer.
    assert scale_quality_value("pressure_iterations", 40, final) == 80
    assert scale_quality_value("pressure_iterations", 40, draft) == 20
    assert scale_quality_value("min_substeps", 1, final) == 2
    assert scale_quality_value("min_substeps", 1, draft) == 1  # floored at 1
    assert scale_quality_value("max_substeps", 3, draft) == 2  # round(1.5) == 2
    # Sampling scales directly, bounded to the backend's 1..64 contract.
    assert scale_quality_value("particles_per_voxel", 8, final) == pytest.approx(16.0)
    assert scale_quality_value("particles_per_voxel", 8, draft) == pytest.approx(4.0)


def test_scale_is_non_destructive_and_repeatable_from_authored_value():
    # The projection always reads the freshly-authored value, so re-applying the
    # tier from that same authored value can never compound (defect #2 guard).
    authored_voxel = 0.1
    final = quality_tier_fidelity("FINAL")
    first = scale_quality_value("voxel_size", authored_voxel, final)
    second = scale_quality_value("voxel_size", authored_voxel, final)
    assert first == second == pytest.approx(0.05)
    assert authored_voxel == 0.1  # authored value is untouched


def test_scale_clamps_sampling_and_voxel_to_backend_bounds():
    # Sampling never exceeds the 0..64 backend contract nor drops below 1.
    assert scale_quality_value(
        "particles_per_voxel", 40, quality_tier_fidelity("FINAL")) == MAX_LIQUID_SAMPLING
    assert scale_quality_value(
        "particles_per_voxel", 1, quality_tier_fidelity("DRAFT")) == 1.0
    # Voxel size stays finite and positive under an extreme refine.
    assert scale_quality_value("voxel_size", 1e-9, 2.0) == MIN_VOXEL_SIZE


def test_capacity_leaf_count_and_cache_signature_include_resolved_quality():
    settings = SimpleNamespace(
        domain_min=BOUNDS[0], domain_max=BOUNDS[1], voxel_size=0.1,
        particles_per_voxel=8.0, topology_expand=0, leaf_capacity=64,
        max_particles=1024, max_whitewater=0, surface_enabled=False,
        surface_resolution_mode=0, surface_resolution_multiplier=1.0,
        surface_voxel_size=0.0, surface_leaf_capacity_scale=1.0,
        surface_max_dense_dim=128, resolved_quality=None)
    quality = resolve_quality(settings, BOUNDS)
    settings.resolved_quality = quality
    resources = {"gpu_total_bytes": 64 * 1024**3,
                 "gpu_free_bytes": 64 * 1024**3}
    plan = plan_initialization(settings, (), (), (), resources)
    assert plan.required_leaf_coords == ()
    record = settings_record(settings)
    assert record["settings"]["resolved_quality"]["padded_voxel_dimensions"] == [16, 24, 16]
    changed = replace(quality, liquid_sampling=9.0, rest_particle_count=9.0)
    settings.resolved_quality = changed
    assert settings_record(settings)["signature"] != record["signature"]
