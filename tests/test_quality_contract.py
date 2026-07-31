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
    SourceSamplingMode,
    SourceSamplingSettings,
    resolve_quality,
    resolve_source_sampling,
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
