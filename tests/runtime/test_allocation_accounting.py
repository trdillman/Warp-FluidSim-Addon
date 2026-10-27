"""Allocation-owner regressions: measure constructed payload, not copied formulas."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import warp as wp

from fluidSim_warp_blender51.sim import memory_safety
from fluidSim_warp_blender51.sim.particles import ParticleSystem
from fluidSim_warp_blender51.sim.solver_settings_model import SolverSettings
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem


def test_admission_cost_imports_do_not_require_blender_or_warp():
    """Preflight planning remains available before GPU runtime import."""
    code = """
import importlib.abc
import sys
class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('warp', 'bpy'):
            raise AssertionError('runtime import attempted: ' + fullname)
sys.meta_path.insert(0, DenyRuntime())
from fluidSim_warp_blender51.sim import memory_safety, capacity_planner, particle_capacity
from fluidSim_warp_blender51.sim import particle_storage_layout, whitewater_storage_layout
assert 'warp' not in sys.modules and 'bpy' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def _candidate_bytes(owner):
    return sum(array.numpy().nbytes for array in (
        owner.candidates, owner.candidate_offsets, owner.candidate_types))


def test_whitewater_candidate_cost_matches_constructor_and_replacement():
    """Both real allocation paths own three int32 channels per liquid slot."""
    owner = WhitewaterSystem(3, "cpu", candidate_capacity=11)
    initial_bytes = _candidate_bytes(owner)
    workspace = owner.allocate_candidate_workspace(19)
    replacement_bytes = sum(array.numpy().nbytes for array in workspace)
    assert initial_bytes // 11 == replacement_bytes // 19
    assert memory_safety.WHITEWATER_EMISSION_BYTES_PER_LIQUID == initial_bytes // 11
    # Allocating a candidate must retain the old workspace until commit.
    assert _candidate_bytes(owner) == initial_bytes
    owner.swap_candidate_workspace(workspace)
    assert _candidate_bytes(owner) == replacement_bytes


def test_whitewater_live_cost_matches_real_capacity_slope():
    small = WhitewaterSystem(3, "cpu", candidate_capacity=11)
    large = WhitewaterSystem(7, "cpu", candidate_capacity=11)
    assert memory_safety.WHITEWATER_BYTES * (7 - 3) == (
        _direct_payload_bytes(large) - _direct_payload_bytes(small))


def _direct_payload_bytes(owner):
    """Unique arrays directly owned by a component, including channel maps."""
    arrays = {}

    def visit(value):
        if isinstance(value, wp.array):
            arrays[int(value.ptr)] = value.numpy().nbytes
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in vars(owner).values():
        visit(value)
    return sum(arrays.values())


@pytest.mark.parametrize("channels", [(), ("color",), ("source_id", "uid"),
                                      ("color", "source_id", "uid")])
def test_particle_optional_allowance_matches_real_double_buffer_payload(channels):
    from fluidSim_warp_blender51.sim.particle_storage_layout import (
        optional_particle_bytes_per_slot,
    )
    empty = ParticleSystem(7, "cpu")
    selected = ParticleSystem(7, "cpu", channels)
    assert (_direct_payload_bytes(selected) - _direct_payload_bytes(empty)
            == 7 * optional_particle_bytes_per_slot(channels))


def test_affine_allowance_matches_both_real_matrix_buffers():
    owner = ParticleSystem(7, "cpu")
    before = _direct_payload_bytes(owner)
    owner.ensure_affine()
    assert (_direct_payload_bytes(owner) - before
            == 7 * memory_safety.affine_bytes_per_particle(SimpleNamespace(sample_type=2)))


def _small_settings(**changes):
    return SolverSettings(domain_min=(0.0, 0.0, 0.0), domain_max=(0.8, 0.8, 0.8),
                          voxel_size=0.1, leaf_capacity=1, max_particles=7,
                          max_whitewater=3, surface_enabled=False, **changes)


def test_initial_estimate_charges_real_optional_channel_payload():
    base = _small_settings()
    selected = replace(base, export_attributes=True, particles_attribute_color=True,
                       particles_attribute_source_id=True, particles_attribute_uid=True)
    actual = (_direct_payload_bytes(ParticleSystem(7, "cpu", ("color", "source_id", "uid")))
              - _direct_payload_bytes(ParticleSystem(7, "cpu")))
    assert memory_safety.estimate_solver_bytes(selected) - memory_safety.estimate_solver_bytes(base) == actual


def test_capillarity_allowance_covers_real_lazy_buffers():
    from fluidSim_warp_blender51.sim.surface import SurfacePipeline
    from fluidSim_warp_blender51.sim.surface_tension import SurfaceTensionSolver
    base = _small_settings()
    owner = SimpleNamespace(device="cpu", grid=SimpleNamespace(voxel_capacity=512),
                            settings=base, profiler=None)
    tension = SurfaceTensionSolver(owner)
    before = _direct_payload_bytes(tension)
    tension.ensure_resources()
    # Only the propagation-buffer owner method is exercised, without building
    # an SDF or launching simulation kernels.
    propagation = SimpleNamespace(_band_closest=None, _band_points=None,
                                  _band_points_tmp=None)
    SurfacePipeline._ensure_band_buffers(propagation, 512, "cpu")
    actual = _direct_payload_bytes(tension) - before + _direct_payload_bytes(propagation)
    allowance = (memory_safety.estimate_solver_bytes(replace(base, surface_tension_enabled=True))
                 - memory_safety.estimate_solver_bytes(base))
    assert allowance >= actual


def test_topology_allowance_covers_all_real_domain_sized_tables():
    from fluidSim_warp_blender51.sim.fields import SparseGrid
    small = SparseGrid((0, 0, 0), (0.8, 0.8, 0.8), 0.1, 1, "cpu")
    large = SparseGrid((0, 0, 0), (4.0, 4.0, 4.0), 0.1, 1, "cpu")
    cells = large.num_leaf_cells - small.num_leaf_cells
    # The simulation owner retains one additional old table across rebuilds.
    old_small = wp.zeros(small.num_leaf_cells, dtype=wp.int32, device="cpu")
    old_large = wp.zeros(large.num_leaf_cells, dtype=wp.int32, device="cpu")
    actual = (_direct_payload_bytes(large) - _direct_payload_bytes(small)
              + old_large.numpy().nbytes - old_small.numpy().nbytes)
    assert cells > 0
    assert memory_safety.TOPOLOGY_BYTES_PER_LEAF_CELL * cells >= actual


def test_growth_charges_complete_replacement_and_real_fixed_bin_payload():
    from fluidSim_warp_blender51.sim.particle_capacity import plan_particle_growth
    from fluidSim_warp_blender51.sim.particles import ParticleBins
    from fluidSim_warp_blender51.sim.particle_storage_layout import bin_voxel_bytes
    bins = ParticleBins(7, 512, "cpu")
    actual_fixed = bins.counts.numpy().nbytes + bins.starts.numpy().nbytes
    assert bin_voxel_bytes(512) == actual_fixed
    parameters = dict(gpu_total_bytes=16 * 1024**3, gpu_free_bytes=8 * 1024**3)
    plan = plan_particle_growth(7, 8, fixed_allocation_bytes=actual_fixed, **parameters)
    assert plan.allowed
    assert plan.peak_allocation_bytes == memory_safety.particle_storage_bytes(plan.target) + actual_fixed
    assert plan.peak_allocation_bytes > memory_safety.particle_storage_bytes(plan.target - 7) + actual_fixed
    sufficient = plan_particle_growth(7, 7, fixed_allocation_bytes=actual_fixed, **parameters)
    assert sufficient.allowed and sufficient.peak_allocation_bytes == 0


def test_base_particle_and_binner_capacity_slope_fits_conservative_allowance():
    from fluidSim_warp_blender51.sim.particles import ParticleBins
    small = _direct_payload_bytes(ParticleSystem(7, "cpu")) + _direct_payload_bytes(ParticleBins(7, 512, "cpu"))
    large = _direct_payload_bytes(ParticleSystem(19, "cpu")) + _direct_payload_bytes(ParticleBins(19, 512, "cpu"))
    assert large - small <= (19 - 7) * memory_safety.PARTICLE_BYTES


def test_growth_guard_costs_actual_retained_channels_and_fixed_bins(monkeypatch):
    from fluidSim_warp_blender51.sim import particle_capacity
    from fluidSim_warp_blender51.sim.particles import ParticleBins
    from fluidSim_warp_blender51.sim.solver import FluidSolver
    particles = ParticleSystem(7, "cpu", ("color", "source_id", "uid"))
    particles.ensure_affine()
    bins = ParticleBins(7, 512, "cpu")
    # Authored settings can already differ from retained storage. A growth
    # transaction copies the existing owner and must cost those actual arrays.
    owner = SimpleNamespace(particles=particles, bins=bins, device="cpu",
                            grid=SimpleNamespace(voxel_capacity=512),
                            settings=_small_settings(), last_particle_growth=None)
    captured = {}

    def plan(current, required, **kwargs):
        captured.update(kwargs)
        return particle_capacity.ParticleGrowthPlan(current, required, 19, 1, 0, False, "observed")

    monkeypatch.setattr(particle_capacity, "plan_particle_growth", plan)
    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _: {
        "gpu_total_bytes": 16 * 1024**3, "gpu_free_bytes": 8 * 1024**3})
    result = FluidSolver.ensure_particle_capacity(owner, 8)
    assert result.status == "BLOCKED"
    base_payload = _direct_payload_bytes(ParticleSystem(7, "cpu"))
    assert captured["extra_particle_bytes"] * 7 == _direct_payload_bytes(particles) - base_payload
    assert captured["fixed_allocation_bytes"] == bins.counts.numpy().nbytes + bins.starts.numpy().nbytes
    assert owner.particles is particles


def test_narrow_band_allowance_covers_real_optional_grid_payload():
    from fluidSim_warp_blender51.sim.fields import SparseGrid
    from fluidSim_warp_blender51.sim.narrow_band_state import NarrowBand
    base = _small_settings()
    grid = SparseGrid((0, 0, 0), (0.8, 0.8, 0.8), 0.1, 1, "cpu")
    owner = SimpleNamespace(device="cpu", settings=base, grid=grid,
                            particles=ParticleSystem(7, "cpu"))
    actual = _direct_payload_bytes(NarrowBand(owner))
    allowance = (memory_safety.estimate_solver_bytes(replace(base, narrow_band_enabled=True))
                 - memory_safety.estimate_solver_bytes(base))
    assert allowance >= actual


def test_secondary_grid_allowance_covers_real_surface_owner_allocation():
    from fluidSim_warp_blender51.sim.fields import SparseGrid
    from fluidSim_warp_blender51.sim.surface import SurfacePipeline
    base = replace(_small_settings(), surface_enabled=True)
    settings = replace(base, surface_resolution_mode=1, surface_resolution_multiplier=2.0)
    grid = SparseGrid((0, 0, 0), (0.8, 0.8, 0.8), 0.1, 1, "cpu")
    owner = SimpleNamespace(device="cpu", settings=settings, grid=grid)
    surface = SurfacePipeline(owner)
    before = _direct_payload_bytes(surface)
    surface.ensure_resources()
    assert surface.grid is not grid
    actual = (_direct_payload_bytes(surface.grid) + _direct_payload_bytes(surface.render_sdf)
              + _direct_payload_bytes(surface.render_tmp) + _direct_payload_bytes(surface.render_extractor))
    assert actual > before
    allowance = memory_safety.estimate_solver_bytes(settings) - memory_safety.estimate_solver_bytes(base)
    assert allowance >= actual
