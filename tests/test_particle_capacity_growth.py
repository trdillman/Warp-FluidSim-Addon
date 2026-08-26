import types
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.particle_capacity import (
    MAX_AUTO_PARTICLES,
    growth_target,
    particle_growth_candidate_bytes,
    plan_particle_growth,
)
from fluidSim_warp_blender51.sim.species import (
    species_transport_allocation_bytes,
)
from fluidSim_warp_blender51.sim.whitewater import WhitewaterSystem


def test_growth_target_adds_geometric_headroom_and_rounds_to_chunks():
    assert growth_target(1_024, 128_000) == 131_072
    assert growth_target(100_000, 100_001) == 196_608


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
def test_growth_target_rejects_nonfinite_factor(factor):
    with pytest.raises(ValueError, match="factor must be finite"):
        growth_target(1_024, 2_048, factor=factor)


def test_growth_guard_charges_full_atomic_replacement_pool():
    unit = 1024**3
    plan = plan_particle_growth(
        1_000_000,
        2_000_000,
        gpu_total_bytes=16 * unit,
        gpu_free_bytes=8 * unit,
    )

    assert plan.allowed is True
    assert plan.target >= 2_000_000
    assert plan.peak_allocation_bytes == plan.target * 136
    assert plan.budget_bytes == int(8 * unit * 0.90)


def test_species_growth_refuses_worked_literal_before_candidate_allocation(
        monkeypatch):
    """Species candidate bytes must be admitted before any replacement allocates."""
    from fluidSim_warp_blender51.sim import memory_safety
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    current = 32_768
    required = current + 1
    target = 65_536
    voxel_capacity = 1_536
    legacy_fixed_estimate = target * 136
    expected_candidate_bytes = target * (136 + 2 * (4 + 16) + 16)
    expected_budget = 10_800_000
    assert legacy_fixed_estimate < expected_budget < expected_candidate_bytes
    species_inventory = species_transport_allocation_bytes(
        target, voxel_capacity, True)
    inventory = particle_growth_candidate_bytes(
        target,
        additional_device_persistent_bytes=(
            species_inventory.particle_resize_candidate_bytes),
    )
    assert inventory.particle_pool_and_bins == target * 128
    assert inventory.whitewater_workspace == target * 8
    assert inventory.additional_device_persistent_bytes == (
        species_inventory.particle_resize_candidate_bytes)
    assert inventory.host_cache_staging == 0
    assert inventory.device_persistent_bytes == expected_candidate_bytes
    assert inventory.device_transaction_peak_bytes == expected_candidate_bytes

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 64 * 1024**2,
        "gpu_free_bytes": 12_000_000,
    })
    allocation_attempts = []

    class CandidateMustNotAllocate:
        capacity = current

        def resized_copy(self, _capacity):
            allocation_attempts.append("particles")
            raise AssertionError("candidate allocation preceded growth admission")

    owner = SimpleNamespace(
        particles=CandidateMustNotAllocate(),
        species=object(),
        grid=SimpleNamespace(voxel_capacity=voxel_capacity),
        device="cpu",
        settings=SolverSettings(
            max_particles=current, species_enabled=True,
            auto_grow_particles=True),
        last_particle_growth=None,
    )
    owner.ensure_particle_capacity = types.MethodType(
        FluidSolver.ensure_particle_capacity, owner)

    result = owner.ensure_particle_capacity(required)

    assert result.status == "BLOCKED"
    assert result.peak_allocation_bytes == expected_candidate_bytes
    assert result.budget_bytes == expected_budget
    assert allocation_attempts == []


def test_species_resize_candidate_reuses_capacity_invariant_device_state(
        monkeypatch):
    """A growth candidate must allocate only particle-sized transport scratch."""
    from fluidSim_warp_blender51.sim import species as species_api

    class FakeArray:
        def __init__(self, count, dtype, device):
            self.count = int(count)
            self.dtype = dtype
            self.device = device
            self.zero_calls = 0

        def zero_(self):
            self.zero_calls += 1

    allocations = []

    def fake_zeros(count, *, dtype, device):
        value = FakeArray(count, dtype, device)
        allocations.append(value)
        return value

    copies = []
    monkeypatch.setattr(species_api.wp, "zeros", fake_zeros)
    monkeypatch.setattr(
        species_api.wp, "copy", lambda *args, **kwargs: copies.append(
            (args, kwargs)))

    original = object.__new__(species_api.SpeciesTransport)
    original.particles = SimpleNamespace(capacity=4)
    original.grid = SimpleNamespace(voxel_capacity=1_536)
    original.device = "cpu"
    original.diffusivity = 0.125
    original.max_diffusion_substeps = 7
    original.palette = object()
    original.particle_delta = FakeArray(4, wp.vec4, "cpu")
    original.mix_volume = FakeArray(1_536, float, "cpu")
    original.species_mass = FakeArray(1_536, wp.vec4, "cpu")
    original.species_delta = FakeArray(1_536, wp.vec4, "cpu")
    original.outgoing_conductance = FakeArray(1_536, float, "cpu")
    original.global_alpha = FakeArray(1, float, "cpu")
    original._receipt = FakeArray(species_api.RECEIPT_SIZE, float, "cpu")
    original._frame_open = True
    original._after_captured = True
    original._last_receipt = object()
    original.frame_diffusion_substeps = 3
    grown_particles = SimpleNamespace(
        capacity=65_536,
        has_optional_channel=lambda name: name in {
            "represented_volume", "species_concentration"},
    )

    grown = original.resized_copy(grown_particles)

    assert [(value.count, value.dtype, value.device) for value in allocations] == [
        (65_536, wp.vec4, "cpu")]
    assert copies == []
    assert grown.particles is grown_particles
    assert grown.particle_delta is allocations[0]
    assert grown.particle_delta is not original.particle_delta
    for name in (
            "grid", "mix_volume", "species_mass", "species_delta",
            "outgoing_conductance", "global_alpha", "_receipt", "palette"):
        assert getattr(grown, name) is getattr(original, name)
    assert grown.diffusivity == original.diffusivity
    assert grown.max_diffusion_substeps == original.max_diffusion_substeps
    assert grown._frame_open is True
    assert grown._after_captured is True
    assert grown._last_receipt is original._last_receipt
    assert grown.frame_diffusion_substeps == 3


def test_species_growth_candidate_charges_only_capacity_axis_allocations():
    from fluidSim_warp_blender51.sim.species import (
        species_transport_allocation_bytes,
    )

    target = 65_536
    inventory = species_transport_allocation_bytes(target, 1_536, True)
    candidate = particle_growth_candidate_bytes(
        target,
        additional_device_persistent_bytes=(
            inventory.particle_resize_candidate_bytes),
    )

    assert inventory.particle_resize_candidate_bytes == target * 56
    assert candidate.additional_device_persistent_bytes == target * 56
    assert candidate.device_transaction_peak_bytes == target * 192


def test_growth_guard_fails_without_changing_the_requested_capacity():
    plan = plan_particle_growth(
        1_000_000,
        4_000_000,
        gpu_total_bytes=1024**3,
        gpu_free_bytes=128 * 1024**2,
    )

    assert plan.allowed is False
    assert plan.current == 1_000_000
    assert plan.required == 4_000_000
    assert "growth budget" in plan.reason


def test_particle_pool_resize_copies_live_state_without_mutating_original():
    from fluidSim_warp_blender51.sim.particles import ParticleSystem

    device = "cpu"
    original = ParticleSystem(4, device)
    positions = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), np.float32)
    wp.copy(original.position, wp.array(positions, dtype=wp.vec3, device=device))
    wp.copy(original.velocity, wp.array(positions * 0.5, dtype=wp.vec3, device=device))
    wp.copy(original.attr("density"), wp.array((2.0, 3.0), dtype=float, device=device))
    wp.copy(original.count, wp.array((2,), dtype=wp.int32, device=device))
    original.host_count = 2

    grown = original.resized_copy(8)

    assert original.capacity == 4
    assert grown.capacity == 8
    assert grown.host_count == 2
    np.testing.assert_allclose(grown.position.numpy()[:2], positions)
    np.testing.assert_allclose(grown.velocity.numpy()[:2], positions * 0.5)
    np.testing.assert_allclose(grown.attr("density").numpy()[:2], (2.0, 3.0))
    assert int(grown.count.numpy()[0]) == 2


def test_solver_growth_swaps_only_after_copy_succeeds(monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety, particle_capacity
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthPlan
    from fluidSim_warp_blender51.sim.particles import ParticleBins, ParticleSystem
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 14 * 1024**3,
    })
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    monkeypatch.setattr(
        particle_capacity,
        "plan_particle_growth",
        lambda *_args, **_kwargs: ParticleGrowthPlan(
            4, 6, 8, 1024, 4096, True),
    )
    particles = ParticleSystem(4, "cpu")
    wp.copy(particles.position, wp.array(
        np.asarray(((1.0, 2.0, 3.0),), np.float32),
        dtype=wp.vec3, device="cpu"))
    wp.copy(particles.count, wp.array((1,), dtype=wp.int32, device="cpu"))
    particles.host_count = 1
    owner = SimpleNamespace(
        particles=particles,
        bins=ParticleBins(4, 8, "cpu"),
        whitewater=WhitewaterSystem(4, "cpu", candidate_capacity=4),
        grid=SimpleNamespace(voxel_capacity=8),
        device="cpu",
        settings=SolverSettings(max_particles=4),
        sourcing=SimpleNamespace(_compiled_batch=object(), last_apply_result={}),
        _host_frame_cache=object(),
        last_particle_growth=None,
    )
    owner.ensure_particle_capacity = types.MethodType(
        FluidSolver.ensure_particle_capacity, owner)

    result = owner.ensure_particle_capacity(6)

    assert result.grew is True
    assert owner.particles.capacity == 8
    assert owner.bins.particle_capacity == 8
    assert owner.whitewater.candidate_capacity == 8
    assert owner.settings.max_particles == 8
    np.testing.assert_allclose(owner.particles.position.numpy()[0], (1.0, 2.0, 3.0))
    assert owner.sourcing._compiled_batch is None


def test_solver_growth_failure_preserves_authoritative_pool(monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety, particle_capacity
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthPlan
    from fluidSim_warp_blender51.sim.particles import ParticleBins, ParticleSystem
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 14 * 1024**3,
    })
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    monkeypatch.setattr(
        particle_capacity,
        "plan_particle_growth",
        lambda *_args, **_kwargs: ParticleGrowthPlan(
            4, 6, 8, 1024, 4096, True),
    )
    particles = ParticleSystem(4, "cpu")
    whitewater = WhitewaterSystem(4, "cpu", candidate_capacity=4)
    old_candidates = whitewater.candidates
    monkeypatch.setattr(
        whitewater, "allocate_candidate_workspace",
        lambda _capacity: (_ for _ in ()).throw(RuntimeError("synthetic OOM")),
    )
    owner = SimpleNamespace(
        particles=particles,
        bins=ParticleBins(4, 8, "cpu"),
        whitewater=whitewater,
        grid=SimpleNamespace(voxel_capacity=8),
        device="cpu",
        settings=SolverSettings(max_particles=4),
        sourcing=SimpleNamespace(_compiled_batch=object(), last_apply_result={}),
        _host_frame_cache=object(),
        last_particle_growth=None,
    )
    owner.ensure_particle_capacity = types.MethodType(
        FluidSolver.ensure_particle_capacity, owner)

    result = owner.ensure_particle_capacity(6)

    assert result.status == "FAILED"
    assert owner.whitewater is whitewater
    assert owner.whitewater.candidates is old_candidates
    assert owner.whitewater.candidate_capacity == 4
    assert owner.particles is particles
    assert owner.particles.capacity == 4
    assert owner.settings.max_particles == 4
    assert "synthetic OOM" in result.reason


def test_species_growth_sync_failure_preserves_authoritative_transport(
        monkeypatch):
    """A post-candidate sync failure must not detach shared species state."""
    from fluidSim_warp_blender51.sim import memory_safety, particle_capacity
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthPlan
    from fluidSim_warp_blender51.sim.particles import ParticleBins, ParticleSystem
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
    from fluidSim_warp_blender51.sim.species import SpeciesPalette, SpeciesTransport

    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: {
        "gpu_total_bytes": 16 * 1024**3,
        "gpu_free_bytes": 14 * 1024**3,
    })
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    monkeypatch.setattr(
        particle_capacity,
        "plan_particle_growth",
        lambda *_args, **_kwargs: ParticleGrowthPlan(
            4, 6, 8, 1_536, 4_096, True),
    )
    particles = ParticleSystem(
        4, "cpu", ("represented_volume", "species_concentration"))
    grid = SimpleNamespace(voxel_capacity=8)
    species = SpeciesTransport(
        particles, grid, "cpu", enabled=True, diffusivity=0.1,
        max_diffusion_substeps=4,
        palette=SpeciesPalette.create(
            ids=("a", "b"),
            absorption_rgb=((0.1, 1.0, 2.0), (2.0, 1.0, 0.1)),
            reference_path_length=1.0))
    receipt_before = np.arange(11, dtype=np.float32)
    wp.copy(species._receipt, wp.array(
        receipt_before, dtype=wp.float32, device="cpu"))
    species._frame_open = True
    species._after_captured = True
    species.frame_diffusion_substeps = 3
    old_receipt = species._receipt
    old_mix_volume = species.mix_volume
    owner = SimpleNamespace(
        particles=particles,
        bins=ParticleBins(4, 8, "cpu"),
        whitewater=WhitewaterSystem(4, "cpu", candidate_capacity=4),
        species=species,
        grid=grid,
        device="cpu",
        settings=SolverSettings(
            max_particles=4, species_enabled=True,
            species_ids=("a", "b")),
        sourcing=SimpleNamespace(_compiled_batch=object(), last_apply_result={}),
        _host_frame_cache=object(),
        last_particle_growth=None,
    )
    owner.ensure_particle_capacity = types.MethodType(
        FluidSolver.ensure_particle_capacity, owner)
    monkeypatch.setattr(
        wp, "synchronize_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("synthetic sync failure")))

    result = owner.ensure_particle_capacity(6)

    assert result.status == "FAILED"
    assert "synthetic sync failure" in result.reason
    assert owner.particles is particles
    assert owner.species is species
    assert owner.species._receipt is old_receipt
    assert owner.species.mix_volume is old_mix_volume
    assert owner.species._frame_open is True
    assert owner.species._after_captured is True
    assert owner.species.frame_diffusion_substeps == 3
    np.testing.assert_array_equal(owner.species._receipt.numpy(), receipt_before)


@pytest.mark.parametrize(("policy", "pause_requested"), (
    ("DEFER", False),
    ("PAUSE", True),
))
def test_growth_failure_policy_preserves_source_batch_for_retry(
        policy, pause_requested):
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthResult
    from fluidSim_warp_blender51.sim.particles import ParticleSystem
    from fluidSim_warp_blender51.sim.sourcing import SourcePipeline

    particles = ParticleSystem(2, "cpu")
    owner = SimpleNamespace(
        device="cpu",
        particles=particles,
        grid=SimpleNamespace(voxel_capacity=1),
        _nvox=1,
        settings=SimpleNamespace(particle_growth_failure_policy=policy),
        ensure_particle_capacity=lambda required: ParticleGrowthResult(
            "BLOCKED", 2, 2, required, 4, reason="guarded"),
    )
    pipeline = SourcePipeline(owner)
    points = np.asarray(((0.0, 0.0, 0.0),) * 3, np.float32)

    assert pipeline.inject_particles(points) == 0
    result = pipeline.last_apply_result
    assert result["admitted"] is False
    assert result["deferred"] is True
    assert result["pause_requested"] is pause_requested
    assert result["growth_status"] == "BLOCKED"
    assert particles.refresh_host_count() == 0


def test_direct_injection_admits_against_device_count_not_stale_host_mirror():
    """A stale host_count must not admit a batch the kernel would then drop."""
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthResult
    from fluidSim_warp_blender51.sim.particles import ParticleSystem
    from fluidSim_warp_blender51.sim.sourcing import SourcePipeline

    particles = ParticleSystem(4, "cpu")
    # Device counter advanced (as ordinary emission does) while the host mirror
    # still reports the pre-emission value.
    particles.count.fill_(3)
    particles.host_count = 0
    owner = SimpleNamespace(
        device="cpu",
        particles=particles,
        grid=SimpleNamespace(voxel_capacity=1),
        _nvox=1,
        settings=SimpleNamespace(particle_growth_failure_policy="DEFER"),
        ensure_particle_capacity=lambda required: ParticleGrowthResult(
            "BLOCKED", 4, 4, required, 4, reason="guarded"),
    )
    pipeline = SourcePipeline(owner)
    points = np.asarray(((0.0, 0.0, 0.0),) * 2, np.float32)

    assert pipeline.inject_particles(points) == 0
    result = pipeline.last_apply_result
    assert result["admitted"] is False
    assert result["required"] == 2
    assert result["available"] == 1
    assert result["emitted"] == 0
    assert particles.refresh_host_count() == 3


def test_growth_hard_limit_is_truthful():
    plan = plan_particle_growth(
        1_000_000,
        MAX_AUTO_PARTICLES + 1,
        gpu_total_bytes=128 * 1024**3,
        gpu_free_bytes=120 * 1024**3,
    )

    assert plan.allowed is False
    assert "hard limit" in plan.reason
