import types
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from fluidSim_warp_blender51.sim.particle_capacity import (
    MAX_AUTO_PARTICLES,
    growth_target,
    plan_particle_growth,
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

    monkeypatch.setattr(
        memory_safety,
        "vram_snapshot",
        lambda _device: {
            "gpu_total_bytes": 16 * 1024**3,
            "gpu_free_bytes": 14 * 1024**3,
        },
    )
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    monkeypatch.setattr(
        particle_capacity,
        "plan_particle_growth",
        lambda *_args, **_kwargs: ParticleGrowthPlan(4, 6, 8, 1024, 4096, True),
    )
    particles = ParticleSystem(4, "cpu")
    wp.copy(
        particles.position,
        wp.array(
            np.asarray(((1.0, 2.0, 3.0),), np.float32), dtype=wp.vec3, device="cpu"
        ),
    )
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
        FluidSolver.ensure_particle_capacity, owner
    )
    def replace_particle_storage(replacement):
        owner.sourcing.particles = replacement

    owner.sourcing.replace_particle_storage = replace_particle_storage
    compiled_batch = owner.sourcing._compiled_batch

    result = owner.ensure_particle_capacity(6)

    assert result.grew is True
    assert owner.particles.capacity == 8
    assert owner.bins.particle_capacity == 8
    assert owner.whitewater.candidate_capacity == 8
    assert owner.settings.max_particles == 8
    np.testing.assert_allclose(owner.particles.position.numpy()[0], (1.0, 2.0, 3.0))
    assert owner.sourcing.particles is owner.particles
    assert owner.sourcing._compiled_batch is compiled_batch


def test_solver_growth_failure_preserves_authoritative_pool(monkeypatch):
    from fluidSim_warp_blender51.sim import memory_safety, particle_capacity
    from fluidSim_warp_blender51.sim.particle_capacity import ParticleGrowthPlan
    from fluidSim_warp_blender51.sim.particles import ParticleBins, ParticleSystem
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    monkeypatch.setattr(
        memory_safety,
        "vram_snapshot",
        lambda _device: {
            "gpu_total_bytes": 16 * 1024**3,
            "gpu_free_bytes": 14 * 1024**3,
        },
    )
    monkeypatch.setattr(memory_safety, "release_cuda_memory", lambda _device: None)
    monkeypatch.setattr(
        particle_capacity,
        "plan_particle_growth",
        lambda *_args, **_kwargs: ParticleGrowthPlan(4, 6, 8, 1024, 4096, True),
    )
    particles = ParticleSystem(4, "cpu")
    whitewater = WhitewaterSystem(4, "cpu", candidate_capacity=4)
    old_candidates = whitewater.candidates
    monkeypatch.setattr(
        whitewater,
        "allocate_candidate_workspace",
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
        FluidSolver.ensure_particle_capacity, owner
    )

    result = owner.ensure_particle_capacity(6)

    assert result.status == "FAILED"
    assert owner.whitewater is whitewater
    assert owner.whitewater.candidates is old_candidates
    assert owner.whitewater.candidate_capacity == 4
    assert owner.particles is particles
    assert owner.particles.capacity == 4
    assert owner.settings.max_particles == 4
    assert "synthetic OOM" in result.reason


@pytest.mark.parametrize(
    ("policy", "pause_requested"),
    (
        ("DEFER", False),
        ("PAUSE", True),
    ),
)
def test_growth_failure_policy_preserves_source_batch_for_retry(
    policy, pause_requested
):
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
            "BLOCKED", 2, 2, required, 4, reason="guarded"
        ),
    )
    pipeline = SourcePipeline(
        device=owner.device,
        grid=owner.grid,
        particles=owner.particles,
        settings=owner.settings,
        active_voxel_count=lambda: owner._nvox,
        current_frame=lambda: 0,
        ensure_particle_capacity=owner.ensure_particle_capacity,
        invalidate_host_cache=lambda: None,
        invalidate_particle_surface=lambda: None,
    )
    points = np.asarray(((0.0, 0.0, 0.0),) * 3, np.float32)

    injection = pipeline.inject_particles(points)
    assert not injection.admitted and injection.emitted == 0
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
            "BLOCKED", 4, 4, required, 4, reason="guarded"
        ),
    )
    pipeline = SourcePipeline(
        device=owner.device,
        grid=owner.grid,
        particles=owner.particles,
        settings=owner.settings,
        active_voxel_count=lambda: owner._nvox,
        current_frame=lambda: 0,
        ensure_particle_capacity=owner.ensure_particle_capacity,
        invalidate_host_cache=lambda: None,
        invalidate_particle_surface=lambda: None,
    )
    points = np.asarray(((0.0, 0.0, 0.0),) * 2, np.float32)

    injection = pipeline.inject_particles(points)
    assert not injection.admitted and injection.emitted == 0
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
