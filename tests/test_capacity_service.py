"""Contracts for the single active capacity assessment/recommendation service."""

from types import SimpleNamespace

from fluidSim_warp_blender51.sim import capacity_service
from fluidSim_warp_blender51.sim.capacity_planner import (
    CapacityPlan,
    CapacitySuggestion,
    CapacityVerdict,
    DemandQuality,
    DemandValue,
)


def _settings():
    return SimpleNamespace(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.6, 1.6, 1.6),
        voxel_size=0.1, particles_per_voxel=8.0,
        surface_resolution_mode=0, surface_resolution_multiplier=1.0,
        surface_voxel_size=0.0, surface_leaf_capacity_scale=1.0,
        surface_max_dense_dim=64, surface_enabled=False,
        topology_expand=0, max_particles=1024, leaf_capacity=64,
        max_whitewater=0)


def _resources():
    return {"gpu_total_bytes": 64 * 1024**3,
            "gpu_free_bytes": 64 * 1024**3}


def test_assess_owns_quality_plan_and_ranked_options():
    prepared = capacity_service.PreparedCapacityInput((), (), ())
    assessment = capacity_service.assess(_settings(), prepared, _resources())
    assert assessment.plan.verdict is CapacityVerdict.ADMITTED
    assert assessment.resolved_quality.padded_voxel_dimensions == (16, 16, 16)
    assert assessment.prepared is prepared
    assert assessment.plan.suggestions == ()


def test_recommend_omits_unreevaluated_semantic_candidates(monkeypatch):
    capacity = CapacitySuggestion(
        "Capacity only", {"max_particles": 2048}, 1, 2,
        key="capacity", priority=0)
    semantic = CapacitySuggestion(
        "Coarser", {"voxel_size": 0.2}, 1, 2,
        key="semantic", priority=10)
    mixed = CapacitySuggestion(
        "Mixed", {"max_particles": 2048, "voxel_size": 0.2}, 1, 2,
        key="mixed", priority=10)
    monkeypatch.setattr(
        capacity_service, "suggest_configurations",
        lambda _plan, _settings: (semantic, mixed, capacity))
    plan = CapacityPlan(
        CapacityVerdict.BLOCKED,
        DemandValue(2048, DemandQuality.EXACT),
        DemandValue(1, DemandQuality.EXACT),
        1, 2, (), (object(),), "sig")

    assert capacity_service.recommend(plan, _settings()) == (capacity,)
