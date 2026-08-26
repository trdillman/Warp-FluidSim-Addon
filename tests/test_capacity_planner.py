from dataclasses import dataclass

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.capacity_planner import (
    CapacityVerdict,
    DemandQuality,
    plan_initialization,
    suggest_configurations,
)


@dataclass(frozen=True)
class Footprint:
    stable_id: str
    particle_demand: int | None
    particle_quality: DemandQuality
    leaf_coords: tuple[tuple[int, int, int], ...] = ()
    staging_bytes: int = 0
    reason: str = ""


def settings(**overrides):
    values = {
        "domain_min": (0.0, 0.0, 0.0),
        "domain_max": (8.0, 8.0, 8.0),
        "voxel_size": 1.0,
        "max_particles": 100,
        "leaf_capacity": 8,
        "particles_per_voxel": 8.0,
        "topology_expand": 0,
        "max_whitewater": 0,
        "surface_enabled": False,
    }
    values.update(overrides)
    return values


def resources(gib=16, free_gib=14):
    unit = 1024**3
    return {"gpu_total_bytes": gib * unit, "gpu_free_bytes": free_gib * unit}


def test_initial_plan_aggregates_seed_and_source_demand_before_admission():
    plan = plan_initialization(
        settings(max_particles=10),
        (Footprint("source", 6, DemandQuality.EXACT, ((0, 0, 0),)),),
        (np.zeros((5, 3), np.float32),),
        (),
        resources(),
    )

    assert plan.verdict is CapacityVerdict.BLOCKED
    assert plan.particle_demand.value == 11
    assert plan.particle_demand.quality is DemandQuality.EXACT
    assert plan.blockers[0].code == "PARTICLE_CAPACITY"
    assert plan.blockers[0].required == 11
    assert plan.blockers[0].available == 10


def test_transaction_residency_is_included_in_peak_plan_and_signature():
    base = plan_initialization(settings(), (), (), (), resources())
    resident_bytes = 14 * 1024**3
    transactional = plan_initialization(
        settings(), (), (), (), resources(),
        additional_transaction_bytes=resident_bytes)

    assert transactional.additional_transaction_bytes == resident_bytes
    assert transactional.transaction_scratch_bytes == (
        base.transaction_scratch_bytes + resident_bytes)
    assert transactional.estimated_bytes == base.estimated_bytes + resident_bytes
    assert transactional.scene_signature != base.scene_signature
    assert any(
        blocker.code == "MEMORY_ESTIMATE"
        for blocker in transactional.blockers)


def test_species_voxel_scratch_is_refused_before_initial_transport_allocation(
        monkeypatch):
    """The exact 40-byte voxel inventory must participate in initial admission."""
    from types import SimpleNamespace

    from fluidSim_warp_blender51.blender import simulation_admission
    from fluidSim_warp_blender51.sim import memory_safety

    disabled_settings = settings(species_enabled=False)
    enabled_settings = settings(
        species_enabled=True, species_diffusivity=0.1,
        species_max_diffusion_substeps=16)
    abundant = resources()
    disabled = plan_initialization(
        disabled_settings, (), (), (), abundant)
    enabled = plan_initialization(
        enabled_settings, (), (), (), abundant)
    # The 8^3 domain contains one sparse leaf even though capacity permits 8.
    expected_transport = 100 * 56 + 512 * 40 + 48
    assert enabled.species_transport_bytes == expected_transport
    assert enabled.estimated_bytes > disabled.estimated_bytes

    boundary_budget = (
        disabled.estimated_bytes
        + (enabled.estimated_bytes - disabled.estimated_bytes) // 2)
    constrained = {
        "gpu_total_bytes": 64 * 1024**3,
        "gpu_free_bytes": (boundary_budget * 10 + 8) // 9,
    }
    plan = plan_initialization(
        enabled_settings, (), (), (), constrained)

    assert disabled.estimated_bytes <= plan.budget_bytes
    assert plan.budget_bytes < enabled.estimated_bytes
    assert plan.verdict is CapacityVerdict.BLOCKED
    assert any(blocker.code == "MEMORY_ESTIMATE" for blocker in plan.blockers)

    allocations = []
    session = SimpleNamespace(
        solver=None,
        error="",
        _set_capacity_plan=lambda *_args, **_kwargs: None,
        set_error=lambda _scene, message: setattr(session, "error", str(message)),
    )
    owner = simulation_admission.SimulationAdmission(
        session,
        scene_identity=lambda _scene: 1,
        snippet_builder=lambda _scene: (),
        applied_state_capture=lambda *_args, **_kwargs: None,
        reset_settings_apply=lambda *_args, **_kwargs: None,
    )
    snapshot = SimpleNamespace(
        capacity_plan=plan,
        settings=enabled_settings,
        frame_start=1,
        release_transients=lambda: None,
    )
    scene = SimpleNamespace(
        frame_start=1,
        fluidsim=SimpleNamespace(status="", snippets=[]),
    )
    runtime_settings = SimpleNamespace(device="cuda:0", **enabled_settings)
    monkeypatch.setattr(
        simulation_admission, "build_settings", lambda _scene: runtime_settings)
    monkeypatch.setattr(memory_safety, "vram_snapshot", lambda _device: constrained)
    monkeypatch.setattr(owner, "_plan_scene", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(
        owner, "_prepare_solver_replacement",
        lambda *_args, **_kwargs: allocations.append("solver"),
    )

    with pytest.raises(RuntimeError, match="stopped initialization before CUDA allocation"):
        owner.initialize(scene)

    assert allocations == []


def test_unknown_particle_demand_fails_closed_without_inventing_a_count():
    plan = plan_initialization(
        settings(),
        (Footprint("dynamic-volume", None, DemandQuality.UNKNOWN,
                   reason="volume metadata unavailable"),),
        (), (), resources(),
    )

    assert plan.verdict is CapacityVerdict.BLOCKED
    assert plan.particle_demand.value is None
    assert plan.particle_demand.quality is DemandQuality.UNKNOWN
    assert any(blocker.code == "PARTICLE_DEMAND_UNKNOWN" for blocker in plan.blockers)
    assert suggest_configurations(plan, settings()) == ()


def test_leaf_union_is_deduplicated_and_expanded_within_domain():
    plan = plan_initialization(
        settings(topology_expand=1, leaf_capacity=8),
        (
            Footprint("a", 1, DemandQuality.EXACT, ((0, 0, 0),)),
            Footprint("b", 1, DemandQuality.EXACT, ((0, 0, 0),)),
        ),
        (), (), resources(),
    )

    # The 8x8x8 domain contains one leaf, so expansion clips to that leaf.
    assert plan.leaf_demand.value == 1
    assert plan.leaf_demand.quality is DemandQuality.EXACT
    assert plan.verdict is CapacityVerdict.ADMITTED


def test_seed_leaf_classification_deduplicates_and_clips_vectorized_payloads():
    plan = plan_initialization(
        settings(domain_max=(24.0, 8.0, 8.0)),
        (),
        (
            np.array([
                [-0.01, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [7.99, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ], np.float32),
            np.array([
                [8.0, 0.0, 0.0],
                [15.99, 0.0, 0.0],
                [24.0, 0.0, 0.0],
            ], np.float32),
        ),
        (),
        resources(),
    )

    assert plan.required_leaf_coords == ((0, 0, 0), (1, 0, 0))


def test_leaf_capacity_is_reported_independently_from_particles():
    plan = plan_initialization(
        settings(domain_max=(16.0, 8.0, 8.0), leaf_capacity=1),
        (Footprint("wide", 1, DemandQuality.EXACT,
                   ((0, 0, 0), (1, 0, 0))),),
        (), (), resources(),
    )

    assert plan.verdict is CapacityVerdict.BLOCKED
    assert any(
        blocker.code == "LEAF_CAPACITY"
        and blocker.required == 2
        and blocker.available == 1
        for blocker in plan.blockers
    )


def test_suggestions_are_immutable_estimates_and_do_not_mutate_settings():
    authored = settings(max_particles=5, leaf_capacity=1,
                        domain_max=(16.0, 8.0, 8.0))
    original = dict(authored)
    plan = plan_initialization(
        authored,
        (Footprint("source", 12, DemandQuality.UPPER_BOUND,
                   ((0, 0, 0), (1, 0, 0))),),
        (), (), resources(),
    )

    suggestions = suggest_configurations(plan, authored)

    assert authored == original
    assert suggestions
    assert suggestions[0].key == "preserve_simulation"
    assert suggestions[0].changes == {
        "max_particles": 12,
        "leaf_capacity": 2,
    }
    assert suggestions[0].priority == 0
    assert all("estimate" in item.disclaimer.lower() for item in suggestions)
    assert all(item.changes for item in suggestions)


def test_planner_exposes_domain_margin_and_balanced_alternatives_without_applying_them():
    authored = settings(
        domain_max=(32.0, 32.0, 32.0), max_particles=1,
        leaf_capacity=64)
    plan = plan_initialization(
        authored,
        (Footprint("center", 2, DemandQuality.UPPER_BOUND,
                   ((1, 1, 1),)),),
        (), (), resources(),
    )

    suggestions = suggest_configurations(plan, authored)
    labels = {item.label for item in suggestions}

    assert "Tighten empty X/Y/Z domain margins" in labels
    assert "Balanced bounds, resolution, and capacity" in labels
    assert authored["domain_max"] == (32.0, 32.0, 32.0)
    tightened = next(
        item for item in suggestions
        if item.label == "Tighten empty X/Y/Z domain margins")
    assert tightened.changes["domain_min"] == (8.0, 8.0, 8.0)
    assert tightened.changes["domain_max"] == (16.0, 16.0, 16.0)


def test_scene_signature_is_stable_for_equivalent_inputs_and_changes_with_demand():
    first = plan_initialization(
        settings(), (Footprint("source", 4, DemandQuality.EXACT),),
        (), (), resources(),
    )
    same = plan_initialization(
        dict(settings()), (Footprint("source", 4, DemandQuality.EXACT),),
        (), (), dict(resources()),
    )
    changed = plan_initialization(
        settings(), (Footprint("source", 5, DemandQuality.EXACT),),
        (), (), resources(),
    )

    assert first.scene_signature == same.scene_signature
    assert first.scene_signature != changed.scene_signature
