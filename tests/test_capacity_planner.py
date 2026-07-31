from dataclasses import dataclass

import numpy as np

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
    assert suggest_configurations(plan, settings(), resources()) == ()


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

    suggestions = suggest_configurations(plan, authored, resources())

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

    suggestions = suggest_configurations(plan, authored, resources())
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
