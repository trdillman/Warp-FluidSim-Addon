from dataclasses import dataclass

import numpy as np

from fluidSim_warp_blender51.sim.capacity_planner import (
    CapacityVerdict,
    DemandQuality,
    _snap_voxel_coarser,
    plan_initialization,
    suggest_configurations,
)
from fluidSim_warp_blender51.sim.memory_safety import estimate_solver_bytes
from fluidSim_warp_blender51.sim.quality_contract import (
    MIN_RESOLUTION_DIVISIONS,
    voxel_size_for_divisions,
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


def test_suggestion_estimates_cost_apic_and_narrow_band_like_the_applied_settings():
    # B51: the suggestion must cost the settings the solver will actually
    # allocate after the option is applied, including the APIC affine buffers
    # and the narrow-band arrays; otherwise the approved option is refused.
    for extra in ({"sample_type": 2}, {"narrow_band_enabled": True},
                  {"sample_type": 2, "narrow_band_enabled": True}):
        authored = settings(
            domain_min=(-2.0, -2.0, 0.0), domain_max=(2.0, 2.0, 4.0),
            voxel_size=0.02, max_particles=1_000_000, leaf_capacity=8192,
            **extra)
        plan = plan_initialization(
            authored,
            (Footprint("src", 20_000_000, DemandQuality.EXACT, ((0, 0, 0),)),),
            (), (), resources())

        suggestions = suggest_configurations(plan, authored)

        assert suggestions, extra
        for item in suggestions:
            applied = dict(authored, **item.changes)
            after_apply = (
                int(estimate_solver_bytes(applied)) + plan.staging_bytes
                + applied["leaf_capacity"] * 512 * 4
                + plan.additional_transaction_bytes)
            assert item.estimated_bytes == after_apply, (extra, item.key)
            assert after_apply <= item.budget_bytes, (extra, item.key)


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


# --- suggestions stay on the resolution-divisions lattice ----------------------


def _lattice_divisions(voxel_size, longest):
    divisions = round(longest / voxel_size)
    assert abs(voxel_size - longest / divisions) < 1e-12
    return divisions


def _sweep_suggestions(authored, footprints, key):
    """Yield (suggestion, plan) for ``key`` across a sweep of live budgets."""
    unit = 1024**2
    for free_mib in range(64, 8192, 64):
        res = {"gpu_total_bytes": 16 * 1024**3, "gpu_free_bytes": free_mib * unit}
        plan = plan_initialization(authored, footprints, (), (), res)
        for item in suggest_configurations(plan, authored):
            if item.key == key:
                yield item, plan


def test_coarser_resolution_is_on_the_divisions_lattice_and_never_finer():
    authored = settings(
        domain_max=(40.0, 20.0, 10.0), voxel_size=0.25, max_particles=1,
        leaf_capacity=64)
    footprints = (Footprint("src", 50_000, DemandQuality.UPPER_BOUND, ((1, 1, 1),)),)
    longest = 40.0
    base_divisions = 160

    found = list(_sweep_suggestions(authored, footprints, "coarser_resolution"))

    assert found
    for item, plan in found:
        voxel = item.changes["voxel_size"]
        divisions = _lattice_divisions(voxel, longest)
        assert voxel == voxel_size_for_divisions(divisions, longest)
        assert MIN_RESOLUTION_DIVISIONS <= divisions < base_divisions
        assert voxel > 0.25
        candidate = dict(authored, **item.changes)
        assert item.estimated_bytes == (
            int(estimate_solver_bytes(candidate)) + plan.staging_bytes
            + candidate["leaf_capacity"] * 512 * 4 + plan.additional_transaction_bytes)
        assert item.estimated_bytes <= item.budget_bytes


def test_balanced_voxel_is_lattice_snapped_for_the_tightened_domain():
    authored = settings(
        domain_max=(32.0, 32.0, 32.0), voxel_size=0.2, max_particles=1,
        leaf_capacity=64)
    footprints = (Footprint("center", 2, DemandQuality.UPPER_BOUND, ((10, 10, 10),)),)

    found = list(_sweep_suggestions(authored, footprints, "balanced"))

    assert found
    for item, _plan in found:
        lo = item.changes.get("domain_min", authored["domain_min"])
        hi = item.changes.get("domain_max", authored["domain_max"])
        longest = max(h - l for l, h in zip(lo, hi))
        voxel = item.changes["voxel_size"]
        divisions = _lattice_divisions(voxel, longest)
        assert voxel == voxel_size_for_divisions(divisions, longest)
        assert voxel >= 0.2 * (1 - 1e-9)  # never finer than the authored voxel
        if divisions > MIN_RESOLUTION_DIVISIONS:
            assert voxel >= 0.2 * 1.10 * (1 - 1e-9)  # never finer than the costed 10%


def test_snap_rounds_toward_fewer_divisions_and_respects_the_minimum():
    assert _snap_voxel_coarser(0.3, 10.0) == 10.0 / 33
    assert _snap_voxel_coarser(0.25, 10.0) == 10.0 / 40  # exact lattice point kept
    assert _snap_voxel_coarser(5.0, 10.0) == 10.0 / MIN_RESOLUTION_DIVISIONS


def test_a_domain_already_at_minimum_divisions_offers_no_coarser_resolution():
    authored = settings(voxel_size=1.0, max_particles=1, leaf_capacity=8)  # 8 divisions
    footprints = (Footprint("src", 50_000, DemandQuality.UPPER_BOUND, ((0, 0, 0),)),)

    assert not list(_sweep_suggestions(authored, footprints, "coarser_resolution"))
