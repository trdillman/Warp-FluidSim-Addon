"""Single public capacity assessment and recommendation authority.

Low-level estimators remain in their focused modules for compatibility and unit
testing.  Active Blender planning calls this service so assessment, ranked
options, quality resolution, and signature generation cannot diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .capacity_planner import (
    CapacityBlocker,
    CapacityPlan,
    CapacityVerdict,
    plan_initialization,
    suggest_configurations,
)
from .memory_safety import safe_memory_budget
from .quality_contract import ResolvedQuality, resolve_quality


@dataclass(frozen=True)
class PreparedCapacityInput:
    source_footprints: tuple[Any, ...]
    seed_payloads: tuple[Any, ...]
    collider_footprints: tuple[Any, ...]


@dataclass(frozen=True)
class CapacityAssessment:
    resolved_quality: ResolvedQuality
    prepared: PreparedCapacityInput
    resources: Mapping[str, Any]
    plan: CapacityPlan


def assess(settings: Any, prepared: PreparedCapacityInput,
           resources: Mapping[str, Any], *,
           additional_transaction_bytes: int = 0) -> CapacityAssessment:
    """Assess one already-prepared host snapshot without geometry/device work."""
    quality = resolve_quality(
        settings, (settings.domain_min, settings.domain_max))
    plan = plan_initialization(
        settings,
        prepared.source_footprints,
        prepared.seed_payloads,
        prepared.collider_footprints,
        resources,
        additional_transaction_bytes=additional_transaction_bytes,
    )
    plan = replace(plan, suggestions=recommend(plan, settings))
    return CapacityAssessment(
        resolved_quality=quality,
        prepared=prepared,
        resources=dict(resources),
        plan=plan,
    )


def refresh_resources(plan: CapacityPlan, settings: Any,
                      resources: Mapping[str, Any]) -> CapacityPlan:
    """Refresh only the live VRAM verdict for an unchanged prepared plan."""
    budget = int(safe_memory_budget(
        int(resources.get("gpu_total_bytes", 0)),
        int(resources.get("gpu_free_bytes", 0)),
        plan.additional_transaction_bytes,
    ))
    blockers = [item for item in plan.blockers
                if item.code != "MEMORY_ESTIMATE"]
    if budget <= 0 or plan.estimated_bytes > budget:
        blockers.append(CapacityBlocker(
            "MEMORY_ESTIMATE", plan.estimated_bytes, budget,
            tuple(item.stable_id for item in plan.source_demands),
            "Conservative estimated allocation exceeds the live VRAM budget.",
        ))
    refreshed = replace(
        plan,
        verdict=(CapacityVerdict.BLOCKED if blockers
                 else CapacityVerdict.ADMITTED),
        budget_bytes=budget,
        blockers=tuple(blockers),
        suggestions=(),
    )
    return replace(refreshed, suggestions=recommend(refreshed, settings))


def recommend(plan: CapacityPlan, settings: Any):
    """Return the one ranked option list used by panel and confirmation flow.

    Capacity-only changes are exact against the prepared demand and therefore
    safe to display directly. Semantic candidates (voxel size, bounds, density)
    require a new host-demand evaluation at candidate quality; until such an
    evaluation is supplied they are omitted rather than presented with stale
    fine-grid demand.
    """
    options = suggest_configurations(plan, settings)
    # With no prepared topology/source demand, semantic options are fully
    # re-evaluated by the candidate quality/memory formulas and are safe to
    # offer. Once geometry contributes demand, only exact capacity changes are
    # shown until a candidate-resolution host recompile supplies fresh demand.
    if not plan.source_demands and not plan.required_leaf_coords:
        return options
    capacity_fields = {"max_particles", "leaf_capacity"}
    return tuple(
        option for option in options
        if set(option.changes).issubset(capacity_fields)
    )
