"""Pure capacity planning for particle sources and sparse leaf topology.

The planner is deliberately Blender- and Warp-independent.  It reports what is
known, bounded, or unknown; it never mutates authored settings and never claims
that a conservative memory estimate guarantees allocation success.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from . import constants as C
from .memory_safety import estimate_solver_bytes, safe_memory_budget
from .quality_contract import resolve_quality
from .surface_tension_math import required_sdf_splat_radius_voxels, required_topology_expand_leaves


class DemandQuality(Enum):
    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    UNKNOWN = "unknown"


class CapacityVerdict(Enum):
    ADMITTED = "admitted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class DemandValue:
    value: int | None
    quality: DemandQuality
    reason: str = ""


@dataclass(frozen=True)
class SourceDemand:
    stable_id: str
    particles: DemandValue
    leaf_coords: tuple[tuple[int, int, int], ...] = ()
    staging_bytes: int = 0
    leaf_quality: DemandQuality = DemandQuality.EXACT
    leaf_reason: str = ""


@dataclass(frozen=True)
class CapacityBlocker:
    code: str
    required: int | None
    available: int | None
    source_ids: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class CapacitySuggestion:
    label: str
    changes: Mapping[str, Any]
    estimated_bytes: int
    budget_bytes: int
    tradeoffs: tuple[str, ...] = ()
    disclaimer: str = "Conservative estimate; this is not an OOM guarantee."
    key: str = ""
    priority: int = 100
    adjustment_score: float = 0.0


@dataclass(frozen=True)
class CapacityPlan:
    verdict: CapacityVerdict
    particle_demand: DemandValue
    leaf_demand: DemandValue
    estimated_bytes: int
    budget_bytes: int
    blockers: tuple[CapacityBlocker, ...]
    source_demands: tuple[SourceDemand, ...]
    scene_signature: str
    staging_bytes: int = 0
    transaction_scratch_bytes: int = 0
    additional_transaction_bytes: int = 0
    required_leaf_coords: tuple[tuple[int, int, int], ...] = ()
    suggestions: tuple[CapacitySuggestion, ...] = field(default_factory=tuple)


def _value(settings: Any, name: str, default: Any) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _settings_dict(settings: Any) -> dict[str, Any]:
    defaults = {
        "domain_min": (-2.0, -2.0, 0.0),
        "domain_max": (2.0, 2.0, 4.0),
        "voxel_size": 0.1,
        "max_particles": 0,
        "leaf_capacity": 8192,
        "particles_per_voxel": 8.0,
        "topology_expand": 1,
        "max_whitewater": 0,
        "particle_radius_scale": 1.2,
        "surface_splat_radius": 2,
        "surface_filter_iterations": 0,
        "surface_dilate_iterations": 0,
        "surface_erode_iterations": 0,
        "surface_smooth_iterations": 1,
        "surface_tension_enabled": False,
        "surface_tension_coefficient": 0.072,
        "surface_tension_band_voxels": 1.5,
        "surface_tension_smoothing_iterations": 1,
        "surface_enabled": True,
        "surface_resolution_mode": 0,
        "surface_resolution_multiplier": 2.0,
        "surface_leaf_capacity_scale": 8.0,
        "surface_max_dense_dim": 512,
        "surface_voxel_size": 0.0,
    }
    return {name: _value(settings, name, default)
            for name, default in defaults.items()}


def _quality(value: Any) -> DemandQuality:
    if isinstance(value, DemandQuality):
        return value
    try:
        return DemandQuality(str(value).lower())
    except ValueError:
        return DemandQuality.UNKNOWN


def _source_demand(footprint: Any) -> SourceDemand:
    stable_id = str(getattr(footprint, "stable_id", "<unknown>"))
    raw = getattr(footprint, "particle_demand", None)
    quality = _quality(getattr(footprint, "particle_quality", DemandQuality.UNKNOWN))
    reason = str(getattr(footprint, "reason", ""))
    if raw is None:
        quality = DemandQuality.UNKNOWN
        particles = DemandValue(None, quality, reason or "particle demand unavailable")
    else:
        count = int(raw)
        if count < 0:
            particles = DemandValue(None, DemandQuality.UNKNOWN,
                                    "negative particle demand")
        else:
            particles = DemandValue(count, quality, reason)
    coords = tuple(sorted({tuple(int(axis) for axis in coord)
                           for coord in getattr(footprint, "leaf_coords", ())}))
    return SourceDemand(
        stable_id=stable_id,
        particles=particles,
        leaf_coords=coords,
        staging_bytes=max(0, int(getattr(footprint, "staging_bytes", 0))),
        leaf_quality=_quality(getattr(
            footprint, "leaf_quality", DemandQuality.EXACT)),
        leaf_reason=str(getattr(footprint, "leaf_reason", "")),
    )


def _seed_count(payload: Any) -> int:
    if isinstance(payload, (int, np.integer)):
        return max(0, int(payload))
    positions = getattr(payload, "positions", payload)
    try:
        return max(0, int(len(positions)))
    except (TypeError, AttributeError):
        raise ValueError("seed payload has no countable positions") from None


def _seed_positions(payload: Any) -> np.ndarray | None:
    positions = getattr(payload, "positions", payload)
    if isinstance(positions, (int, np.integer)):
        return None
    array = np.asarray(positions)
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    return None


def _leaf_dims(settings: Any) -> tuple[int, int, int]:
    lo = _value(settings, "domain_min", (0.0, 0.0, 0.0))
    hi = _value(settings, "domain_max", (0.0, 0.0, 0.0))
    return resolve_quality(settings, (lo, hi)).leaf_dimensions


def leaf_coords_for_world_bounds(settings: Any, minimum: Iterable[float],
                                 maximum: Iterable[float]) -> tuple[tuple[int, int, int], ...]:
    """Return every flat 8³ leaf intersecting clipped world-space bounds."""
    domain_min = np.asarray(
        _value(settings, "domain_min", (0.0, 0.0, 0.0)), np.float64)
    domain_max = np.asarray(
        _value(settings, "domain_max", (0.0, 0.0, 0.0)), np.float64)
    lo_world = np.maximum(np.asarray(tuple(minimum), np.float64), domain_min)
    hi_world = np.minimum(np.asarray(tuple(maximum), np.float64), domain_max)
    if np.any(hi_world < lo_world):
        return ()
    h = max(1.0e-4, float(_value(settings, "voxel_size", 0.1)))
    lo = np.floor((lo_world - domain_min) / h / 8.0).astype(np.int64)
    hi_clamped = np.minimum(hi_world, np.nextafter(domain_max, domain_min))
    hi = np.floor((hi_clamped - domain_min) / h / 8.0).astype(np.int64)
    dims = _leaf_dims(settings)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(dims, np.int64) - 1)
    if np.any(hi < lo):
        return ()
    return tuple(
        (x, y, z)
        for z in range(int(lo[2]), int(hi[2]) + 1)
        for y in range(int(lo[1]), int(hi[1]) + 1)
        for x in range(int(lo[0]), int(hi[0]) + 1)
    )


def _seed_leaf_coords(settings: Any, payloads: Iterable[Any]) -> set[tuple[int, int, int]]:
    lo = np.asarray(_value(settings, "domain_min", (0.0, 0.0, 0.0)), np.float64)
    h = max(1.0e-4, float(_value(settings, "voxel_size", 0.1)))
    dims = _leaf_dims(settings)
    result: set[tuple[int, int, int]] = set()
    for payload in payloads:
        points = _seed_positions(payload)
        if points is None:
            continue
        leaves = np.floor((points.astype(np.float64) - lo) / h / 8.0).astype(np.int64)
        for coord in leaves:
            item = tuple(int(axis) for axis in coord)
            if all(0 <= item[i] < dims[i] for i in range(3)):
                result.add(item)
    return result


def _effective_topology_expand(settings: Any) -> int:
    authored = max(0, int(_value(settings, "topology_expand", 0)))
    if (not bool(_value(settings, "surface_tension_enabled", False))
            or float(_value(settings, "surface_tension_coefficient", 0.0)) <= 0.0):
        return authored
    sdf_filter_iterations = sum(max(0, int(_value(settings, name, 0))) for name in (
        "surface_filter_iterations",
        "surface_dilate_iterations",
        "surface_erode_iterations",
        "surface_smooth_iterations",
    ))
    splat_radius = max(
        int(_value(settings, "surface_splat_radius", 2)),
        required_sdf_splat_radius_voxels(
            0.5 * float(_value(settings, "particle_radius_scale", 1.2)),
            float(_value(settings, "surface_tension_band_voxels", 1.5)),
            int(_value(settings, "surface_tension_smoothing_iterations", 1)),
            sdf_filter_iterations,
        ),
    )
    capillary = required_topology_expand_leaves(splat_radius, C.LEAF_DIM)
    return max(authored, capillary)


def _expanded_leaf_coords(settings: Any, coords: Iterable[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
    dims = _leaf_dims(settings)
    radius = _effective_topology_expand(settings)
    expanded: set[tuple[int, int, int]] = set()
    for coord in coords:
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    item = (coord[0] + dx, coord[1] + dy, coord[2] + dz)
                    if all(0 <= item[i] < dims[i] for i in range(3)):
                        expanded.add(item)
    return tuple(sorted(expanded))


def _expanded_leaf_count(settings: Any, coords: Iterable[tuple[int, int, int]]) -> int:
    return len(_expanded_leaf_coords(settings, coords))


def _aggregate_particles(source_demands: tuple[SourceDemand, ...],
                         seed_count: int) -> DemandValue:
    if any(item.particles.quality is DemandQuality.UNKNOWN
           for item in source_demands):
        reasons = sorted({item.particles.reason for item in source_demands
                          if item.particles.quality is DemandQuality.UNKNOWN
                          and item.particles.reason})
        return DemandValue(None, DemandQuality.UNKNOWN, "; ".join(reasons))
    quality = (DemandQuality.UPPER_BOUND
               if any(item.particles.quality is DemandQuality.UPPER_BOUND
                      for item in source_demands)
               else DemandQuality.EXACT)
    return DemandValue(seed_count + sum(int(item.particles.value or 0)
                                        for item in source_demands), quality)


def _signature(settings: Any, demands: tuple[SourceDemand, ...],
               seed_count: int, leaf_demand: DemandValue,
               additional_transaction_bytes: int = 0) -> str:
    payload = {
        "settings": _settings_dict(settings),
        "additional_transaction_bytes": int(additional_transaction_bytes),
        "sources": [
            {
                "stable_id": item.stable_id,
                "particles": item.particles.value,
                "quality": item.particles.quality.value,
                "reason": item.particles.reason,
                "leaf_coords": item.leaf_coords,
                "leaf_quality": item.leaf_quality.value,
                "leaf_reason": item.leaf_reason,
                "staging_bytes": item.staging_bytes,
            }
            for item in demands
        ],
        "seed_count": seed_count,
        "leaf_demand": {
            "value": leaf_demand.value,
            "quality": leaf_demand.quality.value,
            "reason": leaf_demand.reason,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()


def _collect_particle_demands(
        source_footprints: Iterable[Any],
        seed_payloads: Iterable[Any],
) -> tuple[tuple[SourceDemand, ...], tuple[Any, ...], int, DemandValue]:
    """Normalize source and seed particle demand without mutating inputs."""
    sources = tuple(_source_demand(item) for item in source_footprints)
    seeds = tuple(seed_payloads)
    try:
        seed_count = sum(_seed_count(payload) for payload in seeds)
    except ValueError as exc:
        sources += (SourceDemand(
            "<automatic-seeds>",
            DemandValue(None, DemandQuality.UNKNOWN, str(exc)),
        ),)
        seed_count = 0
    return sources, seeds, seed_count, _aggregate_particles(sources, seed_count)


def _collect_leaf_demand(
        settings: Any,
        sources: tuple[SourceDemand, ...],
        seeds: tuple[Any, ...],
        collider_footprints: Iterable[Any],
) -> tuple[tuple[tuple[int, int, int], ...], DemandValue]:
    """Aggregate source, collider, and seed topology into sparse leaf demand."""
    leaf_coords = {coord for item in sources for coord in item.leaf_coords}
    for footprint in collider_footprints:
        leaf_coords.update(tuple(int(axis) for axis in coord)
                           for coord in getattr(footprint, "leaf_coords", ()))
    leaf_coords.update(_seed_leaf_coords(settings, seeds))
    required_leaf_coords = _expanded_leaf_coords(settings, leaf_coords)
    unknown_reasons = sorted({
        item.leaf_reason or f"{item.stable_id} leaf demand is unavailable"
        for item in sources if item.leaf_quality is DemandQuality.UNKNOWN
    })
    if unknown_reasons:
        demand = DemandValue(
            None, DemandQuality.UNKNOWN, "; ".join(unknown_reasons))
    else:
        quality = (DemandQuality.UPPER_BOUND
                   if any(item.leaf_quality is DemandQuality.UPPER_BOUND
                          for item in sources)
                   else DemandQuality.EXACT)
        demand = DemandValue(len(required_leaf_coords), quality)
    return required_leaf_coords, demand


def _capacity_budget(
        settings: Any,
        sources: tuple[SourceDemand, ...],
        resources: Mapping[str, Any],
        additional_transaction_bytes: int,
) -> tuple[int, int, int, int]:
    """Return staging, scratch, estimated, and safe-budget byte counts."""
    staging_bytes = sum(item.staging_bytes for item in sources)
    transaction_scratch_bytes = (
        max(1, int(_value(settings, "leaf_capacity", 0))) * 512 * 4
        + additional_transaction_bytes)
    estimated_bytes = int(
        estimate_solver_bytes(settings) + staging_bytes
        + transaction_scratch_bytes)
    budget_bytes = int(safe_memory_budget(
        int(resources.get("gpu_total_bytes", 0)),
        int(resources.get("gpu_free_bytes", 0)),
        additional_transaction_bytes,
    ))
    return (
        staging_bytes,
        transaction_scratch_bytes,
        estimated_bytes,
        budget_bytes,
    )


def _capacity_blockers(
        settings: Any,
        sources: tuple[SourceDemand, ...],
        particle_demand: DemandValue,
        leaf_demand: DemandValue,
        estimated_bytes: int,
        budget_bytes: int,
) -> tuple[CapacityBlocker, ...]:
    """Construct the complete admission verdict evidence."""
    blockers: list[CapacityBlocker] = []
    source_ids = tuple(item.stable_id for item in sources)
    particle_capacity = int(_value(settings, "max_particles", 0))
    leaf_capacity = int(_value(settings, "leaf_capacity", 0))
    if particle_demand.quality is DemandQuality.UNKNOWN:
        blockers.append(CapacityBlocker(
            "PARTICLE_DEMAND_UNKNOWN", None, particle_capacity, source_ids,
            particle_demand.reason or "Particle demand cannot be proven safe.",
        ))
    elif int(particle_demand.value or 0) > particle_capacity:
        blockers.append(CapacityBlocker(
            "PARTICLE_CAPACITY", particle_demand.value,
            particle_capacity, source_ids,
            "Initial seeds and liquid sources exceed particle capacity.",
        ))
    if leaf_demand.quality is DemandQuality.UNKNOWN:
        blockers.append(CapacityBlocker(
            "LEAF_DEMAND_UNKNOWN", None, leaf_capacity, source_ids,
            leaf_demand.reason or "Sparse leaf demand cannot be proven safe.",
        ))
    elif int(leaf_demand.value or 0) > leaf_capacity:
        blockers.append(CapacityBlocker(
            "LEAF_CAPACITY", leaf_demand.value, leaf_capacity, source_ids,
            "Initial source topology exceeds sparse leaf capacity.",
        ))
    if budget_bytes <= 0 or estimated_bytes > budget_bytes:
        blockers.append(CapacityBlocker(
            "MEMORY_ESTIMATE", estimated_bytes, budget_bytes, source_ids,
            "Conservative estimated allocation exceeds the live VRAM budget.",
        ))
    return tuple(blockers)


def plan_initialization(settings: Any, source_footprints: Iterable[Any],
                        seed_payloads: Iterable[Any],
                        collider_footprints: Iterable[Any],
                        resources: Mapping[str, Any], *,
                        additional_transaction_bytes: int = 0) -> CapacityPlan:
    """Plan one initial population without allocating or mutating solver state.

    ``additional_transaction_bytes`` accounts for memory that must remain
    resident while the candidate is allocated, such as an authoritative live
    solver during transactional replacement.
    """
    sources, seeds, seed_count, particle_demand = _collect_particle_demands(
        source_footprints, seed_payloads)
    required_leaf_coords, leaf_demand = _collect_leaf_demand(
        settings, sources, seeds, collider_footprints)
    additional_transaction_bytes = max(0, int(additional_transaction_bytes))
    (
        staging_bytes,
        transaction_scratch_bytes,
        estimated_bytes,
        budget_bytes,
    ) = _capacity_budget(
        settings, sources, resources, additional_transaction_bytes)
    blockers = _capacity_blockers(
        settings, sources, particle_demand, leaf_demand,
        estimated_bytes, budget_bytes)
    signature = _signature(
        settings, sources, seed_count, leaf_demand,
        additional_transaction_bytes)
    return CapacityPlan(
        verdict=(CapacityVerdict.BLOCKED if blockers
                 else CapacityVerdict.ADMITTED),
        particle_demand=particle_demand,
        leaf_demand=leaf_demand,
        estimated_bytes=estimated_bytes,
        budget_bytes=budget_bytes,
        blockers=tuple(blockers),
        source_demands=sources,
        scene_signature=signature,
        staging_bytes=staging_bytes,
        transaction_scratch_bytes=transaction_scratch_bytes,
        additional_transaction_bytes=additional_transaction_bytes,
        required_leaf_coords=required_leaf_coords,
    )


def _adjustment_score(base: Mapping[str, Any], changes: Mapping[str, Any]) -> float:
    """Compare only authored-semantic changes; capacities do not alter physics."""
    score = 0.0
    if "voxel_size" in changes:
        old = max(float(base.get("voxel_size") or 0.0), 1.0e-9)
        score += abs(float(changes["voxel_size"]) / old - 1.0)
    if "domain_min" in changes or "domain_max" in changes:
        old_min = tuple(float(value) for value in base["domain_min"])
        old_max = tuple(float(value) for value in base["domain_max"])
        new_min = tuple(float(value) for value in changes.get("domain_min", old_min))
        new_max = tuple(float(value) for value in changes.get("domain_max", old_max))
        old_volume = math.prod(max(old_max[i] - old_min[i], 1.0e-9)
                               for i in range(3))
        new_volume = math.prod(max(new_max[i] - new_min[i], 0.0)
                               for i in range(3))
        score += max(0.0, 1.0 - new_volume / old_volume)
    semantic_count = sum(name in changes
                         for name in ("voxel_size", "domain_min", "domain_max"))
    return score + max(0, semantic_count - 1) * 0.02


def _estimate_candidate_bytes(
        candidate: Mapping[str, Any], plan: CapacityPlan) -> int:
    return int(
        estimate_solver_bytes(candidate) + plan.staging_bytes
        + max(1, int(candidate["leaf_capacity"])) * 512 * 4
        + plan.additional_transaction_bytes)


def _capacity_only_suggestion(
        plan: CapacityPlan,
        base: Mapping[str, Any],
) -> CapacitySuggestion | None:
    if plan.particle_demand.value is None or plan.leaf_demand.value is None:
        return None
    required = {
        "max_particles": max(
            int(base.get("max_particles") or 0),
            int(plan.particle_demand.value)),
        "leaf_capacity": max(
            int(base.get("leaf_capacity") or 0),
            int(plan.leaf_demand.value)),
    }
    changes = {name: value for name, value in required.items()
               if value != base.get(name)}
    candidate = dict(base)
    candidate.update(required)
    estimate = _estimate_candidate_bytes(candidate, plan)
    if not changes or estimate > plan.budget_bytes:
        return None
    return CapacitySuggestion(
        "Preserve authored domain and resolution", changes, estimate,
        int(plan.budget_bytes),
        ("Raises only the fixed GPU capacities required by this scene.",),
        key="preserve_simulation", priority=0,
        adjustment_score=_adjustment_score(base, changes),
    )


def _tightened_domain_candidate(
        plan: CapacityPlan,
        base: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not plan.required_leaf_coords:
        return None
    h = max(1.0e-4, float(base["voxel_size"]))
    authored_min = tuple(float(value) for value in base["domain_min"])
    authored_max = tuple(float(value) for value in base["domain_max"])
    leaf_min = tuple(min(coord[axis] for coord in plan.required_leaf_coords)
                     for axis in range(3))
    leaf_max = tuple(max(coord[axis] for coord in plan.required_leaf_coords)
                     for axis in range(3))
    domain_min = tuple(max(
        authored_min[axis], authored_min[axis] + leaf_min[axis] * 8.0 * h)
        for axis in range(3))
    domain_max = tuple(min(
        authored_max[axis],
        authored_min[axis] + (leaf_max[axis] + 1) * 8.0 * h)
        for axis in range(3))
    if domain_min == authored_min and domain_max == authored_max:
        return None
    candidate = dict(base)
    candidate.update({
        "domain_min": domain_min,
        "domain_max": domain_max,
        "max_particles": max(
            int(base["max_particles"]),
            int(plan.particle_demand.value or 0)),
        "leaf_capacity": max(
            int(base["leaf_capacity"]),
            int(plan.leaf_demand.value or 0)),
    })
    return candidate


def _tightened_domain_suggestion(
        plan: CapacityPlan,
        base: Mapping[str, Any],
        candidate: Mapping[str, Any] | None,
) -> CapacitySuggestion | None:
    if candidate is None:
        return None
    estimate = _estimate_candidate_bytes(candidate, plan)
    if estimate > plan.budget_bytes:
        return None
    changes = {
        name: candidate[name]
        for name in ("domain_min", "domain_max", "max_particles", "leaf_capacity")
        if candidate[name] != base.get(name)
    }
    return CapacitySuggestion(
        "Tighten empty X/Y/Z domain margins", changes,
        estimate, int(plan.budget_bytes),
        ("Changes domain bounds; review collision and boundary margins.",
         "Only the displayed empty margins will be changed after approval."),
        key="tighten_empty_margins", priority=10,
        adjustment_score=_adjustment_score(base, changes),
    )


def _coarser_resolution_suggestion(
        plan: CapacityPlan,
        base: Mapping[str, Any],
) -> CapacitySuggestion | None:
    """Find the first bounded coarser-resolution candidate within budget."""
    candidate = dict(base)
    original_h = max(1.0e-4, float(candidate.get("voxel_size") or 0.1))
    for step in range(1, 33):
        candidate["voxel_size"] = original_h * (1.05 ** step)
        candidate["max_particles"] = max(
            int(candidate.get("max_particles") or 0),
            int(plan.particle_demand.value or 0),
        )
        domain_leaf_count = math.prod(_leaf_dims(candidate))
        required_leaves = min(
            int(plan.leaf_demand.value or 0), domain_leaf_count)
        candidate["leaf_capacity"] = max(
            int(candidate.get("leaf_capacity") or 0), required_leaves)
        estimate = _estimate_candidate_bytes(candidate, plan)
        if estimate > plan.budget_bytes:
            continue
        changes = {
            name: candidate[name]
            for name in ("voxel_size", "max_particles", "leaf_capacity")
            if candidate[name] != base.get(name)
        }
        if not changes:
            return None
        return CapacitySuggestion(
            "Preserve bounds with coarser resolution", changes,
            estimate, int(plan.budget_bytes),
            ("Increases voxel size and reduces spatial resolution.",),
            key="coarser_resolution", priority=10,
            adjustment_score=_adjustment_score(base, changes),
        )
    return None


def _balanced_suggestion(
        plan: CapacityPlan,
        base: Mapping[str, Any],
        tightened: Mapping[str, Any] | None,
) -> CapacitySuggestion | None:
    if tightened is None:
        return None
    balanced = dict(tightened)
    original_h = max(1.0e-4, float(base.get("voxel_size") or 0.1))
    balanced["voxel_size"] = original_h * 1.10
    balanced["leaf_capacity"] = max(
        int(plan.leaf_demand.value or 0),
        min(int(base["leaf_capacity"]), math.prod(_leaf_dims(balanced))))
    estimate = _estimate_candidate_bytes(balanced, plan)
    if estimate > plan.budget_bytes:
        return None
    changes = {
        name: balanced[name]
        for name in ("domain_min", "domain_max", "voxel_size",
                     "max_particles", "leaf_capacity")
        if balanced[name] != base.get(name)
    }
    return CapacitySuggestion(
        "Balanced bounds, resolution, and capacity", changes,
        estimate, int(plan.budget_bytes),
        ("Combines a 10% voxel-size increase with empty-margin tightening.",
         "Review all authored tradeoffs before applying."),
        key="balanced", priority=10,
        adjustment_score=_adjustment_score(base, changes),
    )


def _rank_suggestions(
        suggestions: Iterable[CapacitySuggestion | None],
) -> tuple[CapacitySuggestion, ...]:
    return tuple(sorted(
        (item for item in suggestions if item is not None),
        key=lambda item: (item.priority, item.adjustment_score, item.label),
    ))


def suggest_configurations(
        plan: CapacityPlan, settings: Any) -> tuple[CapacitySuggestion, ...]:
    """Return ranked, non-mutating alternatives for a blocked plan.

    Capacity-only changes rank first because they preserve authored simulation
    semantics.  If the conservative live VRAM budget cannot admit that option,
    semantic alternatives are ranked by their normalized adjustment amount.
    """
    if plan.verdict is CapacityVerdict.ADMITTED:
        return ()
    if (plan.particle_demand.value is None
            or plan.particle_demand.quality is DemandQuality.UNKNOWN):
        # No automatic parameter edit can make an unbounded source truthful.
        return ()
    base = _settings_dict(settings)
    tightened = _tightened_domain_candidate(plan, base)
    return _rank_suggestions((
        _capacity_only_suggestion(plan, base),
        _tightened_domain_suggestion(plan, base, tightened),
        _coarser_resolution_suggestion(plan, base),
        _balanced_suggestion(plan, base, tightened),
    ))
