"""Pure capacity planning for particle sources and sparse leaf topology.

The planner is deliberately Blender- and Warp-independent.  It reports what is
known, bounded, or unknown; it never mutates authored settings and never claims
that a conservative memory estimate guarantees allocation success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from . import constants as C
from .memory_safety import estimate_solver_bytes, safe_memory_budget
from .surface_tension_math import (
    required_sdf_splat_radius_voxels, required_topology_expand_leaves)
from .quality_contract import resolve_quality


class DemandQuality(Enum):
    EXACT = "exact"
    UPPER_BOUND = "upper_bound"
    UNKNOWN = "unknown"


class CapacityVerdict(Enum):
    ADMITTED = "admitted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class DemandValue:
    value: Optional[int]
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
    required: Optional[int]
    available: Optional[int]
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
        "surface_smooth_iterations": 0,
        "surface_tension_enabled": False,
        "surface_tension_coefficient": 0.072,
        "surface_tension_band_voxels": 1.5,
        "surface_tension_smoothing_iterations": 1,
        "surface_enabled": True,
        "surface_resolution_mode": 0,
        "surface_resolution_multiplier": 1.0,
        "surface_leaf_capacity_scale": 1.0,
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


def _seed_positions(payload: Any) -> Optional[np.ndarray]:
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
    sources = tuple(_source_demand(item) for item in source_footprints)
    seeds = tuple(seed_payloads)
    try:
        seed_count = sum(_seed_count(payload) for payload in seeds)
    except ValueError as exc:
        sources += (SourceDemand(
            "<automatic-seeds>", DemandValue(None, DemandQuality.UNKNOWN, str(exc))),)
        seed_count = 0
    particle_demand = _aggregate_particles(sources, seed_count)

    leaf_coords = {coord for item in sources for coord in item.leaf_coords}
    for footprint in collider_footprints:
        leaf_coords.update(tuple(int(axis) for axis in coord)
                           for coord in getattr(footprint, "leaf_coords", ()))
    leaf_coords.update(_seed_leaf_coords(settings, seeds))
    required_leaf_coords = _expanded_leaf_coords(settings, leaf_coords)
    leaf_count = len(required_leaf_coords)
    unknown_leaf_reasons = sorted({
        item.leaf_reason or f"{item.stable_id} leaf demand is unavailable"
        for item in sources if item.leaf_quality is DemandQuality.UNKNOWN
    })
    if unknown_leaf_reasons:
        leaf_demand = DemandValue(
            None, DemandQuality.UNKNOWN, "; ".join(unknown_leaf_reasons))
    else:
        leaf_quality = (DemandQuality.UPPER_BOUND
                        if any(item.leaf_quality is DemandQuality.UPPER_BOUND
                               for item in sources)
                        else DemandQuality.EXACT)
        leaf_demand = DemandValue(leaf_count, leaf_quality)

    staging_bytes = sum(item.staging_bytes for item in sources)
    # WP3 owns one int32 scratch occupancy entry per allocated sparse voxel.
    additional_transaction_bytes = max(
        0, int(additional_transaction_bytes))
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
    blockers: list[CapacityBlocker] = []
    source_ids = tuple(item.stable_id for item in sources)
    if particle_demand.quality is DemandQuality.UNKNOWN:
        blockers.append(CapacityBlocker(
            "PARTICLE_DEMAND_UNKNOWN", None,
            int(_value(settings, "max_particles", 0)), source_ids,
            particle_demand.reason or "Particle demand cannot be proven safe.",
        ))
    elif int(particle_demand.value or 0) > int(_value(settings, "max_particles", 0)):
        blockers.append(CapacityBlocker(
            "PARTICLE_CAPACITY", particle_demand.value,
            int(_value(settings, "max_particles", 0)), source_ids,
            "Initial seeds and liquid sources exceed particle capacity.",
        ))
    if leaf_demand.quality is DemandQuality.UNKNOWN:
        blockers.append(CapacityBlocker(
            "LEAF_DEMAND_UNKNOWN", None,
            int(_value(settings, "leaf_capacity", 0)), source_ids,
            leaf_demand.reason or "Sparse leaf demand cannot be proven safe.",
        ))
    elif leaf_count > int(_value(settings, "leaf_capacity", 0)):
        blockers.append(CapacityBlocker(
            "LEAF_CAPACITY", leaf_count,
            int(_value(settings, "leaf_capacity", 0)), source_ids,
            "Initial source topology exceeds sparse leaf capacity.",
        ))
    if budget_bytes <= 0 or estimated_bytes > budget_bytes:
        blockers.append(CapacityBlocker(
            "MEMORY_ESTIMATE", estimated_bytes, budget_bytes, source_ids,
            "Conservative estimated allocation exceeds the live VRAM budget.",
        ))

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
    budget = int(plan.budget_bytes)
    suggestions: list[CapacitySuggestion] = []

    if plan.particle_demand.value is not None and plan.leaf_demand.value is not None:
        required = {
            "max_particles": max(int(base.get("max_particles") or 0),
                                 int(plan.particle_demand.value)),
            "leaf_capacity": max(int(base.get("leaf_capacity") or 0),
                                 int(plan.leaf_demand.value)),
        }
        changes = {name: value for name, value in required.items()
                   if value != base.get(name)}
        candidate = dict(base)
        candidate.update(required)
        estimate = (estimate_solver_bytes(candidate) + plan.staging_bytes
                    + max(1, int(candidate["leaf_capacity"])) * 512 * 4
                    + plan.additional_transaction_bytes)
        if changes and estimate <= budget:
            suggestions.append(CapacitySuggestion(
                "Preserve authored domain and resolution", changes, int(estimate),
                int(budget), ("Raises only the fixed GPU capacities required by this scene.",),
                key="preserve_simulation", priority=0,
                adjustment_score=_adjustment_score(base, changes),
            ))

    tightened = None
    if plan.required_leaf_coords:
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
        if domain_min != authored_min or domain_max != authored_max:
            tightened = dict(base)
            tightened.update({
                "domain_min": domain_min,
                "domain_max": domain_max,
                "max_particles": max(
                    int(base["max_particles"]),
                    int(plan.particle_demand.value or 0)),
                "leaf_capacity": max(
                    int(base["leaf_capacity"]),
                    int(plan.leaf_demand.value or 0)),
            })
            estimate = (estimate_solver_bytes(tightened) + plan.staging_bytes
                        + int(tightened["leaf_capacity"]) * 512 * 4
                        + plan.additional_transaction_bytes)
            if estimate <= budget:
                changes = {
                    "domain_min": domain_min, "domain_max": domain_max,
                    "max_particles": tightened["max_particles"],
                    "leaf_capacity": tightened["leaf_capacity"],
                }
                changes = {name: value for name, value in changes.items()
                           if value != base.get(name)}
                suggestions.append(CapacitySuggestion(
                    "Tighten empty X/Y/Z domain margins", changes,
                    int(estimate), int(budget),
                    ("Changes domain bounds; review collision and boundary margins.",
                     "Only the displayed empty margins will be changed after approval."),
                    key="tighten_empty_margins", priority=10,
                    adjustment_score=_adjustment_score(base, changes),
                ))

    # Bounded monotonic search.  This preserves authored bounds and density but
    # trades spatial resolution for fewer leaves and a smaller surface grid.
    candidate = dict(base)
    original_h = max(1.0e-4, float(candidate.get("voxel_size") or 0.1))
    for step in range(1, 33):
        candidate["voxel_size"] = original_h * (1.05 ** step)
        if plan.particle_demand.value is not None:
            candidate["max_particles"] = max(
                int(candidate.get("max_particles") or 0),
                int(plan.particle_demand.value),
            )
        # A larger voxel size cannot increase the number of domain leaf cells.
        domain_leaf_count = math.prod(_leaf_dims(candidate))
        required_leaves = min(int(plan.leaf_demand.value or 0), domain_leaf_count)
        candidate["leaf_capacity"] = max(
            int(candidate.get("leaf_capacity") or 0), required_leaves)
        estimate = (estimate_solver_bytes(candidate) + plan.staging_bytes
                    + max(1, int(candidate["leaf_capacity"])) * 512 * 4
                    + plan.additional_transaction_bytes)
        if estimate <= budget:
            changes = {
                name: candidate[name]
                for name in ("voxel_size", "max_particles", "leaf_capacity")
                if candidate[name] != base.get(name)
            }
            if changes:
                suggestions.append(CapacitySuggestion(
                    "Preserve bounds with coarser resolution", changes,
                    int(estimate), int(budget),
                    ("Increases voxel size and reduces spatial resolution.",),
                    key="coarser_resolution", priority=10,
                    adjustment_score=_adjustment_score(base, changes),
                ))
            break

    # Balanced candidates combine only bounded, explicit changes and retain
    # the old demand as an upper bound.  They are intentionally suggestions,
    # not a hidden auto-tuning path.
    if tightened is not None:
        balanced = dict(tightened)
        balanced["voxel_size"] = original_h * 1.10
        balanced["leaf_capacity"] = max(
            int(plan.leaf_demand.value or 0),
            min(int(base["leaf_capacity"]), math.prod(_leaf_dims(balanced))))
        estimate = (estimate_solver_bytes(balanced) + plan.staging_bytes
                    + int(balanced["leaf_capacity"]) * 512 * 4
                    + plan.additional_transaction_bytes)
        if estimate <= budget:
            changes = {
                name: balanced[name]
                for name in ("domain_min", "domain_max", "voxel_size",
                             "max_particles", "leaf_capacity")
                if balanced[name] != base.get(name)
            }
            suggestions.append(CapacitySuggestion(
                "Balanced bounds, resolution, and capacity", changes,
                int(estimate), int(budget),
                ("Combines a 10% voxel-size increase with empty-margin tightening.",
                 "Review all authored tradeoffs before applying."),
                key="balanced", priority=10,
                adjustment_score=_adjustment_score(base, changes),
            ))
    return tuple(sorted(
        suggestions,
        key=lambda item: (item.priority, item.adjustment_score, item.label),
    ))
