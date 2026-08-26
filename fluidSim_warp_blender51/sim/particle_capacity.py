"""Transactional particle-capacity growth planning and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .memory_safety import (
    PARTICLE_BYTES,
    WHITEWATER_EMISSION_BYTES_PER_LIQUID,
    safe_memory_budget,
)


DEFAULT_GROWTH_FACTOR = 1.5
DEFAULT_GROWTH_GRANULARITY = 65_536
MAX_AUTO_PARTICLES = 50_000_000


@dataclass(frozen=True)
class ParticleGrowthCandidateBytes:
    """Device and host contributions of one grow-copy-swap candidate."""

    particle_pool_and_bins: int
    whitewater_workspace: int
    additional_device_persistent_bytes: int = 0
    host_cache_staging: int = 0

    @property
    def device_persistent_bytes(self) -> int:
        return int(
            self.particle_pool_and_bins
            + self.whitewater_workspace
            + self.additional_device_persistent_bytes
        )

    @property
    def device_transaction_peak_bytes(self) -> int:
        # The complete candidate coexists with the authoritative old pools.
        # The guard compares this incremental device demand with live free VRAM.
        return self.device_persistent_bytes


def particle_growth_candidate_bytes(
    target: int,
    *,
    additional_device_persistent_bytes: int = 0,
    host_cache_staging_bytes: int = 0,
) -> ParticleGrowthCandidateBytes:
    """Inventory the allocations performed before a particle-growth swap."""
    target = max(0, int(target))
    return ParticleGrowthCandidateBytes(
        particle_pool_and_bins=target * PARTICLE_BYTES,
        whitewater_workspace=(
            target * WHITEWATER_EMISSION_BYTES_PER_LIQUID),
        additional_device_persistent_bytes=max(
            0, int(additional_device_persistent_bytes)),
        host_cache_staging=max(0, int(host_cache_staging_bytes)),
    )


@dataclass(frozen=True)
class ParticleGrowthPlan:
    current: int
    required: int
    target: int
    peak_allocation_bytes: int
    budget_bytes: int
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class ParticleGrowthResult:
    status: str
    previous: int
    current: int
    required: int
    target: int
    peak_allocation_bytes: int = 0
    budget_bytes: int = 0
    reason: str = ""

    @property
    def grew(self) -> bool:
        return self.status == "GROWN"


def growth_target(
    current: int,
    required: int,
    *,
    factor: float = DEFAULT_GROWTH_FACTOR,
    granularity: int = DEFAULT_GROWTH_GRANULARITY,
    hard_limit: int = MAX_AUTO_PARTICLES,
) -> int:
    """Return a bounded geometric target that contains ``required``."""
    current = max(1, int(current))
    required = max(0, int(required))
    granularity = max(1, int(granularity))
    factor = float(factor)
    if not math.isfinite(factor):
        raise ValueError("particle growth factor must be finite")
    raw = max(required, int(math.ceil(current * max(1.0, factor))))
    rounded = int(math.ceil(raw / granularity) * granularity)
    return min(max(rounded, current), max(current, int(hard_limit)))


def plan_particle_growth(
    current: int,
    required: int,
    *,
    gpu_total_bytes: int,
    gpu_free_bytes: int,
    hard_limit: int = MAX_AUTO_PARTICLES,
    additional_device_persistent_bytes: int = 0,
    host_cache_staging_bytes: int = 0,
) -> ParticleGrowthPlan:
    """Guard an atomic grow-copy-swap against the live free-memory budget.

    The old and new particle pools coexist until copying succeeds, so the peak
    guard charges the complete target pool rather than only the capacity delta.
    ``PARTICLE_BYTES`` includes conservative particle and bin storage; the
    whitewater candidate flag/offset workspace grows in the same transaction.
    Feature owners provide any additional persistent device bytes from their
    own allocation inventory.  Host staging is carried separately and excluded
    from the device transaction peak.
    """
    current = max(1, int(current))
    required = max(0, int(required))
    target = growth_target(current, required, hard_limit=hard_limit)
    budget = int(safe_memory_budget(gpu_total_bytes, gpu_free_bytes))
    allocation = particle_growth_candidate_bytes(
        target,
        additional_device_persistent_bytes=(
            additional_device_persistent_bytes),
        host_cache_staging_bytes=host_cache_staging_bytes,
    )
    peak = allocation.device_transaction_peak_bytes
    if required <= current:
        return ParticleGrowthPlan(
            current, required, current, 0, budget, True, "already sufficient")
    if required > int(hard_limit):
        return ParticleGrowthPlan(
            current, required, target, peak, budget, False,
            f"required particle count {required:,} exceeds the auto-growth hard limit {int(hard_limit):,}",
        )
    if budget <= 0:
        return ParticleGrowthPlan(
            current, required, target, peak, budget, False,
            "live CUDA memory budget is unavailable",
        )
    if peak > budget:
        return ParticleGrowthPlan(
            current, required, target, peak, budget, False,
            f"atomic growth needs an estimated {peak / 1024**3:.2f} GiB peak allocation but the live growth budget is {budget / 1024**3:.2f} GiB",
        )
    return ParticleGrowthPlan(current, required, target, peak, budget, True)
