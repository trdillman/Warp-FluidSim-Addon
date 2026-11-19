"""Sparse-leaf capacity growth planning and result contracts.

The live leaf storage is sized once, when a solver is allocated, and every
per-voxel array is sized from it. Growing it therefore means building a
replacement solver and carrying the simulation state across (see
``SimulationAdmission.grow_leaf_capacity``); this module only decides the
target and describes the outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_LEAF_GROWTH_FACTOR = 1.5
LEAF_GROWTH_GRANULARITY = 256
# The authored Leaf Capacity property tops out here; automatic growth does too.
MAX_AUTO_LEAVES = 262_144
# Each growth builds a replacement solver, so a single step may only do it a
# few times before it is treated as a runaway and the frame is refused.
MAX_GROWTHS_PER_STEP = 3


@dataclass(frozen=True)
class LeafGrowthResult:
    status: str            # GROWN, SUFFICIENT, DISABLED, BLOCKED, FAILED
    previous: int
    current: int
    required: int
    reason: str = ""

    @property
    def grew(self) -> bool:
        return self.status == "GROWN"


def checkpoint_leaf_allocation(payload) -> int | None:
    """Leaves a restored solver must be able to hold, or None when unknown.

    Automatic growth raises the live allocation without rewriting the authored
    Leaf Capacity. The checkpoint records that larger allocation (and the live
    leaf count). A solver built from the authored value alone cannot import it.
    """
    scalars = getattr(payload, "scalars", None)
    getter = getattr(scalars, "get", None)
    if not callable(getter):
        return None
    values = []
    for name in ("leaf_capacity", "leaf_count"):
        raw = getter(name)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    return max(values) if values else None


def leaf_growth_target(
    current: int,
    required: int,
    *,
    total_cells: int,
    factor: float = DEFAULT_LEAF_GROWTH_FACTOR,
    granularity: int = LEAF_GROWTH_GRANULARITY,
    hard_limit: int = MAX_AUTO_LEAVES,
) -> int:
    """Return a bounded geometric target that contains ``required`` if possible.

    ``total_cells`` is the number of leaf cells the whole domain holds: no
    capacity beyond it can ever be used. The result is never below ``current``
    and is smaller than ``required`` only when the domain or ``hard_limit``
    cannot hold it, which the caller reports instead of growing.
    """
    current = max(1, int(current))
    required = max(0, int(required))
    granularity = max(1, int(granularity))
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("leaf growth factor must be a finite number >= 1")
    ceiling = max(1, min(int(total_cells), int(hard_limit)))
    wanted = max(required, int(math.ceil(current * factor)))
    wanted = -(-wanted // granularity) * granularity
    return max(current, min(wanted, ceiling))
