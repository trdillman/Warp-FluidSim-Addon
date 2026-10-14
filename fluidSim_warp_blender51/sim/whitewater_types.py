"""Stable whitewater type values and deterministic host classification.

The solver keeps one whitewater particle pool.  ``type`` is a persistent int32
channel used by device behavior, cache/checkpoint validation, and Blender output
partitioning.  Values 0--2 are release-stable.  Value 3 is reserved for a future
aerosol/dust model and is deliberately rejected by current validators.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Mapping

import numpy as np


class WhitewaterType(IntEnum):
    FOAM = 0
    BUBBLE = 1
    SPRAY = 2


LEGACY_UNCLASSIFIED = -1
RESERVED_DUST_TYPE = 3
WHITEWATER_TYPE_SCHEMA_VERSION = 1
VALID_TYPE_VALUES = tuple(int(value) for value in WhitewaterType)


@dataclass(frozen=True)
class ClassificationPolicy:
    """World-space thresholds shared by host references and device kernels."""

    foam_band: float
    spray_enter_height: float
    spray_exit_height: float
    bubble_enter_depth: float
    bubble_exit_depth: float
    min_spray_speed: float
    spray_exit_speed: float
    min_churn: float

    def __post_init__(self) -> None:
        values = (
            self.foam_band,
            self.spray_enter_height,
            self.spray_exit_height,
            self.bubble_enter_depth,
            self.bubble_exit_depth,
            self.min_spray_speed,
            self.spray_exit_speed,
            self.min_churn,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("whitewater classification thresholds must be finite")
        if self.foam_band <= 0.0:
            raise ValueError("whitewater foam band must be positive")
        # Equal enter/exit thresholds are valid: that is zero hysteresis, which
        # the Type Hysteresis property allows (min=0.0).
        if not 0.0 <= self.spray_exit_height <= self.spray_enter_height:
            raise ValueError("spray exit height must not exceed spray enter height")
        if not 0.0 <= self.bubble_exit_depth <= self.bubble_enter_depth:
            raise ValueError("bubble exit depth must not exceed bubble enter depth")
        if self.min_spray_speed < 0.0 or self.spray_exit_speed < 0.0:
            raise ValueError("whitewater spray speeds must be non-negative")
        if self.spray_exit_speed > self.min_spray_speed:
            raise ValueError("spray exit speed must not exceed enter speed")
        if self.min_churn < 0.0:
            raise ValueError("whitewater classification churn must be non-negative")


def _setting(settings, name: str, default: float) -> float:
    return float(getattr(settings, name, default))


def classification_policy(settings) -> ClassificationPolicy:
    """Build a validated world-space policy from solver or Blender settings.

    ``ww_foam_band`` is authored in voxels.  Hysteresis expands the enter
    thresholds and contracts the exit thresholds symmetrically.  A value of
    0.25 therefore enters spray/bubble at 1.25 bands and exits at 0.75 bands.
    """

    voxel_size = _setting(settings, "voxel_size", 1.0)
    foam_band = _setting(settings, "ww_foam_band", 1.0) * voxel_size
    hysteresis = _setting(settings, "ww_type_hysteresis", 0.25)
    if not math.isfinite(hysteresis) or not 0.0 <= hysteresis < 1.0:
        raise ValueError("ww_type_hysteresis must be finite and in [0, 1)")
    min_spray_speed = _setting(
        settings,
        "ww_spray_min_speed",
        _setting(settings, "ww_min_speed", 0.5),
    )
    return ClassificationPolicy(
        foam_band=foam_band,
        spray_enter_height=foam_band * (1.0 + hysteresis),
        spray_exit_height=foam_band * (1.0 - hysteresis),
        bubble_enter_depth=foam_band * (1.0 + hysteresis),
        bubble_exit_depth=foam_band * (1.0 - hysteresis),
        min_spray_speed=min_spray_speed,
        spray_exit_speed=min_spray_speed * (1.0 - hysteresis),
        min_churn=_setting(settings, "ww_churn_threshold", 0.0),
    )


def classify_sample(
    *,
    sdf: float,
    speed: float,
    churn: float,
    foam_band: float,
    min_spray_speed: float,
    min_churn: float = 0.0,
) -> WhitewaterType:
    """Classify a newly emitted sample using deterministic branch ordering."""

    values = (sdf, speed, churn, foam_band, min_spray_speed, min_churn)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("whitewater classification inputs must be finite")
    foam_band = abs(float(foam_band))
    if foam_band <= 0.0:
        raise ValueError("foam_band must be positive")
    if float(sdf) < -foam_band:
        return WhitewaterType.BUBBLE
    if (
        float(sdf) > foam_band
        and float(speed) >= max(0.0, float(min_spray_speed))
        and float(churn) >= max(0.0, float(min_churn))
    ):
        return WhitewaterType.SPRAY
    return WhitewaterType.FOAM


def transition_sample(
    current: WhitewaterType | int,
    *,
    sdf: float,
    speed: float,
    policy: ClassificationPolicy,
) -> WhitewaterType:
    """Apply hysteretic per-step transitions to one classified particle."""

    try:
        current = WhitewaterType(int(current))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid current whitewater type {current!r}") from exc
    if not math.isfinite(float(sdf)) or not math.isfinite(float(speed)):
        raise ValueError("whitewater transition inputs must be finite")
    sdf = float(sdf)
    speed = max(0.0, float(speed))

    if current is WhitewaterType.BUBBLE:
        if sdf >= -policy.bubble_exit_depth:
            if (
                sdf > policy.spray_enter_height
                and speed >= policy.min_spray_speed
            ):
                return WhitewaterType.SPRAY
            return WhitewaterType.FOAM
        return current

    if current is WhitewaterType.SPRAY:
        if sdf < -policy.bubble_enter_depth:
            return WhitewaterType.BUBBLE
        if (
            sdf <= policy.spray_exit_height
            or speed < policy.spray_exit_speed
        ):
            return WhitewaterType.FOAM
        return current

    if sdf < -policy.bubble_enter_depth:
        return WhitewaterType.BUBBLE
    if (
        sdf > policy.spray_enter_height
        and speed >= policy.min_spray_speed
    ):
        return WhitewaterType.SPRAY
    return current


def validate_type_values(values, *, allow_unclassified: bool = False) -> np.ndarray:
    """Normalize an aligned type channel and reject unsupported/reserved values."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("whitewater type values must be a one-dimensional array")
    if array.dtype.kind not in "iu":
        raise ValueError("whitewater type values must use an integer dtype")
    normalized = np.ascontiguousarray(array, dtype=np.int32)
    allowed = set(VALID_TYPE_VALUES)
    if allow_unclassified:
        allowed.add(LEGACY_UNCLASSIFIED)
    invalid = sorted(set(int(value) for value in np.unique(normalized)) - allowed)
    if invalid:
        raise ValueError(f"whitewater type values contain unsupported values {invalid}")
    return normalized


def partition_type_indices(
    values,
    *,
    allow_unclassified: bool = False,
) -> Mapping[WhitewaterType | int, np.ndarray]:
    """Return stable ascending indices for each type without reordering input."""

    normalized = validate_type_values(
        values, allow_unclassified=allow_unclassified
    )
    result: dict[WhitewaterType | int, np.ndarray] = {
        kind: np.flatnonzero(normalized == int(kind)).astype(np.intp, copy=False)
        for kind in WhitewaterType
    }
    if allow_unclassified:
        result[LEGACY_UNCLASSIFIED] = np.flatnonzero(
            normalized == LEGACY_UNCLASSIFIED
        ).astype(np.intp, copy=False)
    return result
