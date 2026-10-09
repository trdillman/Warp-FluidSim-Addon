"""Pure authoritative resolution and liquid-sampling contract.

This module intentionally imports neither Blender nor Warp.  It is the single
owner of domain ceil/alignment rules, liquid rest-density/spacing inputs, source
effective sampling, and render-surface voxel size.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

from .constants import LEAF_DIM
from .enums import SurfaceResolutionMode

MIN_VOXEL_SIZE = 1.0e-4
MIN_SURFACE_FRACTION = 1.0 / 16.0
MAX_LIQUID_SAMPLING = 64.0
MIN_RESOLUTION_DIVISIONS = 8
# ``size / voxel_size`` is rounded up to whole cells. A voxel size that went
# through float32 (a Blender FloatProperty) can sit a few parts in 1e8 below the
# exact ``longest / divisions``, which would otherwise ceil to divisions + 1 and
# pad a whole extra leaf. float32 epsilon is 6e-8; 1e-6 absorbs it with margin
# while still counting any real overshoot above 1 ppm of a cell.
CELL_COUNT_TOLERANCE = 1.0e-6


def voxel_size_for_divisions(divisions: float, domain_size: float) -> float:
    """Cell size that puts ``divisions`` cells along an axis of ``domain_size``."""
    divisions = max(float(divisions), float(MIN_RESOLUTION_DIVISIONS))
    return float(domain_size) / divisions


def divisions_for_voxel_size(voxel_size: float, domain_size: float) -> int:
    """Whole divisions (>= 8) along ``domain_size`` that best match ``voxel_size``."""
    voxel_size = float(voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel size must be finite and positive")
    return max(MIN_RESOLUTION_DIVISIONS, int(round(float(domain_size) / voxel_size)))

# Scene-level Quality tier -> speed<->fidelity multiplier. FINAL refines,
# DRAFT coarsens; PREVIEW and CUSTOM defer to the authored values verbatim so
# existing scenes (default PREVIEW) keep identical output. The multiplier is
# applied only when projecting authored knobs into owned solver settings; the
# authored values themselves are never mutated (see ``scale_quality_value``).
QUALITY_TIER_FIDELITY = {
    "DRAFT": 0.5,
    "PREVIEW": 1.0,
    "FINAL": 2.0,
    "CUSTOM": 1.0,
}

# Authored solver knobs the Quality tier rescales. ``voxel_size`` scales
# inversely (higher fidelity -> smaller voxels); the iteration and sampling
# knobs scale directly (higher fidelity -> more work).
_QUALITY_ITERATION_FIELDS = frozenset({
    "min_substeps", "max_substeps", "pressure_iterations",
})
_QUALITY_SAMPLING_FIELDS = frozenset({"particles_per_voxel"})
QUALITY_SCALED_FIELDS = (
    frozenset({"voxel_size"}) | _QUALITY_ITERATION_FIELDS | _QUALITY_SAMPLING_FIELDS
)


def quality_tier_fidelity(tier: Any) -> float:
    """Return the fidelity multiplier for a Quality enum value.

    Unknown or missing values resolve to PREVIEW (1.0), so a scene that predates
    the tier — or any non-tier caller — sees authored values unchanged.
    """
    return QUALITY_TIER_FIDELITY.get(str(tier).strip().upper(), 1.0)


def scale_quality_value(field: str, value: Any, fidelity: float) -> Any:
    """Scale one authored knob for a fidelity multiplier, non-destructively.

    Idempotent by construction: the result depends only on the freshly-read
    authored ``value`` and the tier, never on a previously-scaled value, so
    repeated projection can never compound. A fidelity of 1.0 (PREVIEW/CUSTOM),
    or a non-finite/non-positive factor, returns ``value`` untouched.
    """
    if not math.isfinite(fidelity) or fidelity <= 0.0 or fidelity == 1.0:
        return value
    if field == "voxel_size":
        return max(MIN_VOXEL_SIZE, float(value) / fidelity)
    if field in _QUALITY_ITERATION_FIELDS:
        return max(1, int(round(float(value) * fidelity)))
    if field in _QUALITY_SAMPLING_FIELDS:
        # 0 is the documented sentinel for "disable inherited liquid emission"
        # (solver admission, schema-v4 inherited sources, and height sources all
        # honor it); scaling must never turn an authored 0 into a nonzero value.
        authored = float(value)
        if authored <= 0.0:
            return authored
        return max(1.0, min(MAX_LIQUID_SAMPLING, authored * fidelity))
    return value


class SourceSamplingMode(str, Enum):
    INHERIT = "INHERIT"
    MULTIPLIER = "MULTIPLIER"
    LEGACY_ABSOLUTE = "LEGACY_ABSOLUTE"


@dataclass(frozen=True)
class AuthoredQualitySettings:
    voxel_size: float
    liquid_sampling: float = 8.0
    surface_resolution_mode: int = 0
    surface_resolution_multiplier: float = 2.0
    surface_voxel_size: float = 0.0


@dataclass(frozen=True)
class SourceSamplingSettings:
    mode: SourceSamplingMode = SourceSamplingMode.INHERIT
    multiplier: float = 1.0
    legacy_absolute: float = 8.0


@dataclass(frozen=True)
class EffectiveSourceSampling:
    mode: SourceSamplingMode
    value: float
    target_per_voxel: int
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedQuality:
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    domain_size: tuple[float, float, float]
    voxel_size: float
    unpadded_voxel_dimensions: tuple[int, int, int]
    padded_voxel_dimensions: tuple[int, int, int]
    leaf_dimensions: tuple[int, int, int]
    total_domain_leaf_cells: int
    liquid_sampling: float
    rest_particle_count: float
    nominal_particle_spacing: float
    surface_voxel_size: float
    diagnostics: tuple[str, ...] = ()


def _value(source: Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _vec3(value: Any, name: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-vector") from exc
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _align(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _surface_resolution_mode(value: Any) -> int:
    """Normalize Blender enum identifiers and runtime integer modes."""
    if isinstance(value, str):
        member = SurfaceResolutionMode.__members__.get(value.strip().upper())
        if member is not None:
            return int(member)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _surface_voxel_size(authored: AuthoredQualitySettings, voxel_size: float,
                        diagnostics: list[str]) -> float:
    mode = _surface_resolution_mode(authored.surface_resolution_mode)
    if mode == 2:
        custom = float(authored.surface_voxel_size)
        if math.isfinite(custom) and custom > 0.0:
            return max(custom, voxel_size * MIN_SURFACE_FRACTION)
        diagnostics.append("invalid custom surface voxel size; using simulation size")
        return voxel_size
    if mode == 1:
        multiplier = float(authored.surface_resolution_multiplier)
        if not math.isfinite(multiplier):
            diagnostics.append("invalid surface multiplier; using 1x")
            multiplier = 1.0
        multiplier = max(0.25, min(8.0, multiplier))
        return voxel_size / multiplier
    if mode not in (0, 1, 2):
        diagnostics.append("unknown surface resolution mode; using simulation size")
    return voxel_size


def resolve_quality(authored_quality: AuthoredQualitySettings | Any,
                    domain_bounds) -> ResolvedQuality:
    """Resolve authored quality and bounds into immutable runtime facts."""
    if not isinstance(authored_quality, AuthoredQualitySettings):
        authored_quality = AuthoredQualitySettings(
            voxel_size=float(_value(authored_quality, "voxel_size", 0.1)),
            liquid_sampling=float(_value(
                authored_quality, "particles_per_voxel", 8.0)),
            surface_resolution_mode=_surface_resolution_mode(_value(
                authored_quality, "surface_resolution_mode", 0)),
            surface_resolution_multiplier=float(_value(
                authored_quality, "surface_resolution_multiplier", 2.0)),
            surface_voxel_size=float(_value(
                authored_quality, "surface_voxel_size", 0.0)),
        )
    try:
        minimum, maximum = domain_bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("domain_bounds must contain minimum and maximum vectors") from exc
    minimum = _vec3(minimum, "domain minimum")
    maximum = _vec3(maximum, "domain maximum")
    size = tuple(maximum[i] - minimum[i] for i in range(3))
    if any(value <= 0.0 or not math.isfinite(value) for value in size):
        raise ValueError("domain maximum must be greater than minimum on every axis")

    diagnostics: list[str] = []
    clamped_voxel = False
    voxel_size = float(authored_quality.voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel size must be finite and positive")
    if voxel_size < MIN_VOXEL_SIZE:
        voxel_size = MIN_VOXEL_SIZE
        clamped_voxel = True

    unpadded = tuple(
        max(1, int(math.ceil(value / voxel_size * (1.0 - CELL_COUNT_TOLERANCE))))
        for value in size)
    if clamped_voxel:
        diagnostics.append(
            f"resolution capped at {max(unpadded)} divisions by minimum cell "
            f"size {MIN_VOXEL_SIZE:g}")
    padded = tuple(max(LEAF_DIM, _align(value, LEAF_DIM)) for value in unpadded)
    leaves = tuple(value // LEAF_DIM for value in padded)

    sampling = float(authored_quality.liquid_sampling)
    if not math.isfinite(sampling):
        diagnostics.append("invalid liquid sampling; using 8")
        sampling = 8.0
    clamped_sampling = max(0.0, min(MAX_LIQUID_SAMPLING, sampling))
    if clamped_sampling != sampling:
        diagnostics.append("liquid sampling clamped to backend range 0..64")
    spacing = voxel_size / max(clamped_sampling, 1.0) ** (1.0 / 3.0)

    return ResolvedQuality(
        domain_min=minimum,
        domain_max=maximum,
        domain_size=size,
        voxel_size=voxel_size,
        unpadded_voxel_dimensions=unpadded,
        padded_voxel_dimensions=padded,
        leaf_dimensions=leaves,
        total_domain_leaf_cells=int(leaves[0] * leaves[1] * leaves[2]),
        liquid_sampling=clamped_sampling,
        rest_particle_count=clamped_sampling,
        nominal_particle_spacing=spacing,
        surface_voxel_size=_surface_voxel_size(
            authored_quality, voxel_size, diagnostics),
        diagnostics=tuple(diagnostics),
    )


def resolve_source_sampling(
        resolved_quality: ResolvedQuality,
        source_sampling: SourceSamplingSettings | Any) -> EffectiveSourceSampling:
    """Resolve source occupancy once using the shared 0..64 backend contract."""
    if isinstance(source_sampling, SourceSamplingSettings):
        mode = source_sampling.mode
        multiplier = source_sampling.multiplier
        legacy = source_sampling.legacy_absolute
    else:
        raw_mode = _value(source_sampling, "mode", SourceSamplingMode.INHERIT)
        try:
            mode = (raw_mode if isinstance(raw_mode, SourceSamplingMode)
                    else SourceSamplingMode(str(raw_mode).upper()))
        except ValueError:
            mode = SourceSamplingMode.LEGACY_ABSOLUTE
        multiplier = _value(source_sampling, "multiplier", 1.0)
        legacy = _value(source_sampling, "legacy_absolute", 8.0)

    diagnostics: list[str] = []
    try:
        multiplier = float(multiplier)
    except (TypeError, ValueError):
        diagnostics.append("invalid sampling multiplier; using 1x")
        multiplier = 1.0
    try:
        legacy = float(legacy)
    except (TypeError, ValueError):
        diagnostics.append("invalid legacy source sampling; using zero")
        legacy = 0.0
    if mode is SourceSamplingMode.INHERIT:
        value = resolved_quality.liquid_sampling
    elif mode is SourceSamplingMode.MULTIPLIER:
        if not math.isfinite(multiplier):
            diagnostics.append("invalid sampling multiplier; using 1x")
            multiplier = 1.0
        value = resolved_quality.liquid_sampling * max(0.0, multiplier)
    else:
        value = legacy
    if not math.isfinite(value):
        diagnostics.append("invalid source sampling; using zero")
        value = 0.0
    bounded = max(0.0, min(MAX_LIQUID_SAMPLING, float(value)))
    if bounded != value:
        diagnostics.append("source sampling clamped to backend range 0..64")
    return EffectiveSourceSampling(
        mode=mode,
        value=bounded,
        target_per_voxel=int(round(bounded)),
        diagnostics=tuple(diagnostics),
    )
