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

from .enums import SurfaceResolutionMode

LEAF_DIM = 8
MIN_VOXEL_SIZE = 1.0e-4
MIN_SURFACE_FRACTION = 1.0 / 16.0
MAX_LIQUID_SAMPLING = 64.0


class SourceSamplingMode(str, Enum):
    INHERIT = "INHERIT"
    MULTIPLIER = "MULTIPLIER"
    LEGACY_ABSOLUTE = "LEGACY_ABSOLUTE"


@dataclass(frozen=True)
class AuthoredQualitySettings:
    voxel_size: float
    liquid_sampling: float = 8.0
    surface_resolution_mode: int = 0
    surface_resolution_multiplier: float = 1.0
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
                authored_quality, "surface_resolution_multiplier", 1.0)),
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
    voxel_size = float(authored_quality.voxel_size)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel size must be finite and positive")
    if voxel_size < MIN_VOXEL_SIZE:
        diagnostics.append(f"voxel size clamped to {MIN_VOXEL_SIZE:g}")
        voxel_size = MIN_VOXEL_SIZE

    unpadded = tuple(max(1, int(math.ceil(value / voxel_size))) for value in size)
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
        multiplier = float(_value(source_sampling, "multiplier", 1.0))
        legacy = float(_value(source_sampling, "legacy_absolute", 8.0))

    diagnostics: list[str] = []
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
