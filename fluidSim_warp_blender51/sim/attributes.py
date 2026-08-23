"""Authoritative simulation-output attribute schema and host utilities.

This module deliberately has no Blender or Warp dependency.  It is shared by
solver allocation, cache validation, Blender publication, release tooling, and
pure-Python tests so persistent names and data contracts cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np

ATTRIBUTE_SCHEMA_VERSION = 1
SURFACE_TRANSFER_METHOD = "nearest_liquid_particle_v1"


@dataclass(frozen=True)
class AttributeSpec:
    """One stable cache/Blender output attribute contract."""

    key: str
    blender_name: str
    data_type: str
    domain: str
    components: int
    cached: bool
    checkpointed: bool

    @property
    def numpy_dtype(self):
        if self.data_type == "INT":
            return np.dtype(np.int32)
        return np.dtype(np.float32)

    @property
    def bytes_per_element(self) -> int:
        return self.numpy_dtype.itemsize * self.components


ATTRIBUTE_SPECS: dict[str, AttributeSpec] = {
    "velocity": AttributeSpec(
        "velocity", "fluidsim_velocity", "FLOAT_VECTOR", "POINT", 3, True, True),
    "speed": AttributeSpec(
        "speed", "fluidsim_speed", "FLOAT", "POINT", 1, True, False),
    "vorticity": AttributeSpec(
        "vorticity", "fluidsim_vorticity", "FLOAT_VECTOR", "POINT", 3, True, False),
    "age": AttributeSpec(
        "age", "fluidsim_age", "FLOAT", "POINT", 1, True, True),
    "lifetime": AttributeSpec(
        "lifetime", "fluidsim_lifetime", "FLOAT", "POINT", 1, True, True),
    "color": AttributeSpec(
        "color", "fluidsim_color", "FLOAT_COLOR", "POINT", 4, True, True),
    "species": AttributeSpec(
        "species", "fluidsim_species", "FLOAT_COLOR", "POINT", 4, True, False),
    "source_id": AttributeSpec(
        "source_id", "fluidsim_source_id", "INT", "POINT", 1, True, True),
    "uid": AttributeSpec(
        "uid", "fluidsim_uid", "INT", "POINT", 1, True, True),
    "viscosity": AttributeSpec(
        "viscosity", "fluidsim_viscosity", "FLOAT", "POINT", 1, True, True),
    "density": AttributeSpec(
        "density", "fluidsim_density", "FLOAT", "POINT", 1, True, True),
    # Blender's conventional motion-blur attribute is intentionally named
    # ``velocity``.  It stores displacement per scene frame, not the solver's
    # world-units-per-second velocity channel above.
    "motion": AttributeSpec(
        "motion", "velocity", "FLOAT_VECTOR", "POINT", 3, True, False),
}

# Old experimental names accepted on read and removed/replaced on publication.
ATTRIBUTE_ALIASES = {
    "fluid_velocity": "fluidsim_velocity",
    "fluid_speed": "fluidsim_speed",
    "fluid_vorticity": "fluidsim_vorticity",
    "fluid_age": "fluidsim_age",
    "fluid_life": "fluidsim_lifetime",
    "fluid_color": "fluidsim_color",
    "fluid_source_id": "fluidsim_source_id",
    "fluid_uid": "fluidsim_uid",
}

OUTPUT_ATTRIBUTE_KEYS: dict[str, tuple[str, ...]] = {
    "surface": (
        "velocity", "speed", "vorticity", "age", "lifetime", "color", "species",
        "source_id", "viscosity", "density", "motion",
    ),
    "particles": (
        "velocity", "speed", "vorticity", "age", "lifetime", "color", "species",
        "source_id", "uid", "viscosity", "density",
    ),
    "whitewater": (
        "velocity", "speed", "age", "lifetime",
    ),
}

OUTPUT_TOGGLE_NAMES: dict[str, dict[str, str]] = {
    output_kind: {
        key: f"{output_kind}_attribute_{key}"
        for key in keys
    }
    for output_kind, keys in OUTPUT_ATTRIBUTE_KEYS.items()
}

PERSISTENT_OPTIONAL_PARTICLE_CHANNELS = frozenset({"color", "source_id", "uid"})


CACHE_ATTRIBUTE_NAMES: dict[str, dict[str, str]] = {
    "surface": {key: f"surface_{key}" for key in OUTPUT_ATTRIBUTE_KEYS["surface"]},
    "particles": {
        "velocity": "velocities",
        "speed": "particle_speed",
        "vorticity": "particle_vorticity",
        "age": "particle_age",
        "lifetime": "particle_life",
        "color": "particle_color",
        "species": "particle_species",
        "source_id": "particle_source_id",
        "uid": "particle_uid",
        "viscosity": "particle_viscosity",
        "density": "particle_density",
    },
    "whitewater": {
        "velocity": "whitewater_velocities",
        "speed": "whitewater_speed",
        "age": "whitewater_age",
        "lifetime": "whitewater_life",
    },
}


def cache_attribute_name(output_kind: str, key: str) -> str:
    try:
        return CACHE_ATTRIBUTE_NAMES[str(output_kind).lower()][str(key)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported cache attribute {output_kind!r}/{key!r}") from exc


def enabled_attribute_keys(settings, output_kind: str) -> tuple[str, ...]:
    """Return enabled keys in authoritative schema order.

    Missing settings are treated as disabled.  This keeps old cache/playback
    objects readable while new solver instances receive an explicit inventory.
    """
    output_kind = str(output_kind).lower()
    if output_kind not in OUTPUT_ATTRIBUTE_KEYS:
        raise ValueError(f"unknown output kind {output_kind!r}")
    toggles = OUTPUT_TOGGLE_NAMES[output_kind]
    return tuple(
        key for key in OUTPUT_ATTRIBUTE_KEYS[output_kind]
        if bool(getattr(settings, toggles[key], False))
    )


def enabled_attributes(settings, output_kind: str) -> tuple[AttributeSpec, ...]:
    return tuple(ATTRIBUTE_SPECS[key]
                 for key in enabled_attribute_keys(settings, output_kind))


def persistent_particle_channels(settings) -> tuple[str, ...]:
    """Return opt-in per-particle channels required by any selected output."""
    required: set[str] = set()
    for output_kind in ("surface", "particles"):
        required.update(enabled_attribute_keys(settings, output_kind))
    return tuple(
        key for key in ("color", "source_id", "uid")
        if key in required
        and not (key == "color" and bool(getattr(settings, "species_enabled", False)))
    )


def stable_source_id(value: object) -> int:
    """Map persistent source identity to a deterministic positive int32 ID."""
    text = str(value or "unknown")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    result = int.from_bytes(digest[:4], "little") & 0x7FFF_FFFF
    return result or 1


def normalize_color(value: Iterable[float] | None) -> tuple[float, float, float, float]:
    if value is None:
        return (0.1, 0.45, 1.0, 1.0)
    items = tuple(float(component) for component in value)
    if len(items) == 3:
        items = (*items, 1.0)
    if len(items) != 4 or not np.all(np.isfinite(items)):
        raise ValueError("source color must contain three or four finite components")
    return tuple(float(np.clip(component, 0.0, 1.0)) for component in items)


def derive_species_preview_colors(
        concentrations,
        absorption_rgb,
        reference_path_length: float) -> np.ndarray:
    """Derive scene-linear Beer-Lambert preview RGB from species authority."""
    values = np.asarray(concentrations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("species concentrations must have shape (N, 4)")
    if not np.all(np.isfinite(values)):
        raise ValueError("species concentrations contain non-finite values")
    tolerance = 1.0e-6
    if (np.any(values[:, :2] < -tolerance)
            or np.any(values[:, :2].sum(axis=1) > 1.0 + tolerance)
            or np.any(np.abs(values[:, 2:]) > tolerance)):
        raise ValueError("species concentrations violate the two-species simplex")
    absorption = np.asarray(absorption_rgb, dtype=np.float32)
    if (absorption.shape != (2, 3) or not np.all(np.isfinite(absorption))
            or np.any(absorption < 0.0)):
        raise ValueError("species absorption must be two nonnegative RGB triples")
    path = float(reference_path_length)
    if not np.isfinite(path) or path <= 0.0:
        raise ValueError("species reference path length must be finite and positive")
    result = np.ones((len(values), 4), dtype=np.float32)
    if len(values):
        result[:, :3] = np.exp(-path * (values[:, :2] @ absorption))
    return np.ascontiguousarray(result)


def speed_from_velocity(velocity) -> np.ndarray:
    values = np.asarray(velocity, dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    values = values.reshape((-1, 3))
    return np.linalg.norm(values, axis=1).astype(np.float32, copy=False)


def motion_from_velocity(velocity, *, frame_rate: float, time_scale: float = 1.0) -> np.ndarray:
    """Convert world-units/second velocity to Blender displacement per frame."""
    frame_rate = float(frame_rate)
    if not np.isfinite(frame_rate) or frame_rate <= 0.0:
        raise ValueError("frame_rate must be finite and positive")
    time_scale = float(time_scale)
    if not np.isfinite(time_scale) or time_scale < 0.0:
        raise ValueError("time_scale must be finite and non-negative")
    values = np.asarray(velocity, dtype=np.float32).reshape((-1, 3))
    return np.ascontiguousarray(values * np.float32(time_scale / frame_rate))


def validate_attribute_values(spec: AttributeSpec, values, count: int) -> np.ndarray:
    """Normalize one host array and fail closed on schema violations."""
    count = int(count)
    if count < 0:
        raise ValueError("attribute count must be non-negative")
    array = np.asarray(values)
    expected = (count,) if spec.components == 1 else (count, spec.components)
    if array.shape != expected:
        raise ValueError(
            f"{spec.key} has shape {array.shape}; expected {expected}")
    if spec.data_type == "INT":
        if array.dtype.kind not in "iu":
            raise ValueError(f"{spec.key} must use an integer dtype")
        normalized = np.ascontiguousarray(array, dtype=np.int32)
    else:
        if array.dtype.kind not in "fiu":
            raise ValueError(f"{spec.key} must use a numeric dtype")
        normalized = np.ascontiguousarray(array, dtype=np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{spec.key} contains non-finite values")
    return normalized


def _exact_nearest_indices(
    targets: np.ndarray,
    sample_points: np.ndarray,
    sample_indices: np.ndarray,
    *,
    max_distance_pairs: int,
) -> np.ndarray:
    """Return exact nearest indices with a strict temporary-memory bound."""
    if not len(targets):
        return np.zeros(0, dtype=np.intp)
    if not len(sample_points):
        return np.full(len(targets), -1, dtype=np.intp)
    batch = max(1, int(max_distance_pairs) // len(sample_points))
    result = np.empty(len(targets), dtype=np.intp)
    for start in range(0, len(targets), batch):
        stop = min(start + batch, len(targets))
        delta = targets[start:stop, None, :] - sample_points[None, :, :]
        distance2 = np.einsum("...i,...i->...", delta, delta, optimize=True)
        # np.argmin selects the first sample for an exact tie. The sampled
        # index order is stable, so tie resolution is deterministic.
        result[start:stop] = sample_indices[np.argmin(distance2, axis=1)]
    return result


def nearest_sample_indices(
    targets,
    sample_points,
    *,
    cell_size: float | None = None,
    max_samples: int = 120_000,
    max_distance_pairs: int = 2_000_000,
) -> np.ndarray:
    """Map targets to deterministic nearest source-particle indices.

    With ``cell_size`` this uses a sorted collision-free spatial hash and
    searches the target cell plus its 26 neighbors. Surface vertices produced
    from the liquid field are expected to find local particles there. Any
    unmatched target falls back to an exact, memory-bounded search. A stable
    source stride caps preview/readback work without changing ordering.
    """
    targets = np.asarray(targets, dtype=np.float32).reshape((-1, 3))
    source = np.asarray(sample_points, dtype=np.float32).reshape((-1, 3))
    if not len(targets):
        return np.zeros(0, dtype=np.intp)
    if not len(source):
        return np.full(len(targets), -1, dtype=np.intp)
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(source)):
        raise ValueError("nearest-transfer positions must be finite")

    max_samples = max(1, int(max_samples))
    source_indices = np.arange(len(source), dtype=np.intp)
    if len(source) > max_samples:
        stride = int(np.ceil(len(source) / max_samples))
        source = source[::stride]
        source_indices = source_indices[::stride]

    if cell_size is None:
        return _exact_nearest_indices(
            targets, source, source_indices,
            max_distance_pairs=max_distance_pairs,
        )
    cell_size = float(cell_size)
    if not np.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be finite and positive")

    source_cells = np.floor(source / cell_size).astype(np.int64)
    target_cells = np.floor(targets / cell_size).astype(np.int64)
    key_min = np.minimum(source_cells.min(axis=0), target_cells.min(axis=0) - 1)
    key_max = np.maximum(source_cells.max(axis=0), target_cells.max(axis=0) + 1)
    key_dims = key_max - key_min + 1
    stride_y = int(key_dims[2])
    stride_x = int(key_dims[1]) * stride_y
    if int(key_dims[0]) * stride_x > np.iinfo(np.int64).max:
        raise ValueError("attribute lookup coordinate range is too large")

    def linear_keys(cells: np.ndarray) -> np.ndarray:
        shifted = cells - key_min
        return (
            shifted[..., 0] * stride_x
            + shifted[..., 1] * stride_y
            + shifted[..., 2]
        )

    source_linear = linear_keys(source_cells)
    source_order = np.argsort(source_linear, kind="stable")
    unique_keys, starts, counts = np.unique(
        source_linear[source_order], return_index=True, return_counts=True
    )
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
        ],
        dtype=np.int64,
    )
    result = np.full(len(targets), -1, dtype=np.intp)
    query_start = 0
    query_batch = min(4096, len(targets))
    pair_limit = max(1, int(max_distance_pairs))
    while query_start < len(targets):
        query_end = min(query_start + query_batch, len(targets))
        local_count = query_end - query_start
        neighbor_cells = (
            target_cells[query_start:query_end, None, :] + offsets[None, :, :]
        ).reshape((-1, 3))
        neighbor_keys = linear_keys(neighbor_cells)
        bucket_ids = np.searchsorted(unique_keys, neighbor_keys)
        valid = bucket_ids < len(unique_keys)
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] = (
            unique_keys[bucket_ids[valid_positions]]
            == neighbor_keys[valid_positions]
        )
        if not valid.any():
            query_start = query_end
            query_batch = min(4096, len(targets) - query_start or 1)
            continue

        bucket_ids = bucket_ids[valid]
        query_ids = np.repeat(np.arange(local_count, dtype=np.intp), len(offsets))[valid]
        candidate_counts = counts[bucket_ids].astype(np.intp, copy=False)
        pair_count = int(candidate_counts.sum())
        if pair_count > pair_limit and local_count > 1:
            query_batch = max(1, int(local_count * pair_limit / pair_count))
            continue

        pair_offsets = np.cumsum(candidate_counts) - candidate_counts
        repeated_buckets = np.repeat(
            np.arange(len(bucket_ids), dtype=np.intp), candidate_counts
        )
        within_bucket = (
            np.arange(pair_count, dtype=np.intp)
            - np.repeat(pair_offsets, candidate_counts)
        )
        sorted_candidates = starts[bucket_ids[repeated_buckets]] + within_bucket
        sampled_candidates = source_order[sorted_candidates]
        expanded_queries = np.repeat(query_ids, candidate_counts)
        delta = (
            targets[query_start + expanded_queries]
            - source[sampled_candidates]
        )
        distance2 = np.einsum("ij,ij->i", delta, delta, optimize=True)
        best_distance = np.full(local_count, np.inf, dtype=np.float32)
        np.minimum.at(best_distance, expanded_queries, distance2)
        winners = distance2 == best_distance[expanded_queries]
        # Resolve exact ties to the lowest original particle index.
        best_index = np.full(local_count, np.iinfo(np.intp).max, dtype=np.intp)
        np.minimum.at(
            best_index,
            expanded_queries[winners],
            source_indices[sampled_candidates[winners]],
        )
        matched = best_index != np.iinfo(np.intp).max
        result[query_start + np.flatnonzero(matched)] = best_index[matched]
        query_start = query_end
        query_batch = min(4096, len(targets) - query_start or 1)

    missing = np.flatnonzero(result < 0)
    if len(missing):
        result[missing] = _exact_nearest_indices(
            targets[missing], source, source_indices,
            max_distance_pairs=pair_limit,
        )
    return result


def transfer_nearest(
    targets,
    sample_points,
    sample_values,
    *,
    default=0.0,
    cell_size: float | None = None,
    max_samples: int = 120_000,
    max_distance_pairs: int = 2_000_000,
) -> np.ndarray:
    """Transfer scalar/vector values using one bounded nearest lookup."""
    targets = np.asarray(targets, dtype=np.float32).reshape((-1, 3))
    sample_points = np.asarray(sample_points, dtype=np.float32).reshape((-1, 3))
    values = np.asarray(sample_values)
    if len(values) != len(sample_points):
        raise ValueError("sample values must align with sample points")
    trailing = values.shape[1:]
    if not len(targets):
        return np.zeros((0, *trailing), dtype=values.dtype)
    if not len(sample_points):
        result = np.empty((len(targets), *trailing), dtype=values.dtype)
        result[...] = default
        return result
    indices = nearest_sample_indices(
        targets,
        sample_points,
        cell_size=cell_size,
        max_samples=max_samples,
        max_distance_pairs=max_distance_pairs,
    )
    return np.ascontiguousarray(values[indices])


def estimate_attribute_bytes(settings, counts: Mapping[str, int]) -> dict[str, int]:
    """Estimate one uncompressed host frame's enabled attribute payload."""
    result: dict[str, int] = {}
    for output_kind in OUTPUT_ATTRIBUTE_KEYS:
        count = max(0, int(counts.get(output_kind, 0)))
        result[output_kind] = sum(
            spec.bytes_per_element * count
            for spec in enabled_attributes(settings, output_kind)
        )
    result["total"] = sum(result.values())
    return result
