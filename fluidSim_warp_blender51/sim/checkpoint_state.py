"""Pure schema validation for GPU solver checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import constants as C
from .whitewater_types import validate_type_values


@dataclass(frozen=True)
class CheckpointSchema:
    """Runtime capacities and channel inventory required to validate a restore."""

    leaf_capacity: int
    leaf_table_cells: int
    whitewater_capacity: int
    particle_attributes: tuple[str, ...]
    optional_particle_channels: tuple[str, ...]
    field_names: tuple[str, ...]
    whitewater_type_schema_version: int

    def validate(
            self, state: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
        """Validate a host-owned checkpoint without touching GPU state."""
        if not isinstance(state, dict) or set(state) != {
                "arrays", "scalars", "source_state"}:
            raise ValueError("solver checkpoint state has an invalid schema")
        arrays = state["arrays"]
        scalars = state["scalars"]
        if not isinstance(arrays, dict) or not isinstance(scalars, dict):
            raise ValueError("solver checkpoint arrays/scalars must be mappings")
        required_scalars = {
            "frame", "sim_time", "leaf_count", "particle_count",
            "whitewater_count", "leaf_capacity", "particle_capacity",
            "whitewater_capacity", "whitewater_emit_seed",
            "whitewater_type_schema_version",
        }
        missing = required_scalars - set(scalars)
        if missing:
            raise ValueError(
                f"solver checkpoint is missing scalars {sorted(missing)}")
        counts = {
            name: int(scalars[name])
            for name in ("leaf_count", "particle_count", "whitewater_count")
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("solver checkpoint counts must be non-negative")
        if counts["leaf_count"] > self.leaf_capacity:
            raise ValueError("checkpoint leaf count exceeds current capacity")
        if counts["whitewater_count"] > self.whitewater_capacity:
            raise ValueError("checkpoint whitewater count exceeds current capacity")

        required_arrays = {
            "grid_leaf_coords", "grid_old_leaf_table",
            "particle_position", "particle_velocity", "particle_alive",
            "particle_time_offset", "whitewater_position",
            "whitewater_velocity", "whitewater_age", "whitewater_life",
            "whitewater_alive", "whitewater_type",
        }
        required_arrays.update(
            f"particle_attr_{name}" for name in self.particle_attributes)
        required_arrays.update(
            f"particle_attr_{name}" for name in self.optional_particle_channels)
        if (self.optional_particle_channels
                and "particle_optional_channels" not in scalars):
            raise ValueError(
                "checkpoint is missing optional particle inventory metadata")
        saved_channels = tuple(filter(
            None, str(scalars.get("particle_optional_channels", "")).split(",")))
        if saved_channels != self.optional_particle_channels:
            raise ValueError(
                "checkpoint optional particle inventory does not match settings")
        if "uid" in self.optional_particle_channels:
            if "next_particle_uid" not in scalars:
                raise ValueError(
                    "UID-enabled checkpoint is missing next_particle_uid")
            if int(scalars["next_particle_uid"]) <= 0:
                raise ValueError("next_particle_uid must be positive")
        required_arrays.update(
            f"field_{name.replace('.', '_')}" for name in self.field_names)
        missing_arrays = required_arrays - set(arrays)
        if missing_arrays:
            raise ValueError(
                f"solver checkpoint is missing arrays {sorted(missing_arrays)}")
        for name in required_arrays:
            value = np.asarray(arrays[name])
            if value.dtype.hasobject:
                raise ValueError(f"checkpoint array {name} uses object dtype")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError(f"checkpoint array {name} is non-finite")
        if int(scalars["whitewater_type_schema_version"]) != (
                self.whitewater_type_schema_version):
            raise ValueError("unsupported whitewater type checkpoint schema")
        validate_type_values(
            arrays["whitewater_type"], allow_unclassified=False)
        if np.asarray(arrays["grid_leaf_coords"]).shape != (
                counts["leaf_count"], 3):
            raise ValueError("grid_leaf_coords shape does not match leaf_count")
        if np.asarray(arrays["grid_old_leaf_table"]).shape != (
                self.leaf_table_cells,):
            raise ValueError("grid_old_leaf_table has the wrong shape")
        if "particle_affine" in arrays:
            # Optional: empty (0, 9) means "no affine state" (non-APIC writer).
            affine = np.asarray(arrays["particle_affine"])
            if affine.ndim != 2 or affine.shape[1] != 9 or affine.shape[0] not in (
                    0, counts["particle_count"]):
                raise ValueError(
                    "particle_affine shape does not match particle_count")
        active_voxels = counts["leaf_count"] * C.LEAF_VOLUME
        for dotted_name in self.field_names:
            key = f"field_{dotted_name.replace('.', '_')}"
            if np.asarray(arrays[key]).shape != (active_voxels,):
                raise ValueError(f"{key} shape does not match active topology")
        return arrays, scalars, counts
