"""Pure checkpoint schema validation tests."""

import numpy as np
import pytest

from fluidSim_warp_blender51.sim.checkpoint_state import CheckpointSchema
from fluidSim_warp_blender51.sim.whitewater_types import (
    WHITEWATER_TYPE_SCHEMA_VERSION,
)


def schema(*, optional_channels=()) -> CheckpointSchema:
    return CheckpointSchema(
        leaf_capacity=4,
        leaf_table_cells=2,
        whitewater_capacity=8,
        particle_attributes=(),
        optional_particle_channels=tuple(optional_channels),
        field_names=(),
        whitewater_type_schema_version=WHITEWATER_TYPE_SCHEMA_VERSION,
    )


def checkpoint_state() -> dict[str, object]:
    arrays = {
        "grid_leaf_coords": np.empty((0, 3), dtype=np.int32),
        "grid_old_leaf_table": np.full(2, -1, dtype=np.int32),
        "particle_position": np.empty((0, 3), dtype=np.float32),
        "particle_velocity": np.empty((0, 3), dtype=np.float32),
        "particle_alive": np.empty(0, dtype=np.int32),
        "particle_time_offset": np.empty(0, dtype=np.float32),
        "whitewater_position": np.empty((0, 3), dtype=np.float32),
        "whitewater_velocity": np.empty((0, 3), dtype=np.float32),
        "whitewater_age": np.empty(0, dtype=np.float32),
        "whitewater_life": np.empty(0, dtype=np.float32),
        "whitewater_alive": np.empty(0, dtype=np.int32),
        "whitewater_type": np.empty(0, dtype=np.int32),
    }
    scalars = {
        "frame": 0,
        "sim_time": 0.0,
        "leaf_count": 0,
        "particle_count": 0,
        "whitewater_count": 0,
        "leaf_capacity": 4,
        "particle_capacity": 8,
        "whitewater_capacity": 8,
        "whitewater_emit_seed": 1,
        "st_step_seed": 1,
        "st_previous_dt": 0.0,
        "st_reference_mass_cached": 0.0,
        "whitewater_type_schema_version": WHITEWATER_TYPE_SCHEMA_VERSION,
    }
    return {"arrays": arrays, "scalars": scalars, "source_state": {}}


def test_checkpoint_schema_returns_validated_sections():
    state = checkpoint_state()
    arrays, scalars, counts = schema().validate(state)

    assert arrays is state["arrays"]
    assert scalars is state["scalars"]
    assert counts == {"leaf_count": 0, "particle_count": 0,
                      "whitewater_count": 0}


def test_checkpoint_schema_rejects_missing_required_array():
    state = checkpoint_state()
    del state["arrays"]["particle_velocity"]

    with pytest.raises(ValueError, match="missing arrays"):
        schema().validate(state)


def test_checkpoint_schema_rejects_non_finite_arrays():
    state = checkpoint_state()
    state["arrays"]["particle_velocity"] = np.asarray(
        [[np.nan, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        schema().validate(state)


def test_checkpoint_schema_requires_uid_inventory_metadata():
    state = checkpoint_state()
    state["arrays"]["particle_attr_uid"] = np.empty(0, dtype=np.float32)

    with pytest.raises(ValueError, match="optional particle inventory"):
        schema(optional_channels=("uid",)).validate(state)
