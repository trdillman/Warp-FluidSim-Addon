from __future__ import annotations

import json

import numpy as np
import pytest
from fluidsim_testpkg.io.cache import (
    authoritative_payload_counts,
    validate_optional_attributes,
)
from fluidsim_testpkg.sim.attributes import ATTRIBUTE_SCHEMA_VERSION, SURFACE_TRANSFER_METHOD


def _payload():
    return {
        "attribute_schema_version": np.array(ATTRIBUTE_SCHEMA_VERSION, dtype=np.int32),
        "attribute_inventory_json": np.array(json.dumps({
            "surface": ["velocity", "motion"],
            "particles": ["color", "source_id", "uid"],
            "whitewater": ["age", "lifetime", "type"],
        }, sort_keys=True, separators=(",", ":"))),
        "surface_attribute_transfer_method": np.array(SURFACE_TRANSFER_METHOD),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.zeros((1, 3), dtype=np.float32),
        "surface_verts": np.zeros((3, 3), dtype=np.float32),
        "surface_velocity": np.ones((3, 3), dtype=np.float32),
        "surface_motion": np.full((3, 3), 0.25, dtype=np.float32),
        "particle_color": np.ones((2, 4), dtype=np.float32),
        "particle_source_id": np.array([4, 4], dtype=np.int32),
        "particle_uid": np.array([10, 11], dtype=np.int32),
        "whitewater_age": np.array([0.25], dtype=np.float32),
        "whitewater_life": np.array([2.0], dtype=np.float32),
        "whitewater_type": np.array([0], dtype=np.int32),
    }


def test_attribute_inventory_round_trip_validation():
    inventory = validate_optional_attributes(_payload())
    assert inventory == {
        "surface": ("velocity", "motion"),
        "particles": ("color", "source_id", "uid"),
        "whitewater": ("age", "lifetime", "type"),
    }


def test_legacy_payload_without_attribute_schema_remains_readable():
    assert validate_optional_attributes({
        "positions": np.zeros((1, 3), dtype=np.float32),
        "whitewater": np.zeros((0, 3), dtype=np.float32),
    }) == {}


def test_attribute_lengths_align_with_particles():
    payload = _payload()
    payload["particle_color"] = np.ones((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        validate_optional_attributes(payload)


def test_uid_values_are_unique_and_positive():
    payload = _payload()
    payload["particle_uid"] = np.array([10, 10], dtype=np.int32)
    with pytest.raises(ValueError, match="unique positive"):
        validate_optional_attributes(payload)


def test_nonfinite_vector_attribute_is_rejected():
    payload = _payload()
    payload["surface_velocity"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_optional_attributes(payload)


def test_inventory_order_is_strict():
    payload = _payload()
    payload["attribute_inventory_json"] = np.array(json.dumps({
        "surface": ["motion", "velocity"],
        "particles": ["color", "source_id", "uid"],
        "whitewater": ["age", "lifetime", "type"],
    }))
    with pytest.raises(ValueError, match="schema order"):
        validate_optional_attributes(payload)


def test_phase3_schema_remains_readable_without_type():
    payload = _payload()
    payload["attribute_schema_version"] = np.array(1, dtype=np.int32)
    payload["attribute_inventory_json"] = np.array(json.dumps({
        "surface": ["velocity", "motion"],
        "particles": ["color", "source_id", "uid"],
        "whitewater": ["age", "lifetime"],
    }, sort_keys=True, separators=(",", ":")))
    payload.pop("whitewater_type")
    inventory = validate_optional_attributes(payload)
    assert inventory["whitewater"] == ("age", "lifetime")


def test_phase4_type_rejects_reserved_values():
    payload = _payload()
    payload["whitewater_type"] = np.array([3], dtype=np.int32)
    with pytest.raises(ValueError, match="whitewater type"):
        validate_optional_attributes(payload)


def _authoritative_payload(*, schema_version=2, include_type=True):
    payload = {
        "positions": np.zeros((2, 3), dtype=np.float32),
        "velocities": np.zeros((2, 3), dtype=np.float32),
        "particle_age": np.zeros(2, dtype=np.float32),
        "particle_life": np.ones(2, dtype=np.float32),
        "particle_temperature": np.zeros(2, dtype=np.float32),
        "particle_viscosity": np.zeros(2, dtype=np.float32),
        "particle_density": np.ones(2, dtype=np.float32),
        "particle_time_offset": np.zeros(2, dtype=np.float32),
        "whitewater": np.zeros((3, 3), dtype=np.float32),
        "whitewater_velocities": np.zeros((3, 3), dtype=np.float32),
        "whitewater_age": np.zeros(3, dtype=np.float32),
        "whitewater_life": np.ones(3, dtype=np.float32),
        "attribute_schema_version": np.array(schema_version, dtype=np.int32),
    }
    if include_type:
        payload["whitewater_type"] = np.array([0, 1, 2], dtype=np.int32)
    return payload


def test_phase4_authoritative_payload_requires_type_alignment():
    payload = _authoritative_payload(include_type=False)
    with pytest.raises(ValueError, match="whitewater_type"):
        authoritative_payload_counts(payload)


def test_phase3_authoritative_payload_remains_readable_without_type():
    payload = _authoritative_payload(schema_version=1, include_type=False)
    counts = authoritative_payload_counts(payload)
    assert counts["whitewater"]["whitewater"] == 3
    assert "whitewater_type" not in counts["whitewater"]
