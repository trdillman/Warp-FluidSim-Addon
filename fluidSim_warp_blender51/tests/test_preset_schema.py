from __future__ import annotations

import pytest
from fluidsim_testpkg.presets.schema import PresetV1


def _payload(**overrides):
    payload = {
        "schema_version": 1,
        "id": "preview_water",
        "name": "Preview Water",
        "description": "Fast preview",
        "expected_scale": "metres",
        "performance_cost": "Low",
        "requires_reset": True,
        "settings": {"voxel_size": 0.1, "flip_ratio": 0.95},
        "source_version": "1.0.249",
    }
    payload.update(overrides)
    return payload


def test_preset_json_round_trip():
    preset = PresetV1.from_dict(_payload())
    assert PresetV1.from_dict(preset.to_dict()) == preset


def test_unknown_setting_is_rejected():
    with pytest.raises(ValueError, match="unsupported setting"):
        PresetV1.from_dict(_payload(settings={"arbitrary_rna_path": 1}))


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError, match="invalid preset keys"):
        PresetV1.from_dict(_payload(script="dangerous"))


@pytest.mark.parametrize("value", [True, "0.1", float("nan"), 0.001])
def test_voxel_size_type_range_and_finite_contract(value):
    with pytest.raises(ValueError):
        PresetV1.from_dict(_payload(settings={"voxel_size": value}))


def test_enum_identifier_is_strict():
    with pytest.raises(ValueError, match="must be one of"):
        PresetV1.from_dict(_payload(settings={"sample_type": "APIC"}))


def test_phase4_whitewater_controls_are_allow_listed():
    preset = PresetV1.from_dict({
        "schema_version": 1,
        "id": "typed_whitewater",
        "name": "Typed Whitewater",
        "description": "Deterministic typed output",
        "expected_scale": "metres",
        "performance_cost": "high",
        "requires_reset": True,
        "settings": {
            "whitewater_enabled": True,
            "ww_foam_band": 1.0,
            "ww_spray_min_speed": 0.9,
            "ww_type_hysteresis": 0.25,
            "ww_foam_life_multiplier": 1.2,
            "ww_bubble_buoyancy": 1.0,
            "ww_spray_gravity_scale": 1.0,
            "whitewater_output_mode": "BOTH",
        },
    })
    assert preset.settings["whitewater_output_mode"] == "BOTH"


def test_invalid_phase4_output_mode_is_rejected():
    payload = {
        "schema_version": 1,
        "id": "bad_mode",
        "name": "Bad Mode",
        "description": "Bad",
        "expected_scale": "metres",
        "performance_cost": "low",
        "requires_reset": True,
        "settings": {"whitewater_output_mode": "TRIPLICATE_GPU_POOLS"},
    }
    try:
        PresetV1.from_dict(payload)
    except ValueError as exc:
        assert "whitewater_output_mode" in str(exc)
    else:
        raise AssertionError("invalid output mode was accepted")
