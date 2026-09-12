import pytest

from tools.adhesive_whitewater_calibration import (
    CALENDER_WW_EMISSION_PER_EXCESS_VOXEL_SECOND,
    CALENDER_WW_MIN_SPEED_FRACTION_OF_ROLL_TARGET,
    calender_whitewater_calibration,
    calender_whitewater_population_is_valid,
)


def test_calibration_is_resolution_normalized():
    coarse = calender_whitewater_calibration(0.08)
    fine = calender_whitewater_calibration(0.04)

    assert coarse.churn_threshold == pytest.approx(
        2.0 * fine.churn_threshold)
    assert fine.emission_rate == pytest.approx(2.0 * coarse.emission_rate)
    assert coarse.emission_rate * 0.08 == pytest.approx(
        CALENDER_WW_EMISSION_PER_EXCESS_VOXEL_SECOND)
    assert fine.emission_rate * 0.04 == pytest.approx(
        CALENDER_WW_EMISSION_PER_EXCESS_VOXEL_SECOND)


def test_minimum_speed_tracks_fluid_flow_target():
    calibration = calender_whitewater_calibration(
        0.05, roll_strength=0.12, roll_flow=0.8)

    assert calibration.roll_target_speed == pytest.approx(0.15)
    assert calibration.min_speed == pytest.approx(
        CALENDER_WW_MIN_SPEED_FRACTION_OF_ROLL_TARGET * 0.15)


@pytest.mark.parametrize("voxel_size, strength, flow", [
    (0.0, 0.08, 0.8),
    (-0.05, 0.08, 0.8),
    (0.05, 0.0, 0.8),
    (0.05, 0.08, 0.0),
])
def test_calibration_rejects_nonphysical_inputs(voxel_size, strength, flow):
    with pytest.raises(ValueError):
        calender_whitewater_calibration(
            voxel_size, roll_strength=strength, roll_flow=flow)


def test_population_envelope_scales_with_liquid_count():
    assert calender_whitewater_population_is_valid(1_000, 1_000_000)
    assert calender_whitewater_population_is_valid(50_000, 1_000_000)
    assert not calender_whitewater_population_is_valid(999, 1_000_000)
    assert not calender_whitewater_population_is_valid(50_001, 1_000_000)
    assert not calender_whitewater_population_is_valid(1, 0)
