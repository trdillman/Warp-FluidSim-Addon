"""Pure calibration policy for the adhesive-calender whitewater demo."""

from __future__ import annotations

from dataclasses import dataclass


# Churn integrates turbulent speed over time, so it has scene-length units.
# Expressing its trigger and rate in voxel units keeps the response stable when
# the calender builder changes voxel size with particle count.
CALENDER_WW_CHURN_THRESHOLD_VOXELS = 0.25
CALENDER_WW_EMISSION_PER_EXCESS_VOXEL_SECOND = 1.0

# FLUID_FLOW applies (target_velocity * strength - velocity * flow), making
# strength / flow the steady target speed for the authored roll proxy.
CALENDER_ROLL_FLOW_STRENGTH = 0.08
CALENDER_ROLL_FLOW_COUPLING = 0.8
CALENDER_WW_MIN_SPEED_FRACTION_OF_ROLL_TARGET = 0.5

# The secondary phase must be visible while remaining subordinate to liquid.
CALENDER_WW_MIN_PEAK_FRACTION = 0.001
CALENDER_WW_MAX_PEAK_FRACTION = 0.05

# Three fresh production-density reopens first emitted at frame 30. This keeps
# slightly more than 2x timing margin without hiding a broken setup behind the
# full 250-frame simulation.
CALENDER_WW_ACTIVATION_DEADLINE = 64


@dataclass(frozen=True)
class CalenderWhitewaterCalibration:
    churn_threshold: float
    emission_rate: float
    min_speed: float
    roll_target_speed: float


def calender_whitewater_calibration(
    voxel_size: float,
    *,
    roll_strength: float = CALENDER_ROLL_FLOW_STRENGTH,
    roll_flow: float = CALENDER_ROLL_FLOW_COUPLING,
) -> CalenderWhitewaterCalibration:
    """Resolve the dimensionless calender policy into scene-unit settings."""
    voxel_size = float(voxel_size)
    roll_strength = float(roll_strength)
    roll_flow = float(roll_flow)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if roll_strength <= 0.0:
        raise ValueError("roll_strength must be positive")
    if roll_flow <= 0.0:
        raise ValueError("roll_flow must be positive")

    roll_target_speed = roll_strength / roll_flow
    return CalenderWhitewaterCalibration(
        churn_threshold=(
            CALENDER_WW_CHURN_THRESHOLD_VOXELS * voxel_size),
        emission_rate=(
            CALENDER_WW_EMISSION_PER_EXCESS_VOXEL_SECOND / voxel_size),
        min_speed=(
            CALENDER_WW_MIN_SPEED_FRACTION_OF_ROLL_TARGET
            * roll_target_speed),
        roll_target_speed=roll_target_speed,
    )


def calender_whitewater_population_is_valid(
    peak_whitewater: int,
    liquid_particles: int,
) -> bool:
    """Return whether whitewater is visible but subordinate to the liquid."""
    if liquid_particles <= 0 or peak_whitewater < 0:
        return False
    fraction = peak_whitewater / liquid_particles
    return (
        CALENDER_WW_MIN_PEAK_FRACTION
        <= fraction
        <= CALENDER_WW_MAX_PEAK_FRACTION
    )
