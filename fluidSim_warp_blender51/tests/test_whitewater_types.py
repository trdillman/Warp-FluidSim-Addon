from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fluidsim_testpkg.sim.whitewater_types import (
    LEGACY_UNCLASSIFIED,
    RESERVED_DUST_TYPE,
    WhitewaterType,
    classification_policy,
    classify_sample,
    partition_type_indices,
    transition_sample,
    validate_type_values,
)


def test_stable_whitewater_type_values():
    assert int(WhitewaterType.FOAM) == 0
    assert int(WhitewaterType.BUBBLE) == 1
    assert int(WhitewaterType.SPRAY) == 2
    assert RESERVED_DUST_TYPE == 3
    assert LEGACY_UNCLASSIFIED == -1


def test_above_surface_fast_particle_is_spray():
    result = classify_sample(
        sdf=0.2,
        speed=4.0,
        churn=1.0,
        foam_band=0.1,
        min_spray_speed=1.0,
        min_churn=0.5,
    )
    assert result is WhitewaterType.SPRAY


def test_deep_negative_sdf_is_bubble():
    result = classify_sample(
        sdf=-0.5,
        speed=0.5,
        churn=0.8,
        foam_band=0.1,
        min_spray_speed=1.0,
        min_churn=0.5,
    )
    assert result is WhitewaterType.BUBBLE


def test_near_surface_is_foam():
    result = classify_sample(
        sdf=0.02,
        speed=0.4,
        churn=0.8,
        foam_band=0.1,
        min_spray_speed=1.0,
        min_churn=0.5,
    )
    assert result is WhitewaterType.FOAM


def test_low_churn_above_surface_stays_foam():
    result = classify_sample(
        sdf=0.2,
        speed=4.0,
        churn=0.1,
        foam_band=0.1,
        min_spray_speed=1.0,
        min_churn=0.5,
    )
    assert result is WhitewaterType.FOAM


def test_transition_hysteresis_prevents_type_flicker():
    policy = classification_policy(SimpleNamespace(
        ww_foam_band=1.0,
        voxel_size=0.1,
        ww_spray_min_speed=1.0,
        ww_type_hysteresis=0.25,
        ww_churn_threshold=0.5,
    ))
    assert policy.foam_band == pytest.approx(0.1)
    assert transition_sample(
        WhitewaterType.FOAM, sdf=0.13, speed=2.0, policy=policy
    ) is WhitewaterType.SPRAY
    # Still above the lower exit surface and speed thresholds: stay spray.
    assert transition_sample(
        WhitewaterType.SPRAY, sdf=0.09, speed=1.0, policy=policy
    ) is WhitewaterType.SPRAY
    # Re-entering the inner band transitions back to foam.
    assert transition_sample(
        WhitewaterType.SPRAY, sdf=0.07, speed=1.0, policy=policy
    ) is WhitewaterType.FOAM
    assert transition_sample(
        WhitewaterType.FOAM, sdf=-0.13, speed=0.2, policy=policy
    ) is WhitewaterType.BUBBLE
    # A bubble does not flicker at the center threshold.
    assert transition_sample(
        WhitewaterType.BUBBLE, sdf=-0.09, speed=0.2, policy=policy
    ) is WhitewaterType.BUBBLE
    assert transition_sample(
        WhitewaterType.BUBBLE, sdf=-0.07, speed=0.2, policy=policy
    ) is WhitewaterType.FOAM


def test_partition_preserves_combined_order_and_legacy_unclassified():
    values = np.array([0, 2, -1, 1, 0, 2], dtype=np.int32)
    partitions = partition_type_indices(values, allow_unclassified=True)
    np.testing.assert_array_equal(partitions[WhitewaterType.FOAM], [0, 4])
    np.testing.assert_array_equal(partitions[WhitewaterType.BUBBLE], [3])
    np.testing.assert_array_equal(partitions[WhitewaterType.SPRAY], [1, 5])
    np.testing.assert_array_equal(partitions[LEGACY_UNCLASSIFIED], [2])


def test_type_validation_fails_closed_on_reserved_or_unknown_values():
    np.testing.assert_array_equal(
        validate_type_values(np.array([0, 1, 2], dtype=np.int64)),
        np.array([0, 1, 2], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="whitewater type"):
        validate_type_values(np.array([0, RESERVED_DUST_TYPE], dtype=np.int32))
    with pytest.raises(ValueError, match="whitewater type"):
        validate_type_values(np.array([LEGACY_UNCLASSIFIED], dtype=np.int32))
    np.testing.assert_array_equal(
        validate_type_values(
            np.array([LEGACY_UNCLASSIFIED, 0], dtype=np.int32),
            allow_unclassified=True,
        ),
        np.array([LEGACY_UNCLASSIFIED, 0], dtype=np.int32),
    )
