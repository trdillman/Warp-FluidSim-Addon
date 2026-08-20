from __future__ import annotations

import numpy as np

from fluidsim_testpkg.blender.output_publication import (
    BUBBLES_NAME,
    FOAM_NAME,
    SPRAY_NAME,
    WHITEWATER_NAME,
    WHITEWATER_OUTPUT_NAMES,
    _subset_aligned_attributes,
    _whitewater_mode_flags,
)
from fluidsim_testpkg.sim.whitewater_types import (
    LEGACY_UNCLASSIFIED,
    WhitewaterType,
    partition_type_indices,
)


def test_output_names_are_stable_and_unique():
    assert WHITEWATER_OUTPUT_NAMES == (
        WHITEWATER_NAME, FOAM_NAME, BUBBLES_NAME, SPRAY_NAME)
    assert len(set(WHITEWATER_OUTPUT_NAMES)) == 4


def test_output_mode_never_implies_separate_simulation_pools():
    assert _whitewater_mode_flags("COMBINED") == (True, False)
    assert _whitewater_mode_flags("SEPARATE") == (False, True)
    assert _whitewater_mode_flags("BOTH") == (True, True)
    # Unknown persisted values fail safely to the backward-compatible preview.
    assert _whitewater_mode_flags("UNKNOWN") == (True, False)


def test_typed_partition_preserves_alignment_and_excludes_legacy_points():
    types = np.array([2, 0, LEGACY_UNCLASSIFIED, 1, 0], dtype=np.int32)
    positions = np.arange(15, dtype=np.float32).reshape((5, 3))
    velocity = positions * 0.5
    attrs = {"type": types, "velocity": velocity, "constant": np.array(7)}
    partitions = partition_type_indices(types, allow_unclassified=True)
    np.testing.assert_array_equal(partitions[WhitewaterType.FOAM], [1, 4])
    np.testing.assert_array_equal(partitions[WhitewaterType.BUBBLE], [3])
    np.testing.assert_array_equal(partitions[WhitewaterType.SPRAY], [0])
    np.testing.assert_array_equal(partitions[LEGACY_UNCLASSIFIED], [2])
    foam = _subset_aligned_attributes(attrs, partitions[WhitewaterType.FOAM], 5)
    np.testing.assert_array_equal(foam["type"], [0, 0])
    np.testing.assert_array_equal(foam["velocity"], velocity[[1, 4]])
    assert "constant" not in foam
