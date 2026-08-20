from __future__ import annotations

from fluidsim_testpkg.blender.setup_wizard import (
    align_bounds,
    directional_padded_bounds,
    gravity_padding,
    padded_bounds,
)


def test_padded_bounds_uses_voxel_units():
    minimum, maximum = padded_bounds(
        minimum=(0.0, 1.0, 2.0),
        maximum=(2.0, 3.0, 4.0),
        voxel_size=0.1,
        padding_voxels=3,
    )
    assert minimum == (-0.3, 0.7, 1.7)
    assert maximum == (2.3, 3.3, 4.3)


def test_align_bounds_rounds_outward_to_voxel_grid():
    minimum, maximum = align_bounds(
        (-0.01, 0.11, 0.2), (1.01, 1.19, 1.21), 0.1
    )
    assert minimum == (-0.1, 0.1, 0.2)
    assert maximum == (1.1, 1.2, 1.3)


def test_default_gravity_has_four_voxels_below_and_twelve_above():
    assert gravity_padding((0.0, 0.0, -9.81)) == ((8, 8, 4), (8, 8, 12))
    minimum, maximum = directional_padded_bounds(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.1, (0.0, 0.0, -9.81)
    )
    assert minimum == (-0.8, -0.8, -0.4)
    assert maximum == (1.8, 1.8, 2.2)


def test_positive_x_gravity_rotates_above_below_padding():
    assert gravity_padding((1.0, 0.0, 0.0)) == ((12, 8, 8), (4, 8, 8))
