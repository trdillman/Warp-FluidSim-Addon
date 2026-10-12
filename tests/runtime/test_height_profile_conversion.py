"""The height profile conversion is bit-identical to the old expression (P17).

``gather_height_source`` converts the Height Profile image on every sync.
The banded conversion must give exactly the bytes the whole-image expression
gave, for any size (including sizes that are not a multiple of the band),
any scale, and non-finite pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from fluidSim_warp_blender51.blender.scene_inputs import _height_profile


def _old(pixels, width, height, scale):
    rgba = pixels.reshape(height, width, 4)
    return np.ascontiguousarray(
        (rgba[..., 0] * 0.2126 + rgba[..., 1] * 0.7152
         + rgba[..., 2] * 0.0722).T
        * float(scale),
        dtype=np.float32)


@pytest.mark.parametrize("width,height", [
    (2, 2), (3, 5), (17, 16), (16, 17), (64, 33), (257, 131), (512, 512)])
@pytest.mark.parametrize("scale", [1.0, 0.37, -2.5, 0.0, 1e-30])
def test_banded_profile_matches_the_whole_image_expression(width, height, scale):
    rng = np.random.default_rng(width * 1000 + height)
    pixels = rng.random(width * height * 4, dtype=np.float32) * 4.0 - 1.0
    expected = _old(pixels, width, height, scale)
    actual = _height_profile(pixels, width, height, scale)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape == (width, height)
    assert actual.flags["C_CONTIGUOUS"]
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_non_finite_pixels_convert_identically():
    pixels = np.random.default_rng(1).random(40 * 24 * 4, dtype=np.float32)
    pixels[::7] = np.nan
    pixels[3::11] = np.inf
    pixels[5::13] = -np.inf
    assert _height_profile(pixels, 40, 24, 0.5).tobytes() == _old(
        pixels, 40, 24, 0.5).tobytes()
