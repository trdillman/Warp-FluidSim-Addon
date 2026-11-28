"""Native two-view proof must compare pixels, not only camera labels."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def harness():
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_async_viewport_acceptance.py"
    spec = importlib.util.spec_from_file_location("async_view_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_view_oracle_rejects_identical_images_with_different_dimensions():
    with pytest.raises(AssertionError, match="different completed pixels"):
        harness().compare_view_pixels(np.ones((80, 100, 4)), np.ones((80, 101, 4)))


def test_two_view_oracle_reports_actual_rgb_change_on_common_grid():
    first, second = np.ones((80, 100, 4)), np.ones((80, 101, 4))
    first[:, :50, :3] = 0
    assert harness().compare_view_pixels(first, second) == pytest.approx(0.5)
