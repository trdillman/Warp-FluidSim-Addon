"""Warp device choice shared by pytest and the Blender/headless scripts.

Pytest-free on purpose: Blender acceptance scripts import it by putting this
directory on ``sys.path``. The pytest ``cuda_device`` fixture adds the
fail-or-skip policy on top (see ``conftest.select_test_device``).
"""

import os

TEST_DEVICE_ENV = "FLUIDSIM_TEST_DEVICE"
DEFAULT_CUDA_DEVICE = "cuda:0"


def script_device(environ=os.environ) -> str:
    """``FLUIDSIM_TEST_DEVICE`` when set (e.g. ``cuda:1``), else ``cuda:0``."""
    return str(environ.get(TEST_DEVICE_ENV, "")).strip() or DEFAULT_CUDA_DEVICE
