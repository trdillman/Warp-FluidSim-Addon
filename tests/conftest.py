"""Repository-wide pytest runtime isolation."""

from __future__ import annotations

import os
import tempfile

import pytest

# Keep in sync with fluidSim_warp_blender51/tests/conftest.py (that suite also
# runs standalone from the add-on folder).  tests/test_cuda_device_fixture.py
# asserts both copies resolve devices identically, and that the scripts'
# ``_test_device.script_device`` honours the same override.
TEST_DEVICE_ENV = "FLUIDSIM_TEST_DEVICE"
DEFAULT_CUDA_DEVICE = "cuda:0"


def select_test_device(wp, environ=os.environ) -> str:
    """Return the Warp device GPU tests should use.

    ``FLUIDSIM_TEST_DEVICE`` (e.g. ``cuda:1`` or ``cpu``) wins and must name a
    usable device, otherwise the test fails loudly.  Without an override the
    default CUDA device is used, and the test is skipped when CUDA is absent.
    """
    wp.init()
    override = str(environ.get(TEST_DEVICE_ENV, "")).strip()
    if override:
        try:
            wp.get_device(override)
        except Exception as exc:  # noqa: BLE001 - reported as a test failure
            pytest.fail(
                f"{TEST_DEVICE_ENV}={override!r} is not a usable Warp device: {exc}",
                pytrace=False,
            )
        return override
    if not wp.is_cuda_available():
        pytest.skip(f"CUDA not available (set {TEST_DEVICE_ENV} to choose a device)")
    return DEFAULT_CUDA_DEVICE


@pytest.fixture(scope="session")
def cuda_device():
    """Shared Warp device for GPU tests; see :func:`select_test_device`."""
    wp = pytest.importorskip("warp")
    return select_test_device(wp)


def pytest_collection_modifyitems(config, items):
    """Every test that requests ``cuda_device`` carries the ``cuda`` marker."""
    for item in items:
        if "cuda_device" in getattr(item, "fixturenames", ()) and (
            item.get_closest_marker("cuda") is None
        ):
            item.add_marker(pytest.mark.cuda)


@pytest.fixture(scope="session", autouse=True)
def writable_warp_kernel_cache(tmp_path_factory):
    """Keep optional Warp JIT output inside a writable test-owned directory."""
    try:
        import warp as wp
    except ImportError:
        yield
        return
    previous_cache = wp.config.kernel_cache_dir
    previous_tempdir = tempfile.tempdir
    previous_env = {
        name: os.environ.get(name)
        for name in ("TEMP", "TMP", "CUDA_CACHE_PATH")
    }
    runtime_temp = str(tmp_path_factory.mktemp("warp-runtime-temp"))
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp-kernel-cache"))
    tempfile.tempdir = runtime_temp
    for name in previous_env:
        os.environ[name] = runtime_temp
    try:
        yield
    finally:
        wp.config.kernel_cache_dir = previous_cache
        tempfile.tempdir = previous_tempdir
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
