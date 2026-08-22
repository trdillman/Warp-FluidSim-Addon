from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "fluidsim_testpkg"

if PACKAGE_NAME not in sys.modules:
    package = ModuleType(PACKAGE_NAME)
    package.__file__ = str(ROOT / "__init__.py")
    package.__package__ = PACKAGE_NAME
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE_NAME] = package


@pytest.fixture(scope="session", autouse=True)
def _writable_warp_kernel_cache(tmp_path_factory):
    """Keep optional Warp compilation tests hermetic and sandbox-writable."""
    try:
        import warp as wp
    except ImportError:
        yield
        return
    previous = wp.config.kernel_cache_dir
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp-kernel-cache"))
    try:
        yield
    finally:
        wp.config.kernel_cache_dir = previous
