from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
PACKAGE_NAME = "fluidsim_testpkg"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


@pytest.fixture(scope="session", autouse=True)
def _private_effect_module_directory(tmp_path_factory):
    """Keep generated Custom Effects modules out of the real per-user cache."""
    from fluidsim_testpkg.sim.effects import source

    directory = tmp_path_factory.mktemp("effect-modules")
    previous = source.default_module_directory
    source.default_module_directory = lambda: directory
    try:
        yield directory
    finally:
        source.default_module_directory = previous
