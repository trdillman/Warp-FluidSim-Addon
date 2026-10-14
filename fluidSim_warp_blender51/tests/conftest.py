from __future__ import annotations

import os
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

# Keep in sync with the repository tests/conftest.py; this suite also runs
# standalone from the add-on folder, so it cannot import that copy.
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


@pytest.fixture
def demo_gallery_module():
    """``blender.demo_gallery``, imported against a stand-in ``bpy`` outside Blender.

    The stand-in and every add-on module imported against it are removed
    afterwards, so later tests never inherit an empty ``bpy``. Same scoping as
    ``_import_demo_gallery`` in ``test_demo_gallery_contract.py``, which needs
    the module at collection time and so cannot use a fixture.
    """
    import importlib

    package = "fluidSim_warp_blender51"
    try:
        import bpy  # noqa: F401  (inside Blender the real module is used)
    except ImportError:
        pass
    else:
        yield importlib.import_module(f"{package}.blender.demo_gallery")
        return

    def in_package(name: str) -> bool:
        return name == package or name.startswith(f"{package}.")

    already_loaded = {name for name in sys.modules if in_package(name)}
    sys.modules["bpy"] = ModuleType("bpy")
    try:
        yield importlib.import_module(f"{package}.blender.demo_gallery")
    finally:
        sys.modules.pop("bpy", None)
        for name in [n for n in sys.modules if in_package(n) and n not in already_loaded]:
            module = sys.modules.pop(name)
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and getattr(parent, child, None) is module:
                delattr(parent, child)


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
