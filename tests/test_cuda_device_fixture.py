"""Regression gate for the shared ``cuda_device`` fixture (no GPU required).

Both conftests carry the fixture because the add-on suite also runs
standalone; these tests pin its behaviour and keep the two copies identical.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFTESTS = {
    "repo": REPO_ROOT / "tests" / "conftest.py",
    "addon": REPO_ROOT / "fluidSim_warp_blender51" / "tests" / "conftest.py",
}
SHARED_NAMES = (
    "TEST_DEVICE_ENV",
    "DEFAULT_CUDA_DEVICE",
    "select_test_device",
    "cuda_device",
    "pytest_collection_modifyitems",
)


def _load(key):
    spec = importlib.util.spec_from_file_location(
        f"_fluidsim_conftest_under_test_{key}", CONFTESTS[key]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeWarp:
    def __init__(self, *, cuda, devices=("cpu", "cuda:0")):
        self.cuda = cuda
        self.devices = set(devices)
        self.initialised = False

    def init(self):
        self.initialised = True

    def is_cuda_available(self):
        return self.cuda

    def get_device(self, name):
        if name not in self.devices:
            raise RuntimeError(f"Invalid device identifier: {name}")
        return name


@pytest.fixture(params=sorted(CONFTESTS))
def conftest_module(request):
    return _load(request.param)


def test_defaults_to_first_cuda_device(conftest_module):
    wp = _FakeWarp(cuda=True)
    assert conftest_module.select_test_device(wp, environ={}) == "cuda:0"
    assert wp.initialised


def test_skips_cleanly_without_cuda(conftest_module):
    with pytest.raises(pytest.skip.Exception, match="CUDA not available"):
        conftest_module.select_test_device(_FakeWarp(cuda=False), environ={})


def test_override_selects_named_device_even_without_cuda(conftest_module):
    env = {"FLUIDSIM_TEST_DEVICE": " cpu "}
    assert conftest_module.select_test_device(_FakeWarp(cuda=False), environ=env) == "cpu"


def test_unusable_override_fails_loudly(conftest_module):
    env = {"FLUIDSIM_TEST_DEVICE": "cuda:7"}
    with pytest.raises(pytest.fail.Exception, match="cuda:7"):
        conftest_module.select_test_device(_FakeWarp(cuda=True), environ=env)


class _FakeItem:
    def __init__(self, fixturenames, markers=()):
        self.fixturenames = fixturenames
        self.markers = list(markers)

    def get_closest_marker(self, name):
        return next((m for m in self.markers if m.name == name), None)

    def add_marker(self, marker):
        self.markers.append(marker.mark)


def test_collection_hook_marks_cuda_device_users_once(conftest_module):
    gpu = _FakeItem(["cuda_device"])
    already = _FakeItem(["cuda_device"], [pytest.mark.cuda.mark])
    cpu = _FakeItem(["tmp_path"])
    conftest_module.pytest_collection_modifyitems(None, [gpu, already, cpu])
    assert [m.name for m in gpu.markers] == ["cuda"]
    assert [m.name for m in already.markers] == ["cuda"]
    assert cpu.markers == []


def _shared_segments(path):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    segments = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in SHARED_NAMES:
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name not in SHARED_NAMES:
                continue
        else:
            continue
        decorators = "".join(
            "@" + ast.get_source_segment(text, d) + "\n" for d in getattr(node, "decorator_list", ())
        )
        segments[name] = decorators + ast.get_source_segment(text, node)
    return segments


def test_both_conftest_copies_are_identical():
    repo = _shared_segments(CONFTESTS["repo"])
    addon = _shared_segments(CONFTESTS["addon"])
    assert set(repo) == set(SHARED_NAMES)
    assert repo == addon


_LOCAL_FIXTURE = re.compile(r"^\s*def cuda_device\(([^)]*)\)", re.MULTILINE)


def test_test_modules_do_not_redefine_the_shared_fixture():
    """A module may narrow the shared fixture (take it as a parameter) but not replace it."""
    offenders = []
    for directory in (REPO_ROOT / "tests", REPO_ROOT / "fluidSim_warp_blender51" / "tests"):
        for path in sorted(directory.glob("test_*.py")):
            for match in _LOCAL_FIXTURE.finditer(path.read_text(encoding="utf-8")):
                if "cuda_device" not in match.group(1):
                    offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert not offenders, f"local cuda_device fixtures bypass the shared one: {offenders}"
