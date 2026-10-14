from __future__ import annotations

from fluidsim_testpkg.runtime import warp_runtime


class FakeDevice:
    def __init__(self, alias: str, name: str, is_cuda: bool, total_memory: int):
        self.alias = alias
        self.name = name
        self.is_cuda = is_cuda
        self.total_memory = total_memory


class FakeWarp:
    __version__ = "1.18.0"
    __file__ = "/extension/site-packages/warp/__init__.py"

    @staticmethod
    def init():
        return None

    @staticmethod
    def get_devices():
        return [
            FakeDevice("cpu", "CPU", False, 0),
            FakeDevice("cuda:0", "RTX Test", True, 24 * 1024**3),
        ]

    @staticmethod
    def get_cuda_driver_version():
        return 12090

    @staticmethod
    def get_cuda_toolkit_version():
        # Warp 1.17 reports CUDA versions as (major, minor) tuples.
        return (12, 9)


def test_cached_probe_repairs_blender_gui_streams_that_disappear(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    assert warp_runtime.probe_runtime(force=True).available is True

    monkeypatch.setattr(warp_runtime.sys, "stdout", None)
    monkeypatch.setattr(warp_runtime.sys, "stderr", None)

    assert warp_runtime.probe_runtime().available is True

    assert warp_runtime.sys.stdout.write("Warp info\n") == 10
    assert warp_runtime.sys.stderr.write("Warp error\n") == 11
    assert warp_runtime.sys.stdout.closed is False
    assert warp_runtime.sys.stderr.closed is False


def test_probe_reports_cuda_devices(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is True
    assert probe.warp_version == "1.18.0"
    assert probe.module_file == "/extension/site-packages/warp/__init__.py"
    assert probe.cuda_driver_version == 12090
    assert probe.cuda_toolkit_version == 12090
    assert [device.alias for device in probe.cuda_devices] == ["cuda:0"]
    assert probe.cuda_devices[0].name == "RTX Test"
    assert probe.cuda_devices[0].total_memory == 24 * 1024**3


def test_probe_returns_actionable_import_error(monkeypatch):
    def fail():
        raise ModuleNotFoundError("warp")

    monkeypatch.setattr(warp_runtime, "_import_warp", fail)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is False
    assert "Install Warp" in probe.message
    assert "ModuleNotFoundError" in probe.last_error


def test_probe_reports_no_cuda_without_losing_warp_identity(monkeypatch):
    class CpuOnlyWarp(FakeWarp):
        @staticmethod
        def get_devices():
            return [FakeDevice("cpu", "CPU", False, 0)]

        @staticmethod
        def get_cuda_driver_version():
            # New enough for the certified CUDA 13 build: only the device is missing.
            return 13030

    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: CpuOnlyWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is False
    assert probe.warp_version == "1.18.0"
    assert probe.module_file.endswith("warp/__init__.py")
    assert "no CUDA device" in probe.message


def test_selected_device_gate_is_actionable(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert warp_runtime.runtime_status_for_device("cuda:0", probe=probe) == (
        True,
        "Warp 1.18.0 — cuda:0: RTX Test (24.0 GiB)",
    )
    ready, message = warp_runtime.runtime_status_for_device("cuda:1", probe=probe)
    assert ready is False
    assert "selected CUDA device 'cuda:1' is unavailable" in message


def test_cuda_device_items_never_use_an_empty_identifier(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    items = warp_runtime.cuda_device_items(force=True)
    assert items == [("cuda:0", "cuda:0 — RTX Test", "24.0 GiB")]

    def fail():
        raise ModuleNotFoundError("warp")

    monkeypatch.setattr(warp_runtime, "_import_warp", fail)
    items = warp_runtime.cuda_device_items(force=True)
    assert items[0][0] == "__unavailable__"
    assert "Install Warp" in items[0][2]


def test_probe_rejects_warp_below_the_floor(monkeypatch):
    class OldWarp(FakeWarp):
        __version__ = "1.16.0"

    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: OldWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is False
    assert "requires NVIDIA Warp 1.17.0 or newer 1.x" in probe.message
    assert "share the top-level warp module" in probe.message
    assert "restart Blender" in probe.message


def test_probe_loads_newer_untested_warp_with_a_warning(monkeypatch):
    class NewerWarp(FakeWarp):
        __version__ = "1.19.2"

    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: NewerWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is True
    assert "newer than the certified 1.18.0" in probe.warning
    ready, status = warp_runtime.runtime_status_for_device("cuda:0", probe=probe)
    assert ready is True
    assert status.endswith("(untested Warp version)")


import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.18.0", "certified"),
        ("1.18.0+cu13", "certified"),
        ("1.17.0", "older"),
        ("1.17.0+cu12", "older"),
        ("1.17.3", "older"),
        ("1.18.3", "untested"),
        ("1.20.0.dev1", "untested"),
        ("1.16.9", "unsupported"),
        ("2.0.0", "unsupported"),
        ("0.17.0", "unsupported"),
        ("unknown", "unsupported"),
        ("", "unsupported"),
    ],
)
def test_warp_version_support_range(version, expected):
    assert warp_runtime.warp_version_support(version)[0] == expected


def test_cuda_versions_normalize_from_tuples_and_ints():
    assert warp_runtime._cuda_version_int((12, 9)) == 12090
    assert warp_runtime._cuda_version_int((13, 3)) == 13030
    assert warp_runtime._cuda_version_int(12090) == 12090
