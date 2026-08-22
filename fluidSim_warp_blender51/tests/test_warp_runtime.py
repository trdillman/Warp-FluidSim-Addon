from __future__ import annotations

from fluidsim_testpkg.runtime import warp_runtime


class FakeDevice:
    def __init__(self, alias: str, name: str, is_cuda: bool, total_memory: int):
        self.alias = alias
        self.name = name
        self.is_cuda = is_cuda
        self.total_memory = total_memory


class FakeWarp:
    __version__ = "1.16.0"
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
        return 12090


def test_probe_reports_cuda_devices(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is True
    assert probe.warp_version == "1.16.0"
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
    assert "external Warp dependency" in probe.message
    assert "ModuleNotFoundError" in probe.last_error


def test_probe_reports_no_cuda_without_losing_warp_identity(monkeypatch):
    class CpuOnlyWarp(FakeWarp):
        @staticmethod
        def get_devices():
            return [FakeDevice("cpu", "CPU", False, 0)]

    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: CpuOnlyWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is False
    assert probe.warp_version == "1.16.0"
    assert probe.module_file.endswith("warp/__init__.py")
    assert "no CUDA device" in probe.message


def test_selected_device_gate_is_actionable(monkeypatch):
    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: FakeWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert warp_runtime.runtime_status_for_device("cuda:0", probe=probe) == (
        True,
        "Warp 1.16.0 — cuda:0: RTX Test (24.0 GiB)",
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
    assert "external Warp dependency" in items[0][2]


def test_probe_rejects_other_warp_versions(monkeypatch):
    class WrongWarp(FakeWarp):
        __version__ = "1.15.0"

    monkeypatch.setattr(warp_runtime, "_import_warp", lambda: WrongWarp)
    probe = warp_runtime.probe_runtime(force=True)
    assert probe.available is False
    assert "requires external Warp 1.16.0" in probe.message
