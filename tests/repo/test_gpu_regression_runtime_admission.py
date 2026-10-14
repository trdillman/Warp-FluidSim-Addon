"""The installed GPU verifier must admit the lazy owned runtime before importing kernels."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]

def test_gpu_verifier_admits_runtime_before_loading_validation(monkeypatch):
    # Blender is unavailable in host pytest; only its host API is substituted.
    monkeypatch.setitem(__import__("sys").modules, "bpy", SimpleNamespace(
        app=SimpleNamespace(version=(5, 2, 0))))
    namespace = runpy.run_path(str(ROOT / "tools/release/gpu_regression_blender.py"))
    main = namespace["main"]
    state = {"admitted": False}
    probe = SimpleNamespace(
        warp_version="1.18.0", available=True,
        cuda_devices=[SimpleNamespace(alias="cuda:0")])
    def admit(**kwargs):
        state["admitted"] = True
        return probe
    class ValidationReached(Exception):
        pass
    def run_all(**kwargs):
        raise ValidationReached
    def load(name):
        if name.endswith(".runtime.warp_runtime"):
            return SimpleNamespace(probe_runtime=admit)
        if name.endswith(".preferences"):
            return SimpleNamespace(set_configured_device=lambda device: True)
        if name.endswith(".diagnostics.validation"):
            if not state["admitted"]:
                raise ModuleNotFoundError("owned Warp has not been admitted")
            return SimpleNamespace(VALIDATION_TESTS=[], run_all=run_all)
        raise AssertionError(name)
    monkeypatch.setitem(main.__globals__, "importlib", SimpleNamespace(import_module=load))
    monkeypatch.setitem(main.__globals__, "_installed_module", lambda: "bl_ext.fluidsim_ci.fluidsim_warp")
    monkeypatch.setitem(main.__globals__, "_assert_installed_bundled_runtime", lambda *args: None)
    with pytest.raises(ValidationReached):
        main()
