"""The installed GPU verifier must admit the lazy owned runtime before importing kernels."""
import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("already_on_path", [False, True])
def test_gpu_verifier_prefers_provisioned_warp_over_ambient(monkeypatch, tmp_path, already_on_path):
    extensions = tmp_path / "extensions"
    owned = extensions / ".user" / "fluidsim_ci" / "fluidsim_warp" / ".local"
    ambient = tmp_path / "ambient-site"
    for site, version in ((ambient, "1.18.0"), (owned, "1.19.0")):
        package = site / "warp"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"__version__ = {version!r}\n", encoding="utf-8")
    monkeypatch.setenv("BLENDER_USER_EXTENSIONS", str(extensions))
    monkeypatch.syspath_prepend(str(ambient))
    if already_on_path:
        monkeypatch.setattr(sys, "path", [*sys.path, str(owned)])
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    try:
        runpy.run_path(str(ROOT / "tools/release/gpu_regression_blender.py"))
        # Physical Python packages exercise import precedence without initializing Warp/CUDA.
        warp = importlib.import_module("warp")
        assert warp.__version__ == "1.19.0"
        assert Path(warp.__file__).resolve() == (owned / "warp" / "__init__.py").resolve()
        assert sys.path[0] == str(owned)
    finally:
        sys.modules.pop("warp", None)


def test_gpu_verifier_admits_runtime_before_loading_validation(monkeypatch):
    # Blender is unavailable in host pytest; only its host API is substituted.
    monkeypatch.setitem(__import__("sys").modules, "bpy", SimpleNamespace(
        app=SimpleNamespace(version=(5, 2, 0))))
    namespace = runpy.run_path(str(ROOT / "tools/release/gpu_regression_blender.py"))
    main = namespace["main"]
    state = {"admitted": False}
    probe = SimpleNamespace(
        warp_version="1.19.0", available=True,
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
