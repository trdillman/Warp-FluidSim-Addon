"""The installed probe must bootstrap only the extension-owned, provisioned Warp."""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROBE = Path(__file__).resolve().parents[1] / "acceptance/blender_saved_preview_acceptance.py"


def _bootstrap(tmp_path, monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        monkeypatch.delenv(name, raising=False)
    extensions = tmp_path / "extensions"
    installed = extensions / "user_default/fluidsim_warp"
    owned = extensions / ".user/user_default/fluidsim_warp/.local"
    (owned / "warp").mkdir(parents=True)
    (owned / "warp/__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("BLENDER_USER_EXTENSIONS", str(extensions))
    module = "bl_ext.user_default.fluidsim_warp"
    addon = SimpleNamespace(__file__=str(installed / "__init__.py"))
    warp = SimpleNamespace(__file__=str(owned / "warp/__init__.py"))
    runtime_sys = SimpleNamespace(path=["bundled-warp-site"], modules={})
    calls = []

    def enable(name, **kwargs):
        assert name == module
        assert kwargs == {"default_set": True, "persistent": True}
        assert runtime_sys.path[0] == str(owned.resolve())
        calls.append("enable")

    def import_module(name):
        assert calls, "Registration must follow owned-site bootstrap"
        calls.append(name)
        return warp if name == "warp" else addon

    def extension_path_user(name, *, path, create):
        assert (name, path, create) == (module, ".local", False)
        return str(owned)

    monkeypatch.setitem(sys.modules, "bpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "addon_utils", SimpleNamespace())
    spec = importlib.util.spec_from_file_location("saved_preview_installed_probe", PROBE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.sys = runtime_sys
    probe.bpy = SimpleNamespace(utils=SimpleNamespace(extension_path_user=extension_path_user))
    probe.addon_utils = SimpleNamespace(enable=enable)
    probe.importlib = SimpleNamespace(import_module=import_module)
    return SimpleNamespace(
        run=lambda: probe._enable_installed_runtime(module, installed.resolve()),
        addon=addon, warp=warp, owned=owned, sys=runtime_sys, calls=calls, probe=probe,
    )


def test_installed_probe_bootstraps_owned_warp_before_registration(tmp_path, monkeypatch):
    case = _bootstrap(tmp_path, monkeypatch)
    assert case.run() == (case.addon, case.warp, case.owned.resolve())
    assert case.calls == ["enable", "bl_ext.user_default.fluidsim_warp", "warp"]
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Import) and any(alias.name == "warp" for alias in node.names)
                   for node in tree.body), "Never preload Warp at module import"


@pytest.mark.parametrize("name", ["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"])
def test_installed_probe_refuses_developer_overrides(tmp_path, monkeypatch, name):
    case = _bootstrap(tmp_path, monkeypatch)
    monkeypatch.setenv(name, "development-overlay")
    with pytest.raises(AssertionError, match="developer Python import overrides"):
        case.run()
    assert not case.calls


def test_installed_probe_refuses_preloaded_warp(tmp_path, monkeypatch):
    case = _bootstrap(tmp_path, monkeypatch)
    case.sys.modules["warp"] = case.warp
    with pytest.raises(AssertionError, match="preloaded Warp"):
        case.run()
    assert not case.calls


def test_installed_probe_requires_owned_wheel_provisioning(tmp_path, monkeypatch):
    case = _bootstrap(tmp_path, monkeypatch)
    (case.owned / "warp/__init__.py").unlink()
    with pytest.raises(AssertionError, match="Provision the locked wheel"):
        case.run()
    assert not case.calls


def test_installed_probe_rejects_foreign_warp_after_registration(tmp_path, monkeypatch):
    case = _bootstrap(tmp_path, monkeypatch)
    case.warp.__file__ = str(tmp_path / "bundled-warp-1.18/warp/__init__.py")
    with pytest.raises(AssertionError):
        case.run()


def test_solver_device_selector_is_resolved_by_warp(tmp_path, monkeypatch):
    case = _bootstrap(tmp_path, monkeypatch)
    selectors = []

    def get_device(selector):
        selectors.append(selector)
        return SimpleNamespace(is_cuda=True)

    session = SimpleNamespace(active=True, solver=SimpleNamespace(device="cuda:0"))
    case.probe._require_cuda_session(session, SimpleNamespace(get_device=get_device))
    assert selectors == ["cuda:0"]


@pytest.mark.parametrize("active,is_cuda", [(False, True), (True, False)])
def test_solver_device_guard_refuses_inactive_or_cpu_session(tmp_path, monkeypatch, active, is_cuda):
    case = _bootstrap(tmp_path, monkeypatch)
    session = SimpleNamespace(active=active, solver=SimpleNamespace(device="cpu"))
    with pytest.raises(AssertionError, match="CUDA solver required"):
        case.probe._require_cuda_session(
            session, SimpleNamespace(get_device=lambda selector: SimpleNamespace(is_cuda=is_cuda)))


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell wrapper contract")
@pytest.mark.parametrize("stage", ["Produce", "Reopen"])
def test_installed_wrapper_preserves_private_repository_profile(tmp_path, stage):
    repo = PROBE.parents[2]
    wrapper = repo / "tools/gate/run_saved_preview_installed_probe.ps1"
    profile = tmp_path / "private-profile"
    installed = profile / "extensions/ci-repo/fluidsim_warp"
    installed.mkdir(parents=True)
    preferences = profile / "config/userpref.blend"
    preferences.parent.mkdir()
    preferences.write_bytes(b"private repository preferences must survive")
    archive = tmp_path / "candidate.zip"
    archive.write_bytes(b"capture-only test, never installed")
    input_blend = tmp_path / "input.blend"
    input_blend.write_bytes(b"capture-only test, never opened")
    receipt = tmp_path / "arguments.json"
    capture = tmp_path / "capture.ps1"
    capture.write_text(
        "$result = @{ arguments = @($args); config = $env:BLENDER_USER_CONFIG; "
        "extensions = $env:BLENDER_USER_EXTENSIONS; resources = $env:BLENDER_USER_RESOURCES; "
        "pythonpath = $env:PYTHONPATH; pythonhome = $env:PYTHONHOME; "
        "pythonstartup = $env:PYTHONSTARTUP }\n"
        "$result | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath "
        "$env:FLUIDSIM_WRAPPER_RECEIPT -Encoding utf8\n"
        "$global:LASTEXITCODE = 0\n",
        encoding="utf-8",
    )
    env = dict(os.environ, FLUIDSIM_WRAPPER_RECEIPT=str(receipt), PYTHONPATH="developer-overlay",
               PYTHONHOME="developer-home", PYTHONSTARTUP="developer-startup",
               BGATE_LEASE_ID="capture-only", BGATE_LEASE_CLASS="exclusive",
               CUDA_VISIBLE_DEVICES="")  # The child only executes capture.ps1.
    result = subprocess.run(
        [shutil.which("pwsh"), "-NoProfile", "-File", str(wrapper), "-Stage", stage,
         "-Blender", str(capture),
         "-AddonModule", "bl_ext.fluidsim_ci.fluidsim_warp", "-InstalledRoot", str(installed),
         "-Archive", str(archive), "-ProfileRoot", str(profile),
         "-InputBlend", str(input_blend), "-OutputRoot", str(tmp_path / "output")],
        cwd=repo, env=env, text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(receipt.read_text(encoding="utf-8-sig"))
    assert "--factory-startup" not in observed["arguments"]
    assert "--background" in observed["arguments"]
    assert observed["arguments"][observed["arguments"].index("--phase") + 1] == stage.lower()
    assert Path(observed["config"]) == preferences.parent
    assert Path(observed["extensions"]) == profile / "extensions"
    assert Path(observed["resources"]) == profile
    assert all(not observed[key] for key in ("pythonpath", "pythonhome", "pythonstartup"))
    assert preferences.read_bytes() == b"private repository preferences must survive"
