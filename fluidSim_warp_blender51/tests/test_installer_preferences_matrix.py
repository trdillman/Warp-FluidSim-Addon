"""Host controls for the installed native Preferences acceptance matrix."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tests/acceptance/run_installer_preferences_matrix.py"


def _matrix():
    assert SCRIPT.is_file(), "installed Preferences matrix runner is missing"
    spec = importlib.util.spec_from_file_location("installer_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_covers_all_seven_required_scenarios():
    matrix = _matrix()
    assert set(matrix.SCENARIOS) == {
        "no-warp", "good-owned", "damaged-owned", "foreign-preloaded",
        "offline-local", "cancel", "repeated-clicks",
    }


def test_profile_guard_rejects_consumer_paths_and_symlink_escapes(tmp_path):
    matrix = _matrix()
    profile = tmp_path / "owned"
    environment = matrix.profile_environment(profile)
    matrix.require_owned_profile(profile, environment)
    changed = dict(environment, BLENDER_USER_CONFIG=str(tmp_path / "consumer"))
    with pytest.raises(ValueError, match="BLENDER_USER_CONFIG"):
        matrix.require_owned_profile(profile, changed)
    (profile / "config").rmdir()
    (profile / "config").symlink_to(tmp_path / "consumer", target_is_directory=True)
    with pytest.raises(ValueError, match="BLENDER_USER_CONFIG"):
        matrix.require_owned_profile(profile, environment)


def test_profile_environment_is_absolute_offline_and_scrubs_imports(tmp_path, monkeypatch):
    matrix = _matrix()
    monkeypatch.setenv("PYTHONPATH", "/developer/warp")
    monkeypatch.setenv("PYTHONHOME", "/developer/python")
    monkeypatch.setenv("BLENDER_USER_EXTENSIONS", "/consumer/extensions")
    env = matrix.profile_environment(tmp_path / "private")
    for key in matrix.PROFILE_PATHS:
        assert Path(env[key]).is_absolute()
        assert Path(env[key]).is_relative_to(tmp_path)
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize("scenario", ["no-warp", "good-owned", "damaged-owned", "foreign-preloaded", "offline-local", "cancel", "repeated-clicks"])
def test_matrix_requires_native_per_scenario_report(tmp_path, scenario):
    matrix = _matrix()
    report = {"scenario": scenario, "passed": True, "diagnostic": False,
              "certified_runtime": True, "draws": [{"operators": ["fluidsim.install_warp"]}],
              "checks": {name: True for name in matrix.BASE_CHECKS | matrix.REQUIRED_CHECKS[scenario]}}
    with pytest.raises(ValueError, match="controls"):
        matrix.validate_scenario_report(report, scenario)
    report["draws"][0]["operators"].append("fluidsim.install_warp_from_file")
    # Installed-good/damaged cases expose runtime/repair controls instead.
    if scenario in {"good-owned", "damaged-owned"}:
        report["draws"][0]["operators"].append("fluidsim.refresh_runtime_probe")
    matrix.validate_scenario_report(report, scenario)
    report["checks"].pop("scenario_state")
    with pytest.raises(ValueError, match="checks"):
        matrix.validate_scenario_report(report, scenario)
    report["checks"]["scenario_state"] = True
    report["diagnostic"] = True
    with pytest.raises(ValueError, match="certified"):
        matrix.validate_scenario_report(report, scenario)


def _fixtures():
    path = SCRIPT.with_name("_installer_preferences_scenarios.py")
    spec = importlib.util.spec_from_file_location("installer_scenario_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_installer(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    from types import SimpleNamespace

    from fluidSim_warp_blender51 import warp_installer
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("warp/__init__.py", '__version__ = "1.19.0"\n')
        archive.writestr("warp_lang-1.19.0.dist-info/METADATA", "Name: warp-lang\n")
    info = SimpleNamespace(size=wheel.stat().st_size, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    owned = tmp_path / "profile/extensions/owned/.local"
    monkeypatch.setattr(warp_installer, "get_wheel_info_for_platform", lambda: info)
    monkeypatch.setattr(warp_installer, "_default_install_target", lambda create=False: owned)
    # Production fixture code still validates and extracts with real helpers.
    monkeypatch.setattr(warp_installer, "ensure_installed_warp_on_path", lambda: None)
    import sys
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    return warp_installer, wheel, owned


@pytest.mark.parametrize("scenario", ["good-owned", "damaged-owned"])
def test_owned_fixture_extracts_locked_bytes_then_only_explicit_damage(tmp_path, monkeypatch, scenario):
    installer, wheel, owned = _fixture_installer(tmp_path, monkeypatch)
    returned, foreign = _fixtures().prepare_scenario(installer, scenario, wheel, tmp_path / "profile")
    assert returned == owned
    assert not foreign.exists()
    content = (owned / "warp/__init__.py").read_text()
    assert ("INSTALLER_MATRIX_DAMAGED_OWNED_FIXTURE" in content) is (scenario == "damaged-owned")
    assert installer._owned_distribution_version(owned) == "1.19.0"


def test_fixture_rejects_unlocked_wheel_and_foreign_target_before_mutation(tmp_path, monkeypatch):
    installer, wheel, owned = _fixture_installer(tmp_path, monkeypatch)
    fixture = _fixtures()
    wheel.write_bytes(wheel.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="locked"):
        fixture.prepare_scenario(installer, "good-owned", wheel, tmp_path / "profile")
    assert not owned.exists()
    monkeypatch.setattr(installer, "_wheel_matches_lock", lambda *args: True)
    with pytest.raises(ValueError, match="private profile"):
        fixture.prepare_scenario(installer, "good-owned", wheel, tmp_path / "unrelated")
    assert not owned.exists()


@pytest.mark.parametrize("native_failure", [False, True])
def test_runner_isolates_every_process_and_requires_reports(tmp_path, monkeypatch, native_failure):
    import json
    import subprocess
    matrix = _matrix()
    monkeypatch.setattr(matrix, "verify_local_wheel", lambda *args: {})
    package = tmp_path / "package.zip"
    package.write_bytes(b"package fixture")
    wheel = tmp_path / "warp.whl"
    wheel.write_bytes(b"wheel fixture")
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    sentinel = consumer / "unrelated.whl"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setenv("BLENDER_USER_EXTENSIONS", str(consumer))
    environments = []
    def runner(command, **kwargs):
        env = kwargs["env"]
        environments.append(env)
        assert "--offline-mode" in command
        assert "--python-use-system-env" not in command
        assert Path(env["BLENDER_USER_EXTENSIONS"]).is_relative_to(tmp_path / "evidence")
        if "--scenario" in command:
            scenario = command[command.index("--scenario") + 1]
            path = Path(command[command.index("--json-out") + 1])
            if not native_failure:
                report = {"scenario": scenario, "passed": True, "diagnostic": False,
                          "certified_runtime": True, "draws": [{"operators": [
                              "fluidsim.install_warp", "fluidsim.install_warp_from_file",
                              "fluidsim.refresh_runtime_probe"]}], "checks": {
                              name: True for name in matrix.BASE_CHECKS | matrix.REQUIRED_CHECKS[scenario]}}
                path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, "fixture stdout", "")
    result = matrix.run_matrix(Path("blender"), package, wheel, tmp_path / "evidence", runner=runner)
    assert result["passed"] is (not native_failure)
    assert len(environments) == 21
    assert len({env["BLENDER_USER_EXTENSIONS"] for env in environments}) == 7
    assert all(Path(env["BLENDER_USER_EXTENSIONS"]).exists() is native_failure for env in environments)
    assert sentinel.read_bytes() == b"preserve"
    assert len(list((tmp_path / "evidence").rglob("process-*.log"))) == 21


def test_expected_cancellation_does_not_mask_unrelated_operator_failure():
    from types import SimpleNamespace
    fixture = _fixtures()
    def broken(**kwargs):
        raise RuntimeError("operator not registered")
    bpy = SimpleNamespace(ops=SimpleNamespace(fluidsim=SimpleNamespace(install_warp=broken)))
    with pytest.raises(RuntimeError, match="not registered"):
        fixture.invoke_operator(bpy, "fluidsim.install_warp", expected="CANCELLED",
                                message_contains="Online access required")


def test_draw_proxy_preserves_native_rna_owner_and_observes_nested_controls():
    import ast
    from types import SimpleNamespace
    source = SCRIPT.with_name("blender_installer_preferences_acceptance.py")
    tree = ast.parse(source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"_ObservedLayout", "_PreferencesDrawProxy"}]
    calls = []
    owner = SimpleNamespace()
    class Layout:
        def box(self):
            return self
        def row(self, **kwargs):
            return self
        def prop(self, actual_owner, name):
            assert actual_owner is owner
            calls.append(("prop", name))
        def operator(self, identifier):
            calls.append(("operator", identifier))
            return SimpleNamespace(force_reinstall=False)
    owner.layout = Layout()
    namespace = {}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    operators = set()
    proxy = namespace["_PreferencesDrawProxy"](owner, operators)
    row = proxy.layout.box().row(align=True)
    row.prop(proxy, "device")
    row.operator("fluidsim.refresh_runtime_probe")
    assert calls == [("prop", "device"), ("operator", "fluidsim.refresh_runtime_probe")]
    assert operators == {"fluidsim.refresh_runtime_probe"}
