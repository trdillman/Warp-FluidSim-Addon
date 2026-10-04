from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_production_gate", ROOT / "tools" / "run_production_gate.py"
)
run_production_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_production_gate)


def _stub_gate(monkeypatch, commands=None):
    labels = []

    def run(label, command, env=None, reject_output=False, expect_marker=None):
        labels.append(label)
        if commands is not None:
            commands[label] = [str(part) for part in command]

    monkeypatch.setattr(run_production_gate, "_run", run)
    monkeypatch.setattr(
        run_production_gate, "_check_certified_warp", lambda _python, _env: "warp-root")
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_npz", lambda: None)
    monkeypatch.setattr(
        run_production_gate,
        "_check_no_tracked_ignored_files",
        lambda: None,
    )
    monkeypatch.setattr(run_production_gate, "_clear_gate_manifest", lambda: None)
    monkeypatch.setattr(run_production_gate, "_write_gate_manifest", lambda *_args: None)
    monkeypatch.setattr(run_production_gate, "_validate_performance_evidence", lambda: {})
    monkeypatch.setattr(run_production_gate, "_print_gate_metadata", lambda *_args: None)
    return labels


def test_gate_default_reports_headless_only(monkeypatch, capsys):
    labels = _stub_gate(monkeypatch)

    assert run_production_gate.main([]) == 0

    out = capsys.readouterr().out
    assert "interactive GUI acceptance" not in labels
    assert "HEADLESS-GATE-PASS" in out
    assert "FULL-PRODUCTION-GATE-PASS" not in out


def test_gate_include_gui_runs_gui_acceptance(monkeypatch, capsys, tmp_path):
    labels = _stub_gate(monkeypatch)
    report = tmp_path / "artifacts" / "gui_playback_probe.json"
    report.parent.mkdir()
    report.write_text("stale", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert run_production_gate.main(["--include-gui"]) == 0
    assert not report.exists()

    out = capsys.readouterr().out
    assert "interactive GUI acceptance" in labels
    assert "FULL-PRODUCTION-GATE-PASS" in out


def test_gate_validates_release_kernels_on_cuda(monkeypatch):
    commands = {}
    _stub_gate(monkeypatch, commands)

    assert run_production_gate.main([]) == 0

    for check in ("phase4", "phase5", "custom_effects"):
        command = commands[f"{check} kernels on CUDA"]
        assert command[1] == f"tools/release/validate_{check}_warp.py"
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--expected-warp-root") + 1] == "warp-root"


def test_gate_manifest_records_the_certified_version(monkeypatch, tmp_path):
    artifact = tmp_path / "artifacts" / "demo_analysis" / "analysis_report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    performance = tmp_path / "artifacts" / "performance_budget" / "summary.json"
    performance.parent.mkdir()
    from tools.benchmark_versions import _inventory
    from test_performance_budget import _campaign
    performance.write_text(json.dumps({"passed": True, "gitHead": "abc123",
                                       "candidateInventorySha256": _inventory(tmp_path),
                                       "runs": _campaign(),
                                       "referenceCommits": {"1.0.255": "baseline"}}))
    manifest = tmp_path / "artifacts" / "production_gate_manifest.json"
    monkeypatch.setattr(run_production_gate, "ROOT", tmp_path)
    monkeypatch.setattr(run_production_gate, "GATE_MANIFEST", manifest)
    monkeypatch.setattr(run_production_gate, "_git_text", lambda *_args: "abc123")
    monkeypatch.setattr(run_production_gate, "_fluidsim_version", lambda: "9.9.9")

    run_production_gate._write_gate_manifest("blender-exe", "1.17.0")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["fluidsim_version"] == "9.9.9"
    assert data["blender_executable"] == "blender-exe"
    assert data["warp_version"] == "1.17.0"
    assert data["git_head"] == "abc123"
    assert data["gate_verdict"] == "FULL-PRODUCTION-GATE-PASS"
    assert data["performance_artifact_sha256"]


def test_performance_budget_runs_in_headless_and_full_gate(monkeypatch):
    for argv in ([], ["--include-gui"]):
        commands = {}
        _stub_gate(monkeypatch, commands)
        assert run_production_gate.main(argv) == 0
        command = commands["paired demo performance budget"]
        assert "tools/benchmark_versions.py" in command
        assert "--demo-budget" in command
        assert command[command.index("--trials") + 1] == "8"
        assert command[command.index("--frames") + 1] == "48"
        assert list(commands).index("paired demo performance budget") > list(commands).index("demo blend checks")


def test_failed_budget_prevents_any_gate_pass(monkeypatch, capsys):
    import subprocess
    import pytest

    _stub_gate(monkeypatch)
    def reject_budget(label, command, **kwargs):
        if label == "paired demo performance budget":
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(run_production_gate, "_run", reject_budget)
    with pytest.raises(subprocess.CalledProcessError):
        run_production_gate.main([])
    assert "GATE-PASS" not in capsys.readouterr().out


def test_evidence_validation_blocks_headless_pass(monkeypatch, capsys):
    import pytest
    _stub_gate(monkeypatch)
    def changed_bytes():
        raise RuntimeError("source or demo bytes changed")
    monkeypatch.setattr(run_production_gate, "_validate_performance_evidence", changed_bytes)
    with pytest.raises(RuntimeError, match="bytes changed"):
        run_production_gate.main([])
    assert "GATE-PASS" not in capsys.readouterr().out


def test_gate_reads_the_version_from_the_add_on():
    namespace = {}
    source = ROOT / "fluidSim_warp_blender51" / "version.py"
    exec(source.read_text(encoding="utf-8"), namespace)
    assert run_production_gate._fluidsim_version() == namespace["VERSION_STRING"]


def test_gate_blender_lanes_use_the_gate_python_environment(monkeypatch):
    commands = {}
    envs = {}
    labels = _stub_gate(monkeypatch, commands)

    def run(label, command, env=None, reject_output=False, expect_marker=None):
        labels.append(label)
        commands[label] = [str(part) for part in command]
        envs[label] = env

    monkeypatch.setattr(run_production_gate, "_run", run)
    monkeypatch.setenv("BLENDER_EXE", "blender-exe")
    monkeypatch.setenv("FLUIDSIM_USER_PATH_PROFILE", "1")

    assert run_production_gate.main(["--include-gui"]) == 0

    blender_lanes = [label for label, cmd in commands.items() if cmd[0] == "blender-exe"]
    assert len(blender_lanes) >= 20
    for label in blender_lanes:
        assert commands[label][1] == "--python-use-system-env", label
    for label, env in envs.items():
        if label == "git whitespace":
            continue
        assert env is not None and env["PYTHONNOUSERSITE"] == "1", label


def test_gate_blender_lanes_isolate_user_extensions(monkeypatch):
    # --factory-startup Blender sees every installed extension as disabled and
    # its wheel sync deletes their wheels (Warp) from the real profile's
    # extensions/.local, so no Blender lane may use the real extensions dir.
    commands = {}
    envs = {}
    labels = _stub_gate(monkeypatch, commands)

    def run(label, command, env=None, reject_output=False, expect_marker=None):
        labels.append(label)
        commands[label] = [str(part) for part in command]
        envs[label] = env

    monkeypatch.setattr(run_production_gate, "_run", run)
    monkeypatch.setenv("BLENDER_EXE", "blender-exe")
    monkeypatch.setenv("FLUIDSIM_USER_PATH_PROFILE", "1")
    monkeypatch.delenv("BLENDER_USER_EXTENSIONS", raising=False)
    monkeypatch.delenv("BLENDER_USER_RESOURCES", raising=False)

    assert run_production_gate.main(["--include-gui"]) == 0

    private = run_production_gate.GATE_BLENDER_USER_EXTENSIONS
    assert private.is_relative_to(ROOT / "artifacts")
    blender_lanes = [label for label, cmd in commands.items() if cmd[0] == "blender-exe"]
    assert len(blender_lanes) >= 20
    assert any("--factory-startup" in commands[label] for label in blender_lanes)
    for label in blender_lanes:
        assert envs[label]["BLENDER_USER_EXTENSIONS"] == str(private), label


def test_gate_keeps_a_caller_blender_profile(monkeypatch):
    # Any BLENDER_USER_* variable overrides BLENDER_USER_RESOURCES, so the gate
    # must not point extensions away from a caller's own clean profile.
    monkeypatch.delenv("BLENDER_USER_EXTENSIONS", raising=False)
    monkeypatch.setenv("BLENDER_USER_RESOURCES", "clean-profile")
    assert "BLENDER_USER_EXTENSIONS" not in run_production_gate._env()

    monkeypatch.delenv("BLENDER_USER_RESOURCES")
    monkeypatch.setenv("BLENDER_USER_EXTENSIONS", "caller-extensions")
    assert run_production_gate._env()["BLENDER_USER_EXTENSIONS"] == "caller-extensions"


def test_gate_refuses_an_uncertified_warp(monkeypatch, capsys):
    monkeypatch.setattr(run_production_gate, "_warp_version", lambda *_args: "1.16.1")
    monkeypatch.setattr(run_production_gate, "_warp_root", lambda *_args: "user-site")

    try:
        run_production_gate._check_certified_warp("blender-py", {})
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("an uncertified Warp must stop the gate")
    assert "requires certified Warp" in capsys.readouterr().out


def test_gate_accepts_the_certified_warp(monkeypatch):
    certified = run_production_gate._version_constant("CERTIFIED_WARP_VERSION")
    monkeypatch.setattr(run_production_gate, "_warp_version", lambda *_args: certified)
    monkeypatch.setattr(run_production_gate, "_warp_root", lambda *_args: "site")

    assert run_production_gate._check_certified_warp("blender-py", {}) == "site"
