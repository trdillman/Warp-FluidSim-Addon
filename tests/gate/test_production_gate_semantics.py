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


def _write_passing_budget(monkeypatch, tmp_path, status=""):
    """A passing summary for HEAD abc123; ``status`` is git's porcelain output."""
    artifact = tmp_path / "artifacts" / "demo_analysis" / "analysis_report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    performance = tmp_path / "artifacts" / "performance_budget" / "summary.json"
    performance.parent.mkdir()
    from test_performance_budget import _campaign

    from tools.benchmark_versions import REFERENCE_COMMITS, _inventory
    from tools.performance_budget import evaluate_budget
    runs = _campaign()
    inventory = _inventory(tmp_path)
    for run in runs:
        root = Path(run['root'])
        run.update(addon=str(root / 'fluidSim_warp_blender51/__init__.py'), processId=123,
                   rootInventorySha256=inventory if run['label'] == 'candidate' else 'b' * 64,
                   gitHead='abc123' if run['label'] == 'candidate' else REFERENCE_COMMITS[run['referenceVersion']],
                   candidateInventorySha256=inventory)
    performance.write_text(json.dumps(dict(evaluate_budget(runs), gitHead="abc123",
                                       candidateInventorySha256=_inventory(tmp_path),
                                       runs=runs,
                                       referenceCommits=REFERENCE_COMMITS)))
    manifest = tmp_path / "artifacts" / "production_gate_manifest.json"
    monkeypatch.setattr(run_production_gate, "ROOT", tmp_path)
    monkeypatch.setattr(run_production_gate, "GATE_MANIFEST", manifest)
    monkeypatch.setattr(run_production_gate, "_git_text",
                        lambda *args: status if args[0] == "status" else "abc123")
    monkeypatch.setattr(run_production_gate, "_fluidsim_version", lambda: "9.9.9")
    return manifest


def test_gate_manifest_records_the_certified_version(monkeypatch, tmp_path):
    manifest = _write_passing_budget(monkeypatch, tmp_path)

    run_production_gate._write_gate_manifest("blender-exe", "1.17.0")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["fluidsim_version"] == "9.9.9"
    assert data["blender_executable"] == "blender-exe"
    assert data["warp_version"] == "1.17.0"
    assert data["git_head"] == "abc123"
    assert data["gate_verdict"] == "FULL-PRODUCTION-GATE-PASS"
    assert data["performance_artifact_sha256"]
    assert data["performance_budget_reused"] is False


def test_reuse_budget_skips_the_measurement_and_records_it(monkeypatch, capsys):
    commands, validations, manifests = {}, [], []
    _stub_gate(monkeypatch, commands)
    monkeypatch.setattr(run_production_gate, "_validate_performance_evidence",
                        lambda **kwargs: validations.append(kwargs) or {})
    monkeypatch.setattr(run_production_gate, "_write_gate_manifest",
                        lambda *args: manifests.append(args))

    assert run_production_gate.main(["--include-gui", "--reuse-budget"]) == 0

    assert "paired demo performance budget" not in commands
    assert validations == [{"reuse": True}]
    assert manifests[0][2] is True
    assert "FULL-PRODUCTION-GATE-PASS" in capsys.readouterr().out


def test_reused_budget_survives_rebuilt_demo_bytes(monkeypatch, tmp_path):
    manifest = _write_passing_budget(monkeypatch, tmp_path)
    (tmp_path / "demos").mkdir()
    (tmp_path / "demos" / "dam_break.blend").write_bytes(b"rebuilt")

    import pytest
    with pytest.raises(RuntimeError, match="bytes changed"):
        run_production_gate._validate_performance_evidence()
    run_production_gate._write_gate_manifest("blender-exe", "1.17.0", True)

    assert json.loads(manifest.read_text(encoding="utf-8"))["performance_budget_reused"] is True


def test_reused_budget_rejects_modified_sources(monkeypatch, tmp_path):
    import pytest
    _write_passing_budget(monkeypatch, tmp_path, status=" M fluidSim_warp_blender51/solver.py")
    with pytest.raises(RuntimeError, match="sources changed"):
        run_production_gate._validate_performance_evidence(reuse=True)


def test_reused_budget_must_belong_to_this_head(monkeypatch, tmp_path):
    import pytest
    _write_passing_budget(monkeypatch, tmp_path)
    monkeypatch.setattr(run_production_gate, "_git_text",
                        lambda *args: "" if args[0] == "status" else "def456")
    with pytest.raises(RuntimeError, match="for this HEAD"):
        run_production_gate._validate_performance_evidence(reuse=True)


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


def test_quick_tier_skips_evidence_lanes(monkeypatch, capsys):
    commands = {}
    _stub_gate(monkeypatch, commands)

    assert run_production_gate.main(["--quick"]) == 0

    out = capsys.readouterr().out
    assert "QUICK-GATE-PASS" in out
    assert "GATE-PASS\n" not in out.replace("QUICK-GATE-PASS\n", "")
    assert any(label.startswith("Blender acceptance batch") or label.endswith("_acceptance.py")
               for label in commands)
    for skipped in ("demo analysis", "demo blend checks", "interactive GUI acceptance",
                    "paired demo performance budget"):
        assert skipped not in commands


def test_quick_tier_refuses_release_flags():
    import pytest
    for flag in ("--include-gui", "--reuse-budget"):
        with pytest.raises(SystemExit):
            run_production_gate.main(["--quick", flag])


def test_acceptance_batch_covers_every_background_script():
    scripts = [script for _label, script, _args, _marker in run_production_gate._acceptance_entries()]
    assert len(scripts) == len(set(scripts))
    assert set(scripts) == (
        {script for script, _ in run_production_gate.BACKGROUND_ACCEPTANCE}
        | {script for _l, script, _r, _m in run_production_gate.REPORT_ACCEPTANCE}
        | {f"tests/acceptance/blender_{cap}_acceptance.py"
           for cap, _ in run_production_gate.SOURCE_REPORT_ACCEPTANCE})


def test_batch_reruns_alone_every_script_it_did_not_pass(monkeypatch, tmp_path):
    entries = [("a", "tests/acceptance/blender_a.py", (), "A-PASS"),
               ("b", "tests/acceptance/blender_b.py", ("--json-out", "x.json"), "B-PASS"),
               ("c", "tests/acceptance/blender_c.py", (), "C-PASS")]
    calls = []

    def fake_run(label, command, env=None, reject_output=False, expect_marker=None):
        calls.append((label, command, reject_output, expect_marker))
        if label.startswith("Blender acceptance batch"):
            # a passed, b failed, c never ran (crash).
            (tmp_path / "artifacts" / "gate_batch" / "results.json").write_text(json.dumps([
                {"script": "tests/acceptance/blender_a.py", "passed": True, "seconds": 1.0},
                {"script": "tests/acceptance/blender_b.py", "passed": False, "seconds": 2.0},
            ]))
            import subprocess
            raise subprocess.CalledProcessError(3, command)

    monkeypatch.setattr(run_production_gate, "ROOT", tmp_path)
    monkeypatch.setattr(run_production_gate, "_run", fake_run)
    run_production_gate._run_acceptance_batch("blender", {}, entries)

    reruns = [(label, command[-3:], reject, marker) for label, command, reject, marker in calls[1:]]
    assert reruns == [
        ("b", ["--", "--json-out", "x.json"], True, "B-PASS"),
        ("c", ["--factory-startup", "--python", "tests/acceptance/blender_c.py"], True, "C-PASS"),
    ]


def test_parallel_steps_print_in_order_and_raise_the_first_failure(monkeypatch, capsys):
    import subprocess
    import sys as _sys

    import pytest
    steps = [
        ("slow ok", [_sys.executable, "-c", "import time; time.sleep(0.3); print('one')"], {}),
        ("fails", [_sys.executable, "-c", "raise SystemExit(4)"], {}),
        ("fast ok", [_sys.executable, "-c", "print('three')"], {}),
    ]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_production_gate._run_parallel(steps)
    assert exc.value.returncode == 4
    out = capsys.readouterr().out
    assert out.index("== slow ok ==") < out.index("one") < out.index("== fails ==") \
        < out.index("== fast ok ==") < out.index("three")


def test_every_step_reports_its_wall_time(monkeypatch, capsys):
    import sys as _sys
    run_production_gate.STEP_TIMES.clear()
    run_production_gate._run("tiny", [_sys.executable, "-c", "print('hi')"],
                             expect_marker="hi")
    assert [label for label, _ in run_production_gate.STEP_TIMES] == ["tiny"]
    assert "[time] tiny:" in capsys.readouterr().out
