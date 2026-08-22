import time

import pytest

from tools import run_production_gate
from tools import validate_gui_playback_report


def test_production_gate_pass_defaults_to_headless_without_gui(monkeypatch, capsys):
    calls = []

    def fake_run(label, command, env=None, reject_output=False,
                 expect_marker=None):
        calls.append((label, command, env, reject_output, expect_marker))

    monkeypatch.setattr(run_production_gate, "_run", fake_run)
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_npz", lambda: None)
    monkeypatch.setattr(
        run_production_gate, "_check_no_tracked_ignored_files", lambda: None)
    monkeypatch.setattr(run_production_gate, "_clear_gate_manifest", lambda: None)
    monkeypatch.setattr(run_production_gate, "_write_gate_manifest", lambda: None)
    monkeypatch.setenv("BLENDER_EXE", "blender")
    monkeypatch.setenv("BLENDER_PY", "python")
    monkeypatch.delenv("FLUIDSIM_USER_PATH_PROFILE", raising=False)

    run_production_gate.main([])

    out = capsys.readouterr().out
    gui_calls = [
        (label, command, reject_output)
        for label, command, _env, reject_output, _expect_marker in calls
        if label == "interactive GUI acceptance"
    ]

    assert "\nHEADLESS-GATE-PASS" in out
    assert "FULL-PRODUCTION-GATE-PASS" not in out
    assert gui_calls == []
    assert all(
        expect_marker
        for _label, _command, _env, reject_output, expect_marker in calls
        if reject_output
    )
    capacity_calls = [
        (command, marker)
        for label, command, _env, _reject_output, marker in calls
        if label == "source capacity planner acceptance"
    ]
    assert len(capacity_calls) == 1
    assert "tests/blender_capacity_planner_acceptance.py" in capacity_calls[0][0]
    assert capacity_calls[0][1] == "CAPACITY-PLANNER-ACCEPTANCE-PASS"
    admission_calls = [
        (command, marker)
        for label, command, _env, _reject_output, marker in calls
        if label == "tests/blender_simulation_admission_acceptance.py"
    ]
    assert len(admission_calls) == 1
    assert "tests/blender_simulation_admission_acceptance.py" in admission_calls[0][0]
    assert admission_calls[0][1] == "SIMULATION-ADMISSION-ACCEPTANCE-PASS"
    stability_calls = [
        (command, marker)
        for label, command, _env, _reject_output, marker in calls
        if label == "tests/blender_closed_domain_stability_acceptance.py"
    ]
    assert len(stability_calls) == 1
    assert "tests/blender_closed_domain_stability_acceptance.py" in stability_calls[0][0]
    assert stability_calls[0][1] == "CLOSED-DOMAIN-STABILITY-ACCEPTANCE-PASS"


def test_production_gate_include_gui_requires_foreground_gui_acceptance(
        monkeypatch, capsys, tmp_path):
    calls = []
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    stale_report = artifacts / "gui_playback_probe.json"
    stale_report.write_text("stale", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_run(label, command, env=None, reject_output=False,
                 expect_marker=None):
        calls.append((label, command, env, reject_output, expect_marker))

    monkeypatch.setattr(run_production_gate, "_run", fake_run)
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_npz", lambda: None)
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_ignored_files", lambda: None)
    monkeypatch.setattr(run_production_gate, "_clear_gate_manifest", lambda: None)
    monkeypatch.setattr(run_production_gate, "_write_gate_manifest", lambda: None)
    monkeypatch.setenv("BLENDER_EXE", "blender")
    monkeypatch.setenv("BLENDER_PY", "python")

    run_production_gate.main(["--include-gui"])

    out = capsys.readouterr().out
    assert not stale_report.exists()
    gui_calls = [
        (label, command, reject_output, expect_marker)
        for label, command, _env, reject_output, expect_marker in calls
        if label == "interactive GUI acceptance"
    ]

    assert "\nFULL-PRODUCTION-GATE-PASS" in out
    assert gui_calls == [
        (
            "interactive GUI acceptance",
            [
                "blender",
                "--factory-startup",
                "--python",
                "tests/blender_gui_acceptance.py",
            ],
            True,
            "GUI-ACCEPTANCE-PASS",
        )
    ]
    validator_calls = [
        (command, expect_marker)
        for label, command, _env, _reject_output, expect_marker in calls
        if label == "saved-demo GUI playback report"
    ]
    assert len(validator_calls) == 1
    command, marker = validator_calls[0]
    assert command[-2:] == [
        "--expected-blend",
        run_production_gate.os.path.abspath("demos/dam_break.blend"),
    ]
    assert marker == "SAVED-DEMO-GUI-PLAYBACK-PASS"


def test_blender_lane_missing_positive_marker_fails_with_zero_exit(monkeypatch):
    monkeypatch.setattr(
        run_production_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: run_production_gate.subprocess.CompletedProcess(
            args=["blender"], returncode=0, stdout="clean exit without proof\n"),
    )

    with pytest.raises(run_production_gate.subprocess.CalledProcessError):
        run_production_gate._run(
            "Blender lane", ["blender"], reject_output=True,
            expect_marker="LANE-PASS")


def test_playback_validator_recomputes_and_requires_stored_verdict(tmp_path):
    now = time.time()
    blend = tmp_path / "dam_break.blend"
    report = {
        "blend": str(blend.resolve()),
        "generated_at_unix": now,
        "timed_out": False,
        "handler_count": 1,
        "start_frame": 1,
        "end_frame": 4,
        "session_active": True,
        "solver_frame": 3,
        "viewport_baseline_solver_frame": 1,
        "device": "cuda:0",
        "expected_domain_sdf_walls": True,
        "saved_domain_sdf_walls": True,
        "domain_sdf_walls": True,
        "particle_count": 4,
        "viewport_particle_count": 4,
        "viewport_changed": True,
        "error": "",
        "passed": True,
    }
    validate_gui_playback_report.validate_report(
        report, expected_blend=str(blend), now=now)

    report["passed"] = False
    with pytest.raises(RuntimeError, match="disagrees"):
        validate_gui_playback_report.validate_report(
            report, expected_blend=str(blend), now=now)

    report["passed"] = True
    report["particle_count"] = 0
    with pytest.raises(RuntimeError, match="particle_count"):
        validate_gui_playback_report.validate_report(
            report, expected_blend=str(blend), now=now)


def test_blender_lane_traceback_fails_even_with_zero_exit(monkeypatch, capsys):
    def fake_run(*_args, **_kwargs):
        return run_production_gate.subprocess.CompletedProcess(
            args=["blender"],
            returncode=0,
            stdout="Traceback (most recent call last):\nRuntimeError: boom\n",
        )

    monkeypatch.setattr(run_production_gate.subprocess, "run", fake_run)

    try:
        run_production_gate._run("Blender lane", ["blender"], reject_output=True)
    except run_production_gate.subprocess.CalledProcessError as exc:
        assert exc.returncode == 1
    else:
        raise AssertionError("traceback output must fail the gate")

    assert "RuntimeError: boom" in capsys.readouterr().out
