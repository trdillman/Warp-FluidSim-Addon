from tools import run_production_gate


def test_production_gate_pass_requires_foreground_gui_acceptance(monkeypatch, capsys):
    calls = []

    def fake_run(label, command, env=None, reject_output=False):
        calls.append((label, command, env, reject_output))

    monkeypatch.setattr(run_production_gate, "_run", fake_run)
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_npz", lambda: None)
    monkeypatch.setenv("BLENDER_EXE", "blender")
    monkeypatch.setenv("BLENDER_PY", "python")
    monkeypatch.delenv("FLUIDSIM_USER_PATH_PROFILE", raising=False)

    run_production_gate.main()

    out = capsys.readouterr().out
    gui_calls = [
        (label, command, reject_output)
        for label, command, _env, reject_output in calls
        if label == "Blender GUI acceptance"
    ]

    assert "\nPRODUCTION-GATE-PASS" in out
    assert gui_calls == [
        (
            "Blender GUI acceptance",
            ["blender", "--python", "tests/blender_gui_acceptance.py"],
            True,
        )
    ]


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
