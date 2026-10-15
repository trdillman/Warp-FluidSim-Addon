from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from tools.performance_budget import DEMOS

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
    monkeypatch.setattr(run_production_gate, "_reusable_budget_demos", lambda: frozenset())
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


def _selected(path, specs):
    """git's pathspec match for the plain prefixes and excludes the gate uses."""
    if path.endswith(".md") or path.startswith("fluidSim_warp_blender51/docs/"):
        return False
    return any(path == spec or path.startswith(spec + "/")
               for spec in specs if not spec.startswith(":("))


def _write_passing_budget(monkeypatch, tmp_path, status="", diff="", head="abc123"):
    """A passing summary measured on abc123; ``status``/``diff`` are git's output."""
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
    def git_text(*args):
        if args[0] not in ("status", "diff"):
            return head
        specs = args[args.index("--") + 1:]
        paths = [line[3:] if args[0] == "status" else line
                 for line in (status if args[0] == "status" else diff).splitlines()]
        return "\n".join(path for path in paths if _selected(path, specs))
    monkeypatch.setattr(run_production_gate, "_git_text", git_text)
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
    assert data["performance_budget_measured_at"] == {demo: "abc123" for demo in DEMOS}


def test_gate_reuses_a_budget_that_still_covers_the_sources(monkeypatch, capsys):
    """By default a budget measured with the same budget sources is not re-measured."""
    commands, validations, manifests = {}, [], []
    _stub_gate(monkeypatch, commands)
    monkeypatch.setattr(run_production_gate, "_reusable_budget_demos", lambda: frozenset(DEMOS))
    monkeypatch.setattr(run_production_gate, "_rejudge_budget", lambda: None)
    monkeypatch.setattr(run_production_gate, "_validate_performance_evidence",
                        lambda **kwargs: validations.append(kwargs) or {})
    monkeypatch.setattr(run_production_gate, "_write_gate_manifest",
                        lambda *args: manifests.append(args))

    assert run_production_gate.main(["--include-gui"]) == 0

    assert "paired demo performance budget" not in commands
    assert validations == [{"reuse": True}]
    assert manifests[0][2] is True
    out = capsys.readouterr().out
    assert "reused crown_splash, dam_break, paddle_stir, chromatic_vortex" in out
    assert "FULL-PRODUCTION-GATE-PASS" in out


def test_measure_budget_forces_a_fresh_measurement(monkeypatch):
    commands = {}
    _stub_gate(monkeypatch, commands)
    monkeypatch.setattr(run_production_gate, "_reusable_budget_demos", lambda: frozenset(DEMOS))

    assert run_production_gate.main(["--measure-budget"]) == 0
    assert "paired demo performance budget" in commands


def test_reuse_budget_fails_instead_of_measuring(monkeypatch):
    import pytest
    commands = {}
    _stub_gate(monkeypatch, commands)

    with pytest.raises(RuntimeError, match="no passing performance budget covers"):
        run_production_gate.main(["--reuse-budget"])
    assert "paired demo performance budget" not in commands


def test_budget_flags_are_exclusive(capsys):
    import pytest
    with pytest.raises(SystemExit):
        run_production_gate.main(["--reuse-budget", "--measure-budget"])


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
    solver = tmp_path / "fluidSim_warp_blender51" / "solver.py"
    solver.parent.mkdir()
    solver.write_text("edited after measuring", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sources changed"):
        run_production_gate._validate_performance_evidence(reuse=True)


def test_fresh_budget_must_belong_to_this_head(monkeypatch, tmp_path):
    import pytest
    _write_passing_budget(monkeypatch, tmp_path, head="def456")
    with pytest.raises(RuntimeError, match="for this HEAD"):
        run_production_gate._validate_performance_evidence()


def test_budget_from_an_earlier_commit_is_reused_when_sources_are_unchanged(monkeypatch, tmp_path):
    """A docs- or blends-only commit keeps the measured budget."""
    _write_passing_budget(monkeypatch, tmp_path, head="def456")
    assert run_production_gate._validate_performance_evidence(reuse=True)["gitHead"] == "abc123"
    assert run_production_gate._reusable_budget_demos() == frozenset(DEMOS)


def test_budget_from_an_earlier_commit_is_rejected_when_sources_changed(monkeypatch, tmp_path):
    import pytest
    _write_passing_budget(monkeypatch, tmp_path, head="def456",
                          diff="fluidSim_warp_blender51/solver.py")
    assert run_production_gate._reusable_budget_demos() == frozenset()
    with pytest.raises(RuntimeError, match="solver.py"):
        run_production_gate._validate_performance_evidence(reuse=True)


def test_budget_sources_include_saved_workload_inputs_but_exclude_documentation():
    sources = run_production_gate._BUDGET_SOURCES
    assert ":(exclude)*.md" in sources
    assert ":(exclude)fluidSim_warp_blender51/docs" in sources
    assert "fluidSim_warp_blender51" in sources
    assert "tools/demos" in sources
    assert "demos/paddle_stir.blend" in sources
    assert "demos/chromatic_vortex.blend" in sources
    assert "demos/dam_break.blend" not in sources
    assert "demos/crown_splash.blend" not in sources


def test_reuse_source_check_tracks_builder_and_unrebuilt_scenes(monkeypatch, tmp_path):
    files = (
        "tools/demos/build_demo_scenes.py",
        "demos/paddle_stir.blend",
        "demos/chromatic_vortex.blend",
        "demos/dam_break.blend",
        "docs/readme.md",
    )
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original", encoding="utf-8")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid",
        "commit", "-qm", "baseline")
    measured_head = git("rev-parse", "HEAD")
    monkeypatch.setattr(run_production_gate, "ROOT", tmp_path)

    for name in files:
        path = tmp_path / name
        path.write_text("changed", encoding="utf-8")
        changed = run_production_gate._budget_sources_changed_since(measured_head)
        if name in ("demos/dam_break.blend", "docs/readme.md"):
            assert changed == ""
        else:
            assert name in changed
        path.write_text("original", encoding="utf-8")

    (tmp_path / "tools/demos/build_demo_scenes.py").write_text(
        "committed change", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid",
        "commit", "-qm", "changed builder")
    assert "tools/demos/build_demo_scenes.py" in (
        run_production_gate._budget_sources_changed_since(measured_head))


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
    for flag in ("--include-gui", "--reuse-budget", "--measure-budget"):
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


def test_cuda_hidden_scripts_never_share_the_batch_process(monkeypatch, tmp_path):
    # CUDA_VISIBLE_DEVICES=-1 is ignored once an earlier script initialized
    # CUDA in the shared process, so these scripts must start a fresh Blender.
    hidden = sorted(run_production_gate.CUDA_HIDDEN_ACCEPTANCE)
    entries = [(script, script, (), "OK") for script in hidden]
    entries.append(("a", "tests/acceptance/blender_a.py", (), "A-PASS"))
    calls = []

    def fake_run(label, command, env=None, reject_output=False, expect_marker=None):
        calls.append(label)
        if label.startswith("Blender acceptance batch"):
            (tmp_path / "artifacts" / "gate_batch" / "results.json").write_text(json.dumps([
                {"script": "tests/acceptance/blender_a.py", "passed": True, "seconds": 1.0}]))

    monkeypatch.setattr(run_production_gate, "ROOT", tmp_path)
    monkeypatch.setattr(run_production_gate, "_run", fake_run)
    run_production_gate._run_acceptance_batch("blender", {}, entries)

    assert calls == [*hidden, "Blender acceptance batch (1 scripts)"]
    spec = json.loads((tmp_path / "artifacts" / "gate_batch" / "spec.json").read_text())
    assert [entry["script"] for entry in spec["entries"]] == ["tests/acceptance/blender_a.py"]


def test_every_cuda_hiding_acceptance_script_runs_alone():
    root = Path(run_production_gate.ROOT)
    hiding = {
        path.relative_to(root).as_posix()
        for path in (root / "tests" / "acceptance").glob("blender_*.py")
        if 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in path.read_text(encoding="utf-8")}
    assert hiding == run_production_gate.CUDA_HIDDEN_ACCEPTANCE


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


def _budget_with_demo_heads(monkeypatch, tmp_path, *, head="def456", diff="", fresh=(),
                            passed=True):
    """A summary whose ``fresh`` demos were measured on ``head``, the rest on abc123."""
    manifest = _write_passing_budget(monkeypatch, tmp_path, head=head, diff=diff)
    path = tmp_path / "artifacts" / "performance_budget" / "summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for run in report["runs"]:
        if run["workload"] in fresh and run["label"] == "candidate":
            run["gitHead"] = head
    report["demoMeasuredAt"] = {demo: head if demo in fresh else "abc123" for demo in DEMOS}
    report["demoInventory"] = {demo: report["candidateInventorySha256"] for demo in DEMOS}
    if not passed:
        report.update(passed=False, status="invalid", demos={}, reasons=["older evaluator"])
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, manifest


def test_evaluator_only_change_rejudges_the_saved_runs(monkeypatch, tmp_path):
    """A checker change re-judges the raw runs; it never needs a Blender run."""
    path, _ = _budget_with_demo_heads(monkeypatch, tmp_path, diff="tools/performance_budget.py",
                                      passed=False)
    assert run_production_gate._reusable_budget_demos() == frozenset(DEMOS)

    run_production_gate._rejudge_budget()

    assert json.loads(path.read_text(encoding="utf-8"))["passed"] is True
    assert run_production_gate._validate_performance_evidence(reuse=True)["passed"] is True


def test_only_the_demo_whose_input_changed_is_remeasured(monkeypatch, tmp_path):
    _budget_with_demo_heads(monkeypatch, tmp_path, diff="demos/paddle_stir.blend")
    assert run_production_gate._reusable_budget_demos() == frozenset(DEMOS) - {"paddle_stir"}


def test_builder_change_remeasures_only_the_built_demos(monkeypatch, tmp_path):
    _budget_with_demo_heads(monkeypatch, tmp_path, diff="tools/demos/build_demo_scenes.py")
    assert run_production_gate._reusable_budget_demos() == {"paddle_stir", "chromatic_vortex"}


def test_addon_change_remeasures_every_demo(monkeypatch, tmp_path):
    _budget_with_demo_heads(monkeypatch, tmp_path, diff="fluidSim_warp_blender51/solver.py")
    assert run_production_gate._reusable_budget_demos() == frozenset()


def test_demo_with_invalid_saved_evidence_is_remeasured(monkeypatch, tmp_path):
    path, _ = _budget_with_demo_heads(monkeypatch, tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    broken = next(run for run in report["runs"] if run["workload"] == "chromatic_vortex")
    broken["protectedSettings"] = {"tampered": True}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert run_production_gate._reusable_budget_demos() == frozenset(DEMOS) - {"chromatic_vortex"}


def test_mixed_budget_binds_each_demo_to_its_own_measurement(monkeypatch, tmp_path):
    import pytest
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _budget_with_demo_heads(monkeypatch, fresh, diff="demos/paddle_stir.blend",
                            fresh=("paddle_stir",))
    assert run_production_gate._validate_performance_evidence(reuse=True)["passed"] is True

    stale = tmp_path / "stale"
    stale.mkdir()
    _budget_with_demo_heads(monkeypatch, stale, diff="demos/paddle_stir.blend")
    with pytest.raises(RuntimeError, match="paddle_stir.blend"):
        run_production_gate._validate_performance_evidence(reuse=True)


def test_gate_measures_only_the_demos_it_cannot_reuse(monkeypatch):
    commands, validations = {}, []
    _stub_gate(monkeypatch, commands)
    monkeypatch.setattr(run_production_gate, "_reusable_budget_demos",
                        lambda: frozenset(DEMOS) - {"paddle_stir"})
    monkeypatch.setattr(run_production_gate, "_carry_prior_budget", lambda: "carried.json")
    monkeypatch.setattr(run_production_gate, "_validate_performance_evidence",
                        lambda **kwargs: validations.append(kwargs) or {})

    assert run_production_gate.main([]) == 0

    command = commands["paired demo performance budget"]
    start = command.index("--demos") + 1
    assert command[start:start + 1] == ["paddle_stir"]
    assert command[command.index("--carry-summary") + 1] == "carried.json"
    assert validations == [{"reuse": True}]


def test_measure_budget_measures_every_demo_without_carrying(monkeypatch):
    commands = {}
    _stub_gate(monkeypatch, commands)
    monkeypatch.setattr(run_production_gate, "_reusable_budget_demos",
                        lambda: frozenset(DEMOS) - {"paddle_stir"})

    assert run_production_gate.main(["--measure-budget"]) == 0
    command = commands["paired demo performance budget"]
    assert "--demos" not in command and "--carry-summary" not in command


def test_per_demo_sources_exclude_the_evaluator():
    assert "tools/performance_budget.py" not in run_production_gate._BUDGET_SOURCES
    paddle = run_production_gate._budget_sources("paddle_stir")
    assert "demos/paddle_stir.blend" in paddle and "fluidSim_warp_blender51" in paddle
    assert "tools/demos" not in paddle and "demos/chromatic_vortex.blend" not in paddle
    assert "tools/demos" in run_production_gate._budget_sources("dam_break")


def test_acceptance_lanes_split_every_script_exactly_once(monkeypatch):
    batches = []

    def batch(_blender, _env, entries, name="gate_batch"):
        batches.append((name, [entry[1] for entry in entries]))

    labels = _stub_gate(monkeypatch)
    monkeypatch.setattr(run_production_gate, "_run_acceptance_batch", batch)
    monkeypatch.setenv("BLENDER_EXE", "blender-exe")
    assert run_production_gate.main(["--include-gui"]) == 0

    names = [name for name, _ in batches]
    assert sorted(names) == ["gate_batch", "gate_batch_quiet", "gate_batch_solo"]
    # The quiet batch runs alone, after the demo lane rebuilt the demos.
    assert names[-1] == "gate_batch_quiet"
    assert labels.index("demo blend checks") < labels.index("interactive GUI acceptance")
    scripts = [script for _, group in batches for script in group]
    expected = [entry[1] for entry in run_production_gate._acceptance_entries()]
    assert sorted(scripts) == sorted(expected)
    by_name = dict(batches)
    assert set(by_name["gate_batch_quiet"]) == run_production_gate.QUIET_ACCEPTANCE
    assert set(by_name["gate_batch_solo"]) == run_production_gate.SOLO_ACCEPTANCE


def test_quick_tier_runs_one_batch_with_every_script(monkeypatch):
    batches = []
    _stub_gate(monkeypatch)
    monkeypatch.setattr(run_production_gate, "_run_acceptance_batch",
                        lambda _b, _e, entries, name="gate_batch": batches.append(
                            (name, len(entries))))
    assert run_production_gate.main(["--quick"]) == 0
    assert batches == [("gate_batch", len(run_production_gate._acceptance_entries()))]


def test_parallel_lanes_run_their_steps_in_sequence(capsys):
    import sys as _sys

    def lane():
        run_production_gate._run("lane one", [_sys.executable, "-c", "print('first')"],
                                 expect_marker="first")
        run_production_gate._run("lane two", [_sys.executable, "-c", "print('second')"],
                                 expect_marker="second")

    run_production_gate._run_parallel([lane, ("other", [_sys.executable, "-c", "print('x')"], {})])
    out = capsys.readouterr().out
    assert out.index("== lane one ==") < out.index("first") < out.index("== lane two ==") \
        < out.index("second") < out.index("== other ==")
