"""Fresh-child failure propagation and the release lane's fixed contract."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import benchmark_versions as bench
from tools.benchmark_demo_trial import comparable_profiler, translate_reference_settings


def test_fresh_child_exit_failure_retains_log_and_never_reads_stale_json(monkeypatch, tmp_path):
    out = tmp_path / "trial.json"
    out.write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=9, stdout="raw child evidence\n", stderr="actual failure"))
    with pytest.raises(RuntimeError, match="exit 9"):
        bench._run_trial(["blender"], out)
    assert out.with_suffix(".log").read_text() == "raw child evidence\nactual failure"


def test_successful_child_returns_raw_evidence(monkeypatch, tmp_path):
    out = tmp_path / "trial.json"
    def complete(*args, **kwargs):
        out.write_text(json.dumps({"records": [{"hostMs": 17}]}))
        return SimpleNamespace(returncode=0, stdout="complete", stderr="")
    monkeypatch.setattr(bench.subprocess, "run", complete)
    assert bench._run_trial(["blender"], out)["records"][0]["hostMs"] == 17


def test_zero_exit_cannot_reuse_stale_payload(monkeypatch, tmp_path):
    out = tmp_path / "settings.json"
    out.write_text("{}")
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="missing marker"):
        bench._run_trial(["blender"], out, expect_marker="DEMO-SETTINGS-EXTRACT-PASS")
    assert not out.exists()


def test_budget_rejects_shortened_demo_window_before_launch(tmp_path):
    with pytest.raises(SystemExit):
        bench.demo_budget_main(["--demo-budget", "--blender", "unused",
                                "--frames", "8", "--out-dir", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_reference_commit_identity_matches_release_mapping():
    assert bench.REFERENCE_COMMITS == {
        "1.0.255": "9c0d99e62adda14bb7b075a447c01fefe7187da3",
        "1.1.0": "c5919d668f469942200c58274eca13337d3aa624",
    }


def test_legacy_ownership_adapter_preserves_physics_and_effective_output_work():
    properties = {name: None for name in ("max_substeps", "voxel_size",
                  "surface_attribute_velocity", "whitewater_attribute_type")}
    owner = SimpleNamespace(bl_rna=SimpleNamespace(properties=properties))
    translate_reference_settings(owner, {"max_substeps": 24, "voxel_size": 0.012,
        "export_attributes": False, "surface_attribute_velocity": True,
        "whitewater_attribute_type": True}, {}, apply_values=True)
    assert owner.max_substeps == 24 and owner.voxel_size == 0.012
    assert owner.surface_attribute_velocity is False
    assert owner.whitewater_attribute_type is True


def test_reference_adapter_does_not_substitute_historical_defaults():
    owner = SimpleNamespace(bl_rna=SimpleNamespace(properties={
        "surface_tension_enabled": None, "export_attributes": None}),
        surface_tension_enabled=True, export_attributes=False)
    translate_reference_settings(owner, {"surface_tension_enabled": False,
                                        "export_attributes": False}, {}, apply_values=True)
    assert owner.surface_tension_enabled is False


@pytest.mark.parametrize("legacy", [False, True])
def test_comparable_profiler_never_double_counts_nested_stages(legacy):
    from contextlib import contextmanager
    class FakeProfiler:
        def __init__(self):
            self.frame = SimpleNamespace(timers={})
            self.depth = 0
        @contextmanager
        def track(self, phase):
            outer = self.depth == 0
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                elapsed = {"forces": 10, "surface": 4}[phase]
                slot = {"forces": "compute_3", "surface": "compute_11"}[phase]
                self.frame.timers[slot] = self.frame.timers.get(slot, 0) + elapsed
                if legacy or outer:
                    self.frame.timers["solver"] = self.frame.timers.get("solver", 0) + elapsed
    profiler = FakeProfiler()
    state = comparable_profiler(profiler, {"forces": 3, "surface": 11})
    with profiler.track("forces"):
        with profiler.track("surface"):
            pass
    assert state["outerMs"] == 10
    assert profiler.frame.timers["solver"] == (14 if legacy else 10)
    assert profiler.frame.timers["compute_11"] == 4


def test_release_runner_uses_snapshot_references_and_all_64_trials(monkeypatch, tmp_path):
    import io
    import re
    import tarfile

    from test_performance_budget import _campaign

    source = tmp_path / "source"
    for directory in ("fluidSim_warp_blender51", "demos", "tools"):
        (source / directory).mkdir(parents=True)
    (source / "fluidSim_warp_blender51" / "version.py").write_text('VERSION_STRING = "2.0.0"')
    for demo in ("dam_break", "crown_splash", "paddle_stir", "chromatic_vortex"):
        (source / "demos" / f"{demo}.blend").write_bytes(b"saved demo")
    for script in ("repo_addon_loader.py", "benchmark_demo_trial.py", "benchmark_versions.py",
                   "performance_budget.py", "run_production_gate.py"):
        (source / "tools" / script).write_text("# source child")
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **kw: "head\n")
    archives = []
    def archive(command, *, cwd, check):
        assert Path(cwd) == source  # Snapshots are not Git repositories.
        archives.append(command[-1])
        version = next(key for key, commit in bench.REFERENCE_COMMITS.items() if commit == command[-1])
        content = f'VERSION_STRING = "{version}"'.encode()
        member = tarfile.TarInfo("fluidSim_warp_blender51/version.py")
        member.size = len(content)
        with tarfile.open(command[-2].split("=", 1)[1], "w") as bundle:
            bundle.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(bench.subprocess, "run", archive)
    fixtures = _campaign()
    calls = []
    def child(command, out, **kwargs):
        args = command[command.index("--") + 1:]
        calls.append((command, kwargs))
        assert "--python-exit-code" in command and "--factory-startup" in command
        assert kwargs["expect_marker"]
        if args[0] == "--extract":
            Path(out).write_text("{}")
            return {}
        root, blend, _, frames, reference, payload = args
        assert Path(root) != source and Path(blend).is_relative_to(Path(root).parent / "candidate")
        assert frames == "48" and Path(payload).exists()
        demo = Path(blend).stem
        match = re.search(r"_b(\d+)_s(\d+)_", Path(out).stem)
        block, slot = map(int, match.groups())
        row = next(dict(run) for run in fixtures if run["workload"] == demo
                   and run["round"] == block and run["slot"] == slot)
        row["root"] = root
        row["addon"] = str(Path(root) / 'fluidSim_warp_blender51/__init__.py')
        row["rootInventorySha256"] = bench._inventory(Path(root))
        row["processId"] = 123
        row["version"] = reference if row["label"] == "baseline" else "2.0.0"
        return row
    monkeypatch.setattr(bench, "_run_trial", child)
    out_dir = tmp_path / "evidence"
    assert bench.demo_budget_main(["--demo-budget", "--blender", "fake",
                                   "--candidate-root", str(source), "--out-dir", str(out_dir)]) == 0
    report = json.loads((out_dir / "summary.json").read_text())
    assert len(report["runs"]) == 64
    assert len(calls) == 68  # Four schema extracts plus 64 timed children.
    assert set(archives) == set(bench.REFERENCE_COMMITS.values())
    assert report["candidateInventorySha256"] == bench._inventory(source)


def test_pressure_solve_recorder_captures_each_substep_and_restores_method():
    from tools.benchmark_demo_trial import record_pressure_solves

    class Pressure:
        last_pcg_check_every = 5
        last_pcg_execution_mode = "cuda_conditional"
        def project_frame(self, iterations):
            return {"iterations": iterations, "residual": 1e-6,
                    "converged": True, "reason": "converged",
                    "project_velocity": True}

    pressure = Pressure()
    with record_pressure_solves(pressure) as solves:
        first = pressure.project_frame(5)
        second = pressure.project_frame(10)
        assert first["iterations"] == 5
        assert second["iterations"] == 10
        assert [s["iterations"] for s in solves] == [5, 10]
        assert all(s["checkEvery"] == 5 for s in solves)
        assert all(s["executionMode"] == "cuda_conditional" for s in solves)
        assert all(s["converged"] for s in solves)
    assert "project_frame" not in pressure.__dict__


def test_pressure_solve_recorder_restores_existing_override_after_error():
    from tools.benchmark_demo_trial import record_pressure_solves
    def failing():
        raise RuntimeError("pressure failure")
    pressure = SimpleNamespace(project_frame=failing)
    with pytest.raises(RuntimeError, match="pressure failure"), record_pressure_solves(pressure):
        pressure.project_frame()
    assert pressure.project_frame is failing
