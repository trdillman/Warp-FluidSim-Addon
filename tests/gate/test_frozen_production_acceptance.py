"""Truthfulness and process boundaries for the CUDA continuation harness."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_device_is_refused_before_runtime_fixture_work():
    contract = load(
        "tests/acceptance/_frozen_acceptance_contract.py", "frozen_contract"
    )
    with pytest.raises(RuntimeError, match="CUDA_REQUIRED"):
        contract.require_cuda(SimpleNamespace(is_cuda=False))
    contract.require_cuda(SimpleNamespace(is_cuda=True))


def test_difference_report_never_grants_numerical_qualification():
    contract = load(
        "tests/acceptance/_frozen_acceptance_contract.py", "frozen_contract"
    )
    result = contract.array_differences({"p": np.array([1.0])}, {"p": np.array([1.1])})
    assert result["p"]["exact"] is False
    assert result["p"]["max_abs"] == pytest.approx(0.1)
    assert (
        contract.qualification_status() == "pending_compiled_emitter_numerical_policy"
    )
    with pytest.raises(AssertionError):
        contract.assert_exact_state({"p": np.array([1.0])}, {"p": np.array([1.1])})


def test_shape_and_nonfinite_regressions_are_not_hidden_in_diagnostics():
    contract = load(
        "tests/acceptance/_frozen_acceptance_contract.py", "frozen_contract"
    )
    with pytest.raises(AssertionError):
        contract.array_differences({"p": np.array([1.0])}, {"p": np.array([np.nan])})
    result = contract.array_differences({"p": np.ones(1)}, {"p": np.ones(2)})
    assert result["p"]["shape_matches"] is False


def test_runner_stops_after_failed_preflight_and_cannot_print_pass(tmp_path):
    runner = load("tools/gate/run_frozen_input_acceptance.py", "frozen_runner")
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    assert runner.run_phases("blender", tmp_path, "cuda:0", execute=execute) == 1
    assert len(calls) == 1
    assert calls[0][calls[0].index("--phase") + 1] == "preflight"
    report = json.loads((tmp_path / "run-report.json").read_text())
    assert report["qualified"] is False


def test_runner_requires_reports_and_keeps_each_phase_in_fresh_process(tmp_path):
    runner = load("tools/gate/run_frozen_input_acceptance.py", "frozen_runner")
    seen = []

    def execute(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        seen.append(phase)
        profile = kwargs["env"]
        for name in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
            assert Path(profile["BLENDER_USER_" + name]).is_absolute()
        (tmp_path / f"{phase}-report.json").write_text(
            json.dumps(
                {
                    "phase": phase,
                    "status": "structural_checks_passed",
                    "qualified": False,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    assert runner.run_phases("blender", tmp_path, "cuda:0", execute=execute) == 2
    assert seen == ["preflight", "baseline", "partial", "resume", "rebake", "preview"]
    report = json.loads((tmp_path / "run-report.json").read_text())
    assert report["status"] == "pending_compiled_emitter_numerical_policy"
    assert not report["qualified"]


def test_successful_process_without_phase_evidence_fails(tmp_path):
    runner = load("tools/gate/run_frozen_input_acceptance.py", "frozen_runner")
    assert (
        runner.run_phases(
            "blender",
            tmp_path,
            "cuda:0",
            execute=lambda *a, **k: SimpleNamespace(returncode=0),
        )
        == 1
    )


def test_real_phase_entrypoint_records_cuda_refusal_without_touching_addon(
    tmp_path, monkeypatch
):
    import sys

    module = load(
        "tests/acceptance/blender_frozen_production_acceptance.py", "frozen_phase"
    )
    monkeypatch.setitem(
        sys.modules, "bpy", SimpleNamespace(app=SimpleNamespace(version_string="test"))
    )
    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(init=lambda: None, get_cuda_devices=list, __version__="test"),
    )
    for name in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
        monkeypatch.setenv("BLENDER_USER_" + name, str(tmp_path / name))
    assert (
        module.main(
            SimpleNamespace(state_root=tmp_path, phase="baseline", device="cuda:0")
        )
        == 1
    )
    report = json.loads((tmp_path / "baseline-report.json").read_text())
    assert report["status"] == "failed"
    assert "CUDA_REQUIRED" in report["error"]
    assert not (tmp_path / "production-frozen.blend").exists()


def test_runner_refuses_reuse_of_existing_phase_evidence(tmp_path):
    runner = load("tools/gate/run_frozen_input_acceptance.py", "frozen_runner")
    (tmp_path / "partial-report.json").write_text("{}")
    with pytest.raises(ValueError, match="existing evidence"):
        runner.run_phases("blender", tmp_path, "cuda:0")


def test_import_boundary_checks_solver_arrays_and_lifecycle_exactly():
    phase = load(
        "tests/acceptance/blender_frozen_production_acceptance.py", "frozen_phase"
    )
    expected = SimpleNamespace(
        frame=3,
        sim_time=0.125,
        scalars={"count": 1, "blender_frozen_inputs": {"schema_version": 1}},
        source_state={"emitted": ["source"]},
        arrays={
            "particle_position": np.array([[1.0, 2.0, 3.0]], np.float32),
            "blender_frozen_points": np.array([[0.0, 0.0, 0.0]], np.float32),
        },
    )
    actual = SimpleNamespace(
        frame=3,
        sim_time=0.125,
        scalars={"count": 1},
        source_state={"emitted": ["source"]},
        arrays={"particle_position": expected.arrays["particle_position"].copy()},
    )
    phase.assert_restored(expected, actual)
    actual.source_state = {"emitted": []}
    with pytest.raises(AssertionError):
        phase.assert_restored(expected, actual)
    actual.source_state = expected.source_state
    actual.arrays["particle_position"][0, 0] += 0.001
    with pytest.raises(AssertionError):
        phase.assert_restored(expected, actual)


def test_runner_discards_foreign_environment_paths_but_keeps_source_imports(
    tmp_path, monkeypatch
):
    runner = load("tools/gate/run_frozen_input_acceptance.py", "frozen_runner")
    owned_names = (
        "BLENDER_USER_CONFIG",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_EXTENSIONS",
        "BLENDER_USER_DATAFILES",
        "PYTHONUSERBASE",
        "WARP_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TEMP",
        "TMP",
        "TMPDIR",
    )
    for name in owned_names:
        monkeypatch.setenv(name, str(tmp_path / "foreign" / name))
    monkeypatch.setenv("PYTHONHOME", "/foreign/python-home")
    monkeypatch.setenv("PYTHONSTARTUP", "/foreign/startup.py")
    monkeypatch.setenv("PYTHONPATH", "/explicit/source-dependencies")
    evidence = tmp_path / "evidence"
    captured = []

    def execute(_command, **kwargs):
        env = kwargs["env"]
        captured.append(env)
        assert "PYTHONHOME" not in env
        assert "PYTHONSTARTUP" not in env
        assert env["PYTHONPATH"] == "/explicit/source-dependencies"
        for name in owned_names:
            path = Path(env[name])
            assert path.is_absolute()
            assert path.is_relative_to(evidence / "profile")
            assert path.is_dir()
        return SimpleNamespace(returncode=1)

    assert runner.run_phases("blender", evidence, "cuda:0", execute=execute) == 1
    assert len(captured) == 1
