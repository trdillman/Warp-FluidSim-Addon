"""FSPT gate evidence must come from the attempt that emitted its pass marker."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools import run_production_gate as gate

SCRIPT = "tests/acceptance/blender_fspt_async_viewport_acceptance.py"
BATCH_SCRIPT = "tests/acceptance/blender_fspt_async_acceptance.py"
MARKER = "FSPT-ASYNC-VIEWPORT-PASS"
BATCH_MARKER = "FSPT-ASYNC-BACKEND-PASS"
VERIFIED = b'{"status":"VERIFIED"}\n'
BROKER_SCRIPT = "tests/acceptance/blender_fspt_viewport_backend_acceptance.py"


@pytest.fixture
def gate_run(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    root = tmp_path / "artifacts" / "qualification_runs" / "owned-test-run"
    root.mkdir(parents=True)
    monkeypatch.setattr(gate, "_RUN_DIRECTORY", root)
    monkeypatch.setattr(gate, "_FSPT_REPORTS", {}, raising=False)
    return root


def _launch_script(report):
    gate._run("FSPT viewport", ["blender", "--python", SCRIPT, "--", "--json-out", str(report)],
              reject_output=True, expect_marker=MARKER)


@pytest.mark.parametrize(("payload", "error"), [
    (None, "missing"),
    (b'{"status":"FAILED"}', "FAILED"),
    (b'{"status":', "invalid JSON"),
    (b'["VERIFIED"]', "JSON object"),
    (b'{"status":"VERIFIED","value":NaN}', "invalid JSON"),
])
def test_marker_and_zero_exit_do_not_pass_without_verified_report(monkeypatch, gate_run,
                                                                payload, error):
    report = gate_run / "viewport.json"

    def process(command, **_kwargs):
        if payload is not None:
            report.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, MARKER + "\n")

    monkeypatch.setattr(gate.subprocess, "run", process)
    with pytest.raises(RuntimeError, match=error):
        _launch_script(report)
    assert not report.with_suffix(".receipt.json").exists()


def test_rejects_preexisting_verified_report_before_launch(monkeypatch, gate_run):
    report = gate_run / "viewport.json"
    report.write_bytes(VERIFIED)
    launches = []

    def process(command, **_kwargs):
        launches.append(command)
        return subprocess.CompletedProcess(command, 0, MARKER + "\n")

    monkeypatch.setattr(gate.subprocess, "run", process)
    with pytest.raises(RuntimeError, match="already exists"):
        _launch_script(report)
    assert launches == []
    assert report.read_bytes() == VERIFIED


def test_fresh_verified_report_is_hashed_and_bound(monkeypatch, gate_run):
    report = gate_run / "viewport.json"

    def process(command, **_kwargs):
        report.write_bytes(VERIFIED)
        return subprocess.CompletedProcess(command, 0, MARKER + "\n")

    monkeypatch.setattr(gate.subprocess, "run", process)
    _launch_script(report)
    receipt = json.loads(report.with_suffix(".receipt.json").read_text())
    assert receipt["script"] == SCRIPT
    assert receipt["step"] == "FSPT viewport"
    assert receipt["report"] == str(report)
    assert receipt["run_id"] == gate_run.name
    assert receipt["sha256"] == hashlib.sha256(VERIFIED).hexdigest()
    assert gate._fspt_report_manifest()[SCRIPT] == receipt

    report.write_bytes(b'{"status":"VERIFIED","changed":true}')
    with pytest.raises(RuntimeError, match="changed"):
        gate._fspt_report_manifest()


def test_broker_label_alone_cannot_supply_native_evidence(gate_run):
    report = gate_run / "broker.json"
    report.write_bytes(VERIFIED)
    with pytest.raises(RuntimeError, match="evidence"):
        gate._record_fspt_report("FSPT broker", BROKER_SCRIPT, report)
    assert not report.with_suffix(".receipt.json").exists()


def test_verified_json_does_not_replace_the_success_marker(monkeypatch, gate_run):
    report = gate_run / "viewport.json"

    def process(command, **_kwargs):
        report.write_bytes(VERIFIED)
        return subprocess.CompletedProcess(command, 0, "no success marker\n")

    monkeypatch.setattr(gate.subprocess, "run", process)
    with pytest.raises(subprocess.CalledProcessError):
        _launch_script(report)
    assert not report.with_suffix(".receipt.json").exists()


@pytest.mark.parametrize("batch_report", [VERIFIED, b'{"status":"FAILED"}', None])
def test_batch_retains_verified_report_or_retries_with_new_path(monkeypatch, gate_run,
                                                            batch_report):
    requested = []

    def process(command, **_kwargs):
        script = command[command.index("--python") + 1]
        if script == "tools/gate/run_blender_batch.py":
            spec_path, results_path = map(Path, command[-2:])
            entry, = json.loads(spec_path.read_text())["entries"]
            assert entry["script"] == BATCH_SCRIPT
            assert entry["args"][0] == "--json-out"
            report = Path(entry["args"][1])
            requested.append(report)
            assert report.is_relative_to(gate_run) and not report.exists()
            if batch_report is not None:
                report.parent.mkdir(parents=True)
                report.write_bytes(batch_report)
            results_path.write_text(json.dumps([{"script": BATCH_SCRIPT, "passed": True}]))
            return subprocess.CompletedProcess(command, 0, "BATCH-COMPLETE\n")

        assert script == BATCH_SCRIPT
        report = Path(command[command.index("--json-out") + 1])
        requested.append(report)
        assert report.is_relative_to(gate_run) and not report.exists()
        report.parent.mkdir(parents=True)
        report.write_bytes(VERIFIED)
        return subprocess.CompletedProcess(command, 0, BATCH_MARKER + "\n")

    monkeypatch.setattr(gate.subprocess, "run", process)
    gate._run_acceptance_batch("blender", {},
                               [("FSPT backend", BATCH_SCRIPT, (), BATCH_MARKER)])

    assert len(requested) == (1 if batch_report == VERIFIED else 2)
    if batch_report == b'{"status":"FAILED"}':
        assert requested[0].read_bytes() == batch_report
    accepted = requested[-1]
    assert gate._fspt_report_manifest()[BATCH_SCRIPT]["report"] == str(accepted)
    assert accepted.with_suffix(".receipt.json").is_file()
