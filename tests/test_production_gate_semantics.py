from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_production_gate", ROOT / "tools" / "run_production_gate.py"
)
run_production_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_production_gate)


def _stub_gate(monkeypatch):
    labels = []
    monkeypatch.setattr(
        run_production_gate,
        "_run",
        lambda label, command, env=None, reject_output=False: labels.append(label),
    )
    monkeypatch.setattr(run_production_gate, "_check_no_tracked_npz", lambda: None)
    monkeypatch.setattr(
        run_production_gate,
        "_check_no_tracked_ignored_files",
        lambda: None,
    )
    monkeypatch.setattr(run_production_gate, "_print_gate_metadata", lambda *_args: None)
    return labels


def test_gate_default_reports_headless_only(monkeypatch, capsys):
    labels = _stub_gate(monkeypatch)

    assert run_production_gate.main([]) == 0

    out = capsys.readouterr().out
    assert "interactive GUI acceptance" not in labels
    assert "HEADLESS-GATE-PASS" in out
    assert "FULL-PRODUCTION-GATE-PASS" not in out


def test_gate_include_gui_runs_gui_acceptance(monkeypatch, capsys):
    labels = _stub_gate(monkeypatch)

    assert run_production_gate.main(["--include-gui"]) == 0

    out = capsys.readouterr().out
    assert "interactive GUI acceptance" in labels
    assert "FULL-PRODUCTION-GATE-PASS" in out
