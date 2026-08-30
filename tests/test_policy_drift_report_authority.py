from __future__ import annotations

import json
import subprocess
from tools import validate_policy_drift_reports as validator


def test_report_authority_requires_current_gate_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "production_gate_manifest.json"
    manifest.write_text(json.dumps({"gate_verdict": "HEADLESS-GATE-PASS"}),
                        encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)

    errors = validator.validate_report_authority(manifest)

    assert any("not a full-production-gate pass" in error for error in errors)
    assert any("GUI-inclusive" in error for error in errors)


def test_report_authority_does_not_trust_stale_manifest_head(tmp_path, monkeypatch):
    manifest = tmp_path / "production_gate_manifest.json"
    manifest.write_text(json.dumps({
        "gate_verdict": "FULL-PRODUCTION-GATE-PASS",
        "include_gui": True,
        "git_head": "stale",
        "demo_artifact": "missing.json",
        "demo_artifact_sha256": "stale",
    }), encoding="utf-8")
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(
        validator.subprocess, "check_output",
        lambda *args, **kwargs: b"current" if kwargs.get("text") is False else "current\n",
    )

    errors = validator.validate_report_authority(manifest)

    assert "gate manifest HEAD is not an ancestor of current HEAD" in errors
    assert "gate manifest demo artifact is missing" in errors


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                    capture_output=True, text=True)


def _head(cwd):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def test_evidence_only_commit_after_the_proof_head_is_not_drift(
        tmp_path, monkeypatch):
    """A report can never assert its own committed hash: the commit that
    records the gate's proof always lands one commit after that proof HEAD.
    That evidence-only commit must not itself count as drift."""
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "test", cwd=tmp_path)
    report = tmp_path / "POLICY_DRIFT_REPORT.md"
    report.write_text("proof-head content\n", encoding="utf-8")
    _git("add", "POLICY_DRIFT_REPORT.md", cwd=tmp_path)
    _git("commit", "-q", "-m", "proof head", cwd=tmp_path)
    proof_head = _head(tmp_path)

    report.write_text("rebound to proof head\n", encoding="utf-8")
    _git("add", "POLICY_DRIFT_REPORT.md", cwd=tmp_path)
    _git("commit", "-q", "-m", "rebind evidence", cwd=tmp_path)
    current_head = _head(tmp_path)

    errors = validator._non_evidence_drift(proof_head, current_head)

    assert errors == []


def test_nested_package_report_is_gate_evidence():
    assert validator._is_evidence_path(
        "fluidSim_warp_blender51/POLICY_DRIFT_REPORT.md"
    )


def test_non_evidence_commit_after_the_proof_head_is_drift(
        tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "test", cwd=tmp_path)
    (tmp_path / "POLICY_DRIFT_REPORT.md").write_text("v1\n", encoding="utf-8")
    _git("add", "POLICY_DRIFT_REPORT.md", cwd=tmp_path)
    _git("commit", "-q", "-m", "proof head", cwd=tmp_path)
    proof_head = _head(tmp_path)

    (tmp_path / "solver.py").write_text("code change\n", encoding="utf-8")
    _git("add", "solver.py", cwd=tmp_path)
    _git("commit", "-q", "-m", "unrelated code change", cwd=tmp_path)
    current_head = _head(tmp_path)

    errors = validator._non_evidence_drift(proof_head, current_head)

    assert any("solver.py" in error for error in errors)
