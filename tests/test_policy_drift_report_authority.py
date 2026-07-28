from __future__ import annotations

import json
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

    assert "gate manifest HEAD does not match current HEAD" in errors
    assert "gate manifest demo artifact is missing" in errors
