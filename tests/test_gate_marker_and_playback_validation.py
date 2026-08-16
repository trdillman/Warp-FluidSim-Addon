"""Production-gate positive-marker and playback-report truth contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import run_production_gate as gate
from tools import validate_gui_playback_report as validator


def test_run_rejects_zero_exit_without_expected_success_marker(monkeypatch):
    monkeypatch.setattr(
        gate.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="Blender exited cleanly but script never ran\n"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        gate._run("probe", ["blender"], reject_output=True,
                  expect_marker="REAL-PASS")


def test_run_accepts_expected_success_marker(monkeypatch):
    monkeypatch.setattr(
        gate.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="REAL-PASS\n"),
    )
    gate._run("probe", ["blender"], reject_output=True,
              expect_marker="REAL-PASS")


def passing_report(blend: Path, *, now: float):
    return {
        "blend": str(blend.resolve()),
        "generated_at_unix": now,
        "timed_out": False,
        "handler_count": 1,
        "start_frame": 1,
        "end_frame": 5,
        "session_active": True,
        "solver_frame": 4,
        "viewport_baseline_solver_frame": 1,
        "device": "cuda:0",
        "expected_domain_sdf_walls": True,
        "saved_domain_sdf_walls": True,
        "domain_sdf_walls": True,
        "particle_count": 100,
        "viewport_particle_count": 100,
        "viewport_changed": True,
        "error": "",
        # Deliberately irrelevant: validator must re-derive, not trust this.
        "passed": True,
    }


def test_playback_validator_rederives_invariants_and_checks_identity(tmp_path):
    now = time.time()
    expected = tmp_path / "dam_break.blend"
    expected.write_bytes(b"test identity path only")
    report = passing_report(expected, now=now)
    validator.validate_report(
        report, expected_blend=str(expected), now=now, max_age_seconds=120)

    report["particle_count"] = 0
    with pytest.raises(RuntimeError, match="particle_count"):
        validator.validate_report(
            report, expected_blend=str(expected), now=now,
            max_age_seconds=120)


def test_playback_validator_rejects_stale_wrong_blend_and_timeout(tmp_path):
    now = time.time()
    expected = tmp_path / "dam_break.blend"
    report = passing_report(expected, now=now - 1000)
    with pytest.raises(RuntimeError, match="stale"):
        validator.validate_report(
            report, expected_blend=str(expected), now=now,
            max_age_seconds=120)

    report = passing_report(tmp_path / "other.blend", now=now)
    with pytest.raises(RuntimeError, match="blend"):
        validator.validate_report(
            report, expected_blend=str(expected), now=now,
            max_age_seconds=120)

    report = passing_report(expected, now=now)
    report["timed_out"] = True
    with pytest.raises(RuntimeError, match="timed_out"):
        validator.validate_report(
            report, expected_blend=str(expected), now=now,
            max_age_seconds=120)
