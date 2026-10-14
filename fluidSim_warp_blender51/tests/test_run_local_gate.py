from __future__ import annotations

import json
from pathlib import Path

from fluidsim_testpkg import version
from tools.release import run_local_gate

ROOT = Path(__file__).resolve().parents[1]


def _clean_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(run_local_gate.subprocess, "check_output", lambda command, **kwargs: "a" * 40 if "rev-parse" in command else "")
    monkeypatch.setattr(run_local_gate, "check_environment", lambda: {})
    monkeypatch.setattr(run_local_gate, "installed_warp_root", lambda: tmp_path)


def test_local_gate_keeps_pinned_release_requirements():
    assert run_local_gate.pinned_requirements() == {
        "pytest": "9.1.1", "ruff": "0.16.2", "numpy": "2.3.4",
        "warp-lang": version.CERTIFIED_WARP_VERSION,
    }


def test_local_gate_runs_full_cpu_checks_on_clean_commit(tmp_path, monkeypatch):
    _clean_gate(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr(run_local_gate, "run", lambda command, **kwargs: commands.append((command, kwargs)))
    assert run_local_gate.main(["--checks-only", "--output", str(tmp_path)]) == 0
    rendered = [" ".join(command) for command, _ in commands]
    assert len(rendered) == 7
    for required in ("-m ruff check .", "build_extension.py --verify-only", "-m compileall", "-m pytest -q"):
        assert any(required in command for command in rendered)
    for name in run_local_gate.VALIDATORS:
        assert sum(f"validate_{name}_warp.py" in command and "--device cpu" in command and "--report-json" in command for command in rendered) == 1
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1" for _, kwargs in commands)
    report = json.loads((tmp_path / "local-gate-report.json").read_text(encoding="utf-8"))
    assert report["commit"] == "a" * 40
    assert report["cpu_checks_passed"] is True
    assert report["package_smoke_passed"] is False


def test_local_gate_rejects_dirty_checkout_without_pass_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(run_local_gate.subprocess, "check_output", lambda command, **kwargs: "a" * 40 if "rev-parse" in command else " M source.py")
    assert run_local_gate.main(["--checks-only", "--output", str(tmp_path)]) == 1
    assert not (tmp_path / "local-gate-report.json").exists()


def test_local_gate_rejects_wrong_blender_hash(tmp_path, monkeypatch):
    _clean_gate(monkeypatch, tmp_path)
    monkeypatch.setattr(run_local_gate, "run", lambda command, **kwargs: None)
    blender = tmp_path / "blender"
    blender.write_bytes(b"wrong binary")
    assert run_local_gate.main(["--output", str(tmp_path), "--blender", str(blender), "--blender-sha256", "0" * 64]) == 1
    assert not (tmp_path / "local-gate-report.json").exists()


def test_local_gate_requires_installed_smoke_for_package_pass(tmp_path, monkeypatch):
    _clean_gate(monkeypatch, tmp_path)
    monkeypatch.setattr(run_local_gate, "host_platform", lambda: "windows-x64")
    monkeypatch.setattr(run_local_gate, "check_blender_version", lambda blender: ".".join(map(str, version.CERTIFIED_BLENDER_VERSION)))
    blender = tmp_path / "blender"
    blender.write_bytes(b"approved binary")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "--json-report" in command:
            Path(command[command.index("--json-report") + 1]).write_text(
                json.dumps({"release_ready": True, "smoke_platform": "windows-x64"}), encoding="utf-8"
            )

    monkeypatch.setattr(run_local_gate, "run", fake_run)
    assert run_local_gate.main(["--output", str(tmp_path), "--blender", str(blender), "--blender-sha256", run_local_gate.sha256(blender)]) == 0
    assert "--smoke-platform" in commands[-1] and "windows-x64" in commands[-1]
    report = json.loads((tmp_path / "local-gate-report.json").read_text(encoding="utf-8"))
    assert report["package_smoke_passed"] is True
    assert report["blender_sha256"] == run_local_gate.sha256(blender)


def test_local_gate_rejects_wrong_blender_version(tmp_path, monkeypatch):
    blender = tmp_path / "blender"
    blender.write_bytes(b"binary")
    monkeypatch.setattr(run_local_gate.subprocess, "check_output", lambda command, **kwargs: "Blender 5.1.0\n")
    try:
        run_local_gate.check_blender_version(blender)
    except RuntimeError as exc:
        assert "Expected Blender" in str(exc)
    else:
        raise AssertionError("wrong Blender version accepted")


def test_installed_release_oracles_match_the_current_identity():
    for relative in ("../tools/release/smoke_test_blender.py", "../tools/release/validate_phase5_blender.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert version.VERSION_STRING in source or repr(version.VERSION) in source or "version.py" in source


def test_local_gate_rejects_checkout_changed_during_checks(tmp_path, monkeypatch):
    _clean_gate(monkeypatch, tmp_path)
    changed = False

    def git_output(command, **kwargs):
        return "a" * 40 if "rev-parse" in command else (" M source.py" if changed else "")

    def mutate(command, **kwargs):
        nonlocal changed
        changed = True

    monkeypatch.setattr(run_local_gate.subprocess, "check_output", git_output)
    monkeypatch.setattr(run_local_gate, "run", mutate)
    assert run_local_gate.main(["--checks-only", "--output", str(tmp_path)]) == 1
    assert not (tmp_path / "local-gate-report.json").exists()


def test_local_gate_rejects_head_changed_during_checks(tmp_path, monkeypatch):
    _clean_gate(monkeypatch, tmp_path)
    changed = False

    def git_output(command, **kwargs):
        return ("b" if changed else "a") * 40 if "rev-parse" in command else ""

    def mutate(command, **kwargs):
        nonlocal changed
        changed = True

    monkeypatch.setattr(run_local_gate.subprocess, "check_output", git_output)
    monkeypatch.setattr(run_local_gate, "run", mutate)
    assert run_local_gate.main(["--checks-only", "--output", str(tmp_path)]) == 1
    assert not (tmp_path / "local-gate-report.json").exists()
