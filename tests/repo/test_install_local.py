"""Host-side checks for tools/release/install_local.py (no Blender needed)."""

from __future__ import annotations

import json
import subprocess

from tools.release import install_local


def _report(**overrides):
    report = {
        "modules": ["bl_ext.user_default.fluidsim_warp"],
        "fluidsim_version": install_local.release_identity()["VERSION_STRING"],
        "runtime_available": True,
        "runtime_message": "ok",
        "warp_path": "site-packages/warp/__init__.py",
    }
    report.update(overrides)
    return report


def test_default_zip_matches_build_extension_naming():
    identity = install_local.release_identity()
    assert install_local.default_zip().name == (
        f"{identity['EXTENSION_ID']}-{identity['VERSION_STRING']}-windows_x64.zip"
    )


def test_parse_probe_finds_marker_line_amid_blender_noise():
    output = "Blender 5.2\nRead prefs\n" + install_local.PROBE_MARKER + json.dumps({"a": 1}) + "\nquit\n"
    assert install_local.parse_probe(output) == {"a": 1}
    assert install_local.parse_probe("no marker here") is None


def test_ready_report_has_no_problems():
    expected = install_local.release_identity()["VERSION_STRING"]
    assert install_local.check_probe(_report(), expected) == []


def test_stale_version_and_foreign_warp_are_reported():
    expected = install_local.release_identity()["VERSION_STRING"]
    problems = install_local.check_probe(
        _report(fluidsim_version="1.0.255", runtime_available=False,
                warp_path="scripts/warp_foundry_runtime/warp/__init__.py"),
        expected,
    )
    assert any("1.0.255" in p for p in problems)
    assert any("warp_foundry_runtime" in p for p in problems)


def test_missing_or_duplicate_extension_is_not_ready():
    expected = install_local.release_identity()["VERSION_STRING"]
    assert install_local.check_probe(None, expected)
    assert install_local.check_probe(_report(modules=[]), expected)
    assert install_local.check_probe(_report(modules=["a.fluidsim_warp", "b.fluidsim_warp"]), expected)


def test_legacy_copies_found_in_addons_and_extensions(tmp_path):
    (tmp_path / "5.1" / "scripts" / "addons" / "fluidSim_warp_blender51").mkdir(parents=True)
    (tmp_path / "5.1" / "extensions" / "user_default" / "fluidSim_warp_blender51").mkdir(parents=True)
    # The current extension id is not legacy and must not be flagged.
    (tmp_path / "5.2" / "extensions" / "user_default" / "fluidsim_warp").mkdir(parents=True)
    found = install_local.legacy_copies(tmp_path)
    assert sorted(p.relative_to(tmp_path).parts[1] for p in found) == ["extensions", "scripts"]


def test_failed_probe_cannot_report_ready(monkeypatch, capsys):
    output = install_local.PROBE_MARKER + json.dumps(_report())
    monkeypatch.setattr(install_local, "_run", lambda command: subprocess.CompletedProcess(
        command, 37, output, "Blender failed after reporting runtime"))
    monkeypatch.delenv("APPDATA", raising=False)
    assert install_local.main(["--blender", "blender", "--probe-only"]) == 37
    captured = capsys.readouterr()
    assert "READY:" not in captured.out
    assert "NOT READY: Blender probe exited 37" in captured.err


def test_failed_install_does_not_start_probe(monkeypatch, tmp_path, capsys):
    package = tmp_path / "extension.zip"
    package.write_bytes(b"fixture")
    commands = []

    def run(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 19, "", "install failed")

    monkeypatch.setattr(install_local, "_run", run)
    assert install_local.main(["--blender", "blender", "--zip", str(package)]) == 19
    assert len(commands) == 1
    assert "install-file" in commands[0]
    assert "READY:" not in capsys.readouterr().out


def test_successful_install_then_probe_reports_ready(monkeypatch, tmp_path, capsys):
    package = tmp_path / "extension.zip"
    package.write_bytes(b"fixture")
    commands = []

    def run(command):
        commands.append(command)
        output = install_local.PROBE_MARKER + json.dumps(_report()) if len(commands) == 2 else "installed"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(install_local, "_run", run)
    monkeypatch.delenv("APPDATA", raising=False)
    assert install_local.main(["--blender", "blender", "--zip", str(package)]) == 0
    assert "install-file" in commands[0]
    assert "--python-expr" in commands[1]
    assert "READY:" in capsys.readouterr().out


def test_install_and_probe_receive_repaired_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(install_local, "_blender_process_environment",
                        lambda source: {"PROCESSOR_ARCHITECTURE": "AMD64", "owned": "preserved"})

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(install_local.subprocess, "run", run)
    install_local._run(["blender", "install-file"])
    install_local._run(["blender", "--python-expr", "probe"])
    assert len(calls) == 2
    assert all(call["env"] == {"PROCESSOR_ARCHITECTURE": "AMD64", "owned": "preserved"}
               for call in calls)
