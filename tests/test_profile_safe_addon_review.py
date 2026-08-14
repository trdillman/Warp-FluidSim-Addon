"""Focused contracts for the profile-safe Blender review launcher.

These tests never launch Blender.  The owned-process seam is replaced with a
fake that writes the same readiness marker as the in-Blender bootstrap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import launch_profile_safe_addon_review as launcher


def _probe(config_dir: Path) -> dict:
    return {
        "ok": True,
        "blender_version": "5.1.0",
        "blender_version_tuple": [5, 1, 0],
        "blender_binary": str((config_dir / "fake_blender").resolve()),
        "config_dir": str(config_dir.resolve()),
        "enabled_addons": ["io_scene_fbx", "bl_ext.user_default.nexus_clone_warp"],
    }


class _FakeBlenderProcess:
    instances = []
    mutate_on_wait = None
    visible_review_payload = {"ok": True, "checks": []}

    def __init__(self, command, *, env, stdout, stderr, text):
        self.command = list(command)
        self.env = dict(env)
        self.stdout = stdout
        self.stderr = stderr
        self.text = text
        self.pid = 424242
        self.returncode = None
        self.terminated = False
        self.killed = False
        type(self).instances.append(self)

        config_path = Path(env["FLUIDSIM_PROFILE_SAFE_REVIEW_CONFIG"])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        addons_root = Path(config["addons_root"])
        package_root = addons_root / launcher.ADDON_NAME
        package_root.mkdir(parents=True)
        copied_init = package_root / "__init__.py"
        copied_init.write_text("bl_info = {'version': (1, 0, 0)}\n", encoding="utf-8")
        requested_blend = str(Path(config["blend"]).resolve())
        marker = {
            "stage": "ready",
            "ok": True,
            "module": {
                "name": launcher.ADDON_NAME,
                "version": [1, 0, 0],
                "path": str(copied_init.resolve()),
                "under_temp_root": True,
                "runtime_loaded": True,
                "default_enabled": False,
                "preferences_entry_present": False,
                "nonpersistent_enable_requested": True,
                "loaded_module_paths": {
                    launcher.ADDON_NAME: str(copied_init.resolve())
                },
                "outside_temp_modules": [],
            },
            "blend": {
                "requested": requested_blend,
                "active": requested_blend,
                "matches": True,
            },
            "inventory": {
                "source": {"__init__.py": {"size": 1, "sha256": "source"}},
                "copy": {"__init__.py": {"size": 1, "sha256": "source"}},
                "source_count": 1,
                "copy_count": 1,
                "missing": [],
                "extra": [],
                "mismatched": [],
                "complete": True,
            },
            "blender": {
                "version": "5.1.0",
                "version_tuple": [5, 1, 0],
                "binary": command[0],
            },
            "one_package_identity": True,
            "conflicting_addons": {
                launcher.CONFLICTING_ADDONS[0]: {
                    "default_enabled": True,
                    "loaded_before": True,
                    "loaded_after_disable": False,
                }
            },
        }
        Path(config["marker"]).write_text(json.dumps(marker), encoding="utf-8")
        result_path = env.get(launcher.VISIBLE_REVIEW_RESULT_ENV)
        if result_path and type(self).visible_review_payload is not None:
            payload = type(self).visible_review_payload
            if isinstance(payload, str):
                Path(result_path).write_text(payload, encoding="utf-8")
            else:
                Path(result_path).write_text(
                    json.dumps(payload), encoding="utf-8")

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if type(self).mutate_on_wait is not None:
            type(self).mutate_on_wait()
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.fixture(autouse=True)
def _reset_fake_process():
    _FakeBlenderProcess.instances = []
    _FakeBlenderProcess.mutate_on_wait = None
    _FakeBlenderProcess.visible_review_payload = {"ok": True, "checks": []}
    yield
    _FakeBlenderProcess.instances = []
    _FakeBlenderProcess.mutate_on_wait = None
    _FakeBlenderProcess.visible_review_payload = {"ok": True, "checks": []}


def _run_fake_review(tmp_path, monkeypatch, *, with_review_script=False):
    blend = tmp_path / "review.blend"
    blend.write_bytes(b"BLENDER")
    blender = tmp_path / ("blender.exe" if os.name == "nt" else "blender")
    blender.write_bytes(b"fake executable")
    config_dir = tmp_path / "profile" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "userpref.blend").write_bytes(b"preferences")
    (config_dir / "startup.blend").write_bytes(b"startup")
    report_path = tmp_path / "artifacts" / "review.json"

    monkeypatch.setattr(
        launcher,
        "_run_profile_probe",
        lambda *_args, **_kwargs: _probe(config_dir),
    )
    monkeypatch.setattr(launcher.subprocess, "Popen", _FakeBlenderProcess)
    monkeypatch.setattr(launcher, "_find_recent_crash", lambda *_args: None)
    monkeypatch.setenv("FLUIDSIM_TEST_INHERITED", "yes")

    argv = [
        "--blend",
        str(blend),
        "--report",
        str(report_path),
        "--blender",
        str(blender),
        "--startup-timeout",
        "1",
    ]
    if with_review_script:
        review_script = tmp_path / "visible_review.py"
        review_script.write_text("# fake visible review\n", encoding="utf-8")
        argv.extend(["--review-script", str(review_script)])
    args = launcher.build_parser().parse_args(argv)
    return launcher.run_review(args), report_path, config_dir


def test_discover_blender_uses_explicit_then_environment(tmp_path):
    explicit = tmp_path / "explicit-blender"
    environmental = tmp_path / "environment-blender"
    explicit.write_bytes(b"explicit")
    environmental.write_bytes(b"environment")

    assert launcher.discover_blender(explicit, {}) == explicit.resolve()
    assert launcher.discover_blender(
        None, {"FLUIDSIM_BLENDER_EXECUTABLE": str(environmental)}
    ) == environmental.resolve()


def test_discover_blender_fails_closed_for_missing_override(tmp_path):
    with pytest.raises(FileNotFoundError, match="Blender executable not found"):
        launcher.discover_blender(tmp_path / "missing", {})


def test_profile_comparison_uses_addons_existence_and_hash(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    userpref = config / "userpref.blend"
    userpref.write_bytes(b"before")
    before = launcher.snapshot_profile_files(config)
    probe = {"enabled_addons": ["a", "b"]}

    unchanged, drift = launcher.compare_profile_state(probe, probe, before, before)
    assert unchanged is True
    assert drift == []

    userpref.write_bytes(b"after")
    after = launcher.snapshot_profile_files(config)
    unchanged, drift = launcher.compare_profile_state(
        probe, {"enabled_addons": ["a", "c"]}, before, after
    )
    assert unchanged is False
    assert drift == ["enabled_addons", "userpref.blend"]


def test_bootstrap_is_nonpersistent_and_copies_after_conflict_disable():
    source = launcher._bootstrap_source()
    compile(source, "review_bootstrap.py", "exec")

    assert "save_userpref" not in source
    assert "save_homefile" not in source
    assert "persistent=False" in source
    assert "default_set=False" in source
    assert source.index("addon_utils.disable(identity") < source.index(
        "shutil.copytree(SOURCE_ROOT"
    )
    assert "bpy.ops.wm.open_mainfile" not in source
    assert "bpy.ops.wm.quit_blender" not in source
    # Blender 5.1 has already begun RNA teardown when Python atexit runs;
    # addon_utils.disable() there can dereference destroyed RNA collections.
    assert "atexit.register" not in source
    assert "def runtime_cleanup" not in source


def test_success_report_has_complete_safety_schema_and_cleans_only_owned_process(
    tmp_path, monkeypatch
):
    exit_code, report_path, _config_dir = _run_fake_review(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["schema"] == launcher.REPORT_SCHEMA
    assert report["ok"] is True
    assert report["stage"] == "complete"
    assert report["process"]["owned"] is True
    assert report["process"]["pid"] == 424242
    assert report["process"]["returncode"] == 0
    assert "--factory-startup" not in report["process"]["command"]
    assert report["process"]["command"][1] == report["blend"]["requested"]
    assert report["module"]["under_temp_root"] is True
    assert report["module"]["nonpersistent_enable_requested"] is True
    assert report["one_package_identity"] is True
    assert report["inventory"]["complete"] is True
    assert report["profile"]["unchanged"] is True
    assert report["profile"]["before"]["enabled_addons"]
    assert report["profile"]["before"]["files"]["userpref.blend"]["sha256"]
    assert report["profile"]["after"]["files"]["startup.blend"]["sha256"]
    assert report["cleanup"]["owned_process_exited"] is True
    assert report["cleanup"]["temp_copy_removed"] is True
    assert not Path(report["cleanup"]["temp_root"]).exists()
    assert report["crash_report_path"] is None
    assert report["errors"] == []

    process = _FakeBlenderProcess.instances[0]
    assert process.env["FLUIDSIM_TEST_INHERITED"] == "yes"
    assert process.terminated is False
    assert process.killed is False


def test_optional_review_script_command_and_result_are_required(tmp_path, monkeypatch):
    exit_code, report_path, _config_dir = _run_fake_review(
        tmp_path, monkeypatch, with_review_script=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    process = _FakeBlenderProcess.instances[0]

    assert exit_code == 0
    assert report["visible_review"]["requested"] is True
    assert report["visible_review"]["ok"] is True
    assert report["visible_review"]["result"] == {"ok": True, "checks": []}
    assert report["process"]["command"][-2:] == [
        "--python", report["visible_review"]["script"]]
    assert launcher.VISIBLE_REVIEW_RESULT_ENV in process.env


def test_failed_runtime_review_never_publishes_review_ready(tmp_path, monkeypatch):
    stages = []
    write_json = launcher._atomic_write_json

    def record_stage(path, payload):
        if Path(path).name == "review.json":
            stages.append(payload.get("stage"))
        write_json(path, payload)

    monkeypatch.setattr(launcher, "_atomic_write_json", record_stage)
    _FakeBlenderProcess.visible_review_payload = {
        "ok": False,
        "errors": ["viewport simulation did not advance"],
    }

    exit_code, _report_path, _config_dir = _run_fake_review(
        tmp_path, monkeypatch, with_review_script=True)

    assert exit_code == 1
    assert "review_ready" not in stages


def test_manual_launch_only_publishes_identity_ready(
    tmp_path, monkeypatch, capsys
):
    stages = []
    write_json = launcher._atomic_write_json

    def record_stage(path, payload):
        if Path(path).name == "review.json":
            stages.append(payload.get("stage"))
        write_json(path, payload)

    monkeypatch.setattr(launcher, "_atomic_write_json", record_stage)

    exit_code, _report_path, _config_dir = _run_fake_review(
        tmp_path, monkeypatch)

    assert exit_code == 0
    assert "identity_ready" in stages
    assert "review_ready" not in stages
    output = capsys.readouterr().out
    assert "FLUIDSIM-PROFILE-SAFE-IDENTITY-PASS" in output
    assert "FLUIDSIM-PROFILE-SAFE-REVIEW-PASS" not in output


def test_gui_playback_probe_supports_profile_safe_review_without_repo_reload():
    source = Path(__file__).with_name(
        "blender_gui_playback_probe.py").read_text(encoding="utf-8")

    assert "FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT" in source
    assert "FLUIDSIM_PROFILE_SAFE_REVIEW_CONFIG" in source
    assert 'report["ok"] = report["passed"]' in source
    assert "enable_repo_addon" in source
    assert "if not profile_safe_result_path:" in source


def test_failed_runtime_review_still_captures_profile_drift(
    tmp_path, monkeypatch
):
    def mutate_profile():
        (tmp_path / "profile" / "config" / "userpref.blend").write_bytes(
            b"changed by failed review"
        )

    _FakeBlenderProcess.visible_review_payload = {
        "ok": False,
        "errors": ["viewport simulation did not advance"],
    }
    _FakeBlenderProcess.mutate_on_wait = mutate_profile

    exit_code, report_path, _config_dir = _run_fake_review(
        tmp_path, monkeypatch, with_review_script=True
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["profile"]["unchanged"] is False
    assert report["profile"]["drift"] == ["userpref.blend"]
    evidence = Path(report["profile"]["evidence_path"])
    assert (evidence / "before_userpref.blend").read_bytes() == b"preferences"
    assert (evidence / "after_userpref.blend").read_bytes() == (
        b"changed by failed review"
    )
    assert any("reported failure" in error for error in report["errors"])
    assert any("profile drift" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "did not write"),
        ("not-json", "malformed"),
        ({"ok": False, "errors": ["workflow failed"]}, "reported failure"),
    ],
)
def test_optional_review_script_fails_closed(
    tmp_path, monkeypatch, payload, message
):
    _FakeBlenderProcess.visible_review_payload = payload
    exit_code, report_path, _config_dir = _run_fake_review(
        tmp_path, monkeypatch, with_review_script=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["ok"] is False
    assert any(message in error for error in report["errors"])


def test_profile_drift_fails_and_retains_before_after_evidence(tmp_path, monkeypatch):
    def mutate_profile():
        (tmp_path / "profile" / "config" / "userpref.blend").write_bytes(
            b"changed by owned process"
        )

    _FakeBlenderProcess.mutate_on_wait = mutate_profile
    exit_code, report_path, _config_dir = _run_fake_review(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["ok"] is False
    assert report["profile"]["unchanged"] is False
    assert report["profile"]["drift"] == ["userpref.blend"]
    evidence = Path(report["profile"]["evidence_path"])
    assert evidence.is_dir()
    assert (evidence / "before_userpref.blend").read_bytes() == b"preferences"
    assert (evidence / "after_userpref.blend").read_bytes() == b"changed by owned process"
    assert report["cleanup"]["temp_copy_removed"] is True
    assert any("profile drift" in error for error in report["errors"])


def test_report_is_written_when_preflight_fails(tmp_path, monkeypatch):
    blend = tmp_path / "missing.blend"
    report_path = tmp_path / "report.json"
    args = launcher.build_parser().parse_args(
        ["--blend", str(blend), "--report", str(report_path), "--blender", "missing"]
    )
    monkeypatch.setattr(launcher, "_find_recent_crash", lambda *_args: None)

    exit_code = launcher.run_review(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["schema"] == launcher.REPORT_SCHEMA
    assert report["stage"] == "failed"
    assert report["process"]["pid"] is None
    assert report["cleanup"]["owned_process_exited"] is False
    assert any("review blend does not exist" in error for error in report["errors"])
