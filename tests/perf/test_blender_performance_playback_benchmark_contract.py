from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "acceptance" / "blender_performance_playback_benchmark.py"
SPEC = importlib.util.spec_from_file_location("playback_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _checkout(tmp_path):
    for relative in ("fluidSim_warp_blender51/__init__.py",):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return tmp_path


def _passing_report(required_frames=60):
    samples = [0.01] * required_frames
    return {
        "schema": benchmark.REPORT_SCHEMA,
        "measurement_kind": benchmark.MEASUREMENT_KIND,
        "approval_eligible": False,
        "external_paired_comparator_required": True,
        "required_frames": required_frames,
        "target_root_validated": True,
        "git_available": True,
        "git_head": "0123456789abcdef",
        "expected_git_head": "0123456789abcdef",
        "git_tree_dirty": False,
        "addon_enabled": True,
        "scene_registration_marker": True,
        "imported_from_expected_module_root": True,
        "runtime_inventory_matches": True,
        "active_blend_matches": True,
        "process_pid_matches": True,
        "cleanup_owner": "blender_mcp_launcher",
        "registered_handler_count": 1,
        "registered_handler_identity": benchmark.HANDLER_IDENTITY,
        "all_handler_frame_samples": samples,
        "surface_publication_frame_samples": samples,
        "handler_invocations": required_frames * 2,
        "depsgraph_deliveries": required_frames * 2,
        "step_calls_inside_handler": required_frames * 2,
        "surface_publications": required_frames,
        "surface_attribute_profile": "default",
        "requested_surface_attributes": ["motion", "velocity"],
        "published_surface_attributes": ["motion", "velocity"],
        "direct_step_calls": 0,
        "viewport_non_empty": True,
        "viewport_changed": True,
        "session_active": True,
        "error": "",
    }


def test_parse_args_requires_explicit_target_and_supports_attribute_profiles(tmp_path):
    args = benchmark._parse_args([
        "--addon-root", str(tmp_path),
        "--expected-head", "0123456789abcdef",
        "--expected-module-root", str(tmp_path / benchmark.ADDON_NAME),
        "--owned-process-pid", "123",
        "--blend", "demo.blend",
        "--json-out", "report.json",
        "--pair-id", "pair-03",
        "--order", "candidate-first",
        "--abba-slot", "B2",
        "--surface-attributes", "all",
        "--frames", "72",
    ])
    assert args.addon_root == str(tmp_path)
    assert args.surface_attributes == "all"
    assert args.frames == 72
    assert benchmark._trial_variant(args.order, args.abba_slot) == "baseline"

    with pytest.raises(SystemExit):
        benchmark._parse_args([
            "--blend", "demo.blend", "--json-out", "report.json"])
    with pytest.raises(SystemExit):
        benchmark._parse_args([
            "--addon-root", str(tmp_path),
            "--expected-head", "0123456789abcdef",
            "--expected-module-root", str(tmp_path / benchmark.ADDON_NAME),
            "--owned-process-pid", "123", "--blend", "demo.blend",
            "--json-out", "report.json", "--pair-id", "pair-01",
            "--order", "baseline-first", "--abba-slot", "A1",
            "--frames", "59"])


@pytest.mark.parametrize(
    ("handlers", "surfaces", "cadence", "expected"),
    ((60, 60, 1, 60), (60, 60, 2, 121), (90, 30, 3, 92)),
)
def test_cadence_frame_budget_covers_unknown_initial_phase(
        handlers, surfaces, cadence, expected):
    assert benchmark.cadence_frame_budget(
        handlers, surfaces, cadence) == expected


def test_cadence_frame_budget_rejects_non_positive_or_non_integer_values():
    for values in ((0, 60, 1), (60, -1, 1), (60, 60, 0), (60.0, 60, 1)):
        with pytest.raises(ValueError):
            benchmark.cadence_frame_budget(*values)


def test_target_root_validation_and_module_containment(tmp_path):
    root = _checkout(tmp_path)
    module_root = root / benchmark.ADDON_NAME
    assert benchmark.resolve_target_root(root) == root.resolve()
    assert benchmark.resolve_expected_module_root(module_root) == module_root.resolve()
    assert benchmark.module_is_under_target(
        module_root / "__init__.py", module_root)

    outside = tmp_path.parent / "installed" / "fluidSim_warp_blender51.py"
    outside.parent.mkdir(exist_ok=True)
    outside.write_text("", encoding="utf-8")
    assert not benchmark.module_is_under_target(outside, module_root)

    wrong_root = root / "copy"
    wrong_root.mkdir()
    (wrong_root / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="isolated copied addon package"):
        benchmark.resolve_expected_module_root(wrong_root)


def test_git_provenance_records_head_and_dirty_paths(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        stdout = (
            "0123456789abcdef\n" if command[-2:] == ["rev-parse", "HEAD"]
            else " M candidate.py\n?? evidence.json\n")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._git_provenance(tmp_path)
    assert result == {
        "git_available": True,
        "git_head": "0123456789abcdef",
        "git_tree_dirty": True,
        "git_status_porcelain": [" M candidate.py", "?? evidence.json"],
    }
    assert all(command[1:3] == ["-C", str(tmp_path)] for command in calls)


def test_report_accepts_real_handler_and_surface_samples():
    result = benchmark.evaluate_playback_report(_passing_report())
    assert result["passed"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("git_available", False, "git_available"),
        ("git_head", "wrong", "expected_git_head"),
        ("git_tree_dirty", True, "source_tree_clean"),
        ("addon_enabled", False, "addon_enabled"),
        ("scene_registration_marker", False, "scene_registration_marker"),
        ("imported_from_expected_module_root", False,
         "imported_from_expected_module_root"),
        ("runtime_inventory_matches", False, "runtime_inventory_matches"),
        ("active_blend_matches", False, "active_blend_matches"),
        ("process_pid_matches", False, "process_pid_matches"),
        ("direct_step_calls", 1, "no_direct_step"),
        ("surface_publication_frame_samples", [0.01] * 59,
         "surface_publication_samples"),
        ("viewport_changed", False, "viewport_changed"),
        ("published_surface_attributes", ["velocity"], "attribute_contract"),
    ),
)
def test_report_rejects_non_live_or_incomplete_evidence(field, value, failed_check):
    report = _passing_report()
    report[field] = value
    result = benchmark.evaluate_playback_report(report)
    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_report_accepts_off_profile_only_when_no_surface_attributes_publish():
    report = _passing_report()
    report.update({
        "surface_attribute_profile": "off",
        "requested_surface_attributes": [],
        "published_surface_attributes": [],
    })
    assert benchmark.evaluate_playback_report(report)["passed"] is True
    report["published_surface_attributes"] = ["velocity"]
    assert benchmark.evaluate_playback_report(report)["passed"] is False


def test_single_trial_can_pass_raw_checks_but_never_be_approval_eligible():
    result = benchmark.evaluate_playback_report(_passing_report())
    assert result["passed"] is True
    assert result["approval_eligible"] is False
    assert result["external_paired_comparator_required"] is True


def test_protected_finalization_restores_then_writes_without_owning_cleanup():
    calls = []
    written = []

    def restore():
        calls.append("restore")
        raise RuntimeError("restore boom")

    def build(errors):
        calls.append("build")
        return {"passed": False, "error": "; ".join(errors)}

    def write(report):
        calls.append("write")
        written.append(report)

    report, errors = benchmark._protected_finalize(
        restore=restore, build_report=build, write_report=write)
    assert calls == ["restore", "build", "write"]
    assert "restoration failed" in report["error"]
    assert errors and written == [report]


def test_failure_report_is_fail_closed_for_setup_error(tmp_path):
    args = benchmark._parse_args([
        "--addon-root", str(tmp_path), "--expected-head", "abc",
        "--expected-module-root", str(tmp_path / benchmark.ADDON_NAME),
        "--owned-process-pid", "123", "--blend", "demo.blend",
        "--json-out", "report.json", "--pair-id", "pair-01",
        "--order", "baseline-first", "--abba-slot", "A1"])
    report = benchmark._failure_report(args, error="disabled addon")
    assert report["passed"] is False
    assert report["approval_eligible"] is False
    assert report["error"] == "disabled addon"


def test_harness_never_reopens_file_or_requires_preference_membership():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "open_mainfile" not in source
    assert "preferences.addons" not in source
    assert "quit_blender" not in source


@pytest.mark.parametrize("failure", ("missing", "mismatch"))
def test_runtime_inventory_rejects_missing_or_mismatched_tracked_file(
        monkeypatch, tmp_path, failure):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime" / benchmark.ADDON_NAME
    source_file = source / benchmark.ADDON_NAME / "module.py"
    runtime_file = runtime / "module.py"
    source_file.parent.mkdir(parents=True)
    runtime_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"source bytes")
    if failure == "mismatch":
        runtime_file.write_bytes(b"different bytes")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout=b"fluidSim_warp_blender51/module.py\0", stderr=b"")

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    result = benchmark._runtime_inventory(source, runtime)
    assert result["runtime_inventory_matches"] is False
    assert bool(result[f"runtime_inventory_{failure}{'ed' if failure == 'mismatch' else ''}"])


def test_runtime_inventory_accepts_exact_tracked_copy(monkeypatch, tmp_path):
    source = tmp_path / "source"
    runtime = tmp_path / "runtime" / benchmark.ADDON_NAME
    source_file = source / benchmark.ADDON_NAME / "module.py"
    runtime_file = runtime / "module.py"
    source_file.parent.mkdir(parents=True)
    runtime_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"same bytes")
    runtime_file.write_bytes(b"same bytes")
    monkeypatch.setattr(
        benchmark.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=b"fluidSim_warp_blender51/module.py\0", stderr=b""))
    result = benchmark._runtime_inventory(source, runtime)
    assert result["runtime_inventory_matches"] is True
    assert result["runtime_inventory_count"] == 1
    assert len(result["runtime_inventory_digest"]) == 64


def test_animation_start_timer_is_one_shot():
    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    start_block = source.split("    def start():", 1)[1].split(
        "    bpy.app.timers.register(poll", 1)[0]

    assert "bpy.ops.screen.animation_play()" in start_block
    assert "            return None" in start_block
    assert "            return 0.1" not in start_block


def test_protected_finalization_reports_write_failure_after_restore():
    calls = []
    report, errors = benchmark._protected_finalize(
        restore=lambda: calls.append("restore"),
        build_report=lambda _errors: {"passed": False},
        write_report=lambda _report: (_ for _ in ()).throw(OSError("disk full")))
    assert calls == ["restore"]
    assert report == {"passed": False}
    assert any("report write failed" in value for value in errors)
