from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "compare_playback_performance_trials.py"
SPEC = importlib.util.spec_from_file_location("trial_comparator", SCRIPT)
comparator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparator)

BASELINE_HEAD = "a" * 40
CANDIDATE_HEAD = "b" * 40
SLOTS = ("A1", "B1", "B2", "A2")


def _variant(order, slot):
    first = "baseline" if order == "baseline-first" else "candidate"
    second = "candidate" if first == "baseline" else "baseline"
    return first if slot.startswith("A") else second


def _trial(pair_number, slot, *, mutate=None):
    order = "baseline-first" if pair_number % 2 else "candidate-first"
    variant = _variant(order, slot)
    head = BASELINE_HEAD if variant == "baseline" else CANDIDATE_HEAD
    handler = 2.0 if variant == "baseline" else 1.5
    surface = 4.0 if variant == "baseline" else 3.0
    report = {
        "schema": "fluidsim-handler-playback-evidence-v2",
        "measurement_kind": "handler_driven_gui_playback",
        "passed": True,
        "approval_eligible": False,
        "external_paired_comparator_required": True,
        "required_frames": 60,
        "pair_id": f"pair-{pair_number:02d}",
        "order": order,
        "abba_slot": slot,
        "variant": variant,
        "expected_git_head": head,
        "git_head": head,
        "git_available": True,
        "git_tree_dirty": False,
        "target_root_validated": True,
        "addon_enabled": True,
        "scene_registration_marker": True,
        "imported_from_expected_module_root": True,
        "runtime_inventory_matches": True,
        "runtime_inventory_digest": "c" * 64,
        "runtime_inventory_count": 123,
        "runtime_inventory_missing": [],
        "runtime_inventory_mismatched": [],
        "process_pid_matches": True,
        "owned_process_pid": 4321,
        "expected_process_pid": 4321,
        "active_blend_matches": True,
        "cleanup_owner": "blender_mcp_launcher",
        "registered_handler_count": 1,
        "registered_handler_identity": (
            "fluidSim_warp_blender51.blender.playback._on_frame_change"),
        "handler_invocations": 60,
        "depsgraph_deliveries": 60,
        "step_calls_inside_handler": 60,
        "surface_publications": 60,
        "direct_step_calls": 0,
        "viewport_non_empty": True,
        "viewport_changed": True,
        "session_active": True,
        "surface_attribute_profile": "all",
        "all_handler_frame_samples": [9.0] + [handler] * 59,
        "surface_publication_frame_samples": [12.0] + [surface] * 59,
        "peak_memory_bytes": 1000 if variant == "baseline" else 900,
        "checks": {"raw_contract": True, "no_direct_step": True},
        "error": "",
    }
    if mutate:
        mutate(report)
    return report


def _dataset():
    return [
        _trial(pair, slot)
        for pair in range(1, 8)
        for slot in SLOTS
    ]


def test_complete_abba_evidence_reports_cold_warm_pairing_and_direction():
    result = comparator.compare_trials(
        _dataset(), baseline_head=BASELINE_HEAD,
        candidate_head=CANDIDATE_HEAD)
    assert result["evidence_completeness"]["complete"] is True
    assert result["evidence_completeness"]["errors"] == []
    assert len(result["trials"]) == 28
    assert result["trials"][0]["handler"]["cold_first"] == 9.0
    assert result["trials"][0]["handler"]["warm_count"] == 50
    assert result["trials"][0]["handler"]["warm_median"] == 2.0
    assert result["pairs"][0]["handler"]["percent_change"] == -25.0
    assert result["aggregate"]["handler"]["candidate_wins"] == 7
    assert result["aggregate"]["surface"]["candidate_wins"] == 7
    assert result["aggregate"]["handler"]["paired_percent_change_median"] == -25.0
    assert result["memory"]["fields"]["peak_memory_bytes"]["candidate_wins"] == 7
    assert result["optimization_recommendation"]["direction"] == "candidate_faster"
    assert result["optimization_recommendation"]["materiality_threshold_applied"] is False
    assert result["approval_eligible"] is False


def test_missing_abba_slot_fails_evidence_completeness_without_recommendation():
    reports = _dataset()
    reports.pop()
    result = comparator.compare_trials(
        reports, baseline_head=BASELINE_HEAD, candidate_head=CANDIDATE_HEAD)
    assert result["evidence_completeness"]["complete"] is False
    assert any("exactly 28" in error for error in result["evidence_completeness"]["errors"])
    assert result["optimization_recommendation"]["direction"] == "unavailable"


@pytest.mark.parametrize(
    "mutation,expected",
    (
        (lambda report: report.update(order="candidate-first"), "alternating order"),
        (lambda report: report.update(git_head="c" * 40), "declared HEAD"),
        (lambda report: report.update(passed=False), "raw report did not pass"),
        (lambda report: report.update(approval_eligible=True), "approval_eligible=false"),
        (lambda report: report.update(checks={"raw_contract": False}), "checks are not all true"),
        (lambda report: report.update(surface_attribute_profile="default"), "surface profile 'all'"),
        (lambda report: report.update(runtime_inventory_matches=False), "runtime_inventory_matches"),
        (lambda report: report.update(process_pid_matches=False), "process_pid_matches"),
        (lambda report: report.update(expected_process_pid=9999), "process PID binding"),
        (lambda report: report.update(runtime_inventory_digest="bad"), "inventory digest"),
        (lambda report: report.update(active_blend_matches=False), "active_blend_matches"),
        (lambda report: report.update(all_handler_frame_samples=[1.0] * 59), "60 handler samples"),
    ),
)
def test_raw_contract_violations_fail_closed(mutation, expected):
    reports = _dataset()
    mutation(reports[0])
    result = comparator.compare_trials(
        reports, baseline_head=BASELINE_HEAD, candidate_head=CANDIDATE_HEAD)
    assert result["evidence_completeness"]["complete"] is False
    assert any(expected in error for error in result["evidence_completeness"]["errors"])


def test_duplicate_pair_slot_fails_closed():
    reports = _dataset()
    reports[-1] = dict(reports[-2])
    result = comparator.compare_trials(
        reports, baseline_head=BASELINE_HEAD, candidate_head=CANDIDATE_HEAD)
    assert result["evidence_completeness"]["complete"] is False
    assert any("duplicate pair/slot" in error for error in result["evidence_completeness"]["errors"])


def test_cli_writes_deterministic_json_and_returns_success(tmp_path):
    paths = []
    for index, report in enumerate(_dataset()):
        path = tmp_path / f"trial-{index:02d}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "comparison.json"
    argv = [
        "--baseline-head", BASELINE_HEAD,
        "--candidate-head", CANDIDATE_HEAD,
        "--json-out", str(output),
    ]
    for path in reversed(paths):
        argv.extend(("--trial", str(path)))
    assert comparator.main(argv) == 0
    first = output.read_text(encoding="utf-8")
    assert comparator.main(argv) == 0
    assert output.read_text(encoding="utf-8") == first
    assert json.loads(first)["evidence_completeness"]["complete"] is True


def test_full_heads_are_required():
    with pytest.raises(ValueError, match="40-character"):
        comparator.compare_trials(
            _dataset(), baseline_head="abc", candidate_head=CANDIDATE_HEAD)
