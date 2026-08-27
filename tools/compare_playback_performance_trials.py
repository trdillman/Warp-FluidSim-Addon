"""Validate and compare seven four-slot ABBA playback benchmark pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics


SCHEMA = "fluidsim-playback-paired-comparison-v1"
RAW_SCHEMA = "fluidsim-handler-playback-evidence-v2"
MEASUREMENT_KIND = "handler_driven_gui_playback"
HANDLER_IDENTITY = "fluidSim_warp_blender51.blender.playback._on_frame_change"
PAIR_IDS = tuple(f"pair-{index:02d}" for index in range(1, 8))
SLOTS = ("A1", "B1", "B2", "A2")
WARMUP_SAMPLES = 10
REQUIRED_SAMPLES = 60


def _require_head(value, name):
    value = str(value)
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{name} must be a full 40-character Git HEAD")
    return value


def _expected_order(pair_id):
    number = int(pair_id.rsplit("-", 1)[1])
    return "baseline-first" if number % 2 else "candidate-first"


def _expected_variant(order, slot):
    first = "baseline" if order == "baseline-first" else "candidate"
    second = "candidate" if first == "baseline" else "baseline"
    return first if slot.startswith("A") else second


def _percent_change(baseline, candidate):
    if baseline == 0.0:
        return None
    return (candidate - baseline) * 100.0 / baseline


def _p95(values):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot calculate p95 of an empty sequence")
    rank = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sample_stats(samples):
    warm = [float(value) for value in samples[WARMUP_SAMPLES:]]
    return {
        "cold_first": float(samples[0]),
        "warmup_count": WARMUP_SAMPLES,
        "warm_count": len(warm),
        "warm_median": statistics.median(warm),
        "warm_p95": _p95(warm),
    }


def _valid_samples(values):
    return (
        isinstance(values, list)
        and len(values) == REQUIRED_SAMPLES
        and all(
            type(value) in (int, float)
            and math.isfinite(value) and value > 0.0
            for value in values)
    )


def _memory_values(report):
    return {
        key: float(value)
        for key, value in report.items()
        if "memory" in key.lower()
        and type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0.0
    }


def _validate_trial(report, baseline_head, candidate_head, index):
    label = f"trial[{index}]"
    errors = []
    pair_id = report.get("pair_id")
    slot = report.get("abba_slot")
    order = report.get("order")
    if pair_id not in PAIR_IDS:
        errors.append(f"{label}: invalid pair ID {pair_id!r}")
    if slot not in SLOTS:
        errors.append(f"{label}: invalid ABBA slot {slot!r}")
    if pair_id in PAIR_IDS:
        expected_order = _expected_order(pair_id)
        if order != expected_order:
            errors.append(
                f"{label}: pair order violates alternating order; "
                f"expected {expected_order!r}, got {order!r}")
    if order in ("baseline-first", "candidate-first") and slot in SLOTS:
        variant = _expected_variant(order, slot)
        if report.get("variant") != variant:
            errors.append(
                f"{label}: declared variant does not match order/ABBA slot")
        expected_head = baseline_head if variant == "baseline" else candidate_head
        if (report.get("expected_git_head") != expected_head
                or report.get("git_head") != expected_head):
            errors.append(
                f"{label}: raw source does not match declared HEAD for {variant}")

    direct_requirements = {
        "passed": True,
        "approval_eligible": False,
        "external_paired_comparator_required": True,
        "git_available": True,
        "git_tree_dirty": False,
        "target_root_validated": True,
        "addon_enabled": True,
        "scene_registration_marker": True,
        "imported_from_expected_module_root": True,
        "runtime_inventory_matches": True,
        "process_pid_matches": True,
        "active_blend_matches": True,
        "viewport_non_empty": True,
        "viewport_changed": True,
        "session_active": True,
    }
    for field, expected in direct_requirements.items():
        if report.get(field) is not expected:
            if field == "passed":
                errors.append(f"{label}: raw report did not pass")
            elif field == "approval_eligible":
                errors.append(f"{label}: raw trial must declare approval_eligible=false")
            else:
                errors.append(f"{label}: required {field}={expected!r}")
    if report.get("schema") != RAW_SCHEMA:
        errors.append(f"{label}: wrong raw schema")
    if report.get("measurement_kind") != MEASUREMENT_KIND:
        errors.append(f"{label}: wrong measurement kind")
    if report.get("cleanup_owner") != "blender_mcp_launcher":
        errors.append(f"{label}: Blender MCP cleanup ownership is missing")
    inventory_digest = report.get("runtime_inventory_digest")
    if (not isinstance(inventory_digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", inventory_digest) is None):
        errors.append(f"{label}: runtime inventory digest is missing or invalid")
    if type(report.get("runtime_inventory_count")) is not int or report.get(
            "runtime_inventory_count", 0) <= 0:
        errors.append(f"{label}: runtime inventory count is missing or invalid")
    if report.get("runtime_inventory_missing") != []:
        errors.append(f"{label}: runtime inventory has missing tracked files")
    if report.get("runtime_inventory_mismatched") != []:
        errors.append(f"{label}: runtime inventory has mismatched tracked files")
    actual_pid = report.get("owned_process_pid")
    expected_pid = report.get("expected_process_pid")
    if (type(actual_pid) is not int or actual_pid <= 0
            or actual_pid != expected_pid):
        errors.append(f"{label}: exact positive process PID binding is invalid")
    if report.get("surface_attribute_profile") != "all":
        errors.append(f"{label}: raw trial must use surface profile 'all'")
    if report.get("required_frames") != REQUIRED_SAMPLES:
        errors.append(f"{label}: raw trial must require exactly 60 frames")
    if report.get("registered_handler_count") != 1:
        errors.append(f"{label}: registered handler is not unique")
    if report.get("registered_handler_identity") != HANDLER_IDENTITY:
        errors.append(f"{label}: registered handler identity is wrong")
    if report.get("direct_step_calls") != 0:
        errors.append(f"{label}: direct solver step calls were observed")
    for field in (
            "handler_invocations", "depsgraph_deliveries",
            "step_calls_inside_handler", "surface_publications"):
        if report.get(field, 0) < REQUIRED_SAMPLES:
            errors.append(f"{label}: insufficient {field}")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
            value is True for value in checks.values()):
        errors.append(f"{label}: raw report checks are not all true")
    if report.get("error"):
        errors.append(f"{label}: raw report contains an error")
    handler = report.get("all_handler_frame_samples")
    surface = report.get("surface_publication_frame_samples")
    if not _valid_samples(handler):
        errors.append(f"{label}: expected exactly 60 handler samples")
    if not _valid_samples(surface):
        errors.append(f"{label}: expected exactly 60 surface samples")
    return errors


def _paired_metric(trials, sample_field):
    samples = {"baseline": [], "candidate": []}
    for trial in trials:
        samples[trial["variant"]].extend(
            trial[sample_field][WARMUP_SAMPLES:])
    result = {}
    for variant in ("baseline", "candidate"):
        result[f"{variant}_warm_median"] = statistics.median(samples[variant])
        result[f"{variant}_warm_p95"] = _p95(samples[variant])
    result["percent_change"] = _percent_change(
        result["baseline_warm_median"], result["candidate_warm_median"])
    result["p95_percent_change"] = _percent_change(
        result["baseline_warm_p95"], result["candidate_warm_p95"])
    return result


def _aggregate_pair_metrics(pairs, field):
    changes = [pair[field]["percent_change"] for pair in pairs]
    p95_changes = [pair[field]["p95_percent_change"] for pair in pairs]
    return {
        "paired_percent_changes": changes,
        "paired_percent_change_mean": statistics.mean(changes),
        "paired_percent_change_median": statistics.median(changes),
        "paired_p95_percent_changes": p95_changes,
        "paired_p95_percent_change_mean": statistics.mean(p95_changes),
        "paired_p95_percent_change_median": statistics.median(p95_changes),
        "candidate_wins": sum(value < 0.0 for value in changes),
        "baseline_wins": sum(value > 0.0 for value in changes),
        "ties": sum(value == 0.0 for value in changes),
    }


def _memory_summary(reports):
    fields = sorted({key for report in reports for key in _memory_values(report)})
    summary = {}
    for field in fields:
        pair_changes = []
        complete = True
        for pair_id in PAIR_IDS:
            pair = [report for report in reports if report["pair_id"] == pair_id]
            values = {"baseline": [], "candidate": []}
            for report in pair:
                memory = _memory_values(report)
                if field in memory:
                    values[report["variant"]].append(memory[field])
            if len(values["baseline"]) != 2 or len(values["candidate"]) != 2:
                complete = False
                break
            pair_changes.append(_percent_change(
                statistics.mean(values["baseline"]),
                statistics.mean(values["candidate"])))
        if complete and all(value is not None for value in pair_changes):
            summary[field] = {
                "paired_percent_changes": pair_changes,
                "paired_percent_change_median": statistics.median(pair_changes),
                "candidate_wins": sum(value < 0.0 for value in pair_changes),
                "baseline_wins": sum(value > 0.0 for value in pair_changes),
                "ties": sum(value == 0.0 for value in pair_changes),
            }
        else:
            summary[field] = {"complete_across_pairs": False}
    return {"fields": summary}


def compare_trials(reports, *, baseline_head, candidate_head):
    baseline_head = _require_head(baseline_head, "baseline_head")
    candidate_head = _require_head(candidate_head, "candidate_head")
    reports = [dict(report) for report in reports]
    errors = []
    if len(reports) != 28:
        errors.append(f"expected exactly 28 raw trials, received {len(reports)}")
    seen = set()
    for index, report in enumerate(reports):
        key = (report.get("pair_id"), report.get("abba_slot"))
        if key in seen:
            errors.append(f"trial[{index}]: duplicate pair/slot {key!r}")
        seen.add(key)
        errors.extend(_validate_trial(
            report, baseline_head, candidate_head, index))
    expected = {(pair_id, slot) for pair_id in PAIR_IDS for slot in SLOTS}
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"missing pair/slot trials: {missing}")

    base = {
        "schema": SCHEMA,
        "approval_eligible": False,
        "baseline_head": baseline_head,
        "candidate_head": candidate_head,
        "warmup_samples_per_trial": WARMUP_SAMPLES,
        "evidence_completeness": {
            "complete": not errors,
            "errors": errors,
            "required_pairs": 7,
            "required_slots_per_pair": list(SLOTS),
            "raw_trial_count": len(reports),
        },
    }
    if errors:
        return {
            **base,
            "trials": [],
            "pairs": [],
            "aggregate": {},
            "memory": {"fields": {}},
            "optimization_recommendation": {
                "direction": "unavailable",
                "reason": "evidence is incomplete",
                "materiality_threshold_applied": False,
            },
        }

    slot_order = {slot: index for index, slot in enumerate(SLOTS)}
    reports.sort(key=lambda value: (
        value["pair_id"], slot_order[value["abba_slot"]]))
    trial_results = []
    for report in reports:
        trial_results.append({
            "pair_id": report["pair_id"],
            "abba_slot": report["abba_slot"],
            "order": report["order"],
            "variant": report["variant"],
            "head": report["git_head"],
            "handler": _sample_stats(report["all_handler_frame_samples"]),
            "surface": _sample_stats(
                report["surface_publication_frame_samples"]),
            "memory": _memory_values(report),
        })
    pairs = []
    for pair_id in PAIR_IDS:
        pair_trials = [report for report in reports if report["pair_id"] == pair_id]
        pairs.append({
            "pair_id": pair_id,
            "order": _expected_order(pair_id),
            "handler": _paired_metric(
                pair_trials, "all_handler_frame_samples"),
            "surface": _paired_metric(
                pair_trials, "surface_publication_frame_samples"),
        })
    aggregate = {
        field: _aggregate_pair_metrics(pairs, field)
        for field in ("handler", "surface")
    }
    handler = aggregate["handler"]
    surface = aggregate["surface"]
    candidate_faster = (
        handler["paired_percent_change_median"] < 0.0
        and surface["paired_percent_change_median"] < 0.0
        and handler["candidate_wins"] > handler["baseline_wins"]
        and surface["candidate_wins"] > surface["baseline_wins"])
    baseline_faster = (
        handler["paired_percent_change_median"] > 0.0
        and surface["paired_percent_change_median"] > 0.0
        and handler["baseline_wins"] > handler["candidate_wins"]
        and surface["baseline_wins"] > surface["candidate_wins"])
    direction = (
        "candidate_faster" if candidate_faster
        else "baseline_faster" if baseline_faster
        else "mixed")
    return {
        **base,
        "trials": trial_results,
        "pairs": pairs,
        "aggregate": aggregate,
        "memory": _memory_summary(reports),
        "optimization_recommendation": {
            "direction": direction,
            "reason": (
                "paired median direction and pair win counts agree on both "
                "handler and surface-publication timings"
                if direction != "mixed"
                else "handler and surface paired directions do not agree"),
            "materiality_threshold_applied": False,
        },
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", action="append", required=True)
    parser.add_argument("--baseline-head", required=True)
    parser.add_argument("--candidate-head", required=True)
    parser.add_argument("--json-out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.trial
    ]
    result = compare_trials(
        reports, baseline_head=args.baseline_head,
        candidate_head=args.candidate_head)
    output = Path(args.json_out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    complete = result["evidence_completeness"]["complete"]
    print(
        "PLAYBACK-PAIRED-EVIDENCE-COMPLETE"
        if complete else "PLAYBACK-PAIRED-EVIDENCE-INCOMPLETE",
        flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
