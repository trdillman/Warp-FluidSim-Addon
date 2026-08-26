#!/usr/bin/env python3
"""Run and validate five independent species-transport profiler processes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = Path(__file__).with_name("profile_species_transport.py").resolve()
_PROFILE_SPEC = importlib.util.spec_from_file_location(
    "profile_species_transport_for_runner", PROFILE_PATH)
PROFILE = importlib.util.module_from_spec(_PROFILE_SPEC)
_PROFILE_SPEC.loader.exec_module(PROFILE)

REPETITIONS = 5
THRESHOLDS = {
    "repetitions": REPETITIONS,
    "transport_only_budget": 0.12,
    "diffusion_on_budget": 0.35,
    "transport_only_scaled_mad": 0.02,
    "diffusion_on_scaled_mad": 0.04,
    "transport_only_p95_delta_ms": 2.0,
    "diffusion_on_p95_delta_ms": 4.0,
    "disabled_median_ms": 10.0,
    "mad_scale": 1.4826,
    "upper_bound_scale": 2.5,
}
CHECK_KEYS = frozenset({
    "child_receipts_valid", "child_identity", "production_children",
    "transport_only_stability", "diffusion_on_stability",
    "transport_only_upper_bound", "diffusion_on_upper_bound",
    "transport_only_p95_delta_ms", "diffusion_on_p95_delta_ms",
    "disabled_median",
})


def _utc_now():
    return datetime.now(timezone.utc)


def _current_runner_identity():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {
        "git_head": head,
        "runtime_dependency_tree_sha256": (
            PROFILE._runtime_dependency_tree_sha256()),
        "runner_sha256": _sha256(__file__),
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _numbers_match(left, right):
    try:
        left_value, right_value = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(left_value) and math.isfinite(right_value)
            and math.isclose(left_value, right_value, rel_tol=1.0e-9,
                             abs_tol=1.0e-9))


def _scaled_mad(values):
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return THRESHOLDS["mad_scale"] * mad


def _aggregate_metrics(children):
    result = {}
    for case in ("transport_only", "diffusion_on"):
        medians = [
            float(child["metrics"][f"{case}_overhead_fraction"])
            for child in children]
        median = statistics.median(medians)
        scaled_mad = _scaled_mad(medians)
        result[case] = {
            "process_medians": medians,
            "median": median,
            "scaled_mad": scaled_mad,
            "upper_bound": median + THRESHOLDS["upper_bound_scale"] * scaled_mad,
            "worst_child_p95_delta_ms": max(
                float(child["metrics"][f"{case}_p95_delta_ms"])
                for child in children),
        }
    disabled = [float(child["metrics"]["disabled_median_ms"])
                for child in children]
    result["disabled_medians_ms"] = disabled
    result["max_disabled_median_ms"] = max(disabled)
    return result


def _identity_errors(children, paths):
    if len(children) != REPETITIONS or len(paths) != REPETITIONS:
        return ["child_count"]
    provenance = [child.get("provenance", {}) for child in children]
    pids = [item.get("process_id") for item in provenance]
    errors = []
    if len(set(pids)) != REPETITIONS or any(
            not isinstance(pid, int) or pid <= 0 for pid in pids):
        errors.append("child_identity")
    for key in (
            "executable", "git_head", "runtime_dependency_tree_sha256",
            "profiler_sha256", "cuda_device"):
        if any(item.get(key) != provenance[0].get(key)
               for item in provenance[1:]):
            errors.append("child_identity")
            break
    if any(child.get("configuration") != children[0].get("configuration")
           for child in children[1:]):
        errors.append("child_identity")
    for item, path in zip(provenance, paths):
        command = item.get("command")
        executable = item.get("executable")
        if (not isinstance(command, list) or not command
                or command[0] != executable or "--json-out" not in command):
            errors.append("child_identity")
            continue
        output_index = command.index("--json-out") + 1
        if (output_index >= len(command)
                or Path(command[output_index]).resolve() != path.resolve()):
            errors.append("child_identity")
    return sorted(set(errors))


def _aggregate_checks(children, paths, metrics):
    child_valid = [PROFILE.validate_report(child) == [] for child in children]
    identity_valid = _identity_errors(children, paths) == []
    production = all(
        child.get("evidence_kind") == "production"
        and child.get("status") == "pass"
        for child in children)
    return {
        "child_receipts_valid": len(children) == REPETITIONS and all(child_valid),
        "child_identity": identity_valid,
        "production_children": len(children) == REPETITIONS and production,
        "transport_only_stability": (
            metrics["transport_only"]["scaled_mad"]
            <= THRESHOLDS["transport_only_scaled_mad"]),
        "diffusion_on_stability": (
            metrics["diffusion_on"]["scaled_mad"]
            <= THRESHOLDS["diffusion_on_scaled_mad"]),
        "transport_only_upper_bound": (
            metrics["transport_only"]["upper_bound"]
            <= THRESHOLDS["transport_only_budget"]),
        "diffusion_on_upper_bound": (
            metrics["diffusion_on"]["upper_bound"]
            <= THRESHOLDS["diffusion_on_budget"]),
        "transport_only_p95_delta_ms": (
            metrics["transport_only"]["worst_child_p95_delta_ms"]
            <= THRESHOLDS["transport_only_p95_delta_ms"]),
        "diffusion_on_p95_delta_ms": (
            metrics["diffusion_on"]["worst_child_p95_delta_ms"]
            <= THRESHOLDS["diffusion_on_p95_delta_ms"]),
        "disabled_median": (
            metrics["max_disabled_median_ms"]
            <= THRESHOLDS["disabled_median_ms"]),
    }


def _runner_provenance(command=None):
    return {
        "timestamp_utc": _utc_now().isoformat(),
        "executable": str(Path(sys.executable).resolve()),
        "process_id": os.getpid(),
        "command": list(command or sys.argv),
        **_current_runner_identity(),
    }


def _validate_runner_provenance(provenance):
    required = {
        "timestamp_utc", "executable", "process_id", "command", "git_head",
        "runtime_dependency_tree_sha256", "runner_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        return ["runner_provenance"]
    errors = []
    try:
        current = _current_runner_identity()
    except (OSError, subprocess.SubprocessError):
        return ["runner_identity_unavailable"]
    comparisons = {
        "git_head": "stale_runner_git_head",
        "runtime_dependency_tree_sha256": (
            "stale_runner_runtime_dependency_tree"),
        "runner_sha256": "stale_runner_hash",
    }
    for key, error in comparisons.items():
        if provenance.get(key) != current[key]:
            errors.append(error)
    try:
        timestamp = datetime.fromisoformat(str(provenance["timestamp_utc"]))
        age = (_utc_now() - timestamp).total_seconds()
        if (age > PROFILE.MAX_EVIDENCE_AGE_SECONDS
                or age < -PROFILE.MAX_FUTURE_CLOCK_SKEW_SECONDS):
            errors.append("stale_runner_timestamp")
    except (TypeError, ValueError):
        errors.append("runner_provenance")
    return errors


def build_aggregate_report(child_paths, *, command=None):
    """Build an aggregate while retaining every child as a separate receipt."""
    paths = [Path(path).resolve() for path in child_paths]
    children = [_read_json(path) for path in paths]
    manifests = []
    for index, (path, child) in enumerate(zip(paths, children)):
        provenance = child.get("provenance", {})
        manifests.append({
            "index": index,
            "receipt_path": str(path),
            "receipt_sha256": _sha256(path),
            "process_id": provenance.get("process_id"),
            "executable": provenance.get("executable"),
            "command": provenance.get("command"),
        })
    metrics = _aggregate_metrics(children)
    checks = _aggregate_checks(children, paths, metrics)
    return {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "thresholds": dict(THRESHOLDS),
        "runner_provenance": _runner_provenance(command),
        "children": manifests,
        "metrics": metrics,
        "checks": checks,
    }


def _metrics_match(supplied, expected):
    if not isinstance(supplied, dict) or set(supplied) != set(expected):
        return False
    for key, expected_value in expected.items():
        value = supplied.get(key)
        if isinstance(expected_value, dict):
            if not _metrics_match(value, expected_value):
                return False
        elif isinstance(expected_value, list):
            if (not isinstance(value, list) or len(value) != len(expected_value)
                    or any(not _numbers_match(left, right)
                           for left, right in zip(value, expected_value))):
                return False
        elif not _numbers_match(value, expected_value):
            return False
    return True


def validate_aggregate_report(report):
    """Reload children and recompute all aggregate metrics and checks."""
    errors = []
    if not isinstance(report, dict):
        return ["report", "status"]
    if report.get("schema_version") != 1:
        errors.append("schema_version")
    if report.get("thresholds") != THRESHOLDS:
        errors.append("thresholds")
    errors.extend(_validate_runner_provenance(report.get("runner_provenance")))
    manifests = report.get("children")
    required_manifest = {
        "index", "receipt_path", "receipt_sha256", "process_id",
        "executable", "command",
    }
    if (not isinstance(manifests, list) or len(manifests) != REPETITIONS
            or any(not isinstance(item, dict) or set(item) != required_manifest
                   for item in manifests)
            or [item.get("index") for item in manifests]
            != list(range(REPETITIONS))):
        errors.append("child_count")
        manifests = []

    children = []
    paths = []
    for manifest in manifests:
        path = Path(manifest["receipt_path"]).resolve()
        if not path.is_file():
            errors.append("child_receipt_missing")
            continue
        if _sha256(path) != manifest["receipt_sha256"]:
            errors.append("child_receipt_hash")
            continue
        try:
            child = _read_json(path)
        except (OSError, ValueError, TypeError):
            errors.append("child_receipt_json")
            continue
        provenance = child.get("provenance", {})
        if (manifest["process_id"] != provenance.get("process_id")
                or manifest["executable"] != provenance.get("executable")
                or manifest["command"] != provenance.get("command")):
            errors.append("child_identity")
        children.append(child)
        paths.append(path)

    recomputed_checks = None
    if len(children) == REPETITIONS:
        identity_errors = _identity_errors(children, paths)
        errors.extend(identity_errors)
        child_errors = [PROFILE.validate_report(child) for child in children]
        if any(child_errors):
            errors.append("child_receipts_valid")
            errors.extend(
                error for child_result in child_errors for error in child_result
                if error.startswith("stale_")
                or error == "current_identity_unavailable")
        if any(child.get("evidence_kind") != "production"
               or child.get("status") != "pass" for child in children):
            errors.append("production_children")
        metrics = _aggregate_metrics(children)
        if not _metrics_match(report.get("metrics"), metrics):
            errors.append("metrics_consistency")
        recomputed_checks = _aggregate_checks(children, paths, metrics)
        supplied_checks = report.get("checks")
        if (not isinstance(supplied_checks, dict)
                or set(supplied_checks) != CHECK_KEYS
                or supplied_checks != recomputed_checks):
            errors.append("checks")
        for key, value in recomputed_checks.items():
            if not value:
                errors.append(key)
    else:
        errors.append("child_count")

    expected_status = (
        "pass" if recomputed_checks is not None
        and all(recomputed_checks.values()) and not errors else "fail")
    if report.get("status") != expected_status or errors:
        errors.append("status")
    return sorted(set(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--particles", type=int, default=200_000)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--child-dir")
    args = parser.parse_args(argv)
    output = Path(args.json_out).resolve()
    child_dir = (Path(args.child_dir).resolve() if args.child_dir
                 else output.with_name(f"{output.stem}-children"))
    child_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = [child_dir / f"species-transport-{index + 1:02d}.json"
             for index in range(REPETITIONS)]
    existing = [path for path in [output, *paths] if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite retained profile evidence: "
            + ", ".join(str(path) for path in existing))

    # Make the owned evidence paths visible before the first child snapshots
    # git status. Porcelain status then stays identical as each child replaces
    # its own placeholder, including when the evidence directory is untracked.
    output.touch()
    for path in paths:
        path.touch()

    for path in paths:
        command = [
            str(Path(sys.executable).resolve()), str(PROFILE_PATH),
            "--device", args.device, "--frames", str(args.frames),
            "--warmup", str(args.warmup), "--particles", str(args.particles),
            "--json-out", str(path.resolve()),
        ]
        subprocess.run(command, cwd=ROOT, check=False)
    missing = []
    for path in paths:
        try:
            _read_json(path)
        except (OSError, ValueError, TypeError):
            missing.append(path)
    if missing:
        raise RuntimeError(
            "profiler subprocess did not retain receipt(s): "
            + ", ".join(str(path) for path in missing))

    report = build_aggregate_report(paths, command=[
        str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
        *(list(argv) if argv is not None else sys.argv[1:]),
    ])
    validation_errors = validate_aggregate_report(report)
    if validation_errors:
        report["status"] = "fail"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf8")
    print("SPECIES-TRANSPORT-REPETITIONS-" + report["status"].upper())
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
