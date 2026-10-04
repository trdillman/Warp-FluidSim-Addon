"""Independently validate a fresh Blender GUI playback probe report."""

from __future__ import annotations

import argparse
import json
import ntpath
import os
import time


DEFAULT_MAX_AGE_SECONDS = 120.0
_RAW_FIELDS = (
    "blend",
    "generated_at_unix",
    "timed_out",
    "handler_count",
    "start_frame",
    "end_frame",
    "session_active",
    "solver_frame",
    "viewport_baseline_solver_frame",
    "device",
    "expected_domain_sdf_walls",
    "saved_domain_sdf_walls",
    "domain_sdf_walls",
    "particle_count",
    "viewport_particle_count",
    "viewport_changed",
    "error",
    "passed",
)


def _normalized_path(value):
    raw = os.fspath(value)
    if ntpath.isabs(raw):
        normalized = ntpath.normpath(raw)
    else:
        normalized = os.path.normpath(os.path.abspath(raw))
    # Case-fold and unify separators even when tests run under POSIX but compare
    # paths written by Blender on Windows.
    return normalized.replace("\\", "/").casefold()


def _same_path(left, right):
    return _normalized_path(left) == _normalized_path(right)


def validate_report(
    report,
    *,
    expected_blend=None,
    now=None,
    max_age_seconds=DEFAULT_MAX_AGE_SECONDS,
):
    """Raise ``RuntimeError`` unless independent invariants all pass."""
    now = time.time() if now is None else float(now)
    errors = []
    invariant_errors = [
        f"{field} is missing" for field in _RAW_FIELDS if field not in report
    ]

    generated = report.get("generated_at_unix")
    if type(generated) not in (int, float):
        if "generated_at_unix" in report:
            errors.append("generated_at_unix must be numeric")
    else:
        age = now - float(generated)
        if age < -5.0:
            errors.append(f"generated_at_unix is in the future ({age:.1f}s)")
        elif age > float(max_age_seconds):
            errors.append(
                f"report is stale ({age:.1f}s > {max_age_seconds:.1f}s)")

    if expected_blend:
        blend = report.get("blend")
        if not isinstance(blend, str) or not _same_path(blend, expected_blend):
            errors.append(
                f"blend identity mismatch: report={blend!r}, "
                f"expected={_normalized_path(expected_blend)!r}")

    if report.get("timed_out") is not False:
        invariant_errors.append(
            f"timed_out must be false, got {report.get('timed_out')!r}")
    if report.get("handler_count") != 1:
        invariant_errors.append(
            f"handler_count must be 1, got {report.get('handler_count')!r}")

    start = report.get("start_frame")
    end = report.get("end_frame")
    if type(start) is not int or type(end) is not int or end < start + 3:
        invariant_errors.append(
            f"timeline did not advance 3 frames: start={start!r}, end={end!r}")
    if report.get("session_active") is not True:
        invariant_errors.append("session_active is not true")

    solver_frame = report.get("solver_frame")
    baseline = report.get("viewport_baseline_solver_frame")
    if type(solver_frame) is not int or solver_frame < 3:
        invariant_errors.append(
            f"solver_frame must be >=3, got {solver_frame!r}")
    if type(baseline) is not int or type(solver_frame) is not int \
            or solver_frame <= baseline:
        invariant_errors.append(
            "solver did not advance beyond viewport baseline: "
            f"baseline={baseline!r}, solver={solver_frame!r}")
    if not isinstance(report.get("device"), str) \
            or not report["device"].startswith("cuda"):
        invariant_errors.append(f"device is not CUDA: {report.get('device')!r}")

    expected_walls = report.get("expected_domain_sdf_walls")
    if type(expected_walls) is not bool:
        invariant_errors.append(
            "expected_domain_sdf_walls must be a boolean, "
            f"got {expected_walls!r}")
    if report.get("saved_domain_sdf_walls") != expected_walls:
        invariant_errors.append(
            "saved_domain_sdf_walls does not match expectation")
    if report.get("domain_sdf_walls") != expected_walls:
        invariant_errors.append(
            "runtime domain_sdf_walls does not match expectation")

    particle_count = report.get("particle_count")
    if type(particle_count) is not int or particle_count <= 0:
        invariant_errors.append(
            f"particle_count must be >0, got {particle_count!r}")
    viewport_count = report.get("viewport_particle_count")
    if type(viewport_count) is not int or viewport_count <= 0:
        invariant_errors.append(
            f"viewport_particle_count must be >0, got {viewport_count!r}")
    if report.get("viewport_changed") is not True:
        invariant_errors.append("viewport_changed is not true")
    if report.get("error"):
        invariant_errors.append(
            f"probe error is non-empty: {report.get('error')!r}")

    recomputed_passed = not invariant_errors
    stored_passed = report.get("passed")
    if type(stored_passed) is not bool:
        errors.append(f"stored passed must be a boolean, got {stored_passed!r}")
    elif stored_passed != recomputed_passed:
        errors.append(
            "stored passed disagrees with recomputed verdict: "
            f"stored={stored_passed}, recomputed={recomputed_passed}")
    elif not stored_passed:
        errors.append("stored and recomputed playback verdicts are false")

    errors.extend(invariant_errors)
    if errors:
        raise RuntimeError(
            "saved-demo GUI timeline playback failed:\n- "
            + "\n- ".join(errors))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument(
        "--expected-blend",
        default=os.environ.get("FLUIDSIM_PLAYBACK_BLEND"),
    )
    parser.add_argument(
        "--max-age-seconds", type=float, default=DEFAULT_MAX_AGE_SECONDS)
    args = parser.parse_args(argv)
    with open(args.report, encoding="utf-8") as stream:
        report = json.load(stream)
    validate_report(
        report,
        expected_blend=args.expected_blend,
        max_age_seconds=args.max_age_seconds,
    )
    print("SAVED-DEMO-GUI-PLAYBACK-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
