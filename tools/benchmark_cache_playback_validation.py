"""Run and validate fresh-process cache-playback benchmark evidence.

The worker is ``tests/blender_cache_playback_benchmark.py`` and must run in
Blender 5.1.  This module deliberately imports no Blender modules, so report
validation and paired-campaign orchestration run under ordinary system Python.
Every campaign leg is a new Blender process.  Seven counterbalanced ABBA/BAAB
pairs are required; the resulting observations are never approval-eligible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

WORKER_SCHEMA = "fluidsim-cache-playback-benchmark-v1"
CAMPAIGN_SCHEMA = "fluidsim-cache-playback-abba-v1"
WORKER_RESULT = "CACHE-PLAYBACK-BENCHMARK-PASS"
CAMPAIGN_RESULT = "CACHE-PLAYBACK-ABBA-PASS"
MEASUREMENT_KIND = "targeted_cache_playback_handler_equivalent"
DEFAULT_PAIRS = 7
MIN_CACHE_FRAMES = 70
WARMUP_FRAMES = 10
MEASURED_FRAMES = 60
PACKAGE_NAME = "fluidSim_warp_blender51"
TIMING_FIELDS = (
    "total_seconds",
    "read_frame_seconds",
    "metadata_load_seconds",
    "signature_bundle_seconds",
    "publication_seconds",
)
COUNT_FIELDS = (
    "play_cache_frame",
    "read_frame",
    "manifest_metadata_load",
    "signature_bundle",
    "output_publication",
)
CORRECTNESS_SAMPLE_FIELDS = (
    "published_payload_sha256",
    "published_particle_count",
    "published_whitewater_count",
    "published_surface_vertex_count",
    "published_surface_triangle_count",
    "status",
)
CORRECTNESS_FIXTURE_FIELDS = (
    "mesh_name",
    "role",
    "static",
    "vertex_count",
    "triangle_count",
    "matrix_world_sha256",
    "signature_before",
    "signature_after",
)
CORRECTNESS_CACHE_FIELDS = (
    "format_version",
    "frame_start",
    "frame_end",
    "committed_frame_count",
    "manifest_sha256_before",
    "manifest_sha256_after",
    "cache_tree_sha256_before",
    "cache_tree_sha256_after",
    "metadata_identity",
)


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normal_path(value: object) -> str:
    if not isinstance(value, (str, os.PathLike)):
        return ""
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _path_within(path: object, parent: object) -> bool:
    normalized_path = _normal_path(path)
    normalized_parent = _normal_path(parent)
    try:
        return os.path.commonpath((normalized_path, normalized_parent)) == normalized_parent
    except ValueError:
        return False


def _nearest_rank_p95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one sample is required")
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def alternating_abba_schedule(pairs: int = DEFAULT_PAIRS) -> list[dict[str, object]]:
    """Return alternating ABBA/BAAB legs to counterbalance execution drift."""
    if type(pairs) is not int or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    schedule: list[dict[str, object]] = []
    for pair_index in range(pairs):
        order = (
            ("baseline", "candidate", "candidate", "baseline")
            if pair_index % 2 == 0
            else ("candidate", "baseline", "baseline", "candidate")
        )
        for position, subject in enumerate(order, 1):
            schedule.append({
                "pair": pair_index + 1,
                "position": position,
                "subject": subject,
            })
    return schedule


def _identity_errors(identity: object) -> list[str]:
    if not isinstance(identity, dict):
        return ["code_identity must be an object"]
    errors: list[str] = []
    for key in ("addon_root", "git_root"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            errors.append(f"code_identity.{key} is required")
    if not isinstance(identity.get("git_head"), str) or len(identity["git_head"]) != 40:
        errors.append("code_identity.git_head must be a full commit id")
    if identity.get("checkout_dirty") is not False:
        errors.append("benchmark requires a clean checkout")
    if identity.get("addon_dirty") is not False:
        errors.append("benchmark requires a clean add-on tree")
    for key in ("addon_tree_sha256", "worker_sha256"):
        if not _valid_digest(identity.get(key)):
            errors.append(f"code_identity.{key} must be sha256")
    return errors


def validate_worker_report(report: object) -> list[str]:
    """Fail-closed validation for one fresh Blender playback observation."""
    if not isinstance(report, dict):
        return ["report must be an object"]
    errors: list[str] = []
    if report.get("schema") != WORKER_SCHEMA:
        errors.append(f"schema must be {WORKER_SCHEMA}")
    if report.get("result") != WORKER_RESULT:
        errors.append(f"result must be {WORKER_RESULT}")
    if report.get("measurement_kind") != MEASUREMENT_KIND:
        errors.append(f"measurement_kind must be {MEASUREMENT_KIND}")
    if report.get("approval_eligible") is not False:
        errors.append("targeted benchmark must disable approval eligibility")

    identity = report.get("code_identity")
    errors.extend(_identity_errors(identity))
    addon_root = identity.get("addon_root", "") if isinstance(identity, dict) else ""

    environment = report.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
    else:
        if not str(environment.get("blender_version", "")).startswith("5.1"):
            errors.append("environment must identify Blender 5.1")
        for key in (
            "blender_executable", "python_executable", "python_version",
            "addon_module_path", "playback_module_path", "cache_module_path",
        ):
            if not isinstance(environment.get(key), str) or not environment[key]:
                errors.append(f"environment.{key} is required")
        if type(environment.get("process_id")) is not int or environment["process_id"] < 1:
            errors.append("environment.process_id must be positive")
        if environment.get("fresh_process_import") is not True:
            errors.append("environment must prove a fresh process import")
        if environment.get("background") is not True:
            errors.append("environment must prove background Blender")
        expected_paths = {
            "addon_module_path": Path(addon_root) / PACKAGE_NAME / "__init__.py",
            "playback_module_path": Path(addon_root) / PACKAGE_NAME / "blender" / "playback.py",
            "cache_module_path": Path(addon_root) / PACKAGE_NAME / "io" / "cache.py",
        }
        for key, expected in expected_paths.items():
            actual = environment.get(key)
            if addon_root and isinstance(actual, str) and _normal_path(actual) != _normal_path(expected):
                errors.append(f"environment.{key} does not resolve inside the recorded add-on")

    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("configuration must be an object")
        configuration = {}
    if configuration.get("cache_frames") != MIN_CACHE_FRAMES:
        errors.append(f"configuration.cache_frames must be {MIN_CACHE_FRAMES}")
    if configuration.get("warmup_frames") != WARMUP_FRAMES:
        errors.append(f"configuration.warmup_frames must be {WARMUP_FRAMES}")
    if configuration.get("measured_frames") != MEASURED_FRAMES:
        errors.append(f"configuration.measured_frames must be {MEASURED_FRAMES}")
    expected_warmup = list(range(1, WARMUP_FRAMES + 1))
    expected_measured = list(range(WARMUP_FRAMES + 1, MIN_CACHE_FRAMES + 1))
    if configuration.get("warmup_sequence") != expected_warmup:
        errors.append("warmup_sequence must cover frames 1..10 exactly")
    if configuration.get("measured_sequence") != expected_measured:
        errors.append("measured_sequence must cover 60 distinct frames 11..70")
    if configuration.get("distinct_measured_frames") is not True:
        errors.append("configuration must affirm distinct measured frames")
    if configuration.get("cache_construction_outside_timed_region") is not True:
        errors.append("cache construction must be outside the timed region")
    if configuration.get("timing_scope") != "play_cache_frame":
        errors.append("timing_scope must be play_cache_frame")
    playback_read_api = configuration.get("playback_read_api")
    signature_validation_mode = configuration.get("signature_validation_mode")
    valid_runtime_modes = {
        ("read_frame", "per_frame"),
        ("read_frame_for_playback", "static_generation"),
    }
    if (playback_read_api, signature_validation_mode) not in valid_runtime_modes:
        errors.append(
            "configuration must identify a supported playback read/signature mode"
        )

    fixture = report.get("fixture")
    if not isinstance(fixture, dict):
        errors.append("fixture must be an object")
    else:
        if fixture.get("role") != "SOURCE" or fixture.get("static") is not True:
            errors.append("fixture must be a static mesh source")
        if (
            type(fixture.get("vertex_count")) is not int
            or fixture["vertex_count"] < 4096
            or type(fixture.get("triangle_count")) is not int
            or fixture["triangle_count"] < 8000
        ):
            errors.append("fixture must be mesh-source-heavy")
        if not _valid_digest(fixture.get("matrix_world_sha256")):
            errors.append("fixture matrix digest must be sha256")
        before = fixture.get("signature_before")
        after = fixture.get("signature_after")
        if not isinstance(before, dict) or any(
            not _valid_digest(before.get(key))
            for key in ("scene_key", "settings_signature", "source_signature")
        ):
            errors.append("fixture signature_before must contain exact signature digests")
        if before != after:
            errors.append("static fixture signature changed during playback")

    cache = report.get("cache_provenance")
    if not isinstance(cache, dict):
        errors.append("cache_provenance must be an object")
    else:
        if cache.get("destination_initially_empty") is not True:
            errors.append("cache destination must be clean before fixture construction")
        if cache.get("outside_addon_root") is not True:
            errors.append("cache must be outside the add-on root")
        if cache.get("format_version") != 2:
            errors.append("cache format_version must be 2")
        if (
            cache.get("frame_start") != 1
            or cache.get("frame_end") != MIN_CACHE_FRAMES
            or cache.get("committed_frame_count") != MIN_CACHE_FRAMES
        ):
            errors.append("cache must commit exactly frames 1..70")
        for key in (
            "manifest_sha256_before", "manifest_sha256_after",
            "cache_tree_sha256_before", "cache_tree_sha256_after",
            "metadata_identity",
        ):
            if not _valid_digest(cache.get(key)):
                errors.append(f"cache_provenance.{key} must be sha256")
        if (
            cache.get("immutable_after_build") is not True
            or cache.get("manifest_sha256_before") != cache.get("manifest_sha256_after")
            or cache.get("cache_tree_sha256_before") != cache.get("cache_tree_sha256_after")
        ):
            errors.append("immutable cache changed during playback")
        if addon_root and isinstance(cache.get("cache_path"), str):
            cache_path = _normal_path(cache["cache_path"])
            root_path = _normal_path(addon_root)
            if _path_within(cache_path, root_path):
                errors.append("cache path must be external to the add-on root")

    counts = report.get("publication_counts")
    if not isinstance(counts, dict):
        errors.append("publication_counts must be an object")
    else:
        expected_counts = {
            "play_cache_frame": MEASURED_FRAMES,
            "read_frame": MEASURED_FRAMES,
            "manifest_metadata_load": MEASURED_FRAMES,
            "signature_bundle": (
                0 if signature_validation_mode == "static_generation"
                else MEASURED_FRAMES
            ),
            "output_publication": MEASURED_FRAMES,
        }
        for key, expected_count in expected_counts.items():
            if counts.get(key) != expected_count:
                errors.append(
                    f"{key} count must equal {expected_count}, got {counts.get(key)!r}"
                )
    warmup_counts = report.get("warmup_counts")
    if not isinstance(warmup_counts, dict):
        errors.append("warmup_counts must be an object")
    else:
        expected_warmup_counts = {
            "play_cache_frame": WARMUP_FRAMES,
            "read_frame": WARMUP_FRAMES,
            "manifest_metadata_load": WARMUP_FRAMES,
            "signature_bundle": (
                1 if signature_validation_mode == "static_generation"
                else WARMUP_FRAMES
            ),
            "output_publication": WARMUP_FRAMES,
        }
        for key, expected_count in expected_warmup_counts.items():
            if warmup_counts.get(key) != expected_count:
                errors.append(
                    f"warmup {key} count must equal {expected_count}, "
                    f"got {warmup_counts.get(key)!r}"
                )

    contention = report.get("contention")
    if not isinstance(contention, dict):
        errors.append("contention must be an object")
    else:
        if contention.get("query_available") is not True:
            errors.append("contention.query_available must be true")
        if contention.get("classification") != "zero-other-gpu-processes":
            errors.append("contention classification must prove zero other GPU processes")
        if contention.get("other_gpu_processes") != 0:
            errors.append("contention must prove zero other GPU processes")
        if not isinstance(contention.get("method"), str) or not contention["method"]:
            errors.append("contention.method is required")
        if (
            not isinstance(contention.get("query_executable"), str)
            or not contention["query_executable"]
        ):
            errors.append("contention.query_executable is required")
        if contention.get("query_command") != [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]:
            errors.append("contention query_command must identify the bounded process query")
        observations = contention.get("samples")
        if (
            not isinstance(observations, list)
            or len(observations) != 2
            or [
                item.get("phase") if isinstance(item, dict) else None
                for item in observations
            ] != ["before", "after"]
        ):
            errors.append("contention must contain before and after query samples")
            observations = []
        other_pids: set[int] = set()
        own_pid = environment.get("process_id") if isinstance(environment, dict) else None
        for index, observation in enumerate(observations):
            if observation.get("query_available") is not True:
                errors.append(f"contention sample {index} query_available must be true")
            if observation.get("returncode") != 0:
                errors.append(f"contention sample {index} query must succeed")
            for key in ("raw_stdout_lines", "raw_stderr_lines"):
                value = observation.get(key)
                if not isinstance(value, list) or any(
                    not isinstance(line, str)
                    for line in (value if isinstance(value, list) else ())
                ):
                    errors.append(f"contention sample {index}.{key} must be a string list")
            classifications = observation.get("classifications")
            if not isinstance(classifications, list):
                errors.append(f"contention sample {index}.classifications must be an array")
                continue
            for classified in classifications:
                if not isinstance(classified, dict):
                    errors.append("contention classifications must be unambiguous")
                    continue
                classification = classified.get("classification")
                pid = classified.get("pid")
                if (
                    classification not in {"self", "other"}
                    or type(pid) is not int
                    or pid < 1
                    or not isinstance(classified.get("raw"), str)
                    or not classified["raw"]
                    or not isinstance(classified.get("process_name"), str)
                    or not classified["process_name"]
                    or not isinstance(classified.get("gpu_uuid"), str)
                    or not classified["gpu_uuid"]
                ):
                    errors.append("contention classifications must be unambiguous")
                    continue
                if classification == "self" and pid != own_pid:
                    errors.append("contention self classification must match worker PID")
                if classification == "other":
                    if pid == own_pid:
                        errors.append("contention other classification cannot use worker PID")
                    other_pids.add(pid)
        if other_pids:
            errors.append("contention classifications contain other GPU processes")
        if contention.get("other_gpu_processes") != len(other_pids):
            errors.append("contention other_gpu_processes does not match raw classifications")

    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != MEASURED_FRAMES:
        errors.append("samples must contain exactly 60 measured frames")
        samples = []
    sample_frames = [sample.get("frame") for sample in samples if isinstance(sample, dict)]
    if sample_frames != expected_measured or len(set(sample_frames)) != MEASURED_FRAMES:
        errors.append("samples must contain 60 distinct measured frames in order")
    payload_digests: list[object] = []
    totals: list[float] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"sample {index} must be an object")
            continue
        for key in TIMING_FIELDS:
            value = sample.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                errors.append(f"sample {index}.{key} must be finite and non-negative")
        total = sample.get("total_seconds")
        if isinstance(total, (int, float)) and not isinstance(total, bool) and math.isfinite(total):
            totals.append(float(total))
        digest = sample.get("published_payload_sha256")
        payload_digests.append(digest)
        if not _valid_digest(digest):
            errors.append(f"sample {index} payload digest must be sha256")
        for key in (
            "published_particle_count", "published_whitewater_count",
            "published_surface_vertex_count", "published_surface_triangle_count",
        ):
            if type(sample.get(key)) is not int or sample[key] < 0:
                errors.append(f"sample {index}.{key} must be a non-negative integer")
        frame = sample.get("frame")
        particle_count = sample.get("published_particle_count")
        if type(frame) is int and type(particle_count) is int:
            expected_status = f"Cache frame {frame}: {particle_count:,} particles"
            if sample.get("status") != expected_status:
                errors.append(f"sample {index} status does not match exact playback status")
    if len(payload_digests) == MEASURED_FRAMES and len(set(payload_digests)) != MEASURED_FRAMES:
        errors.append("published payload digests must change on every measured frame")
    if len(totals) == MEASURED_FRAMES:
        median = statistics.median(totals)
        p95 = _nearest_rank_p95(totals)
        reported_median = report.get("sample_median_seconds")
        reported_p95 = report.get("sample_p95_seconds")
        if (
            not isinstance(reported_median, (int, float))
            or isinstance(reported_median, bool)
            or not math.isfinite(reported_median)
            or not math.isclose(float(reported_median), median, rel_tol=1e-12, abs_tol=1e-15)
        ):
            errors.append("sample_median_seconds must be recomputed from raw samples")
        if (
            not isinstance(reported_p95, (int, float))
            or isinstance(reported_p95, bool)
            or not math.isfinite(reported_p95)
            or not math.isclose(float(reported_p95), p95, rel_tol=1e-12, abs_tol=1e-15)
        ):
            errors.append("sample_p95_seconds must be nearest-rank p95 of raw samples")
    if report.get("error_count") != 0 or report.get("errors") != []:
        errors.append("passing worker report must contain no errors")
    return errors


def assemble_campaign_report(
    legs: list[dict[str, object]], *, baseline_root: Path, candidate_root: Path,
) -> dict[str, object]:
    """Assemble raw legs without converting observations into approval claims."""
    totals: dict[str, list[float]] = {"baseline": [], "candidate": []}
    for leg in legs:
        subject = leg.get("subject")
        report = leg.get("report")
        if subject in totals and isinstance(report, dict):
            totals[subject].extend(
                float(sample["total_seconds"])
                for sample in report.get("samples", [])
                if isinstance(sample, dict) and isinstance(sample.get("total_seconds"), (int, float))
            )
    summaries = {}
    for subject, values in totals.items():
        summaries[subject] = {
            "sample_count": len(values),
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": _nearest_rank_p95(values) if values else None,
        }
    return {
        "schema": CAMPAIGN_SCHEMA,
        "result": CAMPAIGN_RESULT,
        "measurement_kind": MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "paired raw observation only; no standalone keep or speedup decision",
        "configuration": {
            "pairs": DEFAULT_PAIRS,
            "legs_per_pair": 4,
            "warmup_frames_per_leg": WARMUP_FRAMES,
            "measured_frames_per_leg": MEASURED_FRAMES,
            "schedule": "alternating ABBA/BAAB",
        },
        "subjects": {
            "baseline": _normal_path(baseline_root),
            "candidate": _normal_path(candidate_root),
        },
        "legs": legs,
        "summaries": summaries,
        "error_count": 0,
        "errors": [],
    }


def _mapping_parity_errors(
    reference: object,
    actual: object,
    *,
    fields: tuple[str, ...],
    prefix: str,
) -> list[str]:
    if not isinstance(reference, dict) or not isinstance(actual, dict):
        return []
    return [
        f"{prefix}.{field} mismatch"
        for field in fields
        if reference.get(field) != actual.get(field)
    ]


def _worker_correctness_parity_errors(
    reference: dict[str, object],
    actual: dict[str, object],
    *,
    pair: int,
    leg_index: int,
) -> list[str]:
    prefix = f"pair {pair} leg {leg_index}"
    errors = _mapping_parity_errors(
        reference.get("fixture"),
        actual.get("fixture"),
        fields=CORRECTNESS_FIXTURE_FIELDS,
        prefix=f"{prefix} fixture",
    )
    errors.extend(_mapping_parity_errors(
        reference.get("cache_provenance"),
        actual.get("cache_provenance"),
        fields=CORRECTNESS_CACHE_FIELDS,
        prefix=f"{prefix} cache_provenance",
    ))

    reference_samples = reference.get("samples")
    actual_samples = actual.get("samples")
    if not isinstance(reference_samples, list) or not isinstance(actual_samples, list):
        return errors
    for reference_sample, actual_sample in zip(reference_samples, actual_samples):
        if not isinstance(reference_sample, dict) or not isinstance(actual_sample, dict):
            continue
        frame = reference_sample.get("frame")
        sample_prefix = f"{prefix} frame {frame}"
        if actual_sample.get("frame") != frame:
            errors.append(f"{sample_prefix} frame mismatch")
        errors.extend(_mapping_parity_errors(
            reference_sample,
            actual_sample,
            fields=CORRECTNESS_SAMPLE_FIELDS,
            prefix=sample_prefix,
        ))
    return errors


def validate_campaign_report(report: object) -> list[str]:
    """Validate all 28 legs, their order, roots, and process isolation."""
    if not isinstance(report, dict):
        return ["campaign report must be an object"]
    errors: list[str] = []
    if report.get("schema") != CAMPAIGN_SCHEMA:
        errors.append(f"schema must be {CAMPAIGN_SCHEMA}")
    if report.get("result") != CAMPAIGN_RESULT:
        errors.append(f"result must be {CAMPAIGN_RESULT}")
    if report.get("measurement_kind") != MEASUREMENT_KIND:
        errors.append(f"measurement_kind must be {MEASUREMENT_KIND}")
    if report.get("approval_eligible") is not False:
        errors.append("campaign must disable approval eligibility")
    configuration = report.get("configuration")
    if not isinstance(configuration, dict) or configuration.get("pairs") != DEFAULT_PAIRS:
        errors.append("campaign requires exactly seven pairs")
    subjects = report.get("subjects")
    if not isinstance(subjects, dict):
        errors.append("subjects must identify baseline and candidate roots")
        subjects = {}
    expected = alternating_abba_schedule()
    legs = report.get("legs")
    if not isinstance(legs, list) or len(legs) != len(expected):
        errors.append("campaign must contain exactly 28 alternating ABBA legs")
        legs = []
    process_ids: list[int] = []
    pair_reports: dict[int, list[tuple[int, str, dict[str, object]]]] = {}
    for index, leg in enumerate(legs):
        if not isinstance(leg, dict):
            errors.append(f"leg {index} must be an object")
            continue
        if index < len(expected) and any(
            leg.get(key) != expected[index][key] for key in ("pair", "position", "subject")
        ):
            errors.append(f"leg {index} violates alternating ABBA schedule")
        worker = leg.get("report")
        for error in validate_worker_report(worker):
            errors.append(f"leg {index}: {error}")
        if isinstance(worker, dict):
            identity = worker.get("code_identity", {})
            environment = worker.get("environment", {})
            process_id = environment.get("process_id") if isinstance(environment, dict) else None
            if type(process_id) is int:
                process_ids.append(process_id)
            subject = leg.get("subject")
            expected_root = subjects.get(subject) if subject in ("baseline", "candidate") else None
            actual_root = identity.get("addon_root") if isinstance(identity, dict) else None
            if expected_root and _normal_path(actual_root) != _normal_path(expected_root):
                errors.append(f"leg {index} does not resolve the recorded {subject} root")
            pair = leg.get("pair")
            if type(pair) is int and subject in ("baseline", "candidate"):
                pair_reports.setdefault(pair, []).append((index, subject, worker))
    for pair, entries in sorted(pair_reports.items()):
        baselines = [entry for entry in entries if entry[1] == "baseline"]
        if not baselines:
            continue
        reference_index, _subject, reference = baselines[0]
        for leg_index, _subject, worker in entries:
            if leg_index == reference_index:
                continue
            errors.extend(_worker_correctness_parity_errors(
                reference,
                worker,
                pair=pair,
                leg_index=leg_index,
            ))
    if len(process_ids) == len(legs) and len(set(process_ids)) != len(process_ids):
        errors.append("every campaign leg must run in a fresh process")
    summaries = report.get("summaries")
    if not isinstance(summaries, dict):
        errors.append("summaries must be an object")
    else:
        expected_count = DEFAULT_PAIRS * 2 * MEASURED_FRAMES
        for subject in ("baseline", "candidate"):
            summary = summaries.get(subject)
            if not isinstance(summary, dict) or summary.get("sample_count") != expected_count:
                errors.append(f"{subject} summary must contain {expected_count} samples")
    if report.get("error_count") != 0 or report.get("errors") != []:
        errors.append("passing campaign report must contain no errors")
    return errors


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    baseline = args.baseline_root.resolve()
    candidate = args.candidate_root.resolve()
    worker = (candidate / "tests" / "blender_cache_playback_benchmark.py").resolve()
    if not worker.is_file():
        raise ValueError(f"candidate lacks cache-playback worker: {worker}")
    work_dir = args.work_dir.resolve()
    if any(_path_within(work_dir, root) for root in (baseline, candidate)):
        raise ValueError("campaign work-dir must be outside both measured checkouts")
    work_dir.mkdir(parents=True, exist_ok=True)
    legs: list[dict[str, object]] = []
    for index, scheduled in enumerate(alternating_abba_schedule(args.pairs), 1):
        root = baseline if scheduled["subject"] == "baseline" else candidate
        report_path = work_dir / f"leg_{index:02d}_{scheduled['subject']}.json"
        cache_dir = work_dir / f"cache_leg_{index:02d}_{scheduled['subject']}"
        command = [
            str(args.blender), "--background", "--factory-startup", "--python", str(worker), "--",
            "--addon-root", str(root), "--cache-dir", str(cache_dir), "--json-out", str(report_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise RuntimeError(f"campaign leg {index} failed ({scheduled['subject']}): {detail}")
        raw = _load_json(report_path)
        errors = validate_worker_report(raw)
        if errors:
            raise RuntimeError(f"campaign leg {index} report invalid: {'; '.join(errors)}")
        legs.append({**scheduled, "report_path": str(report_path), "report": raw})
    campaign = assemble_campaign_report(
        legs, baseline_root=baseline, candidate_root=candidate,
    )
    errors = validate_campaign_report(campaign)
    if errors:
        raise RuntimeError("invalid cache-playback campaign: " + "; ".join(errors))
    return campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-report", type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_report is not None:
            payload = _load_json(args.validate_report)
            errors = (
                validate_campaign_report(payload)
                if isinstance(payload, dict) and payload.get("schema") == CAMPAIGN_SCHEMA
                else validate_worker_report(payload)
            )
            if errors:
                raise ValueError("; ".join(errors))
            print(f"CACHE-PLAYBACK-REPORT-PASS: {args.validate_report.resolve()}")
            return 0
        required = {
            "blender": args.blender,
            "baseline-root": args.baseline_root,
            "candidate-root": args.candidate_root,
            "work-dir": args.work_dir,
            "json-out": args.json_out,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("campaign mode requires " + ", ".join(f"--{name}" for name in missing))
        if args.pairs != DEFAULT_PAIRS:
            raise ValueError(f"campaign evidence requires exactly {DEFAULT_PAIRS} pairs")
        campaign = run_campaign(args)
        _atomic_json(args.json_out, campaign)
    except Exception as exc:  # noqa: BLE001 - fail-closed evidence boundary
        print(f"CACHE-PLAYBACK-VALIDATION-FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{CAMPAIGN_RESULT}: {args.json_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
