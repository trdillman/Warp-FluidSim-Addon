"""System-Python contracts for the cache-playback benchmark evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import benchmark_cache_playback_validation as benchmark

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _load_worker_module():
    path = Path(__file__).with_name("blender_cache_playback_benchmark.py")
    spec = importlib.util.spec_from_file_location("_cache_playback_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(root: str, *, head: str = "1" * 40) -> dict[str, object]:
    return {
        "addon_root": root,
        "git_root": root,
        "git_head": head,
        "checkout_dirty": False,
        "addon_dirty": False,
        "addon_tree_sha256": HEX_A,
        "worker_sha256": HEX_B,
    }


def _worker_report(
    root: str = "C:/bench/a", *, pid: int = 101, mode: str = "baseline",
) -> dict[str, object]:
    frames = list(range(11, 71))
    generation_cached = mode == "candidate"
    samples = [
        {
            "frame": frame,
            "total_seconds": 0.010 + frame / 1_000_000,
            "read_frame_seconds": 0.002,
            "metadata_load_seconds": 0.001,
            "signature_bundle_seconds": 0.0 if generation_cached else 0.003,
            "publication_seconds": 0.004,
            "published_payload_sha256": f"{frame:064x}",
            "published_particle_count": 4096,
            "published_whitewater_count": 0,
            "published_surface_vertex_count": 0,
            "published_surface_triangle_count": 0,
            "status": f"Cache frame {frame}: 4,096 particles",
        }
        for frame in frames
    ]
    return {
        "schema": benchmark.WORKER_SCHEMA,
        "result": benchmark.WORKER_RESULT,
        "measurement_kind": benchmark.MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw observation only",
        "code_identity": _identity(root),
        "environment": {
            "blender_version": "5.1.0",
            "blender_executable": "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe",
            "python_executable": "C:/Program Files/Blender Foundation/Blender 5.1/5.1/python/bin/python.exe",
            "python_version": "3.11.0",
            "addon_module_path": root + "/fluidSim_warp_blender51/__init__.py",
            "playback_module_path": root + "/fluidSim_warp_blender51/blender/playback.py",
            "cache_module_path": root + "/fluidSim_warp_blender51/io/cache.py",
            "process_id": pid,
            "fresh_process_import": True,
            "background": True,
        },
        "configuration": {
            "cache_frames": 70,
            "warmup_frames": 10,
            "measured_frames": 60,
            "warmup_sequence": list(range(1, 11)),
            "measured_sequence": frames,
            "distinct_measured_frames": True,
            "cache_construction_outside_timed_region": True,
            "timing_scope": "play_cache_frame",
            "playback_read_api": (
                "read_frame_for_playback" if generation_cached else "read_frame"
            ),
            "signature_validation_mode": (
                "static_generation" if generation_cached else "per_frame"
            ),
        },
        "fixture": {
            "mesh_name": "CachePlaybackStaticMeshSource",
            "role": "SOURCE",
            "static": True,
            "vertex_count": 4225,
            "triangle_count": 8192,
            "matrix_world_sha256": HEX_C,
            "signature_before": {"scene_key": HEX_A, "settings_signature": HEX_B, "source_signature": HEX_C},
            "signature_after": {"scene_key": HEX_A, "settings_signature": HEX_B, "source_signature": HEX_C},
        },
        "cache_provenance": {
            "cache_path": "C:/bench-cache/run-101",
            "destination_initially_empty": True,
            "outside_addon_root": True,
            "format_version": 2,
            "frame_start": 1,
            "frame_end": 70,
            "committed_frame_count": 70,
            "manifest_sha256_before": HEX_A,
            "manifest_sha256_after": HEX_A,
            "cache_tree_sha256_before": HEX_B,
            "cache_tree_sha256_after": HEX_B,
            "immutable_after_build": True,
            "metadata_identity": HEX_C,
        },
        "publication_counts": {
            "play_cache_frame": 60,
            "read_frame": 60,
            "manifest_metadata_load": 60,
            "signature_bundle": 0 if generation_cached else 60,
            "output_publication": 60,
        },
        "warmup_counts": {
            "play_cache_frame": 10,
            "read_frame": 10,
            "manifest_metadata_load": 10,
            "signature_bundle": 1 if generation_cached else 10,
            "output_publication": 10,
        },
        "contention": {
            "other_gpu_processes": 0,
            "query_available": True,
            "classification": "zero-other-gpu-processes",
            "method": "bounded nvidia-smi compute-process query immediately before and after measured segment",
            "query_executable": "C:/Windows/System32/nvidia-smi.exe",
            "query_command": [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            "samples": [
                {
                    "phase": phase,
                    "query_available": True,
                    "returncode": 0,
                    "raw_stdout_lines": [],
                    "raw_stderr_lines": [],
                    "classifications": [],
                }
                for phase in ("before", "after")
            ],
        },
        "samples": samples,
        "sample_median_seconds": 0.0100405,
        "sample_p95_seconds": 0.010067,
        "error_count": 0,
        "errors": [],
    }


def test_alternating_abba_schedule_has_seven_counterbalanced_pairs():
    schedule = benchmark.alternating_abba_schedule()

    assert len(schedule) == 28
    assert [leg["subject"] for leg in schedule[:8]] == [
        "baseline", "candidate", "candidate", "baseline",
        "candidate", "baseline", "baseline", "candidate",
    ]
    assert [leg["pair"] for leg in schedule] == [
        pair for pair in range(1, 8) for _ in range(4)
    ]
    assert [leg["position"] for leg in schedule] == [
        position for _ in range(7) for position in range(1, 5)
    ]


def test_worker_validator_accepts_exact_distinct_playback_evidence():
    assert benchmark.validate_worker_report(_worker_report()) == []


@pytest.mark.parametrize("mode", ["baseline", "candidate"])
def test_worker_validator_accepts_truthful_baseline_and_candidate_telemetry(mode):
    assert benchmark.validate_worker_report(_worker_report(mode=mode)) == []


def test_worker_validator_rejects_measured_signature_rebuild_for_cached_candidate():
    report = _worker_report(mode="candidate")
    report["publication_counts"]["signature_bundle"] = 60

    assert any(
        "signature_bundle count must equal 0" in error
        for error in benchmark.validate_worker_report(report)
    )


def test_worker_validator_requires_one_warmup_signature_for_cached_candidate():
    report = _worker_report(mode="candidate")
    report["warmup_counts"]["signature_bundle"] = 0

    assert any(
        "warmup signature_bundle count must equal 1" in error
        for error in benchmark.validate_worker_report(report)
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda contention: contention.update(query_available=False), "query_available"),
        (lambda contention: contention.update(other_gpu_processes=1), "zero other GPU"),
        (lambda contention: contention.update(classification="ambiguous"), "classification"),
        (lambda contention: contention["samples"].pop(), "before and after"),
        (
            lambda contention: contention["samples"][0]["classifications"].append({
                "raw": "broken", "classification": "ambiguous",
            }),
            "unambiguous",
        ),
    ],
)
def test_worker_validator_fails_closed_on_unproven_contention(mutate, message):
    report = _worker_report(mode="candidate")
    mutate(report["contention"])

    assert any(message in error for error in benchmark.validate_worker_report(report))


def test_worker_validator_reports_malformed_summary_without_raising():
    report = _worker_report()
    report["sample_median_seconds"] = None

    assert any(
        "sample_median_seconds" in error
        for error in benchmark.validate_worker_report(report)
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report: report.update(approval_eligible=True), "approval eligibility"),
        (lambda report: report["code_identity"].update(checkout_dirty=True), "clean checkout"),
        (lambda report: report["configuration"].update(warmup_frames=9), "warmup_frames"),
        (lambda report: report["configuration"].update(measured_frames=59), "measured_frames"),
        (lambda report: report["samples"].__setitem__(1, copy.deepcopy(report["samples"][0])), "distinct measured frames"),
        (lambda report: report["fixture"].update(vertex_count=4095), "mesh-source-heavy"),
        (lambda report: report["fixture"].update(signature_after={"scene_key": HEX_B, "settings_signature": HEX_B, "source_signature": HEX_C}), "static fixture signature"),
        (lambda report: report["cache_provenance"].update(format_version=1), "cache format_version"),
        (lambda report: report["cache_provenance"].update(cache_tree_sha256_after=HEX_C), "cache changed"),
        (lambda report: report["publication_counts"].update(signature_bundle=59), "signature_bundle count"),
        (lambda report: report["samples"][0].update(published_payload_sha256=report["samples"][1]["published_payload_sha256"]), "payload digests"),
        (lambda report: report["samples"][0].update(status="wrong"), "status"),
    ],
)
def test_worker_validator_fails_closed_on_invalid_evidence(mutate, message):
    report = _worker_report()
    mutate(report)

    assert any(message in error for error in benchmark.validate_worker_report(report))


def test_campaign_validator_requires_exact_schedule_fresh_processes_and_roots():
    schedule = benchmark.alternating_abba_schedule()
    reports = []
    for index, leg in enumerate(schedule):
        root = "C:/bench/a" if leg["subject"] == "baseline" else "C:/bench/b"
        report = _worker_report(root, pid=1000 + index)
        reports.append({**leg, "report": report})

    campaign = benchmark.assemble_campaign_report(
        reports,
        baseline_root=Path("C:/bench/a"),
        candidate_root=Path("C:/bench/b"),
    )

    assert campaign["approval_eligible"] is False
    assert campaign["configuration"]["pairs"] == 7
    assert benchmark.validate_campaign_report(campaign) == []

    duplicate_pid = copy.deepcopy(campaign)
    duplicate_pid["legs"][1]["report"]["environment"]["process_id"] = 1000
    assert any("fresh process" in error for error in benchmark.validate_campaign_report(duplicate_pid))

    wrong_root = copy.deepcopy(campaign)
    wrong_root["legs"][1]["report"]["code_identity"]["addon_root"] = "C:/bench/a"
    assert any("candidate root" in error for error in benchmark.validate_campaign_report(wrong_root))


def _campaign_report():
    reports = []
    for index, leg in enumerate(benchmark.alternating_abba_schedule()):
        root = "C:/bench/a" if leg["subject"] == "baseline" else "C:/bench/b"
        reports.append({
            **leg,
            "report": _worker_report(root, pid=2000 + index),
        })
    return benchmark.assemble_campaign_report(
        reports,
        baseline_root=Path("C:/bench/a"),
        candidate_root=Path("C:/bench/b"),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["samples"][0].update(
                published_payload_sha256=HEX_C),
            "frame 11.published_payload_sha256",
        ),
        (
            lambda report: (
                report["samples"][0].update(
                    published_particle_count=4097,
                    status="Cache frame 11: 4,097 particles",
                )
            ),
            "frame 11.published_particle_count",
        ),
        (
            lambda report: (
                report["fixture"]["signature_before"].update(
                    settings_signature=HEX_C),
                report["fixture"]["signature_after"].update(
                    settings_signature=HEX_C),
            ),
            "fixture.signature_before",
        ),
        (
            lambda report: report["cache_provenance"].update(
                metadata_identity=HEX_A),
            "cache_provenance.metadata_identity",
        ),
        (
            lambda report: report["cache_provenance"].update(
                manifest_sha256_before=HEX_C,
                manifest_sha256_after=HEX_C,
            ),
            "cache_provenance.manifest_sha256_before",
        ),
    ],
)
def test_campaign_validator_rejects_cross_subject_correctness_mismatch(
        mutate, message):
    campaign = _campaign_report()
    candidate = campaign["legs"][1]["report"]
    mutate(candidate)

    # The adversarial report remains individually valid; only cross-subject
    # correspondence can detect that the candidate changed simulation output.
    assert benchmark.validate_worker_report(candidate) == []
    assert any(
        message in error
        for error in benchmark.validate_campaign_report(campaign)
    )


def test_system_python_validate_cli_accepts_worker_report(tmp_path):
    report_path = tmp_path / "worker.json"
    report_path.write_text(json.dumps(_worker_report()), encoding="utf-8")
    script = Path(benchmark.__file__).resolve()

    result = subprocess.run(
        [sys.executable, str(script), "--validate-report", str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "CACHE-PLAYBACK-REPORT-PASS" in result.stdout


def test_worker_helpers_are_system_python_safe_and_payload_digest_is_exact():
    np = pytest.importorskip("numpy")
    worker = _load_worker_module()
    first = {
        "positions": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        "frame": np.array(11, dtype=np.int32),
    }
    reordered = {"frame": first["frame"], "positions": first["positions"]}
    changed_value = {**first, "frame": np.array(12, dtype=np.int32)}
    changed_dtype = {**first, "frame": np.array(11, dtype=np.int64)}

    assert worker.payload_sha256(first) == worker.payload_sha256(reordered)
    assert worker.payload_sha256(first) != worker.payload_sha256(changed_value)
    assert worker.payload_sha256(first) != worker.payload_sha256(changed_dtype)


def test_worker_rejects_cache_inside_addon_or_nonempty_cache(tmp_path):
    worker = _load_worker_module()
    addon = tmp_path / "addon"
    addon.mkdir()
    inside = addon / "cache"
    outside = tmp_path / "cache"

    with pytest.raises(ValueError, match="outside"):
        worker.validate_cache_destination(addon, inside)

    outside.mkdir()
    (outside / "old.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="empty"):
        worker.validate_cache_destination(addon, outside)


def test_worker_classifies_nvidia_smi_rows_against_its_own_pid():
    worker = _load_worker_module()

    classified = worker.classify_nvidia_smi_process_rows(
        "123, blender.exe, GPU-a\n456, renderer.exe, GPU-a\n",
        current_pid=123,
    )

    assert classified == [
        {
            "raw": "123, blender.exe, GPU-a",
            "pid": 123,
            "process_name": "blender.exe",
            "gpu_uuid": "GPU-a",
            "classification": "self",
        },
        {
            "raw": "456, renderer.exe, GPU-a",
            "pid": 456,
            "process_name": "renderer.exe",
            "gpu_uuid": "GPU-a",
            "classification": "other",
        },
    ]


@pytest.mark.parametrize("raw", ["N/A", "123, blender.exe", "pid, name, uuid"])
def test_worker_rejects_ambiguous_nvidia_smi_rows(raw):
    worker = _load_worker_module()

    with pytest.raises(ValueError, match="ambiguous"):
        worker.classify_nvidia_smi_process_rows(raw, current_pid=123)
