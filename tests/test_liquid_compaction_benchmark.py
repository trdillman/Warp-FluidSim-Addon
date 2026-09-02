from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "benchmark_liquid_compaction.py"


def _module():
    spec = importlib.util.spec_from_file_location("liquid_compaction_benchmark", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exposes_fresh_process_campaign_and_validation_controls():
    run = subprocess.run(
        [sys.executable, str(TOOL), "run", "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    for option in (
        "--baseline-root", "--candidate-root", "--blender-python", "--pairs",
        "--device", "--iterations", "--warmup", "--particle-counts", "--json-out",
    ):
        assert option in run.stdout
    validate = subprocess.run(
        [sys.executable, str(TOOL), "validate", "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert validate.returncode == 0
    assert "report" in validate.stdout


@pytest.mark.parametrize(
    ("pairs", "iterations", "warmup", "counts", "match"),
    (
        (0, 30, 3, (50_000,), "pairs"),
        (1, 29, 3, (50_000,), "iterations"),
        (1, 30, 0, (50_000,), "warmup"),
        (1, 30, 3, (49_999,), "particle-counts"),
        (1, 30, 3, (50_000, 50_000), "duplicates"),
    ),
)
def test_configuration_fails_closed(pairs, iterations, warmup, counts, match):
    module = _module()
    with pytest.raises(ValueError, match=match):
        module.validate_configuration(
            pairs=pairs, iterations=iterations, warmup=warmup, particle_counts=counts,
        )


def test_seven_pairs_use_abba_balanced_order():
    module = _module()
    assert [module.pair_order(pair) for pair in range(1, 8)] == [
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
    ]


def test_death_masks_cover_zero_one_half_and_last_dead_stably():
    module = _module()
    import numpy as np

    masks = {name: module.death_mask(name, 10) for name in module.DEATH_PATTERNS}
    assert [int((masks[name] == 0).sum()) for name in module.DEATH_PATTERNS] == [0, 1, 5, 1]
    assert masks["last-dead"][-1] == 0
    for name, mask in masks.items():
        survivors = np.flatnonzero(mask).astype(np.int32)
        assert np.array_equal(survivors, np.sort(survivors)), name


def test_code_identity_binds_exact_checkout_and_addon_tree():
    module = _module()
    identity = module.resolve_code_identity(ROOT)
    assert Path(identity["addon_root"]) == ROOT.resolve()
    assert len(identity["git_head"]) == 40
    assert len(identity["addon_tree_sha256"]) == 64
    assert isinstance(identity["dirty"], bool)


def _worker(module, role: str, pair: int, process_id: int, *, last_count: int):
    digest = "c" * 64
    scenarios = []
    for pattern in module.DEATH_PATTERNS:
        survivor_count = {"zero-death": 50_000, "one-death": 49_999,
                          "half-death": 25_000, "last-dead": 49_999}[pattern]
        channel_digests = {name: digest for name in module.CHANNEL_NAMES}
        scenarios.append({
            "particle_count": 50_000,
            "death_pattern": pattern,
            "optional_channels": list(module.OPTIONAL_CHANNELS),
            "calibration": {
                "fixture_seed": 1729,
                "death_count": 50_000 - survivor_count,
                "expected_survivor_count": survivor_count,
                "survivor_indices_sha256": digest,
                "expected_channel_digests": dict(channel_digests),
            },
            "samples_seconds": [0.01] * 30,
            "sample_median_seconds": 0.01,
            "sample_min_seconds": 0.01,
            "sample_max_seconds": 0.01,
            "device_synchronized_warmup": True,
            "device_synchronized_samples": True,
            "observed_survivor_counts": [survivor_count] * 30,
            "telemetry": {
                "array_scans_per_iteration": [1] * 30,
                "kernel_launches_per_iteration": [6 + last_count] * 30,
                "k_last_offset_count_per_iteration": [last_count] * 30,
                "k_compact_alive_scatter_per_iteration": [1] * 30,
                "k_compact_vec4_scatter_per_iteration": [1] * 30,
                "k_compact_int_scatter_per_iteration": [2] * 30,
            },
            "output": {
                "count_array": [survivor_count],
                "survivor_indices_sha256": digest,
                "channel_digests": dict(channel_digests),
                "exact_stable_survivor_arrays": True,
            },
        })
    return {
        "schema": module.WORKER_SCHEMA,
        "result": module.WORKER_RESULT,
        "measurement_kind": module.MEASUREMENT_KIND,
        "approval_eligible": False,
        "role": role,
        "pair": pair,
        "code_identity": {
            "addon_root": str(ROOT), "git_head": "a" * 40,
            "addon_tree_sha256": "b" * 64, "dirty": True, "addon_dirty": True,
        },
        "seam_provenance": {
            "callable": "ParticleSystem.compact", "module_path": "particles.py",
            "source_sha256": "d" * 64, "benchmark_sha256": "e" * 64,
        },
        "environment": {
            "blender_version": "5.1",
            "python_executable": sys.executable, "python_version": sys.version,
            "warp_version": "test", "warp_path": "warp.py", "device": "cuda:0",
            "process_id": process_id, "fresh_process_import": True,
        },
        "configuration": {
            "iterations": 30, "warmup": 3, "particle_counts": [50_000], "seed": 1729,
        },
        "scenarios": scenarios,
        "error_count": 0,
        "errors": [],
    }


def _campaign(module):
    records = []
    process_id = 100
    for pair_number in range(1, 3):
        order = module.pair_order(pair_number)
        reports = {
            "baseline": _worker(module, "baseline", pair_number, process_id, last_count=1),
            "candidate": _worker(module, "candidate", pair_number, process_id + 1, last_count=0),
        }
        process_id += 2
        records.append({
            "pair": pair_number,
            "order": list(order),
            "legs": [reports[role] for role in order],
            "comparison": module._comparison(reports["baseline"], reports["candidate"]),
        })
    return {
        "schema": module.SCHEMA,
        "result": module.RESULT,
        "measurement_kind": module.MEASUREMENT_KIND,
        "approval_eligible": False,
        "interpretation": "raw paired observation only; no standalone approval claim",
        "campaign_provenance": {
            "fresh_processes": True, "unique_worker_processes": 4,
            "benchmark_sha256": "f" * 64, "order_policy": "AB/BA alternating (ABBA balanced)",
        },
        "code_identities": {
            "baseline": records[0]["legs"][0]["code_identity"],
            "candidate": records[0]["legs"][1]["code_identity"],
        },
        "configuration": {
            "pairs": 2, "iterations": 30, "warmup": 3,
            "particle_counts": [50_000], "seed": 1729, "device": "cuda:0",
        },
        "pairs": records,
        "launch_change": {
            "kernel": "k_last_offset_count",
            "baseline_launches": 240,
            "candidate_launches": 0,
            "removed_in_candidate": True,
        },
        "error_count": 0,
        "errors": [],
    }


def test_worker_validator_accepts_exact_synchronized_raw_observation():
    module = _module()
    assert module.validate_worker_report(_worker(module, "baseline", 1, 123, last_count=1)) == []


def test_worker_validator_rejects_approval_unsynchronized_samples_and_channel_drift():
    module = _module()
    report = _worker(module, "candidate", 1, 123, last_count=0)
    report["approval_eligible"] = True
    report["scenarios"][0]["device_synchronized_samples"] = False
    report["scenarios"][1]["output"]["channel_digests"]["uid"] = "9" * 64

    errors = " ".join(module.validate_worker_report(report))
    assert "approval eligibility" in errors
    assert "device-synchronized" in errors
    assert "exact stable survivor" in errors


def test_campaign_validator_requires_unique_processes_exact_outputs_and_removed_kernel():
    module = _module()
    report = _campaign(module)
    assert module.validate_campaign_report(report) == []

    report["pairs"][1]["legs"][0]["environment"]["process_id"] = 100
    report["pairs"][0]["legs"][1]["scenarios"][0]["output"]["count_array"] = [49_998]
    report["launch_change"]["removed_in_candidate"] = False
    errors = " ".join(module.validate_campaign_report(report))
    assert "unique worker process" in errors
    assert "outputs differ" in errors
    assert "k_last_offset_count" in errors
