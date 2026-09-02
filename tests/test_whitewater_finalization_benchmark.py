from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "benchmark_whitewater_finalization.py"


def _module():
    spec = importlib.util.spec_from_file_location("whitewater_finalization_benchmark", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exposes_campaign_worker_and_system_python_validation_commands():
    result = subprocess.run(
        [sys.executable, str(TOOL), "run", "--help"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--baseline-root", "--candidate-root", "--blender-python", "--pairs",
        "--device", "--iterations", "--warmup", "--live-counts", "--json-out",
    ):
        assert option in result.stdout
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
        (1, 30, 3, (49_999,), "live-counts"),
        (1, 30, 3, (50_000, 50_000), "duplicates"),
    ),
)
def test_configuration_fails_closed(pairs, iterations, warmup, counts, match):
    module = _module()
    with pytest.raises(ValueError, match=match):
        module.validate_configuration(
            pairs=pairs, iterations=iterations, warmup=warmup, live_counts=counts,
        )


def test_seven_pairs_use_alternating_ab_ba_order():
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


def test_code_identity_binds_exact_checkout_and_addon_tree():
    module = _module()
    identity = module.resolve_code_identity(ROOT)
    assert Path(identity["addon_root"]) == ROOT.resolve()
    assert len(identity["git_head"]) == 40
    assert len(identity["addon_tree_sha256"]) == 64
    assert isinstance(identity["dirty"], bool)


def _worker(module, role: str, pair: int, process_id: int, *, type_launches: int):
    digest = "c" * 64
    role_digest = "b" * 64 if role == "baseline" else "6" * 64
    source_digest = "d" * 64 if role == "baseline" else "7" * 64
    count = 50_000
    expected = [16_667, 16_667, 16_666, 0]
    output_digests = {
        key: digest for key in (
            "position_sha256", "velocity_sha256", "age_sha256", "life_sha256",
            "alive_sha256", "type_sha256", "density_sha256",
        )
    }
    return {
        "schema": module.WORKER_SCHEMA,
        "result": module.WORKER_RESULT,
        "measurement_kind": module.MEASUREMENT_KIND,
        "approval_eligible": False,
        "role": role,
        "pair": pair,
        "code_identity": {
            "addon_root": str(ROOT / role), "git_head": "a" * 40,
            "addon_tree_sha256": role_digest, "dirty": True, "addon_dirty": True,
        },
        "seam_provenance": {
            "callable": "FluidSolver._finalize_whitewater", "module_path": "solver.py",
            "source_sha256": source_digest, "benchmark_sha256": "e" * 64,
        },
        "environment": {
            "python_executable": sys.executable, "python_version": sys.version,
            "warp_version": "test", "warp_path": "warp.py", "device": "cuda:0",
            "process_id": process_id, "process_instance_token": f"worker-{process_id}",
            "fresh_process_import": True,
        },
        "configuration": {
            "iterations": 30, "warmup": 3, "live_counts": [count], "seed": 1729,
        },
        "scenarios": [{
            "live_count": count,
            "calibration": {
                "fixture_seed": 1729, "expected_live_count": count,
                "expected_type_count_array": expected,
                "input_array_digests": {key: digest for key in module.FIXTURE_CHANNELS},
            },
            "samples_seconds": [0.01] * 30,
            "sample_median_seconds": 0.01,
            "sample_min_seconds": 0.01,
            "sample_max_seconds": 0.01,
            "device_synchronized_samples": True,
            "observed_live_counts": [count] * 30,
            "observed_type_count_arrays": [expected] * 30,
            "telemetry": {
                "wp_launches_per_iteration": [8] * 30,
                "type_count_launches_per_iteration": [type_launches] * 30,
                "density_launches_per_iteration": [1] * 30,
                "compact_calls_per_iteration": [1] * 30,
            },
            "output_arrays": {
                "count_array": [count], "type_count_array": expected,
                "digests": output_digests,
            },
        }],
        "error_count": 0,
        "errors": [],
    }


def _campaign(module):
    pairs = []
    process_id = 100
    for pair_number in range(1, 3):
        order = module.pair_order(pair_number)
        reports = {
            "baseline": _worker(module, "baseline", pair_number, process_id, type_launches=2),
            "candidate": _worker(module, "candidate", pair_number, process_id + 1, type_launches=1),
        }
        process_id += 2
        pairs.append({
            "pair": pair_number, "order": list(order),
            "legs": [reports[role] for role in order],
            "comparison": module._comparison(reports["baseline"], reports["candidate"]),
        })
    return {
        "schema": module.SCHEMA,
        "result": module.RESULT,
        "measurement_kind": module.MEASUREMENT_KIND,
        "approval_eligible": False,
        "campaign_provenance": {
            "fresh_processes": True, "unique_worker_processes": 4,
            "benchmark_sha256": "e" * 64,
        },
        "code_identities": {
            leg["role"]: leg["code_identity"] for leg in pairs[0]["legs"]
        },
        "configuration": {
            "pairs": 2, "iterations": 30, "warmup": 3,
            "live_counts": [50_000], "seed": 1729, "device": "cuda:0",
        },
        "pairs": pairs,
        "error_count": 0,
        "errors": [],
    }


def test_raw_validator_accepts_observed_baseline_and_candidate_launch_counts():
    module = _module()
    report = _campaign(module)
    assert module.validate_campaign_report(report) == []


def test_system_python_validate_command_accepts_campaign_report(tmp_path):
    module = _module()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(_campaign(module)), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL), "validate", str(path)], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "WHITEWATER-FINALIZATION-REPORT-PASS" in result.stdout


def test_validator_rejects_approval_claim_process_reuse_and_output_drift():
    module = _module()
    report = _campaign(module)
    report["approval_eligible"] = True
    report["pairs"][1]["legs"][0]["environment"]["process_instance_token"] = "worker-100"
    report["pairs"][0]["legs"][1]["scenarios"][0]["output_arrays"]["digests"]["type_sha256"] = "9" * 64

    errors = " ".join(module.validate_campaign_report(report))
    assert "approval eligibility" in errors
    assert "unique worker process" in errors
    assert "outputs differ" in errors


def test_validator_rejects_identical_baseline_and_candidate_code_identity():
    module = _module()
    report = _campaign(module)
    baseline = report["code_identities"]["baseline"]
    report["code_identities"]["candidate"] = baseline.copy()
    for record in report["pairs"]:
        candidate = next(leg for leg in record["legs"] if leg["role"] == "candidate")
        candidate["code_identity"] = baseline.copy()
        candidate["seam_provenance"]["source_sha256"] = next(
            leg for leg in record["legs"] if leg["role"] == "baseline"
        )["seam_provenance"]["source_sha256"]

    errors = " ".join(module.validate_campaign_report(report))
    assert "distinct add-on roots" in errors
    assert "distinct add-on tree digests" in errors
    assert "distinct finalizer source digests" in errors


def test_worker_validator_rejects_candidate_with_duplicate_type_reduction():
    module = _module()
    report = _worker(module, "candidate", 1, 123, type_launches=2)

    errors = " ".join(module.validate_worker_report(report))
    assert "candidate type-count launches must equal 1" in errors


def test_worker_validator_rejects_unsynchronized_or_inexact_observation():
    module = _module()
    report = _worker(module, "baseline", 1, 123, type_launches=2)
    report["scenarios"][0]["device_synchronized_samples"] = False
    report["scenarios"][0]["observed_type_count_arrays"][4] = [0, 0, 0, 0]

    errors = " ".join(module.validate_worker_report(report))
    assert "device-synchronized" in errors
    assert "exact type-count arrays" in errors
