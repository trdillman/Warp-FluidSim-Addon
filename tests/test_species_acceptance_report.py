from __future__ import annotations

import sys
import types
import importlib.util
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import simulation_admission as admission
from fluidSim_warp_blender51.sim.capacity_planner import (
    DemandQuality,
    plan_initialization,
)
from fluidSim_warp_blender51.sim.memory_safety import (
    estimate_solver_bytes,
    estimate_species_cache_staging_bytes,
    estimate_species_transport_bytes,
)


GIB = 1024**3


_ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "blender_species_transport_acceptance",
    Path(__file__).with_name("blender_species_transport_acceptance.py"))
_ACCEPTANCE = importlib.util.module_from_spec(_ACCEPTANCE_SPEC)
_ACCEPTANCE_SPEC.loader.exec_module(_ACCEPTANCE)
_PROFILE_SPEC = importlib.util.spec_from_file_location(
    "profile_species_transport",
    Path(__file__).parents[1] / "tools" / "profile_species_transport.py")
_PROFILE = importlib.util.module_from_spec(_PROFILE_SPEC)
_PROFILE_SPEC.loader.exec_module(_PROFILE)
_PROFILE_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_species_transport_profile_repetitions",
    Path(__file__).parents[1] / "tools" /
    "run_species_transport_profile_repetitions.py")
_PROFILE_RUNNER = importlib.util.module_from_spec(_PROFILE_RUNNER_SPEC)
_PROFILE_RUNNER_SPEC.loader.exec_module(_PROFILE_RUNNER)


def test_profiler_backend_probe_accepts_disabled_solver_without_species_attribute():
    assert _PROFILE._species_backend(SimpleNamespace()) == "disabled"
    assert _PROFILE._species_backend(SimpleNamespace(species=None)) == "disabled"
    assert _PROFILE._species_backend(SimpleNamespace(species=SimpleNamespace(
        bounds_reduction_backend="native_cuda"))) == "native_cuda"


def _resources():
    return {"gpu_total_bytes": 64 * GIB, "gpu_free_bytes": 64 * GIB}


def _settings(**changes):
    values = dict(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(2.4, 0.8, 0.8),
        voxel_size=0.1,
        leaf_capacity=64,
        max_particles=128,
        max_whitewater=0,
        surface_enabled=False,
        topology_expand=0,
        frame_rate=24.0,
        time_scale=1.0,
        min_substeps=1,
        species_enabled=False,
        species_diffusivity=0.0,
        species_max_diffusion_substeps=16,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _source(coords=((1, 0, 0),)):
    return SimpleNamespace(
        stable_id="dye-source",
        particle_demand=1,
        particle_quality=DemandQuality.EXACT,
        leaf_coords=tuple(coords),
        staging_bytes=0,
        reason="",
    )


def test_disabled_species_estimate_is_byte_for_byte_unchanged():
    settings = _settings()
    baseline = estimate_solver_bytes(settings)
    explicit_disabled = _settings(
        species_enabled=False,
        species_diffusivity=999.0,
        species_max_diffusion_substeps=1,
    )

    assert estimate_species_transport_bytes(explicit_disabled) == 0
    assert estimate_species_cache_staging_bytes(explicit_disabled) == 0
    assert estimate_solver_bytes(explicit_disabled) == baseline


def test_enabled_species_estimate_charges_exact_transport_and_cache_staging():
    disabled = _settings()
    enabled = _settings(species_enabled=True)

    transport = estimate_species_transport_bytes(enabled)
    cache_staging = estimate_species_cache_staging_bytes(enabled)

    assert transport > 0
    assert cache_staging == enabled.max_particles * 20
    assert estimate_solver_bytes(enabled) - estimate_solver_bytes(disabled) == (
        transport + cache_staging
    )


def test_transport_estimate_matches_named_particle_and_voxel_inventory():
    from fluidSim_warp_blender51.sim.species import (
        SpeciesTransport,
        species_transport_allocation_bytes,
    )

    particle_capacity, voxel_capacity = 128, 1536
    # Live + replacement represented-volume/concentration channels, persistent
    # particle preview, one scalar + one vec4 local-mixture grid, and receipt.
    expected = particle_capacity * 56 + voxel_capacity * 20 + 11 * 4
    inventory = species_transport_allocation_bytes(
        particle_capacity, voxel_capacity, True)
    assert inventory.particle_channels_bytes == particle_capacity * 40
    assert inventory.particle_preview_bytes == particle_capacity * 16
    assert inventory.voxel_scratch_bytes == voxel_capacity * 20
    assert inventory.receipt_bytes == 44
    assert inventory.device_persistent_bytes == expected
    assert inventory.host_cache_staging_bytes == 0
    assert SpeciesTransport.estimate_bytes(
        particle_capacity, voxel_capacity, True) == inventory.device_persistent_bytes


def test_profile_parses_decimal_warp_memory_report_peak():
    report = """Allocation Tracking Report
  Peak usage:
    cuda:0           13.44 MB
    cpu              20 B
"""
    assert _PROFILE._peak_cuda_bytes(report) == 13_440_000


def test_profile_overhead_uses_adjacent_disabled_pair_without_clamping():
    assert _PROFILE._paired_overhead(12.0, 10.0, 14.0) == 0.0
    assert _PROFILE._paired_overhead(9.0, 10.0, 10.0) == pytest.approx(-0.1)


def test_species_release_thresholds_are_frozen_and_not_legacy_budgets():
    assert dict(_PROFILE.SOLVER_GATES["configured_inactive"]) == {
        "median_fraction": 0.01, "p95_delta_ms": 0.25}
    assert dict(_PROFILE.SOLVER_GATES["transport"]) == {
        "median_fraction": 0.02, "p95_delta_ms": 0.50}
    assert dict(_PROFILE.SOLVER_GATES["mixing_n1_n4"]) == {
        "median_fraction": 0.05, "p95_delta_ms": 1.00}
    with pytest.raises(TypeError):
        _PROFILE.SOLVER_GATES["transport"]["median_fraction"] = 0.12


def test_each_feature_sample_has_adjacent_disabled_brackets():
    metrics = _PROFILE.paired_case_metrics([
        {"case": "disabled", "ms": 10.0},
        {"case": "mixing_n2", "ms": 10.4},
        {"case": "disabled", "ms": 10.2},
    ])

    assert metrics["mixing_n2"]["baseline_ms"] == pytest.approx(10.1)
    assert metrics["mixing_n2"]["median_fraction"] == pytest.approx(
        10.4 / 10.1 - 1.0)
    with pytest.raises(ValueError, match="disabled_before"):
        _PROFILE.paired_case_metrics([
            {"case": "transport", "ms": 10.1},
            {"case": "disabled", "ms": 10.0},
            {"case": "disabled", "ms": 10.0},
        ])


def test_production_release_gate_rejects_recorded_unsupported_cases():
    inventory = {
        field: ({} if field in {"launch_inventory", "readback_inventory"}
                else None)
        for field in _PROFILE.RELEASE_INVENTORY_FIELDS
    }
    report = {
        "cases": {
            name: {"status": "supported", "inventory": dict(inventory)}
            for name in _PROFILE.SOLVER_CASES
        }
    }
    report["cases"]["mixing_n8"] = {
        "status": "unsupported",
        "reason": "pre-redesign fixed species width",
        "inventory": dict(inventory),
    }

    assert _PROFILE.validate_release_cases(
        report, require_production=False) == []
    reloaded = json.loads(json.dumps(report, sort_keys=True))
    assert _PROFILE.validate_release_cases(
        reloaded, require_production=False) == []
    assert _PROFILE.validate_release_cases(
        report, require_production=True) == ["unsupported_cases"]


def _strict_release_fixture(frames=3):
    samples = []
    cases = {}
    supported = (
        "configured_inactive", "transport", "mixing_n1", "mixing_n2")
    feature_ms = {
        "configured_inactive": 10.05,
        "transport": 10.1,
        "mixing_n1": 10.2,
        "mixing_n2": 10.3,
    }
    for name in supported:
        for _index in range(frames):
            samples.extend((
                {"case": "disabled", "ms": 10.0,
                 "reduction_backend": "disabled"},
                {"case": name, "ms": feature_ms[name],
                 "reduction_backend": "native_cuda"},
                {"case": "disabled", "ms": 10.0,
                 "reduction_backend": "disabled"},
            ))
    invariants = {
        "active_particles": 200_000, "active_leaves": 27,
        "active_voxels": 13_824, "solver_substeps": 1,
    }
    for name in _PROFILE.SOLVER_CASES:
        supported_case = name not in {"mixing_n4", "mixing_n8"}
        cases[name] = {
            "status": "supported" if supported_case else "unsupported",
            "reason": "" if supported_case else "fixed-width storage",
            "inventory": {
                **invariants,
                "active_species": {
                    "disabled": 0, "configured_inactive": 0,
                    "transport": 2, "mixing_n1": 1, "mixing_n2": 2,
                }.get(name),
                "chunk_count": 0 if name in {
                    "disabled", "configured_inactive"} else (
                    1 if supported_case else None),
                "launch_inventory": {}, "readback_inventory": {},
                "cuda_allocation_bytes": 0,
                "reduction_backend": (
                    "disabled" if name == "disabled" else
                    "native_cuda" if supported_case else None),
            },
        }
    return _PROFILE.build_release_gate(
        samples=samples, cases=cases,
        configuration={
            "frames": frames,
            "warmup": _PROFILE.RELEASE_MINIMUMS["warmup"],
            "particles": 200_000,
        },
        timed_context_provenance={
            "disabled": "disabled",
            "configured_inactive": "native_cuda", "transport": "native_cuda",
            "mixing_n1": "native_cuda", "mixing_n2": "native_cuda",
        },
        production=frames >= _PROFILE.RELEASE_MINIMUMS["frames"],
    )


def test_strict_solver_release_validator_recomputes_metrics_and_required_n2():
    report = _strict_release_fixture(frames=_PROFILE.RELEASE_MINIMUMS["frames"])
    assert _PROFILE.validate_release_gate(report, require_production=True) == []

    forged = json.loads(json.dumps(report))
    forged["metrics"]["mixing_n2"]["median_fraction"] = 0.0
    assert "metrics_consistency" in _PROFILE.validate_release_gate(forged)

    forged_inventory = json.loads(json.dumps(report))
    for case in forged_inventory["cases"].values():
        case["inventory"]["active_particles"] = 199_999
    assert "inventory_contract" in _PROFILE.validate_release_gate(
        forged_inventory)

    missing_n2 = json.loads(json.dumps(report))
    missing_n2["cases"]["mixing_n2"]["status"] = "unsupported"
    missing_n2["cases"]["mixing_n2"]["reason"] = "fabricated unsupported"
    assert "required_supported_cases" in _PROFILE.validate_release_gate(missing_n2)


def test_strict_solver_release_rejects_fabricated_native_timed_provenance():
    report = _strict_release_fixture(frames=_PROFILE.RELEASE_MINIMUMS["frames"])
    forged = json.loads(json.dumps(report))
    forged["timed_context_provenance"]["transport"] = "warp_atomic"

    errors = _PROFILE.validate_release_gate(forged, require_production=True)

    assert "timed_context_provenance" in errors


def test_strict_solver_runtime_case_plan_executes_task1a_cases():
    plan = _PROFILE.release_case_plan()

    assert [name for name, case in plan.items() if case["status"] == "supported"] == [
        "disabled", "configured_inactive", "transport", "mixing_n1",
        "mixing_n2",
    ]
    assert plan["mixing_n4"]["status"] == "unsupported"
    assert plan["mixing_n8"]["status"] == "unsupported"


def _profile_audit_case(name):
    inventories = {
        "diffusion": ({"species_diffusion_kernel": 1}
                      if name == "diffusion_on" else {}),
        "lifecycle_other": (
            {"native_cuda::fluid_species_postcompact_reduce": 1}
            if name != "disabled" else {"solver_kernel": 1}),
    }
    species_launches = sum(
        count for phase in inventories.values()
        for kernel, count in phase.items() if "species" in kernel)
    return {
        "active_particles": 200_000,
        "active_leaf_count": 27,
        "active_voxel_count": 13_824,
        "solver_substeps_max": 1,
        "species_substeps_max": 2 if name == "diffusion_on" else 0,
        "species_kernel_launches": species_launches,
        "all_kernel_launches": sum(
            sum(phase.values()) for phase in inventories.values()),
        "kernel_launches_by_phase": inventories,
        "substep_device_to_host_transfers": 0,
        "tracked_peak_cuda_bytes": 10_000_000,
        "species_object_present": name != "disabled",
        "species_bounds_reduction_backend": (
            "disabled" if name == "disabled" else "native_cuda"),
    }


def _passing_profile_receipt(*, frames=60, warmup=15, diagnostic=False):
    disabled = [10.0] * (2 * frames + 1)
    transport = [10.5] * frames
    diffusion = [12.0] * frames
    transport_overheads = [0.05] * frames
    diffusion_overheads = [0.2] * frames
    cases = {
        name: _profile_audit_case(name)
        for name in ("disabled", "transport_only", "diffusion_on")
    }
    identity = _PROFILE._current_evidence_identity()
    report = {
        "schema_version": 2,
        "status": "diagnostic" if diagnostic else "pass",
        "evidence_kind": "diagnostic" if diagnostic else "production",
        "provenance": {
            "timestamp_utc": _PROFILE._utc_now().isoformat(),
            "executable": sys.executable,
            "process_id": 12345,
            "command": [sys.executable, "tools/profile_species_transport.py"],
            **identity,
            "cuda_device": {
                "alias": "cuda:0", "name": "GPU", "ordinal": 0,
            },
            "python_version": "3.13",
            "warp_version": "1.16.0",
        },
        "configuration": {
            "frames": frames, "warmup": warmup, "particles": 200_000,
            "diffusivity_k2": 0.06, "diffusion_substeps": 2,
        },
        "budgets": dict(_PROFILE.BUDGETS),
        "timing": {
            "method": "adjacent-disabled-brackets",
            "instrumentation_active": False,
            "raw_samples_ms": {
                "disabled": disabled,
                "transport_only": transport,
                "diffusion_on": diffusion,
            },
            "paired_overheads": {
                "transport_only": transport_overheads,
                "diffusion_on": diffusion_overheads,
            },
            "context_backends": {
                "disabled": "disabled",
                "transport_only": "native_cuda",
                "diffusion_on": "native_cuda",
            },
        },
        "audit": {
            "instrumentation_active": True,
            "cases": cases,
            "source_ir_hotpath": _PROFILE._source_ir_hotpath_audit(),
            "allocation": {
                "admitted_species_device_bytes": 11_476_524,
                "admitted_species_cache_staging_bytes": 4_000_000,
                "measured_species_peak_increment_bytes": 11_706_054,
            },
        },
        "metrics": {
            "disabled_median_ms": 10.0,
            "transport_only_overhead_fraction": 0.05,
            "diffusion_on_overhead_fraction": 0.2,
            "transport_only_p95_delta_ms": 0.5,
            "diffusion_on_p95_delta_ms": 2.0,
            "admitted_species_device_bytes": 11_476_524,
            "admitted_species_cache_staging_bytes": 4_000_000,
            "measured_species_peak_increment_bytes": 11_706_054,
            "allocation_relative_error": 0.01999995817548937,
        },
        "checks": {
            "disabled_zero_species_resources": True,
            "transport_only_overhead": True, "diffusion_on_overhead": True,
            "diffusion_on_k2": True, "allocation_matches_admission": True,
            "no_substep_host_synchronization": True,
            "timed_samples_uninstrumented": True,
            "adjacent_disabled_brackets": True,
            "raw_sample_counts": True,
            "cuda_runtime": True,
        },
    }
    return report


def test_profile_receipt_validator_requires_identity_budgets_and_k2():
    assert _PROFILE.validate_report(_passing_profile_receipt()) == []
    broken = _passing_profile_receipt()
    broken["provenance"].pop("warp_version")
    broken["metrics"]["transport_only_overhead_fraction"] = 0.13
    broken["audit"]["cases"]["diffusion_on"]["species_substeps_max"] = 1
    errors = _PROFILE.validate_report(broken)
    assert "provenance" in errors
    assert "metrics_consistency" in errors
    assert "diffusion_on_k2" in errors
    assert "status" in errors


def test_profile_receipt_rejects_timed_fallback_even_when_audit_is_native():
    forged = _passing_profile_receipt()
    forged["timing"]["context_backends"]["transport_only"] = "warp_atomic"
    forged["checks"]["cuda_runtime"] = False

    errors = _PROFILE.validate_report(forged)

    assert "cuda_identity" in errors
    assert "status" in errors


def test_profile_receipt_validator_recomputes_raw_metrics_and_exact_checks():
    forged = _passing_profile_receipt()
    forged["timing"]["raw_samples_ms"]["transport_only"] = [25.0] * 60
    forged["checks"]["invented_check"] = True

    errors = _PROFILE.validate_report(forged)

    assert "metrics_consistency" in errors
    assert "paired_overheads_consistency" in errors
    assert "checks" in errors
    assert "status" in errors


def test_profile_receipt_validator_recomputes_admission_bytes():
    forged = _passing_profile_receipt()
    forged["audit"]["allocation"]["admitted_species_device_bytes"] += 4
    metrics, pairs = _PROFILE._recomputed_metrics(forged)
    forged["timing"]["paired_overheads"] = pairs
    forged["metrics"] = metrics
    forged["checks"] = _PROFILE._recomputed_checks(forged, metrics)

    assert "admission_consistency" in _PROFILE.validate_report(forged)


def test_profile_source_ir_audit_exposes_current_hotpath_state_and_hashes():
    audit = _PROFILE._source_ir_hotpath_audit()

    assert set(audit["source_sha256"]) == {
        "fluidSim_warp_blender51/sim/cuda_reduction.py",
        "fluidSim_warp_blender51/sim/fields.py",
        "fluidSim_warp_blender51/sim/particles.py",
        "fluidSim_warp_blender51/sim/species.py",
    }
    assert audit["compaction"]["duplicate_species_scatter_active"] is False
    assert audit["compaction"]["duplicate_species_scatter_paths"] == 0
    assert audit["compaction"]["scatter_receipt_atomic_add_sites"] == 2
    assert audit["compaction"]["scatter_receipt_atomic_min_sites"] == 0
    assert audit["compaction"]["scatter_receipt_atomic_max_sites"] == 0
    assert audit["compaction"]["native_postcompact_reducer_declared"] is True
    assert audit["compaction"]["native_device_count_pointer"] is True
    assert audit["compaction"]["native_float_cas_sites"] >= 2
    assert audit["limiter"]["global_alpha_direct_write_sites"] == 0
    assert audit["limiter"]["global_alpha_atomic_min_sites"] == 0
    assert audit["limiter"][
        "g2p_global_alpha_atomic_candidate_lt_one_guarded"] is False
    assert audit["limiter"][
        "particle_alpha_atomic_candidate_lt_one_guarded"] is False
    assert audit["limiter"][
        "delta_apply_min_alpha_receipt_tid_zero_guarded"] is False
    assert audit["scratch"]["face_delta_declared"] is False
    assert audit["scratch"]["face_delta_declaration_sites"] == 0
    assert audit["scratch"]["face_conductance_declared"] is False
    assert audit["scratch"]["face_conductance_declaration_sites"] == 0
    assert audit["neighbor"]["interior_fast_path_present"] is False


def test_profile_validator_rejects_forged_source_ir_and_launch_inventory():
    forged = _passing_profile_receipt()
    forged["audit"]["source_ir_hotpath"]["source_sha256"][
        "fluidSim_warp_blender51/sim/species.py"] = "0" * 64
    forged["audit"]["source_ir_hotpath"]["limiter"][
        "g2p_global_alpha_atomic_candidate_lt_one_guarded"] = True
    forged["audit"]["cases"]["diffusion_on"]["all_kernel_launches"] += 1

    errors = _PROFILE.validate_report(forged)

    assert "source_ir_hotpath" in errors
    assert "audit_launch_inventory" in errors


def test_profile_cuda_check_rejects_silent_species_reduction_fallback():
    forged = _passing_profile_receipt()
    forged["audit"]["cases"]["transport_only"][
        "species_bounds_reduction_backend"] = "warp_atomic"
    metrics, _pairs = _PROFILE._recomputed_metrics(forged)
    forged["checks"] = _PROFILE._recomputed_checks(forged, metrics)

    assert forged["checks"]["cuda_runtime"] is False


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("git_head", "0" * 40, "stale_git_head"),
        ("runtime_dependency_tree_sha256", "0" * 64,
         "stale_runtime_dependency_tree"),
        ("profiler_sha256", "0" * 64, "stale_profiler_hash"),
    ],
)
def test_profile_receipt_rejects_stale_current_identity(field, value, expected):
    stale = _passing_profile_receipt()
    stale["provenance"][field] = value

    assert expected in _PROFILE.validate_report(stale)


def test_profile_receipt_rejects_timestamp_older_than_frozen_max_age():
    stale = _passing_profile_receipt()
    stale["provenance"]["timestamp_utc"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=_PROFILE.MAX_EVIDENCE_AGE_SECONDS + 1)
    ).isoformat()

    assert "stale_timestamp" in _PROFILE.validate_report(stale)


def test_profile_receipt_rejects_changed_bytes_at_same_dirty_dependency_path(
        tmp_path, monkeypatch):
    relative = Path("fluidSim_warp_blender51") / "sim" / "dirty_dependency.py"
    dependency = tmp_path / relative
    dependency.parent.mkdir(parents=True)
    dependency.write_text("VALUE = 1\n", encoding="utf8")
    original_hash = _PROFILE._runtime_dependency_tree_sha256(
        root=tmp_path, tracked_relative_paths=(relative,))
    report = _passing_profile_receipt()
    report["provenance"]["runtime_dependency_tree_sha256"] = original_hash
    hash_helper = _PROFILE._runtime_dependency_tree_sha256
    monkeypatch.setattr(
        _PROFILE, "_runtime_dependency_tree_sha256",
        lambda: hash_helper(
            root=tmp_path, tracked_relative_paths=(relative,)))

    dependency.write_text("VALUE = 2\n", encoding="utf8")

    assert "stale_runtime_dependency_tree" in _PROFILE.validate_report(report)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda report: report.pop("provenance"), "provenance"),
        (lambda report: report["provenance"].pop("cuda_device"), "cuda_identity"),
        (lambda report: report["configuration"].update(warmup=14),
         "production_minimums"),
        (lambda report: report["timing"]["raw_samples_ms"]["disabled"].pop(),
         "raw_sample_counts"),
    ],
)
def test_profile_receipt_validator_rejects_missing_decision_grade_evidence(
        mutation, expected):
    broken = _passing_profile_receipt()
    mutation(broken)
    assert expected in _PROFILE.validate_report(broken)


def test_short_profile_is_diagnostic_and_rejected_as_production_evidence():
    diagnostic = _passing_profile_receipt(
        frames=3, warmup=1, diagnostic=True)

    assert _PROFILE.validate_report(diagnostic, require_production=False) == []
    assert "production_evidence" in _PROFILE.validate_report(diagnostic)


def test_timed_schedule_never_installs_warp_or_memory_wrappers():
    launch = object()
    synchronize = object()
    synchronize_device = object()
    array_numpy = object()
    tracker = SimpleNamespace(active=False)
    fake_wp = SimpleNamespace(
        launch=launch,
        synchronize=synchronize,
        synchronize_device=lambda _device: None,
        array=SimpleNamespace(numpy=array_numpy),
        ScopedMemoryTracker=tracker,
    )

    class Solver:
        def __init__(self, backend):
            self.species = (None if backend == "disabled" else SimpleNamespace(
                bounds_reduction_backend=backend))

        def step_frame(self):
            assert fake_wp.launch is launch
            assert fake_wp.synchronize is synchronize
            assert fake_wp.array.numpy is array_numpy
            assert tracker.active is False
            return SimpleNamespace(
                substeps=1,
                species_receipt=SimpleNamespace(diffusion_substeps=0),
            )

    contexts = {
        "disabled": {"solver": Solver("disabled")},
        "transport_only": {"solver": Solver("warp_atomic")},
        "diffusion_on": {"solver": Solver("native_cuda")},
    }
    ticks = iter(float(value) for value in range(100))

    samples, work = _PROFILE._run_timed_schedule(
        contexts, frames=2, device="cuda:0", wp_module=fake_wp,
        clock=lambda: next(ticks))

    assert len(samples["disabled"]) == 5
    assert len(samples["transport_only"]) == 2
    assert len(samples["diffusion_on"]) == 2
    assert work["transport_only"]["reduction_backends"] == {"warp_atomic"}
    assert work["diffusion_on"]["reduction_backends"] == {"native_cuda"}


def _write_profile_child(
        directory, index, *, transport=0.05, diffusion=0.20,
        baseline=10.0, executable=None, process_id=None):
    path = directory / f"child-{index:02d}.json"
    report = _passing_profile_receipt()
    report["provenance"]["process_id"] = process_id or 20_000 + index
    report["provenance"]["executable"] = executable or sys.executable
    report["provenance"]["command"] = [
        report["provenance"]["executable"],
        str(Path(_PROFILE.__file__).resolve()),
        "--frames", "60", "--warmup", "15", "--particles", "200000",
        "--device", "cuda:0", "--json-out", str(path.resolve()),
    ]
    report["timing"]["raw_samples_ms"]["disabled"] = [baseline] * 121
    report["timing"]["raw_samples_ms"]["transport_only"] = [
        baseline * (1.0 + transport)] * 60
    report["timing"]["raw_samples_ms"]["diffusion_on"] = [
        baseline * (1.0 + diffusion)] * 60
    metrics, paired = _PROFILE._recomputed_metrics(report)
    report["timing"]["paired_overheads"] = paired
    report["metrics"] = metrics
    report["checks"] = _PROFILE._recomputed_checks(report, metrics)
    report["status"] = "pass" if all(report["checks"].values()) else "fail"
    path.write_text(json.dumps(report, indent=2), encoding="utf8")
    return path


def _passing_aggregate(tmp_path, child_changes=None):
    child_changes = child_changes or {}
    paths = []
    for index in range(5):
        changes = child_changes.get(index, {})
        paths.append(_write_profile_child(tmp_path, index, **changes))
    return _PROFILE_RUNNER.build_aggregate_report(paths), paths


def test_aggregate_receipt_recomputes_frozen_mad_bound_and_retains_children(
        tmp_path):
    aggregate, paths = _passing_aggregate(tmp_path)

    assert _PROFILE_RUNNER.validate_aggregate_report(aggregate) == []
    assert len(aggregate["children"]) == 5
    assert {entry["receipt_path"] for entry in aggregate["children"]} == {
        str(path.resolve()) for path in paths}
    assert aggregate["metrics"]["transport_only"]["median"] == pytest.approx(0.05)
    assert aggregate["metrics"]["transport_only"]["scaled_mad"] == 0.0
    assert aggregate["metrics"]["transport_only"]["upper_bound"] == pytest.approx(0.05)


def test_aggregate_receipt_rejects_mutually_consistent_stale_children(tmp_path):
    paths = [_write_profile_child(tmp_path, index) for index in range(5)]
    for path in paths:
        child = json.loads(path.read_text(encoding="utf8"))
        child["provenance"]["git_head"] = "0" * 40
        path.write_text(json.dumps(child), encoding="utf8")
    aggregate = _PROFILE_RUNNER.build_aggregate_report(paths)

    errors = _PROFILE_RUNNER.validate_aggregate_report(aggregate)

    assert "stale_git_head" in errors
    assert "child_receipts_valid" in errors


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("git_head", "0" * 40, "stale_runner_git_head"),
        ("runtime_dependency_tree_sha256", "0" * 64,
         "stale_runner_runtime_dependency_tree"),
        ("runner_sha256", "0" * 64, "stale_runner_hash"),
        ("timestamp_utc", "2000-01-01T00:00:00+00:00",
         "stale_runner_timestamp"),
    ],
)
def test_aggregate_receipt_rejects_stale_runner_provenance(
        tmp_path, field, value, expected):
    aggregate, _paths = _passing_aggregate(tmp_path)
    aggregate["runner_provenance"][field] = value

    assert expected in _PROFILE_RUNNER.validate_aggregate_report(aggregate)


def test_aggregate_receipt_rejects_deleted_slow_child(tmp_path):
    aggregate, paths = _passing_aggregate(
        tmp_path, {4: {"transport": 0.11}})
    paths[4].unlink()

    errors = _PROFILE_RUNNER.validate_aggregate_report(aggregate)

    assert "child_receipt_missing" in errors
    assert "status" in errors


def test_aggregate_receipt_requires_exactly_five_manifested_children(tmp_path):
    aggregate, _paths = _passing_aggregate(tmp_path)
    aggregate["children"].pop()

    assert "child_count" in _PROFILE_RUNNER.validate_aggregate_report(aggregate)


def test_aggregate_receipt_rejects_mismatched_process_identity(tmp_path):
    other_executable = str((tmp_path / "other-python.exe").resolve())
    aggregate, _paths = _passing_aggregate(
        tmp_path, {3: {"executable": other_executable}})

    errors = _PROFILE_RUNNER.validate_aggregate_report(aggregate)

    assert "child_identity" in errors
    assert "status" in errors


def test_aggregate_receipt_rejects_unstable_process_medians(tmp_path):
    aggregate, _paths = _passing_aggregate(
        tmp_path,
        {
            0: {"transport": 0.00}, 1: {"transport": 0.01},
            2: {"transport": 0.02}, 3: {"transport": 0.05},
            4: {"transport": 0.08},
        })

    errors = _PROFILE_RUNNER.validate_aggregate_report(aggregate)

    assert "transport_only_stability" in errors
    assert "status" in errors


def test_aggregate_receipt_rejects_recomputed_upper_bound_over_budget(tmp_path):
    aggregate, _paths = _passing_aggregate(
        tmp_path,
        {
            0: {"transport": 0.08}, 1: {"transport": 0.09},
            2: {"transport": 0.10}, 3: {"transport": 0.11},
            4: {"transport": 0.12},
        })

    assert "transport_only_upper_bound" in (
        _PROFILE_RUNNER.validate_aggregate_report(aggregate))


@pytest.mark.parametrize(
    ("child_changes", "expected"),
    [
        ({0: {"baseline": 10.01}}, "disabled_median"),
        ({0: {"transport": -0.30}}, "transport_only_p95_delta_ms"),
        ({0: {"diffusion": -0.50}}, "diffusion_on_p95_delta_ms"),
    ],
)
def test_aggregate_receipt_rejects_absolute_tail_or_baseline_limit(
        tmp_path, child_changes, expected):
    aggregate, _paths = _passing_aggregate(tmp_path, child_changes)
    assert expected in _PROFILE_RUNNER.validate_aggregate_report(aggregate)


def test_aggregate_receipt_rejects_short_diagnostic_children(tmp_path):
    aggregate, paths = _passing_aggregate(tmp_path)
    child = json.loads(paths[0].read_text(encoding="utf8"))
    child["configuration"].update(frames=3, warmup=1)
    child["evidence_kind"] = "diagnostic"
    child["status"] = "diagnostic"
    for name in ("transport_only", "diffusion_on"):
        child["timing"]["raw_samples_ms"][name] = child[
            "timing"]["raw_samples_ms"][name][:3]
        child["timing"]["paired_overheads"][name] = child[
            "timing"]["paired_overheads"][name][:3]
    child["timing"]["raw_samples_ms"]["disabled"] = child[
        "timing"]["raw_samples_ms"]["disabled"][:7]
    metrics, pairs = _PROFILE._recomputed_metrics(child)
    child["timing"]["paired_overheads"] = pairs
    child["metrics"] = metrics
    child["checks"] = _PROFILE._recomputed_checks(child, metrics)
    paths[0].write_text(json.dumps(child), encoding="utf8")
    aggregate["children"][0]["receipt_sha256"] = hashlib.sha256(
        paths[0].read_bytes()).hexdigest()

    errors = _PROFILE_RUNNER.validate_aggregate_report(aggregate)

    assert "production_children" in errors
    assert "status" in errors


def test_capacity_plan_captures_species_bytes_and_admitted_halo():
    settings = _settings(species_enabled=True)
    plan = plan_initialization(settings, (_source(),), (), (), _resources())

    assert plan.species_transport_bytes == estimate_species_transport_bytes(settings)
    assert plan.species_cache_staging_bytes == estimate_species_cache_staging_bytes(settings)
    assert plan.species_halo_leaf_count == 2
    assert plan.required_leaf_coords == ((0, 0, 0), (1, 0, 0), (2, 0, 0))


def test_single_admission_snapshot_owns_species_capacity_demand(monkeypatch):
    from fluidSim_warp_blender51.sim import source_regions

    class Compiler:
        def compile(self, _plans, _context):
            return SimpleNamespace(footprints=(), diagnostics=(), sources=())

        def release(self):
            pass

    plan = SimpleNamespace(
        blockers=(), species_transport_bytes=1234,
        species_cache_staging_bytes=567,
        species_halo_leaf_count=9, species_diffusion_substeps=3)
    monkeypatch.setattr(source_regions, "SourceCompiler", Compiler)
    monkeypatch.setattr(admission, "_plan_capacity", lambda *_a, **_k: plan)
    depsgraph = SimpleNamespace(updates=[])
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.context = SimpleNamespace(
        evaluated_depsgraph_get=lambda: depsgraph)
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    scene = SimpleNamespace(
        frame_current=1, frame_start=1, objects=[],
        fluidsim=SimpleNamespace(
            domain_object=None, profiling_enabled=False,
            profiling_sync=False, height_enabled=False))
    settings = _settings(device="cpu")

    snapshot = admission.capture_snapshot(
        scene, settings, _resources(), depsgraph=depsgraph, scene_identity=1)
    try:
        assert snapshot.capacity_plan is plan
        assert snapshot.species_transport_bytes == 1234
        assert snapshot.species_cache_staging_bytes == 567
        assert snapshot.species_halo_leaf_count == 9
        assert snapshot.species_diffusion_substeps == 3
    finally:
        snapshot.release_transients()


def test_excessive_diffusion_substeps_block_capacity_before_allocation():
    settings = _settings(
        species_enabled=True,
        species_diffusivity=0.1,
        species_max_diffusion_substeps=1,
    )
    plan = plan_initialization(settings, (_source(),), (), (), _resources())

    blocker = next(item for item in plan.blockers
                   if item.code == "SPECIES_DIFFUSION_SUBSTEPS")
    assert blocker.required == 3
    assert blocker.available == 1


def test_admission_rejects_nonfinite_species_timing_inputs():
    settings = _settings(species_enabled=True, species_diffusivity=0.1,
                         frame_rate=0.0)
    plan = plan_initialization(settings, (_source(),), (), (), _resources())

    assert any(item.code == "SPECIES_DIFFUSION_SETTINGS"
               for item in plan.blockers)


def _passing_receipt():
    return {
        "schema_version": 2,
        "status": "pass",
        "runtime": {
            "blender_version": "5.1.0", "blender_executable": "blender",
            "python_version": "3.11", "addon_version": [1, 0, 0],
            "warp_version": "1.16.0", "device_alias": "cuda:0",
            "device_name": "GPU",
        },
        "thresholds": dict(_ACCEPTANCE.THRESHOLDS),
        "metrics": {
            "closed_box_relative_conservation": 1.0e-7,
            "min_concentration": 0.0,
            "max_loading": 1.0,
            "gaussian_relative_l2": 0.01,
            "face_seam_relative_l2": 1.0e-7,
            "edge_seam_relative_l2": 1.0e-7,
            "corner_seam_relative_l2": 1.0e-7,
            "checkpoint_relative_l2": 1.0e-7,
            "substep_device_to_host_transfers": 0,
            "long_run_frames": 240,
            "long_run_receipt_frames": 240,
            "long_run_solver_substeps_min": 4,
            "long_run_diffusion_substeps_min": 4,
            "measured_source_amount": 0.0,
            "max_ledger_residual": 1.0e-7,
            "max_receipt_continuity_error": 1.0e-7,
            "particle_mixing_index_delta": 0.1,
            "reopen_particle_mixing_index_first": 0.1,
            "reopen_particle_mixing_index_second": 0.2,
            "reopen_particle_mixing_index_delta": 0.1,
            "reopen_diffusion_substeps_min": 4,
            "reopen_limiter_alpha_min": 0.75,
            "particle_color_distribution_delta": 0.1,
            "particle_color_samples_min": 8,
            "surface_color_delta": 0.1,
            "surface_color_samples_min": 8,
            "lifecycle_source_amount_a": 0.25,
            "lifecycle_source_amount_b": 0.5,
            "lifecycle_sink_amount_a": 0.25,
            "lifecycle_sink_amount_b": 0.5,
            "lifecycle_max_ledger_residual": 1.0e-7,
            "lifecycle_removed_particle_count": 8,
            "solver_substeps": 4,
            "cache_frames_written": 2,
            "cache_frames_read": 2,
        },
        "provenance": {
            "solver": "FluidSolver.step_frame",
            "cache_write": "io.cache.write_frame",
            "cache_read": "io.cache.read_frame",
            "playback": "blender.playback.play_cache_frame",
            "reopened_blend": True,
            "measured_segment": {
                "advance": "FluidSolver.step_frame",
                "seeded_before_measurement": True,
                "post_baseline_injection_count": 0,
                "active_source_count": 0,
                "direct_species_transport_calls": 0,
            },
            "lifecycle_boundary_segment": {
                "advance": "FluidSolver.step_frame",
                "source_descriptor": "SourceShapeDesc",
                "sink_descriptor": "SourceShapeDesc",
                "source_frame_count": 1,
                "sink_frame_count": 1,
            },
        },
        "checks": {
            "mixing_off_identity": True, "finite": True,
            "real_solver_long_run": True,
            "zero_measured_sources": True,
            "ledger_closure": True,
            "diffusion_caused_particle_mixing": True,
            "derived_particle_color_distribution_change": True,
            "reopen_diffusion_only_mixing": True,
            "nonempty_publication_samples": True,
            "real_solver_lifecycle_boundaries": True,
            "saved_reopen_surface_color": True,
            "real_solver_step_frame": True,
            "multi_cfl_substeps": True,
            "cache_write_read": True,
            "playback_after_reopen": True,
        },
    }


def test_acceptance_receipt_validator_requires_all_thresholds_and_identity():
    assert _ACCEPTANCE.validate_report(_passing_receipt()) == []

    broken = _passing_receipt()
    broken["runtime"].pop("device_name")
    broken["metrics"]["gaussian_relative_l2"] = 0.021
    broken["checks"]["saved_reopen_surface_color"] = False
    errors = _ACCEPTANCE.validate_report(broken)
    assert errors == [
        "gaussian_relative_l2", "runtime_identity",
        "saved_reopen_surface_color", "status",
    ]


def test_acceptance_receipt_rejects_direct_transport_or_fabricated_playback_proof():
    broken = _passing_receipt()
    broken["provenance"]["measured_segment"][
        "advance"] = "SpeciesTransport.diffuse"
    broken["provenance"]["measured_segment"][
        "direct_species_transport_calls"] = 240

    errors = _ACCEPTANCE.validate_report(broken)
    assert "integrated_provenance" in errors
    assert "status" in errors


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (lambda receipt: receipt["provenance"]["measured_segment"].update(
            post_baseline_injection_count=1), "integrated_provenance"),
        (lambda receipt: receipt["provenance"]["measured_segment"].update(
            active_source_count=1), "integrated_provenance"),
        (lambda receipt: receipt["metrics"].update(
            measured_source_amount=0.25), "zero_measured_sources"),
        (lambda receipt: receipt["metrics"].update(
            long_run_frames=239), "real_solver_long_run"),
        (lambda receipt: receipt["metrics"].update(
            long_run_solver_substeps_min=1), "multi_cfl_substeps"),
        (lambda receipt: receipt["metrics"].update(
            long_run_diffusion_substeps_min=0),
         "diffusion_caused_particle_mixing"),
        (lambda receipt: receipt["metrics"].update(
            max_ledger_residual=3.0e-6), "ledger_closure"),
        (lambda receipt: receipt["metrics"].update(
            particle_mixing_index_delta=0.0),
         "diffusion_caused_particle_mixing"),
        (lambda receipt: receipt["metrics"].update(
            particle_color_distribution_delta=0.0),
         "derived_particle_color_distribution_change"),
    ),
)
def test_acceptance_receipt_recomputes_real_solver_measured_segment_claims(
        mutate, expected):
    broken = _passing_receipt()
    mutate(broken)

    errors = _ACCEPTANCE.validate_report(broken)
    assert expected in errors
    assert "status" in errors


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (lambda receipt: receipt["provenance"][
            "lifecycle_boundary_segment"].update(
                advance="SpeciesTransport.diffuse"),
         "integrated_provenance"),
        (lambda receipt: receipt["metrics"].update(
            reopen_limiter_alpha_min=float("nan")), "finite_metrics"),
        (lambda receipt: receipt["metrics"].update(
            reopen_limiter_alpha_min=-0.01), "reopen_limiter_alpha_range"),
        (lambda receipt: receipt["metrics"].update(
            reopen_limiter_alpha_min=1.01), "reopen_limiter_alpha_range"),
        (lambda receipt: receipt["metrics"].update(
            reopen_particle_mixing_index_delta=0.0),
         "reopen_diffusion_only_mixing"),
        (lambda receipt: receipt["metrics"].update(
            reopen_particle_mixing_index_second=0.15),
         "reopen_diffusion_only_mixing"),
        (lambda receipt: receipt["metrics"].update(
            reopen_diffusion_substeps_min=0),
         "reopen_diffusion_only_mixing"),
        (lambda receipt: receipt["metrics"].update(
            particle_color_samples_min=0), "nonempty_publication_samples"),
        (lambda receipt: receipt["metrics"].update(
            surface_color_samples_min=0), "nonempty_publication_samples"),
        (lambda receipt: receipt["metrics"].update(
            lifecycle_source_amount_a=0.0),
         "real_solver_lifecycle_boundaries"),
        (lambda receipt: receipt["metrics"].update(
            lifecycle_sink_amount_b=0.0),
         "real_solver_lifecycle_boundaries"),
        (lambda receipt: receipt["metrics"].update(
            lifecycle_max_ledger_residual=3.0e-6),
         "real_solver_lifecycle_boundaries"),
        (lambda receipt: receipt["metrics"].update(
            lifecycle_removed_particle_count=0),
         "real_solver_lifecycle_boundaries"),
    ),
)
def test_acceptance_receipt_validates_reopen_and_lifecycle_boundary_fields(
        mutate, expected):
    broken = _passing_receipt()
    mutate(broken)

    errors = _ACCEPTANCE.validate_report(broken)
    assert expected in errors
    assert "status" in errors


def test_steady_mixing_off_frame_launches_no_species_work(monkeypatch):
    import warp as wp

    particles, _grid, transport = _ACCEPTANCE._fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device="cuda:0", diffusivity=0.0)
    transport.begin_frame()
    first = transport.finalize_frame_receipt()
    launches = []
    original_launch = wp.launch

    def counted_launch(kernel, *args, **kwargs):
        launches.append(str(getattr(kernel, "key", kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", counted_launch)
    monkeypatch.setattr(
        type(transport._receipt), "numpy",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("steady D=0 receipt crossed device/host boundary")))

    transport.diffuse(1.0 / 24.0)
    second = transport.finalize_frame_receipt()

    assert launches == []
    assert second.before == first.after
    assert second.after == first.after
    assert second.diffusion_substeps == 0


def test_one_local_mix_step_uses_only_fused_transfer_launches(monkeypatch):
    import warp as wp

    _particles, _grid, transport = _ACCEPTANCE._fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device="cuda:0", diffusivity=0.1)
    transport.begin_frame()
    launches = []
    original_launch = wp.launch

    def counted_launch(kernel, *args, **kwargs):
        launches.append(str(getattr(kernel, "key", kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", counted_launch)
    transport.diffuse(0.5)

    assert len(launches) == 2
    assert any("particle_set_grid_data_with_species" in key for key in launches)
    assert any("staggered_field_to_particle_with_species" in key
               for key in launches)
    assert not any(term in key for key in launches for term in (
        "conductance", "flux", "alpha", "delta", "apply"))


def test_diffusion_frame_reuses_prior_before_and_finalizes_after_once(monkeypatch):
    import warp as wp

    _particles, _grid, transport = _ACCEPTANCE._fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device="cuda:0", diffusivity=0.1)
    transport.begin_frame()
    baseline = transport.finalize_frame_receipt()
    launches = []
    native_launches = []
    original_launch = wp.launch
    original_native_launch = transport._bounds_reducer.launch

    def counted_launch(kernel, *args, **kwargs):
        launches.append(str(getattr(kernel, "key", kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", counted_launch)
    monkeypatch.setattr(
        transport._bounds_reducer, "launch",
        lambda *args, **kwargs: (
            native_launches.append("fluid_species_postcompact_reduce"),
            original_native_launch(*args, **kwargs),
        )[1],
    )
    transport.diffuse(0.5)
    receipt = transport.finalize_frame_receipt()

    assert len(launches) == 3  # receipt init + fused P2G + fused G2P
    assert native_launches == ["fluid_species_postcompact_reduce"]
    assert receipt.before == pytest.approx(baseline.after, abs=1.0e-6)
    assert receipt.after == pytest.approx(baseline.after, abs=1.0e-6)
    assert receipt.residual == pytest.approx((0.0, 0.0), abs=2.0e-6)


def test_two_diffuse_calls_capture_after_once_at_frame_finalize():
    particles, _grid, transport = _ACCEPTANCE._fixture(
        ((6, 0, 0), (7, 0, 0)), ((1, 0), (0, 1)),
        device="cuda:0", diffusivity=0.1)
    transport.begin_frame()
    baseline = transport.finalize_frame_receipt()

    transport.diffuse(0.25)
    transport.diffuse(0.25)
    receipt = transport.finalize_frame_receipt()
    amounts = _ACCEPTANCE._amounts(particles)

    assert receipt.before == pytest.approx(baseline.after, abs=1.0e-6)
    assert receipt.after == pytest.approx(amounts, abs=1.0e-6)
    assert receipt.residual == pytest.approx((0.0, 0.0), abs=2.0e-6)


def test_final_cfl_compaction_captures_after_without_receipt_reduction(monkeypatch):
    import warp as wp
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.125, leaf_capacity=8, max_particles=4,
        max_whitewater=0, frame_rate=24.0, cfl=0.01,
        min_substeps=1, max_substeps=4, gravity=(0.0, 0.0, 0.0),
        pressure_iterations=8, relax_enabled=False, surface_enabled=False,
        whitewater_enabled=False, species_enabled=True,
        species_diffusivity=0.1, species_max_diffusion_substeps=16,
        device="cuda:0")
    solver = FluidSolver(settings)
    try:
        solver.inject_particles_exact(
            ((0.4, 0.5, 0.5), (0.6, 0.5, 0.5)),
            velocity=(0.1, 0.0, 0.0),
            species_concentration=(0.25, 0.5, 0.0, 0.0))
        solver.species.finalize_frame_receipt()
        launches = []
        original_launch = wp.launch

        def counted_launch(kernel, *args, **kwargs):
            launches.append(str(getattr(kernel, "key", kernel)))
            return original_launch(kernel, *args, **kwargs)

        monkeypatch.setattr(wp, "launch", counted_launch)
        stats = solver.step_frame()

        assert stats.substeps == 4
        assert stats.species_receipt.diffusion_substeps == 4
        assert stats.species_receipt.residual == pytest.approx(
            (0.0, 0.0), abs=2.0e-6)
        assert not any("species_bounds_budget_reduce" in key
                       or "species_sink_and_removal" in key
                       for key in launches)
        assert sum("species_compact_alive_scatter" in key
                   for key in launches) == stats.substeps
    finally:
        solver.release()


def test_particle_growth_retargets_species_and_preserves_authoritative_state():
    import numpy as np
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25, leaf_capacity=1, max_particles=2,
        max_whitewater=0, surface_enabled=False, whitewater_enabled=False,
        gravity=(0.0, 0.0, 0.0), species_enabled=True,
        species_diffusivity=0.1, auto_grow_particles=True, device="cuda:0")
    solver = FluidSolver(settings)
    try:
        solver.inject_particles_exact(
            ((0.3, 0.3, 0.3), (0.55, 0.3, 0.3)),
            species_concentration=(0.25, 0.5, 0.0, 0.0))
        old_particles, old_species = solver.particles, solver.species
        old_species.finalize_frame_receipt()
        before_c = old_particles.attr("species_concentration").numpy()[:2].copy()
        before_v = old_particles.attr("represented_volume").numpy()[:2].copy()
        last_receipt = old_species._last_receipt

        result = solver.ensure_particle_capacity(6)

        assert result.status == "GROWN"
        assert solver.particles is not old_particles
        assert solver.species is not old_species
        assert solver.species.particles is solver.particles
        assert solver.species.particle_preview.shape[0] == solver.particles.capacity
        assert not hasattr(solver.species, "particle_delta")
        assert solver.species._last_receipt == last_receipt
        np.testing.assert_array_equal(
            solver.particles.attr("species_concentration").numpy()[:2], before_c)
        np.testing.assert_array_equal(
            solver.particles.attr("represented_volume").numpy()[:2], before_v)
        solver._rebuild_topology()
        solver.species.diffuse(0.01)
        receipt = solver.species.finalize_frame_receipt()
        assert np.all(np.isfinite(
            solver.particles.attr("species_concentration").numpy()[:2]))
        assert receipt.residual == pytest.approx((0.0, 0.0), abs=2.0e-6)
    finally:
        solver.release()


def test_particle_growth_species_allocation_failure_preserves_old_solver(monkeypatch):
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25, leaf_capacity=1, max_particles=2,
        max_whitewater=0, surface_enabled=False, whitewater_enabled=False,
        gravity=(0.0, 0.0, 0.0), species_enabled=True,
        species_diffusivity=0.1, auto_grow_particles=True, device="cuda:0")
    solver = FluidSolver(settings)
    try:
        solver.inject_particles_exact(
            ((0.3, 0.3, 0.3),),
            species_concentration=(0.25, 0.5, 0.0, 0.0))
        old_particles, old_bins, old_species = (
            solver.particles, solver.bins, solver.species)

        def fail_species_candidate(_particles):
            raise RuntimeError("injected species scratch allocation failure")

        monkeypatch.setattr(
            old_species, "resized_copy", fail_species_candidate, raising=False)
        result = solver.ensure_particle_capacity(6)

        assert result.status == "FAILED"
        assert "species scratch allocation failure" in result.reason
        assert solver.particles is old_particles
        assert solver.bins is old_bins
        assert solver.species is old_species
        assert solver.species.particles is solver.particles
    finally:
        solver.release()


def test_live_species_topology_enforces_one_leaf_halo_when_authored_zero():
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0), domain_max=(24.0, 8.0, 8.0),
        voxel_size=1.0, leaf_capacity=3, max_particles=1,
        max_whitewater=0, topology_expand=0, surface_enabled=False,
        whitewater_enabled=False, gravity=(0.0, 0.0, 0.0),
        species_enabled=True, species_diffusivity=0.1, device="cuda:0")
    solver = FluidSolver(settings)
    try:
        solver.inject_particles_exact(
            ((8.5, 0.5, 0.5),),
            species_concentration=(1.0, 0.0, 0.0, 0.0))
        solver._rebuild_topology()
        coords = {tuple(map(int, value))
                  for value in solver.grid.leaf_coords.numpy()[:3]}
        assert coords == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    finally:
        solver.release()


def test_sourcing_reports_whether_particle_sink_scan_is_required():
    from fluidSim_warp_blender51.sim.enums import SourceOperation
    from fluidSim_warp_blender51.sim.sourcing import SourcePipeline

    sourcing = object.__new__(SourcePipeline)
    sourcing.plans = ()
    sourcing.sources = []
    assert sourcing.has_particle_sinks() is False
    sourcing.plans = (SimpleNamespace(operation=SourceOperation.SINK),)
    assert sourcing.has_particle_sinks() is True
