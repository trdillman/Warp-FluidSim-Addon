"""Reject false FSPT performance success; these synthetic rows certify no runtime."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import ast
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "tests/render/fixtures/fspt_acceptance_contract_v1.json"


def evaluator():
    path = ROOT / "tools/gate/fspt_contract.py"
    assert path.is_file(), "The FSPT contract evaluator is not implemented"
    spec = importlib.util.spec_from_file_location("fspt_contract_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


@pytest.fixture
def contract():
    return json.loads(CONTRACT_PATH.read_text())


@pytest.fixture
def trial(contract):
    # 120 warmup + 600 measured solved states, 30 completed steps/s and 24-Hz
    # simulation time. Expectations below are calculated independently by hand.
    lanes = {}
    for lane in "ABCD":
        states = [{"solver_frame": i, "state_revision": i * 5, "epoch": 1,
                   "simulation_time": i / 24, "completed_at": i / 30,
                   "geometry_sha256": digest(f"geometry-{i}")}
                  for i in range(1, 721)]
        lanes[lane] = {"states": states, "warmup_states": 120,
                       "solver_settings_sha256": digest("authored-solver"),
                       "surface_settings_sha256": digest("authored-surface"),
                       "required_outputs_sha256": digest("required-output-obligations"),
                       "demand_schedule_sha256": digest("matched-demand"),
                       "environment_fingerprint": digest("actual-environment")}
    presentations = [
        {"solver_frame": i, "state_revision": i * 5, "epoch": 1,
         "surface_generation": i, "state_ready_at": i / 30,
         "render_completed_at": i / 30 + 0.008, "presented_at": i / 30 + 0.02,
         "geometry_sha256": digest(f"geometry-{i}"),
         "pixels_sha256": digest(f"actual-pixels-{i}")}
        for i in range(121, 721)]
    metrics = {
        "fresnel_absolute_error": (0.001, 0.0001),
        "slab_transmission_relative_error": (0.005, 0.001),
        "snell_angle_degrees_error": (0.05, 0.01),
        "tir_leakage_fraction": (0.0, 0.000005),
        "ordinary_luminance_nmae": (0.04, 0.005),
        "caustic_luminance_nmae": (0.08, 0.01),
        "thin_feature_precision": (0.99, 0.005),
        "thin_feature_recall": (0.99, 0.005),
        "ghost_residual_after_two_new_frames": (0.005, 0.001),
    }
    return {
        "contract_id": contract["contract_id"],
        "contract_sha256": hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "candidate_sha": "a" * 40,
        "fixture_id": "dam_break_authored",
        "fixture_sha256": "05e1918423e48fdf22ea702ff5ca2db1c8f7464be865619d382ad5d8118b92f8",
        "profile": "live_native1080p_spp1",
        "settings": deepcopy(contract["profiles"]["live_native1080p_spp1"]["settings"]),
        "lanes": lanes,
        "presentations": presentations,
        "presentation_scope": "draw_submission",
        "quality": {
            "confidence_level": 0.99, "stochastic_analytic_rays": 1048576,
            "reference_sha256": digest("independent-reference"),
            "reference_frozen_at": "2026-11-27T12:00:00Z",
            "candidate_started_at": "2026-11-27T13:00:00Z",
            "mask_sha256": {name: digest(name) for name in
                            ("ordinary_liquid", "caustic", "thin_features", "disocclusion")},
            "mask_pixels": {name: 512 for name in
                            ("ordinary_liquid", "caustic", "thin_features", "disocclusion")},
            "reference_convergence_ordinary_nmae": 0.002,
            "reference_convergence_caustic_nmae": 0.004,
            "nonfinite_radiance_count": 0,
            "metrics": {name: {"value": value, "ci99_half_width": uncertainty}
                        for name, (value, uncertainty) in metrics.items()},
        },
        "memory": {"peak_combined_bytes": 8 * 1024**3,
                   "physical_device_bytes": 16 * 1024**3},
    }


def test_valid_joint_trial_reports_recomputed_rate_and_scope(trial, contract):
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "PASS", result
    assert result["scope"] == "contract_evaluation_only"
    assert result["metrics"]["fresh_states"] == 600
    assert result["metrics"]["solver_wall_ratio"] == pytest.approx(1.0)
    assert result["metrics"]["fresh_states_per_second"] == pytest.approx(600 / 20.02)
    assert result["metrics"]["simulated_seconds_per_wall_second"] == pytest.approx(1.25)


def test_zero_overhead_with_no_rendered_images_cannot_pass(trial, contract):
    trial["presentations"] = []
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert any("fresh" in reason for reason in result["reasons"])


@pytest.mark.parametrize("frozen", ["geometry_sha256", "pixels_sha256"])
def test_moving_labels_with_frozen_content_do_not_count_as_fresh(trial, contract, frozen):
    for row in trial["presentations"]:
        row[frozen] = digest("frozen")
        if frozen == "geometry_sha256":
            trial["lanes"]["D"]["states"][row["solver_frame"] - 1][frozen] = row[frozen]
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert result["metrics"]["fresh_states"] == 1


def test_repeated_presentations_do_not_raise_solved_state_throughput(trial, contract):
    trial["presentations"] = [deepcopy(trial["presentations"][0]) for _ in range(20000)]
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert result["metrics"]["fresh_states"] <= 1


def test_changed_content_with_reused_publication_generation_is_not_fresh(trial, contract):
    for row in trial["presentations"]:
        row["surface_generation"] = 121
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert any("generation" in reason for reason in result["reasons"])


def test_quality_point_estimate_inside_limit_fails_when_confidence_bound_exceeds_it(trial, contract):
    trial["quality"]["metrics"]["ordinary_luminance_nmae"] = {
        "value": 0.099, "ci99_half_width": 0.002}
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert any("ordinary_luminance_nmae" in reason for reason in result["reasons"])


@pytest.mark.parametrize("mutation", ["boolean_samples", "boolean_epoch", "boolean_mask_count",
                                     "boolean_nonfinite_count", "nonstr_date"])
def test_malformed_typed_evidence_fails_without_raising(trial, contract, mutation):
    if mutation == "boolean_samples":
        trial["settings"]["samples_per_image"] = True
    elif mutation == "boolean_epoch":
        trial["presentations"][0]["epoch"] = True
    elif mutation == "boolean_mask_count":
        trial["quality"]["mask_pixels"]["caustic"] = True
    elif mutation == "boolean_nonfinite_count":
        trial["quality"]["nonfinite_radiance_count"] = False
    elif mutation == "nonstr_date":
        trial["quality"]["reference_frozen_at"] = 0
    assert evaluator().evaluate_live_trial(trial, contract)["status"] == "FAIL"


@pytest.mark.parametrize("report", [None, {}, [], {"contract_id": "unknown"}])
def test_missing_trial_cannot_produce_a_pass(report, contract):
    assert evaluator().evaluate_live_trial(report, contract)["status"] == "FAIL"


@pytest.mark.parametrize("field,value", [
    ("internal_width", 960), ("samples_per_image", 0), ("max_bounces", 6),
    ("max_transmission_bounces", 6), ("denoising", "UNQUALIFIED"),
    ("temporal_reuse", True), ("adaptive_sampling", True), ("radiance_clamp", 1.0),
])
def test_cheaper_quality_configuration_cannot_inherit_profile_pass(trial, contract, field, value):
    trial["settings"][field] = value
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert any("quality settings" in reason for reason in result["reasons"])


@pytest.mark.parametrize("field", ["solver_settings_sha256", "surface_settings_sha256",
                                  "required_outputs_sha256", "demand_schedule_sha256"])
def test_changed_solver_or_output_demand_fails_matched_comparison(trial, contract, field):
    trial["lanes"]["D"][field] = digest("reduced-fidelity")
    assert evaluator().evaluate_live_trial(trial, contract)["status"] == "FAIL"


def test_completed_step_wall_and_tail_must_both_pass(trial, contract):
    rows = trial["lanes"]["D"]["states"]
    for row in rows:
        row["completed_at"] *= 1.06
    for row in trial["presentations"]:
        row["state_ready_at"] *= 1.06
        row["render_completed_at"] = row["state_ready_at"] + 0.008
        row["presented_at"] = row["state_ready_at"] + 0.02
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert result["metrics"]["solver_wall_ratio"] == pytest.approx(1.06)
    assert any("p95" in reason for reason in result["reasons"])


def test_fresh_but_old_images_fail_age_gate(trial, contract):
    for row in trial["presentations"]:
        row["presented_at"] += 0.30
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert any("age" in reason for reason in result["reasons"])


def test_subrealtime_solver_cannot_pass_by_repeated_fast_images(trial, contract):
    for lane in trial["lanes"].values():
        for row in lane["states"]:
            row["completed_at"] *= 2
    for row in trial["presentations"]:
        row["state_ready_at"] *= 2
        row["render_completed_at"] = row["state_ready_at"] + 0.008
        row["presented_at"] = row["state_ready_at"] + 0.02
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL"
    assert result["metrics"]["simulated_seconds_per_wall_second"] == pytest.approx(0.625)


@pytest.mark.parametrize("mutation", ["vram", "quality", "thin_recall", "uncertainty",
                                     "mask", "reference_after_candidate", "warmup", "missing_lane",
                                     "wrong_fixture", "wrong_contract", "nan_clock", "wrong_epoch"])
def test_insufficient_or_inconsistent_evidence_fails_closed(trial, contract, mutation):
    if mutation == "vram":
        trial["memory"]["peak_combined_bytes"] = 15 * 1024**3
    elif mutation == "quality":
        trial["quality"]["metrics"]["ordinary_luminance_nmae"]["value"] = 0.11
    elif mutation == "thin_recall":
        trial["quality"]["metrics"]["thin_feature_recall"]["value"] = 0.94
    elif mutation == "uncertainty":
        trial["quality"]["metrics"]["fresnel_absolute_error"]["ci99_half_width"] = 0.002
    elif mutation == "mask":
        trial["quality"]["mask_pixels"]["thin_features"] = 0
    elif mutation == "reference_after_candidate":
        trial["quality"]["reference_frozen_at"] = "2026-11-28T12:00:00Z"
    elif mutation == "warmup":
        trial["lanes"]["A"]["warmup_states"] = 0
    elif mutation == "missing_lane":
        del trial["lanes"]["C"]
    elif mutation == "wrong_fixture":
        trial["fixture_sha256"] = digest("different-surface-recipe")
    elif mutation == "wrong_contract":
        trial["contract_sha256"] = digest("old-acceptance-thresholds")
    elif mutation == "nan_clock":
        trial["lanes"]["D"]["states"][180]["completed_at"] = float("nan")
    elif mutation == "wrong_epoch":
        trial["presentations"][0]["epoch"] = 99
    result = evaluator().evaluate_live_trial(trial, contract)
    assert result["status"] == "FAIL", mutation
    assert result["reasons"]


def test_contract_identity_is_independent_of_mapping_order(contract):
    module = evaluator()
    reordered = dict(reversed(list(contract.items())))
    assert module.contract_sha256(reordered) == module.contract_sha256(contract)
    changed = deepcopy(contract)
    changed["joint_live"]["fresh_states_per_second_min"] = 1
    assert module.contract_sha256(changed) != module.contract_sha256(contract)


def test_contract_preserves_existing_release_budget_authority(contract):
    # Inspect literals without importing or mutating the production budget tool.
    tree = ast.parse((ROOT / "tools/performance_budget.py").read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    existing = contract["existing_release_gate"]
    assert existing["demos"] == list(constants["DEMOS"])
    for field, constant in {
        "base_host_ratio_max": "BUDGET_RATIO",
        "trials_per_variant": "MIN_TRIALS_PER_VARIANT",
        "existing_early_stop_trials_per_variant": "EARLY_STOP_TRIALS_PER_VARIANT",
        "existing_early_stop_margin": "EARLY_STOP_MARGIN",
        "residual_ratio_limit": "RESIDUAL_RATIO_LIMIT",
        "owned_multigrid_residual_ratio_limit": "MULTIGRID_RESIDUAL_RATIO_LIMIT",
    }.items():
        assert existing[field] == constants[constant], field


def test_contract_freezes_joint_targets_and_preserves_required_product_scope(contract):
    live = contract["joint_live"]
    assert live["fresh_states_per_second_min"] == 24
    assert live["simulated_seconds_per_wall_second_min"] == 1
    assert live["solver_wall_ratio_max"] == live["solver_p95_step_interval_ratio_max"] == 1.05
    assert live["p95_state_ready_to_present_ms_max"] == 100
    assert live["max_state_ready_to_present_ms_max"] == 250
    assert live["peak_combined_vram_fraction_max"] == 0.8
    assert contract["fixed_renderer"]["new_images_per_second_min"] == 30
    for profile in contract["profiles"].values():
        settings = profile["settings"]
        assert (settings["internal_width"], settings["internal_height"]) == (1920, 1080)
        assert settings["max_bounces"] == settings["max_transmission_bounces"] == 12
        assert profile["ordinary_nmae_max"] == (0.1 if profile["mode"] == "LIVE" else 0.05)
        assert profile["caustic_nmae_max"] == (0.2 if profile["mode"] == "LIVE" else 0.1)
    product = contract["required_product_compatibility"]
    assert {"full_required_Principled_inputs", "normal_bump", "fluidsim_color",
            "fluidsim_viscosity", "mixed_shaders", "node_groups"} <= set(product["materials"])
    assert {"depth_dependent_absorption", "nested_dielectric_media"} <= set(product["liquid_interior"])
    assert {"depth_of_field_final", "motion_blur_final",
            "topology_changing_fluid_shutter_samples"} <= set(product["geometry_camera"])
    assert set(contract["export"]["required_passes"]) == {
        "Combined", "Alpha", "Depth", "Normal", "Albedo", "Emission", "Environment",
        "Direct", "Indirect", "Transmission"}

