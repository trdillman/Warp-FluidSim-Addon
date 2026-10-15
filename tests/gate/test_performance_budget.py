"""Focused tests for the production performance-budget evidence reducer."""

from __future__ import annotations

import pytest

from tools.performance_budget import (
	DEMOS,
	NEUTRAL_ABSENT_EFFECTIVE_SETTINGS,
	PROTECTED_SETTINGS,
	REFERENCE_VERSIONS,
	SOLVER_TIMER_CONTRACT,
	abba_schedule,
	evaluate_budget,
)


def _records(host, solver, *, frame1_multiplier=1.0, residual=1.0e-5, iterations=40):
	frame_dt = 1.0 / 24.0
	return [
		{
			"frame": frame,
			"hostMs": host * (frame1_multiplier if frame == 1 else 1.0),
			"substeps": 2,
			"activeParticles": 10_000 + frame,
			"pressureIterations": iterations,
			"pressureResidual": residual,
			"frameDt": frame_dt,
			"simTimeBefore": (frame - 1) * frame_dt,
			"simTimeAfter": frame * frame_dt,
			"timers": {
				"solver": solver * (frame1_multiplier if frame == 1 else 1.0),
				"compute_p2g": solver * 0.25,
				"compute_pressure": solver * 0.25,
			},
		}
		for frame in range(1, 49)
	]


def _campaign(*, baseline=100.0, candidate=100.0, solver_candidate=None,
			  slot_drift=0.0, frame1_multiplier=1.0, candidate_residual=1.0e-5,
			  candidate_iterations=40):
	if solver_candidate is None:
		solver_candidate = candidate
	runs = []
	settings = {
		"voxel_size": 0.03,
		"min_substeps": 1,
		"max_substeps": 16,
		"domain_min": [0.0, -1.0, 0.0],
		"domain_max": [5.6, 1.0, 1.6],
		"surface_enabled": True,
		"whitewater_enabled": True,
		"viscosity_enabled": False,
		"viscosity_constant": 1.0,
		"surface_tension_enabled": False,
		"surface_tension_coefficient": 0.072,
		"max_particles": 600_000,
		"max_whitewater": 200_000,
	}
	effective_settings = {
		"time_scale": 1.0,
		"frame_rate": 24,
		"voxel_size": 0.03,
		"min_substeps": 1,
		"max_substeps": 16,
		"pressure_tolerance": 1.0e-5,
		"device": "CUDA",
	}
	attribute_inventory = {
		"surface": [],
		"particles": [],
		"whitewater": ["type"],
	}
	assert set(settings) == set(PROTECTED_SETTINGS)
	for demo_index, demo in enumerate(DEMOS):
		for block, slot, label in abba_schedule():
			value = baseline if label == "baseline" else candidate
			solver = baseline if label == "baseline" else solver_candidate
			value += slot * slot_drift
			solver += slot * slot_drift
			runs.append({
				"label": label,
				"round": block,
				"slot": slot,
				"workload": demo,
				"records": _records(
					value, solver, frame1_multiplier=frame1_multiplier,
					residual=1.0e-5 if label == "baseline" else candidate_residual,
					iterations=40 if label == "baseline" else candidate_iterations),
				"rigidBodies": 0,
				"profilerSync": True,
				"solverTimerContract": SOLVER_TIMER_CONTRACT,
				"device": "cuda:0",
				"blendSha256": f"{demo_index + 1:064x}",
				"settingsSha256": f"{demo_index + 10:064x}",
				"protectedSettings": dict(settings),
				"effectiveSettings": dict(effective_settings),
				"attributeInventory": {
					key: list(value) for key, value in attribute_inventory.items()},
				"referenceVersion": REFERENCE_VERSIONS[demo],
				"version": (REFERENCE_VERSIONS[demo]
							if label == "baseline" else "2.1.0-dev"),
				"warp": "1.17.2",
				"root": f"C:/bench/{label}",
			})
	return runs


def test_abba_schedule_has_eight_trials_per_variant_and_reverses_order():
	schedule = abba_schedule()
	assert len(schedule) == 16
	assert [entry[2] for entry in schedule[:4]] == [
		"baseline", "candidate", "candidate", "baseline"]
	assert [entry[2] for entry in schedule[4:8]] == [
		"candidate", "baseline", "baseline", "candidate"]
	assert sum(label == "baseline" for _, _, label in schedule) == 8
	assert sum(label == "candidate" for _, _, label in schedule) == 8
	with pytest.raises(ValueError):
		abba_schedule(7)


def test_host_time_over_budget_is_a_regression():
	result = evaluate_budget(_campaign(candidate=106.0))
	assert result["passed"] is False
	assert result["status"] == "regression"
	assert result["demos"]["dam_break"]["hostRatio"] == pytest.approx(1.06)
	assert result["demos"]["dam_break"]["solverRatio"] == pytest.approx(1.06)


def test_linear_slot_drift_is_cancelled_by_each_abba_block():
	result = evaluate_budget(_campaign(slot_drift=10.0))
	assert result["passed"] is True
	assert result["status"] == "pass"
	for demo in DEMOS:
		assert result["demos"][demo]["hostRatio"] == pytest.approx(1.0)
		assert result["demos"][demo]["solverRatio"] == pytest.approx(1.0)


def test_host_time_decides_even_when_solver_time_is_flat():
	# The user waits on wall time; solver time is a diagnostic only.
	result = evaluate_budget(_campaign(candidate=108.0, solver_candidate=100.0))
	assert result["status"] == "regression"
	assert result["demos"]["paddle_stir"]["solverRatio"] == pytest.approx(1.0)
	result = evaluate_budget(_campaign(candidate=100.0, solver_candidate=120.0))
	assert result["status"] == "pass"


def _order_split(baseline_first, candidate_first):
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			factor = baseline_first if run["round"] % 2 == 0 else candidate_first
			for record in run["records"]:
				record["hostMs"] *= factor
				record["timers"]["solver"] *= factor
	return evaluate_budget(runs)


def test_order_balanced_host_ratio_decides_and_reports_the_spread():
	result = _order_split(1.16, 0.90)
	assert result["status"] == "pass", result["demos"]["dam_break"]["reasons"]
	demo = result["demos"]["dam_break"]
	assert demo["orders"]["baseline-first"]["hostRatio"] == pytest.approx(1.16)
	assert demo["hostRatio"] == pytest.approx(1.03)
	assert demo["hostOrderSpread"] == pytest.approx(0.26)
	assert _order_split(1.20, 0.96)["status"] == "regression"


def test_missing_trial_is_invalid():
	runs = _campaign()
	runs.pop()
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"


@pytest.mark.parametrize("mutate", [
	lambda run: run["records"][1].update(hostMs=0),
	lambda run: run["records"][1].update(timers={"solver": 10.0}),
])
def test_malformed_numeric_evidence_is_invalid(mutate):
	runs = _campaign()
	mutate(runs[0])
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"


def test_zero_disabled_stage_is_allowed_and_stage_ratios_are_reported():
	runs = _campaign(candidate=102.0)
	for run in runs:
		for record in run["records"]:
			record["timers"]["compute_disabled"] = 0.0
	result = evaluate_budget(runs)
	assert result["passed"] is True
	stages = result["demos"]["dam_break"]["stages"]
	assert stages["compute_p2g"]["ratio"] == pytest.approx(1.02)
	assert stages["compute_disabled"]["ratio"] is None
	assert result["demos"]["dam_break"]["orders"][
		"baseline-first"]["stages"]["compute_p2g"] == pytest.approx(1.02)


@pytest.mark.parametrize("value", [None, "legacy_aggregate_solver_timer"])
def test_solver_timer_contract_is_required_exactly(value):
	runs = _campaign()
	if value is None:
		runs[0].pop("solverTimerContract")
	else:
		runs[0]["solverTimerContract"] = value
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("solverTimerContract" in reason for reason in result["reasons"])


def test_every_demo_is_compared_with_the_previous_release():
	assert set(REFERENCE_VERSIONS.values()) == {"3.2.1"}


def test_wrong_reference_is_invalid():
	runs = _campaign()
	runs[0]["referenceVersion"] = "1.1.0"
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("referenceVersion" in reason for reason in result["reasons"])


def test_mismatched_workload_provenance_is_invalid():
	runs = _campaign()
	runs[16]["blendSha256"] = "f" * 64
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("blendSha256" in reason for reason in result["reasons"])


def test_trial_assigned_to_the_wrong_workload_is_invalid():
	runs = _campaign()
	run = runs[0]
	run["workload"] = "dam_break"
	run["referenceVersion"] = REFERENCE_VERSIONS["dam_break"]
	run["version"] = REFERENCE_VERSIONS["dam_break"]
	run["blendSha256"] = runs[16]["blendSha256"]
	run["settingsSha256"] = runs[16]["settingsSha256"]
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"


@pytest.mark.parametrize("mutation, expected_reason", [
	(lambda run: run.update(settingsSha256="f" * 64), "settingsSha256"),
	(lambda run: run["protectedSettings"].update(min_substeps=2),
	 "protectedSettings"),
])
def test_settings_provenance_must_match_every_trial(mutation, expected_reason):
	runs = _campaign()
	mutation(runs[1])
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any(expected_reason in reason for reason in result["reasons"])


def test_missing_protected_setting_or_wrong_baseline_version_is_invalid():
	runs = _campaign()
	runs[0]["protectedSettings"].pop("max_substeps")
	runs[3]["version"] = "9.9.9"
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("protectedSettings missing max_substeps" in reason
			   for reason in result["reasons"])
	assert any("baseline version" in reason for reason in result["reasons"])


def test_common_effective_physics_setting_must_match():
	runs = _campaign()
	runs[1]["effectiveSettings"]["pressure_tolerance"] = 1.0e-3
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("effectiveSettings.pressure_tolerance" in reason
			   for reason in result["reasons"])


def test_known_neutral_candidate_only_settings_are_reported_and_allowed():
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"].update(NEUTRAL_ABSENT_EFFECTIVE_SETTINGS)
	result = evaluate_budget(runs)
	assert result["passed"] is True
	assert result["status"] == "pass"
	diagnostic = result["demos"]["dam_break"]["effectiveSettings"]
	assert diagnostic["neutralAbsentFields"]["species_enabled"] == {
		"presentIn": "candidate", "neutralValue": False}
	assert "export_attributes" not in NEUTRAL_ABSENT_EFFECTIVE_SETTINGS


def test_unknown_or_non_neutral_candidate_only_setting_is_invalid():
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"]["new_enabled_feature"] = True
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("new_enabled_feature" in reason for reason in result["reasons"])

	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"]["species_enabled"] = True
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert any("species_enabled" in reason for reason in result["reasons"])


def test_export_attributes_absence_is_not_a_neutral_schema_difference():
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"]["export_attributes"] = False
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("export_attributes" in reason for reason in result["reasons"])


def test_attribute_inventory_must_match_every_trial():
	runs = _campaign()
	runs[1]["attributeInventory"]["particles"].append("velocity")
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("attributeInventory must match" in reason
			   for reason in result["reasons"])


def test_naturally_adaptive_candidate_substeps_are_diagnostic_only():
	runs = _campaign(candidate=80.0)
	for run in runs:
		if (run["workload"] == "dam_break" and run["round"] == 0
				and run["label"] == "candidate"):
			for record in run["records"][1:]:
				record["substeps"] = 1
	result = evaluate_budget(runs)
	assert result["passed"] is True
	assert result["status"] == "pass"
	block = result["demos"]["dam_break"]["blocks"][0]
	assert block["substeps"]["candidate"] < block["substeps"]["baseline"]
	assert block["substeps"]["candidateToBaselineRatio"] == pytest.approx(0.5)


def test_candidate_running_fewer_particles_is_invalid_even_when_nonempty():
	runs = _campaign(candidate=80.0)
	for run in runs:
		if (run["workload"] == "dam_break" and run["round"] == 0
				and run["label"] == "candidate"):
			run["records"][1]["activeParticles"] -= 1
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("candidate activeParticles is below baseline" in reason
			   for reason in result["reasons"])


def test_shortened_or_discontinuous_simulated_time_is_invalid():
	runs = _campaign()
	runs[0]["records"][1]["simTimeAfter"] -= 0.01
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("simulated-time advance" in reason for reason in result["reasons"])

	runs = _campaign()
	runs[0]["records"][2]["simTimeBefore"] += 0.01
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert any("simulated time is not sequential" in reason
			   for reason in result["reasons"])


def test_frame_duration_must_match_between_versions():
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			for index, record in enumerate(run["records"]):
				record["frameDt"] = 1.0 / 30.0
				record["simTimeBefore"] = index / 30.0
				record["simTimeAfter"] = (index + 1) / 30.0
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("frameDt differs" in reason for reason in result["reasons"])


def test_missing_frame_is_invalid():
	runs = _campaign()
	runs[0]["records"].pop(10)
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"
	assert any("frames 1 through 48" in reason for reason in result["reasons"])


@pytest.mark.parametrize("field, value", [
	("substeps", 0),
	("activeParticles", 0),
	("pressureIterations", -1),
	("pressureIterations", 4.0),
	("pressureResidual", float("nan")),
	("pressureResidual", None),
])
def test_warm_frames_require_real_workload_evidence(field, value):
	runs = _campaign()
	runs[0]["records"][1][field] = value
	result = evaluate_budget(runs)
	assert result["passed"] is False
	assert result["status"] == "invalid"


def test_frame_one_is_excluded_from_trial_medians():
	normal = evaluate_budget(_campaign())
	runs = _campaign(frame1_multiplier=1000.0)
	# Only candidate frame 1 receives an additional, extreme perturbation.
	for run in runs:
		if run["label"] == "baseline":
			run["records"][0]["hostMs"] /= 1000.0
			run["records"][0]["timers"]["solver"] /= 1000.0
		result = evaluate_budget(runs)
	assert result["passed"] is True
	assert result["status"] == "pass"
	assert result["demos"]["chromatic_vortex"]["hostRatio"] == normal["demos"]["chromatic_vortex"]["hostRatio"]


def _quality(unpadded, padded, leaves):
	return {"voxel_size": 0.01, "unpadded_voxel_dimensions": unpadded,
			"padded_voxel_dimensions": padded, "leaf_dimensions": leaves,
			"total_domain_leaf_cells": leaves[0] * leaves[1] * leaves[2],
			"liquid_sampling": 8.0}


def _with_quality(baseline, candidate):
	runs = _campaign()
	for run in runs:
		run["effectiveSettings"]["resolved_quality"] = dict(
			baseline if run["label"] == "baseline" else candidate)
	return runs


def test_3_0_schema_changes_are_neutral_while_their_switches_are_off():
	runs = _campaign()
	for run in runs:
		if run["label"] == "baseline":
			run["effectiveSettings"].update(
				spacetime_enabled=False, spacetime_seed=7, spacetime_variable_projection=True)
		else:
			run["effectiveSettings"].update(
				narrow_band_enabled=False, narrow_band_voxels=3.0,
				pressure_preconditioner="JACOBI")
	result = evaluate_budget(runs)
	assert result["status"] == "pass", result["reasons"]
	fields = result["demos"]["dam_break"]["effectiveSettings"]["neutralAbsentFields"]
	assert fields["narrow_band_voxels"] == {
		"presentIn": "candidate", "inertWhile": {"narrow_band_enabled": False}}
	assert fields["spacetime_seed"]["presentIn"] == "baseline"


@pytest.mark.parametrize("switch, value", [
	("narrow_band_enabled", True), ("pressure_preconditioner", "MGPCG")])
def test_an_enabled_3_0_feature_is_not_a_neutral_absence(switch, value):
	runs = _campaign()
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"].update(
				narrow_band_enabled=False, narrow_band_voxels=3.0,
				pressure_preconditioner="JACOBI")
			run["effectiveSettings"][switch] = value
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"
	assert any(switch in reason for reason in result["reasons"])


def test_resolution_divisions_lattice_snap_is_allowed_and_recorded():
	runs = _with_quality(_quality([81, 81, 51], [88, 88, 56], [11, 11, 7]),
						 _quality([80, 80, 50], [80, 80, 56], [10, 10, 7]))
	result = evaluate_budget(runs)
	assert result["status"] == "pass", result["reasons"]
	snap = result["demos"]["crown_splash"]["effectiveSettings"]["latticeSnap"]
	assert snap["baseline"]["padded_voxel_dimensions"] == [88, 88, 56]
	assert snap["candidatePaddedCellRatio"] == round(80 * 80 / (88 * 88), 4)


@pytest.mark.parametrize("candidate", [
	_quality([78, 80, 50], [80, 80, 56], [10, 10, 7]),  # more than one voxel
	{**_quality([80, 80, 50], [80, 80, 56], [10, 10, 7]), "voxel_size": 0.02},
])
def test_resolved_quality_beyond_a_lattice_snap_is_invalid(candidate):
	runs = _with_quality(_quality([81, 81, 51], [88, 88, 56], [11, 11, 7]), candidate)
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"
	assert any("resolved_quality" in reason for reason in result["reasons"])


def test_resolved_quality_must_match_across_trials_of_one_variant():
	runs = _with_quality(_quality([81, 81, 51], [88, 88, 56], [11, 11, 7]),
						 _quality([80, 80, 50], [80, 80, 56], [10, 10, 7]))
	runs[1]["effectiveSettings"]["resolved_quality"] = _quality(
		[81, 81, 50], [88, 88, 56], [11, 11, 7])
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"


def _multigrid_campaign(**kwargs):
	runs = _campaign(**kwargs)
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"]["pressure_preconditioner"] = "MULTIGRID"
	return runs


def test_4_0_multigrid_default_is_an_owned_change_with_the_plan_budget():
	result = evaluate_budget(_multigrid_campaign(candidate=108.0))
	assert result["status"] == "pass", result["reasons"]
	demo = result["demos"]["dam_break"]
	assert demo["budgetRatio"] == 1.10
	assert demo["effectiveSettings"]["intendedChanges"]["pressure_preconditioner"] == {
		"baseline": "JACOBI", "candidate": "MULTIGRID", "budgetRatio": 1.10}
	assert "pressure_preconditioner" not in demo["effectiveSettings"]["neutralAbsentFields"]


def test_multigrid_is_owned_when_both_versions_report_the_preconditioner():
	# 3.2.1 reports the field too: JACOBI -> MULTIGRID is the owned change.
	runs = _multigrid_campaign(candidate=108.0)
	for run in runs:
		if run["label"] == "baseline":
			run["effectiveSettings"]["pressure_preconditioner"] = "JACOBI"
	result = evaluate_budget(runs)
	assert result["status"] == "pass", result["reasons"]
	assert result["demos"]["dam_break"]["budgetRatio"] == 1.10
	for run in runs:
		run["effectiveSettings"]["pressure_preconditioner"] = (
			"MULTIGRID" if run["label"] == "baseline" else "JACOBI")
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"
	assert any("pressure_preconditioner" in reason for reason in result["reasons"])


def test_multigrid_allowance_does_not_widen_other_runs_or_exceed_ten_percent():
	assert evaluate_budget(_campaign(candidate=108.0))["status"] == "regression"
	result = evaluate_budget(_multigrid_campaign(candidate=112.0))
	assert result["status"] == "regression"
	assert any("1.100x budget" in reason
			   for reason in result["demos"]["dam_break"]["reasons"])


def _coupled(runs, candidate_bodies=1, baseline_bodies=0):
	for run in runs:
		if run["workload"] == "crown_splash":
			run["rigidBodies"] = candidate_bodies if run["label"] == "candidate" else baseline_bodies
	return runs


def test_rigid_coupling_is_an_owned_change_that_multiplies_with_multigrid():
	result = evaluate_budget(_coupled(_campaign(candidate=104.0)))
	crown = result["demos"]["crown_splash"]
	assert crown["budgetRatio"] == pytest.approx(1.05)
	assert crown["effectiveSettings"]["intendedChanges"]["rigidBodies"] == {
		"baseline": 0, "candidate": 1, "budgetRatio": 1.05}
	result = evaluate_budget(_coupled(_multigrid_campaign(candidate=114.0)))
	assert result["demos"]["crown_splash"]["status"] == "pass"
	assert result["demos"]["dam_break"]["status"] == "regression"
	assert result["demos"]["crown_splash"]["budgetRatio"] == pytest.approx(1.155)
	assert result["demos"]["dam_break"]["budgetRatio"] == pytest.approx(1.10)
	result = evaluate_budget(_coupled(_multigrid_campaign(candidate=116.0)))
	assert result["demos"]["crown_splash"]["status"] == "regression"


@pytest.mark.parametrize("mutate, expected", [
	(lambda runs: _coupled(runs, candidate_bodies=0, baseline_bodies=1), "rigidBodies differ"),
	(lambda runs: runs[0].update(rigidBodies=2), "rigidBodies must be constant"),
	(lambda runs: runs[0].pop("rigidBodies"), "rigidBodies must be a non-negative integer"),
])
def test_rigid_body_evidence_must_be_consistent(mutate, expected):
	runs = _campaign()
	mutate(runs)
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"
	assert any(expected in reason for reason in result["reasons"])


def test_a_faster_but_less_accurate_candidate_fails():
	result = evaluate_budget(_campaign(candidate=90.0, candidate_residual=1.2e-5))
	assert result["passed"] is False
	assert result["status"] == "less_accurate"
	pressure = result["demos"]["dam_break"]["pressure"]
	assert pressure["residual"]["ratio"] == pytest.approx(1.2)
	assert evaluate_budget(_campaign(candidate_residual=1.05e-5))["status"] == "pass"


def test_regression_outranks_less_accurate():
	result = evaluate_budget(_campaign(candidate=120.0, candidate_residual=1.0e-3))
	assert result["status"] == "regression"


def test_zero_baseline_residual_requires_a_zero_candidate_residual():
	runs = _campaign(candidate_residual=0.0)
	for run in runs:
		for record in run["records"]:
			record["pressureResidual"] = 0.0
	assert evaluate_budget(runs)["status"] == "pass"
	for run in runs:
		if run["label"] == "candidate":
			for record in run["records"]:
				record["pressureResidual"] = 1.0e-9
	result = evaluate_budget(runs)
	assert result["status"] == "less_accurate"
	assert result["demos"]["dam_break"]["pressure"]["residual"]["ratio"] is None


def test_pressure_iteration_reduction_is_reported():
	result = evaluate_budget(_campaign(candidate_iterations=10))
	assert result["status"] == "pass"
	iterations = result["demos"]["dam_break"]["pressure"]["iterations"]
	assert iterations == {"baseline": 40, "candidate": 10, "reduction": 4.0}


def test_multigrid_only_in_the_baseline_is_still_an_omission():
	runs = _campaign()
	for run in runs:
		if run["label"] == "baseline":
			run["effectiveSettings"]["pressure_preconditioner"] = "MULTIGRID"
	result = evaluate_budget(runs)
	assert result["status"] == "invalid"
	assert any("pressure_preconditioner" in reason for reason in result["reasons"])


def _learned_guess_campaign(value):
	runs = _multigrid_campaign(candidate=100.0)
	for run in runs:
		if run["label"] == "candidate":
			run["effectiveSettings"]["pressure_learned_guess"] = value
	return runs


def test_4_0_learned_guess_off_is_a_neutral_absence():
	result = evaluate_budget(_learned_guess_campaign(False))
	assert result["status"] == "pass", result["reasons"]
	assert result["demos"]["dam_break"]["effectiveSettings"]["neutralAbsentFields"][
		"pressure_learned_guess"] == {"presentIn": "candidate", "neutralValue": False}


def test_learned_guess_on_is_still_an_omission():
	result = evaluate_budget(_learned_guess_campaign(True))
	assert result["status"] == "invalid"
	assert any("pressure_learned_guess" in reason for reason in result["reasons"])
