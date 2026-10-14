"""Pure validation and evaluation for the production performance budget.

The runner is deliberately kept out of this module.  It supplies fresh-process
raw reports in :func:`abba_schedule` order; this module rejects incomplete or
unverifiable reports and reduces valid evidence to one decision per demo.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict

DEMOS = ("crown_splash", "dam_break", "paddle_stir", "chromatic_vortex")
REFERENCE_VERSIONS = {
	"crown_splash": "1.0.255",
	"dam_break": "1.0.255",
	"paddle_stir": "1.0.255",
	"chromatic_vortex": "1.1.0",
}
BUDGET_RATIO = 1.05
MIN_TRIALS_PER_VARIANT = 8
EXPECTED_FRAMES = tuple(range(1, 49))
SOLVER_TIMER_CONTRACT = "sum_of_outermost_synchronized_phase_spans"
_LABELS = ("baseline", "candidate")
_ORDERS = ("baseline-first", "candidate-first")
PROTECTED_SETTINGS = (
	"voxel_size",
	"min_substeps",
	"max_substeps",
	"domain_min",
	"domain_max",
	"surface_enabled",
	"whitewater_enabled",
	"viscosity_enabled",
	"viscosity_constant",
	"surface_tension_enabled",
	"surface_tension_coefficient",
	"max_particles",
	"max_whitewater",
)
NEUTRAL_ABSENT_EFFECTIVE_SETTINGS = {
	"species_enabled": False,
	"species_mixing_rate": 8.0,
	# 3.0: narrow band (off) and the explicit preconditioner (1.x pressure PCG
	# was inverse-diagonal, i.e. Jacobi) are new; ST-FLIP's switch is removed.
	"narrow_band_enabled": False,
	"pressure_preconditioner": "JACOBI",
	"spacetime_enabled": False,
}
# Settings that only one variant reports and that do nothing while their
# switch holds the given value: field -> (switch, inert value). The switch
# must itself be a neutral absence for the same variant.
INERT_WHEN_SWITCHED_OFF = {
	"narrow_band_voxels": ("narrow_band_enabled", False),
	**{f"spacetime_{name}": ("spacetime_enabled", False) for name in (
		"adaptive_jitter", "gas_density", "jitter_strength", "liquid_density",
		"local_cfl", "max_local_substeps", "min_density", "phase_eta",
		"phase_threshold", "reference_mass", "resync_surface", "seed",
		"variable_projection")},
}
# resolved_quality keys that 3.0's Resolution Divisions lattice may snap
# (owner decision 2026-10-09: compare per frame, record the difference).
LATTICE_SNAP_KEYS = (
	"unpadded_voxel_dimensions", "padded_voxel_dimensions", "leaf_dimensions",
	"total_domain_leaf_cells")
ATTRIBUTE_OUTPUT_KINDS = ("surface", "particles", "whitewater")


def _lattice_snap(demo_runs):
	"""Return ``(snap diagnostic or None, invalid reason or None)``.

	Within a variant resolved_quality must be identical. Across variants only
	the grid dimensions may differ, by at most one unpadded voxel per axis
	(the Resolution Divisions lattice snap); everything else must match.
	"""
	per_label = {}
	for run in demo_runs:
		per_label.setdefault(run["label"], []).append(run["effectiveSettings"]["resolved_quality"])
	for values in per_label.values():
		if any(value != values[0] for value in values[1:]):
			return None, "differs across trials"
	firsts = {label: values[0] for label, values in per_label.items()}
	if len({json.dumps(value, sort_keys=True) for value in firsts.values()}) == 1:
		return None, None
	if not all(isinstance(value, dict) for value in firsts.values()):
		return None, "differs across variants or trials"
	stripped = [{key: item for key, item in value.items() if key not in LATTICE_SNAP_KEYS}
				for value in firsts.values()]
	if any(value != stripped[0] for value in stripped[1:]):
		return None, "differs across variants beyond the grid dimensions"
	try:
		dims = [list(value["unpadded_voxel_dimensions"]) for value in firsts.values()]
		if any(len(dim) != 3 for dim in dims) or any(
				abs(int(a) - int(b)) > 1 for a, b in zip(dims[0], dims[1])):
			return None, "grid differs by more than a one-voxel lattice snap"
	except (KeyError, TypeError, ValueError):
		return None, "differs across variants or trials"
	snap = {label: {key: value.get(key) for key in LATTICE_SNAP_KEYS}
			for label, value in firsts.items()}
	baseline_cells = math.prod(firsts["baseline"]["padded_voxel_dimensions"])
	candidate_cells = math.prod(firsts["candidate"]["padded_voxel_dimensions"])
	snap["candidatePaddedCellRatio"] = round(candidate_cells / baseline_cells, 4)
	return snap, None


def abba_schedule(trials=MIN_TRIALS_PER_VARIANT):
	"""Return ``(block, slot, label)`` entries for balanced four-slot blocks.

	``trials`` counts trials *per variant*.  Every block contributes two trials
	to each variant, and adjacent blocks reverse which variant runs first.
	"""
	if type(trials) is not int or trials < MIN_TRIALS_PER_VARIANT or trials % 2:
		raise ValueError("trials must be an even integer of at least 8 per variant")
	schedule = []
	for block in range(trials // 2):
		labels = (
			("baseline", "candidate", "candidate", "baseline")
			if block % 2 == 0 else
			("candidate", "baseline", "baseline", "candidate")
		)
		schedule.extend((block, slot, label) for slot, label in enumerate(labels))
	return schedule


def _positive_number(value):
	return (
		type(value) in (int, float)
		and math.isfinite(value)
		and value > 0.0
	)


def _finite_number(value):
	return type(value) in (int, float) and math.isfinite(value)


def _nonnegative_number(value):
	return (
		type(value) in (int, float)
		and math.isfinite(value)
		and value >= 0.0
	)


def _sha256(value):
	return (
		isinstance(value, str)
		and len(value) == 64
		and all(character in "0123456789abcdefABCDEF" for character in value)
	)


def _validate_run(run, index):
	errors = []
	name = f"run[{index}]"
	if not isinstance(run, dict):
		return [f"{name}: report must be an object"]

	workload = run.get("workload")
	if workload not in REFERENCE_VERSIONS:
		errors.append(f"{name}: unknown workload {workload!r}")
	label = run.get("label")
	if label not in _LABELS:
		errors.append(f"{name}: label must be baseline or candidate")
	block = run.get("round")
	if type(block) is not int or block < 0:
		errors.append(f"{name}: round must be a non-negative integer block")
	slot = run.get("slot")
	if type(slot) is not int or slot not in range(4):
		errors.append(f"{name}: slot must be an integer from 0 through 3")

	if workload in REFERENCE_VERSIONS:
		expected = REFERENCE_VERSIONS[workload]
		if run.get("referenceVersion") != expected:
			errors.append(
				f"{name}: {workload} referenceVersion must be {expected}")
		version = run.get("version")
		if not isinstance(version, str) or not version.strip():
			errors.append(f"{name}: actual version provenance is missing")
		elif label == "baseline" and version != expected:
			errors.append(
				f"{name}: baseline version must be the {expected} reference")
	if run.get("profilerSync") is not True:
		errors.append(f"{name}: profilerSync must be true")
	if run.get("solverTimerContract") != SOLVER_TIMER_CONTRACT:
		errors.append(
			f"{name}: solverTimerContract must be {SOLVER_TIMER_CONTRACT}")
	device = run.get("device")
	if not isinstance(device, str) or not device.lower().startswith("cuda"):
		errors.append(f"{name}: device must identify CUDA")
	for field in ("blendSha256", "settingsSha256"):
		if not _sha256(run.get(field)):
			errors.append(f"{name}: {field} provenance is missing or invalid")
	for field in ("warp", "root"):
		if not isinstance(run.get(field), str) or not run[field].strip():
			errors.append(f"{name}: {field} provenance is missing")
	settings = run.get("protectedSettings")
	if not isinstance(settings, dict):
		errors.append(f"{name}: protectedSettings must be an object")
	else:
		missing_settings = [field for field in PROTECTED_SETTINGS
							if field not in settings]
		if missing_settings:
			errors.append(
				f"{name}: protectedSettings missing {', '.join(missing_settings)}")
	effective = run.get("effectiveSettings")
	if (not isinstance(effective, dict) or not effective
			or not all(isinstance(key, str) and key for key in effective)):
		errors.append(f"{name}: effectiveSettings must be a non-empty object")
	attribute_inventory = run.get("attributeInventory")
	if not isinstance(attribute_inventory, dict):
		errors.append(f"{name}: attributeInventory must be an object")
	else:
		if set(attribute_inventory) != set(ATTRIBUTE_OUTPUT_KINDS):
			errors.append(
				f"{name}: attributeInventory must contain exactly "
				f"{', '.join(ATTRIBUTE_OUTPUT_KINDS)}")
		for output_kind in ATTRIBUTE_OUTPUT_KINDS:
			attributes = attribute_inventory.get(output_kind)
			if (not isinstance(attributes, list)
					or not all(isinstance(value, str) and value for value in attributes)):
				errors.append(
					f"{name}: attributeInventory.{output_kind} must be a list "
					"of non-empty attribute names")

	records = run.get("records")
	if not isinstance(records, list):
		errors.append(f"{name}: records must be a list")
		return errors
	frames = [record.get("frame") for record in records
			  if isinstance(record, dict)]
	if len(records) != len(EXPECTED_FRAMES) or tuple(frames) != EXPECTED_FRAMES:
		errors.append(f"{name}: records must contain frames 1 through 48 exactly once")
	previous_time_after = None
	stage_inventory = None
	authored_dt = None
	if isinstance(effective, dict):
		if _positive_number(effective.get("time_scale")) and _positive_number(effective.get("frame_rate")):
			authored_dt = effective["time_scale"] / effective["frame_rate"]
		else:
			errors.append(f"{name}: effectiveSettings requires positive time_scale and frame_rate")
	for record_index, record in enumerate(records):
		frame_name = f"{name}.records[{record_index}]"
		if not isinstance(record, dict):
			errors.append(f"{frame_name}: record must be an object")
			continue
		if not _positive_number(record.get("hostMs")):
			errors.append(f"{frame_name}: hostMs must be positive and finite")
		frame_dt = record.get("frameDt")
		time_before = record.get("simTimeBefore")
		time_after = record.get("simTimeAfter")
		if not _positive_number(frame_dt):
			errors.append(f"{frame_name}: frameDt must be positive and finite")
		elif authored_dt is not None and not math.isclose(frame_dt, authored_dt, rel_tol=1.0e-8, abs_tol=1.0e-8):
			errors.append(f"{frame_name}: frameDt differs from effectiveSettings authored duration")
		if not _finite_number(time_before) or not _finite_number(time_after):
			errors.append(
				f"{frame_name}: simTimeBefore and simTimeAfter must be finite")
		elif _positive_number(frame_dt) and not math.isclose(
				time_after - time_before, frame_dt,
				rel_tol=1.0e-8, abs_tol=1.0e-8):
			errors.append(
				f"{frame_name}: simulated-time advance does not match frameDt")
		if (previous_time_after is not None and _finite_number(time_before)
				and not math.isclose(time_before, previous_time_after,
							 rel_tol=1.0e-8, abs_tol=1.0e-8)):
			errors.append(f"{frame_name}: simulated time is not sequential")
		if _finite_number(time_after):
			previous_time_after = time_after
		if record.get("frame") in EXPECTED_FRAMES[1:]:
			if type(record.get("substeps")) is not int or record["substeps"] <= 0:
				errors.append(f"{frame_name}: substeps must be a positive integer")
			if not _positive_number(record.get("activeParticles")):
				errors.append(f"{frame_name}: activeParticles must be positive")
		timers = record.get("timers")
		if not isinstance(timers, dict):
			errors.append(f"{frame_name}: timers must be an object")
			continue
		if not _positive_number(timers.get("solver")):
			errors.append(f"{frame_name}: timers.solver must be positive and finite")
		stage_names = {key for key in timers if isinstance(key, str) and key.startswith("compute_")}
		if stage_inventory is None:
			stage_inventory = stage_names
		elif stage_names != stage_inventory:
			errors.append(f"{frame_name}: compute stage inventory changed within trial")
		if not any(_positive_number(timers.get(key)) for key in ("compute_2", "compute_p2g")):
			errors.append(f"{frame_name}: mandatory P2G stage must be positive")
		if not any(_positive_number(timers.get(key)) for key in ("compute_5", "compute_pressure")):
			errors.append(f"{frame_name}: mandatory pressure stage must be positive")
		compute = [value for key, value in timers.items()
				   if isinstance(key, str) and key.startswith("compute_")]
		if not compute:
			errors.append(f"{frame_name}: at least one compute_* stage is required")
		elif not all(_nonnegative_number(value) for value in compute):
			errors.append(f"{frame_name}: every compute_* stage must be finite and non-negative")
		elif not any(value > 0.0 for value in compute):
			errors.append(f"{frame_name}: at least one compute_* stage must be positive")
	return errors


def _trial_medians(run):
	warm = run["records"][1:]  # Frame 1 is initialization/warm-up by contract.
	stage_names = set.intersection(*(
		{key for key in record["timers"] if key.startswith("compute_")}
		for record in warm
	))
	return {
		"host": statistics.median(record["hostMs"] for record in warm),
		"solver": statistics.median(record["timers"]["solver"] for record in warm),
		"stages": {
			stage: statistics.median(record["timers"][stage] for record in warm)
			for stage in sorted(stage_names)
		},
	}


def _order_for_block(block_runs):
	return "baseline-first" if block_runs[0]["label"] == "baseline" else "candidate-first"


def _invalid(reasons):
	return {"passed": False, "status": "invalid", "demos": {}, "reasons": reasons}


def evaluate_budget(runs):
	"""Validate raw reports and enforce the per-demo 5 percent budget.

	A confirmed regression requires both host time and synchronized solver time
	to exceed the limit with the two ABBA orders agreeing.  A host/solver
	disagreement or order split is retained as inconclusive evidence and blocks
	the production gate.  Invalid or incomplete evidence always fails.
	"""
	if not isinstance(runs, list):
		return _invalid(["runs must be a list of raw reports"])
	errors = []
	for index, run in enumerate(runs):
		errors.extend(_validate_run(run, index))
	if errors:
		return _invalid(errors)

	by_demo = defaultdict(list)
	for run in runs:
		by_demo[run["workload"]].append(run)
	missing = [demo for demo in DEMOS if demo not in by_demo]
	if missing:
		return _invalid([f"missing demo evidence: {', '.join(missing)}"])

	demo_results = {}
	effective_settings_diagnostics = {}
	invalid_reasons = []
	for demo in DEMOS:
		demo_runs = by_demo[demo]
		for field in ("blendSha256", "settingsSha256", "warp", "device"):
			values = {run[field] for run in demo_runs}
			if len(values) != 1:
				invalid_reasons.append(f"{demo}: {field} must match across both variants")
		first_settings = demo_runs[0]["protectedSettings"]
		if any(run["protectedSettings"] != first_settings for run in demo_runs[1:]):
			invalid_reasons.append(
				f"{demo}: protectedSettings must match across every trial")
		first_inventory = demo_runs[0]["attributeInventory"]
		if any(run["attributeInventory"] != first_inventory
				for run in demo_runs[1:]):
			invalid_reasons.append(
				f"{demo}: attributeInventory must match across every trial")
		for record_index, frame in enumerate(EXPECTED_FRAMES):
			frame_dts = [run["records"][record_index]["frameDt"]
						 for run in demo_runs]
			if any(not math.isclose(value, frame_dts[0],
					rel_tol=1.0e-8, abs_tol=1.0e-8) for value in frame_dts[1:]):
				invalid_reasons.append(
					f"{demo}: frame {frame} frameDt differs across variants or trials")
		effective_keys = {}
		for label in _LABELS:
			label_runs = [run for run in demo_runs if run["label"] == label]
			if not label_runs:
				effective_keys[label] = set()
				continue
			stage_sets = {frozenset(key for key in run["records"][0]["timers"] if key.startswith("compute_")) for run in label_runs}
			if len(stage_sets) != 1:
				invalid_reasons.append(f"{demo}: {label} compute stage inventory changes between trials")
			key_sets = {frozenset(run["effectiveSettings"]) for run in label_runs}
			if len(key_sets) != 1:
				invalid_reasons.append(
					f"{demo}: {label} effectiveSettings schema changes between trials")
			effective_keys[label] = set.intersection(
				*(set(run["effectiveSettings"]) for run in label_runs))
		common_effective = set.intersection(*effective_keys.values())
		all_effective = set.union(*effective_keys.values())
		neutral_absences = {}
		lattice_snap = None
		if not common_effective:
			invalid_reasons.append(
				f"{demo}: variants have no common effectiveSettings fields")
		else:
			for field in sorted(common_effective):
				values = [run["effectiveSettings"][field] for run in demo_runs]
				if field == "resolved_quality":
					snap, reason = _lattice_snap(demo_runs)
					if reason:
						invalid_reasons.append(f"{demo}: effectiveSettings.resolved_quality {reason}")
					elif snap:
						lattice_snap = snap
				elif any(value != values[0] for value in values[1:]):
					invalid_reasons.append(
						f"{demo}: effectiveSettings.{field} differs across variants or trials")
		for field in sorted(all_effective - common_effective):
			present_labels = [label for label in _LABELS
							  if field in effective_keys[label]]
			neutral = NEUTRAL_ABSENT_EFFECTIVE_SETTINGS.get(field, object())
			present_values = [
				run["effectiveSettings"][field]
				for run in demo_runs if field in run["effectiveSettings"]
			]
			switch, inert = INERT_WHEN_SWITCHED_OFF.get(field, (None, object()))
			if (len(present_labels) == 1
					and field in NEUTRAL_ABSENT_EFFECTIVE_SETTINGS
					and all(value == neutral for value in present_values)):
				neutral_absences[field] = {
					"presentIn": present_labels[0],
					"neutralValue": neutral,
				}
			elif (len(present_labels) == 1
					and NEUTRAL_ABSENT_EFFECTIVE_SETTINGS.get(switch, object()) == inert
					and switch not in common_effective
					and all(run["effectiveSettings"].get(switch, object()) == inert
							for run in demo_runs if field in run["effectiveSettings"])):
				neutral_absences[field] = {
					"presentIn": present_labels[0],
					"inertWhile": {switch: inert},
				}
			else:
				invalid_reasons.append(
					f"{demo}: effectiveSettings.{field} is an unsupported cross-version omission")
		effective_settings_diagnostics[demo] = {
			"commonFields": sorted(common_effective),
			"neutralAbsentFields": neutral_absences,
		}
		if lattice_snap:
			effective_settings_diagnostics[demo]["latticeSnap"] = lattice_snap
		roots = {label: {run["root"] for run in demo_runs if run["label"] == label}
				 for label in _LABELS}
		if any(len(values) != 1 for values in roots.values()):
			invalid_reasons.append(f"{demo}: each variant must have one stable root")
		elif roots["baseline"] == roots["candidate"]:
			invalid_reasons.append(f"{demo}: baseline and candidate roots must differ")

		counts = {label: sum(run["label"] == label for run in demo_runs)
				  for label in _LABELS}
		if any(count < MIN_TRIALS_PER_VARIANT for count in counts.values()):
			invalid_reasons.append(
				f"{demo}: requires at least 8 trials per baseline and candidate")

		blocks = defaultdict(list)
		for run in demo_runs:
			blocks[run["round"]].append(run)
		block_rows = []
		trial_medians = [_trial_medians(run) for run in demo_runs]
		common_stages = set.intersection(*(
			set(values["stages"]) for values in trial_medians
		))
		for block, block_runs in sorted(blocks.items()):
			block_runs.sort(key=lambda run: run["slot"])
			labels = tuple(run["label"] for run in block_runs)
			valid_orders = (
				("baseline", "candidate", "candidate", "baseline"),
				("candidate", "baseline", "baseline", "candidate"),
			)
			if (len(block_runs) != 4
					or tuple(run["slot"] for run in block_runs) != (0, 1, 2, 3)
					or labels not in valid_orders):
				invalid_reasons.append(
					f"{demo}: round {block} is not one complete ABBA/BAAB block")
				continue
			trial_values = [(run, _trial_medians(run)) for run in block_runs]
			row = {"block": block, "order": _order_for_block(block_runs)}
			baseline_substeps = sum(
				record["substeps"] for run in block_runs if run["label"] == "baseline"
				for record in run["records"][1:])
			candidate_substeps = sum(
				record["substeps"] for run in block_runs if run["label"] == "candidate"
				for record in run["records"][1:])
			# Substeps are adaptive solver output and can legitimately differ across
			# implementations.  Record them for diagnosis; the workload invariants
			# are the authored min/max limits and equal physical-time advancement.
			row["substeps"] = {
				"baseline": baseline_substeps,
				"candidate": candidate_substeps,
				"candidateToBaselineRatio": candidate_substeps / baseline_substeps,
			}
			particle_ratios = []
			for record_index, frame in enumerate(EXPECTED_FRAMES[1:], start=1):
				baseline_particles = statistics.median(
					run["records"][record_index]["activeParticles"]
					for run in block_runs if run["label"] == "baseline")
				candidate_particles = statistics.median(
					run["records"][record_index]["activeParticles"]
					for run in block_runs if run["label"] == "candidate")
				particle_ratios.append(candidate_particles / baseline_particles)
				if candidate_particles < baseline_particles:
					invalid_reasons.append(
						f"{demo}: round {block} frame {frame} candidate activeParticles "
						"is below baseline")
			row["activeParticlesMinimumRatio"] = min(particle_ratios)
			for metric in ("host", "solver"):
				baseline = [value[metric] for run, value in trial_values
							if run["label"] == "baseline"]
				candidate = [value[metric] for run, value in trial_values
							 if run["label"] == "candidate"]
				row[f"{metric}Ratio"] = (
					statistics.median(candidate) / statistics.median(baseline))
			row["stages"] = {}
			for stage in sorted(common_stages):
				baseline = statistics.median(
					value["stages"][stage] for run, value in trial_values
					if run["label"] == "baseline")
				candidate = statistics.median(
					value["stages"][stage] for run, value in trial_values
					if run["label"] == "candidate")
				row["stages"][stage] = {
					"baselineMedian": baseline,
					"candidateMedian": candidate,
					"ratio": candidate / baseline if baseline > 0.0 else None,
				}
			block_rows.append(row)
		if not common_stages or not any(
			stage["ratio"] is not None
			for row in block_rows for stage in row["stages"].values()
		):
			invalid_reasons.append(
				f"{demo}: no common comparable compute_* stage across variants")

		orders = {order: [row for row in block_rows if row["order"] == order]
				  for order in _ORDERS}
		if any(len(rows) < 2 for rows in orders.values()):
			invalid_reasons.append(
				f"{demo}: requires at least two complete blocks in each order")
			continue
		if invalid_reasons:
			continue

		host_ratio = statistics.median(row["hostRatio"] for row in block_rows)
		solver_ratio = statistics.median(row["solverRatio"] for row in block_rows)
		order_diagnostics = {}
		for order, rows in orders.items():
			order_diagnostics[order] = {
				"blocks": len(rows),
				"hostRatio": statistics.median(row["hostRatio"] for row in rows),
				"solverRatio": statistics.median(row["solverRatio"] for row in rows),
				"stages": {
					stage: statistics.median(ratios) if ratios else None
					for stage in sorted(common_stages)
					for ratios in [[
						row["stages"][stage]["ratio"] for row in rows
						if row["stages"][stage]["ratio"] is not None
					]]
				},
			}
		order_split = any(
			(order_diagnostics[_ORDERS[0]][f"{metric}Ratio"] <= BUDGET_RATIO)
			!= (order_diagnostics[_ORDERS[1]][f"{metric}Ratio"] <= BUDGET_RATIO)
			for metric in ("host", "solver")
		)
		if order_split:
			status = "inconclusive"
			reasons = ["ABBA orders disagree on the budget direction"]
		elif host_ratio <= BUDGET_RATIO and solver_ratio <= BUDGET_RATIO:
			status = "pass"
			reasons = []
		elif host_ratio > BUDGET_RATIO and solver_ratio > BUDGET_RATIO:
			status = "regression"
			reasons = ["host and synchronized solver medians exceed the 5% budget"]
		else:
			status = "inconclusive"
			reasons = ["host and synchronized solver evidence disagree"]
		stage_diagnostics = {}
		for stage in sorted(common_stages):
			ratios = [row["stages"][stage]["ratio"] for row in block_rows
					  if row["stages"][stage]["ratio"] is not None]
			stage_diagnostics[stage] = {
				"ratio": statistics.median(ratios) if ratios else None,
				"orders": {
					order: order_diagnostics[order]["stages"][stage]
					for order in _ORDERS
				},
			}
		demo_results[demo] = {
			"passed": status == "pass",
			"status": status,
			"referenceVersion": REFERENCE_VERSIONS[demo],
			"trials": counts,
			"hostRatio": host_ratio,
			"solverRatio": solver_ratio,
			"orders": order_diagnostics,
			"stages": stage_diagnostics,
			"effectiveSettings": effective_settings_diagnostics[demo],
			"attributeInventory": first_inventory,
			"blocks": block_rows,
			"reasons": reasons,
		}

	if invalid_reasons:
		return _invalid(invalid_reasons)
	statuses = {result["status"] for result in demo_results.values()}
	if "regression" in statuses:
		status = "regression"
	elif "inconclusive" in statuses:
		status = "inconclusive"
	else:
		status = "pass"
	return {
		"passed": status == "pass",
		"status": status,
		"demos": demo_results,
		"reasons": [],
	}
