"""Pure FSPT P0 trial evaluation; it cannot certify runtime or artifact authenticity.

Use beside the existing production gate, after its source/runtime admission.
The frozen JSON is an engineering contract, never a passing performance receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "tests/render/fixtures/fspt_acceptance_contract_v1.json"


def load_contract(path=None):
    with Path(path or CONTRACT_PATH).open(encoding="utf-8") as stream:
        return json.load(stream)


def contract_sha256(contract):
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def _hash(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _quality(report, profile, contract, reasons):
    quality = report["quality"]
    policy = contract["quality"]
    if quality["confidence_level"] != policy["confidence_level"]:
        reasons.append("quality confidence level differs from the frozen 99% contract")
    if (type(quality["stochastic_analytic_rays"]) is not int
            or quality["stochastic_analytic_rays"] < policy["stochastic_analytic_rays_min"]):
        reasons.append("analytic rendering has insufficient sampled rays")
    if not _hash(quality["reference_sha256"]):
        reasons.append("quality reference has no immutable SHA256")
    if any(not isinstance(quality[field], str)
           for field in ("reference_frozen_at", "candidate_started_at")):
        raise ValueError("reference and candidate timestamps must be ISO-8601 strings")
    reference_time = datetime.fromisoformat(quality["reference_frozen_at"].replace("Z", "+00:00"))
    candidate_time = datetime.fromisoformat(quality["candidate_started_at"].replace("Z", "+00:00"))
    if reference_time.tzinfo is None or candidate_time.tzinfo is None or reference_time >= candidate_time:
        reasons.append("reference and masks must be frozen before the candidate run")
    for name in policy["mask_roles"]:
        count = quality["mask_pixels"][name]
        if (not _hash(quality["mask_sha256"][name])
                or type(count) is not int or count <= 0):
            reasons.append(f"quality mask {name} is missing, mutable or empty")
    final = contract["profiles"]["final_native1080p_spp1024"]
    fraction = policy["reference_convergence_fraction_of_final_gate"]
    for name, limit in (("ordinary", final["ordinary_nmae_max"]), ("caustic", final["caustic_nmae_max"])):
        value = quality[f"reference_convergence_{name}_nmae"]
        if not _number(value) or value > fraction * limit:
            reasons.append(f"independent {name} reference has not converged")
    if type(quality["nonfinite_radiance_count"]) is not int or quality["nonfinite_radiance_count"] != 0:
        reasons.append("quality evidence contains nonfinite radiance")

    maximums = dict(policy["analytic_limits"])
    maximums.update(ordinary_luminance_nmae=profile["ordinary_nmae_max"],
                    caustic_luminance_nmae=profile["caustic_nmae_max"],
                    ghost_residual_after_two_new_frames=policy["image_limits"]["ghost_residual_max"])
    minimums = {"thin_feature_precision": policy["image_limits"]["thin_feature_precision_min"],
                "thin_feature_recall": policy["image_limits"]["thin_feature_recall_min"]}
    for name, limit in (maximums | minimums).items():
        measured = quality["metrics"][name]
        value, uncertainty = measured["value"], measured["ci99_half_width"]
        if not _number(value) or not _number(uncertainty):
            reasons.append(f"quality metric {name} is missing or nonfinite")
            continue
        # For coverage rates, uncertainty is bounded against their permissible
        # error (1 - required coverage), not against the coverage value itself.
        tolerance = 1 - limit if name in minimums else limit
        if uncertainty > tolerance * policy["ci_half_width_max_fraction_of_tolerance"]:
            reasons.append(f"quality metric {name} has excessive uncertainty")
        if name in minimums:
            if value > 1 or value - uncertainty < limit:
                reasons.append(f"quality metric {name} fails its 99% lower bound")
        elif value + uncertainty > limit:
            reasons.append(f"quality metric {name} fails its 99% upper bound")


def _read_lanes(report, contract, fixture, reasons):
    lanes = report["lanes"]
    required = contract["protocol"]["required_lanes"]
    if any(name not in lanes for name in required):
        raise ValueError("paired A/B/C/D lane is missing")
    for key in (*contract["protocol"]["matched_inputs"], "environment_fingerprint"):
        values = [lanes[name][key] for name in required]
        if any(not _hash(value) for value in values) or len(set(values)) != 1:
            reasons.append(f"paired lanes differ in {key}")
    demands = [lanes[name]["demand_schedule_sha256"] for name in contract["protocol"]["matched_demand_lanes"]]
    if any(not _hash(value) for value in demands) or len(set(demands)) != 1:
        reasons.append("B/C/D do not use the same recorded surface capture demand schedule")

    result = {}
    for name in required:
        lane = lanes[name]
        rows = lane["states"]
        warmup = lane["warmup_states"]
        if type(warmup) is not int or warmup < contract["protocol"]["warmup_states_per_lane"]:
            raise ValueError(f"{name}: incomplete warmup states")
        if not isinstance(rows, list) or len(rows) - warmup < contract["protocol"]["measured_states_per_trial_min"]:
            raise ValueError(f"{name}: incomplete measured solved-state sequence")
        previous = None
        for row in rows:
            if any(type(row[field]) is not int or row[field] < 1
                   for field in ("solver_frame", "state_revision", "epoch")):
                raise ValueError(f"{name}: invalid actual solved-state identity")
            if not _number(row["completed_at"]) or not _number(row["simulation_time"]):
                raise ValueError(f"{name}: nonfinite or negative completed-state clock")
            if previous is not None:
                if (row["epoch"] != previous["epoch"]
                        or row["solver_frame"] != previous["solver_frame"] + 1
                        or row["state_revision"] <= previous["state_revision"]
                        or row["completed_at"] <= previous["completed_at"]):
                    raise ValueError(f"{name}: duplicate, missing, reversed or reset solved states")
                delta = row["simulation_time"] - previous["simulation_time"]
                if not math.isclose(delta, 1 / fixture["authored_hz"], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{name}: actual simulation timestep changed")
            previous = row
        measured = rows[warmup:]
        before = rows[warmup - 1]
        intervals = [later["completed_at"] - earlier["completed_at"]
                     for earlier, later in zip(rows[warmup - 1:-1], measured)]
        result[name] = {"rows": measured, "start": before["completed_at"],
                        "end": rows[-1]["completed_at"], "intervals": intervals,
                        "wall": rows[-1]["completed_at"] - before["completed_at"],
                        "simulated": rows[-1]["simulation_time"] - before["simulation_time"]}
    expected = [(row["solver_frame"], row["simulation_time"]) for row in result["A"]["rows"]]
    for name in required[1:]:
        observed = [(row["solver_frame"], row["simulation_time"]) for row in result[name]["rows"]]
        if observed != expected:
            reasons.append(f"{name}: paired lanes do not contain identical completed solved steps")
    return result


def _presentations(report, lane, reasons):
    if report["presentation_scope"] != "draw_submission":
        reasons.append("presentation must name the qualified draw-submission measurement boundary")
    states = {(r["epoch"], r["state_revision"], r["solver_frame"]): r for r in lane["rows"]}
    accepted, ages = [], []
    identities, geometry, pixels, observed_sources = set(), set(), set(), set()
    last_present = lane["start"]
    last_generation = 0
    for row in report["presentations"]:
        if any(type(row[field]) is not int or row[field] < 1
               for field in ("epoch", "state_revision", "solver_frame")):
            reasons.append("presentation has an invalid solved-state identity")
            continue
        key = (row["epoch"], row["state_revision"], row["solver_frame"])
        state = states.get(key)
        if state is None:
            reasons.append("presentation refers to an unknown or retired solved state")
            continue
        ready, complete, presented = (row[field] for field in
                                      ("state_ready_at", "render_completed_at", "presented_at"))
        if (any(not _number(value) for value in (ready, complete, presented))
                or not math.isclose(ready, state["completed_at"], rel_tol=0, abs_tol=1e-6)
                or not ready <= complete <= presented or presented < last_present):
            reasons.append("presentation has an invalid ready/completion/presentation clock")
            continue
        if (not _hash(row["geometry_sha256"]) or not _hash(row["pixels_sha256"])
                or row["geometry_sha256"] != state["geometry_sha256"]):
            reasons.append("presentation content differs from its captured solved geometry")
            continue
        generation = row["surface_generation"]
        if (type(generation) is not int or generation < last_generation or generation < 1
                or (key not in observed_sources and generation <= last_generation)):
            reasons.append("presentation has an invalid surface generation")
            continue
        last_present, last_generation = presented, generation
        observed_sources.add(key)
        ages.append((presented - ready) * 1000)
        # A later timestamp, generation or draw count cannot turn retained
        # geometry/pixels into a genuinely new image of a solved state.
        if key in identities or row["geometry_sha256"] in geometry or row["pixels_sha256"] in pixels:
            continue
        identities.add(key)
        geometry.add(row["geometry_sha256"])
        pixels.add(row["pixels_sha256"])
        accepted.append(row)
    end = max(lane["end"], last_present)
    return {"fresh_states": len(accepted), "fresh_states_per_second": len(accepted) / (end - lane["start"]),
            "p95_state_ready_to_present_ms": _p95(ages) if ages else None,
            "max_state_ready_to_present_ms": max(ages) if ages else None,
            "presentation_window_seconds": end - lane["start"]}


def evaluate_live_trial(report, contract=None):
    """Recompute one paired raw trial's joint decision; return no runtime receipt.

    Full qualification also needs order-balanced repetitions, the retained
    ten-minute soak, visual/optical review, source/runtime admission and installed
    gates. Callers must use the tracked contract; custom contracts are not a
    permission to relabel a failed run with relaxed requirements.
    """
    contract = load_contract() if contract is None else contract
    reasons, metrics = [], {}
    try:
        if contract.get("schema_version") != 1:
            raise ValueError("unsupported contract schema")
        if (report["contract_id"] != contract["contract_id"]
                or report["contract_sha256"] != contract_sha256(contract)):
            reasons.append("report is not bound to this frozen acceptance contract")
        if not _hash(report["candidate_sha"], 40):
            reasons.append("candidate source commit is missing")
        profile = contract["profiles"][report["profile"]]
        if profile["mode"] != "LIVE":
            reasons.append("a non-Live profile cannot qualify a Live throughput trial")
        if contract_sha256(report["settings"]) != contract_sha256(profile["settings"]):
            reasons.append("quality settings differ from the frozen profile")
        fixture = contract["fixtures"][report["fixture_id"]]
        if report["fixture_sha256"] != fixture["sha256"]:
            reasons.append("fixture SHA256 differs from the frozen authored scene")
        lanes = _read_lanes(report, contract, fixture, reasons)
        baseline, live = lanes["A"], lanes["D"]
        metrics.update(solver_wall_ratio=live["wall"] / baseline["wall"],
                       solver_p95_step_interval_ratio=_p95(live["intervals"]) / _p95(baseline["intervals"]),
                       simulated_seconds_per_wall_second=live["simulated"] / live["wall"],
                       completed_solved_steps=len(live["rows"]))
        metrics.update(_presentations(report, live, reasons))
        for before, after in (("A", "B"), ("B", "C"), ("C", "D")):
            metrics[f"{after}_minus_{before}_milliseconds"] = 1000 * (lanes[after]["wall"] - lanes[before]["wall"])
        memory = report["memory"]
        if not _number(memory["physical_device_bytes"], positive=True) or not _number(memory["peak_combined_bytes"]):
            raise ValueError("invalid measured combined VRAM or device capacity")
        metrics["peak_combined_vram_fraction"] = memory["peak_combined_bytes"] / memory["physical_device_bytes"]
        gates = contract["joint_live"]
        for name in ("fresh_states_per_second", "simulated_seconds_per_wall_second"):
            if metrics[name] < gates[name + "_min"]:
                reasons.append(f"{name} is below the frozen minimum")
        for name in ("solver_wall_ratio", "solver_p95_step_interval_ratio", "peak_combined_vram_fraction"):
            if metrics[name] > gates[name + "_max"] + 1e-12:
                reasons.append(f"{name} exceeds the frozen maximum")
        for name in ("p95_state_ready_to_present_ms", "max_state_ready_to_present_ms"):
            if metrics[name] is None or metrics[name] > gates[name + "_max"]:
                reasons.append(f"image age {name} is missing or exceeds the frozen maximum")
        _quality(report, profile, contract, reasons)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        reasons.append(f"incomplete or invalid evidence: {type(exc).__name__}: {exc}")
    return {"status": "FAIL" if reasons else "PASS", "scope": "contract_evaluation_only",
            "contract_id": contract.get("contract_id"), "metrics": metrics,
            "reasons": list(dict.fromkeys(reasons))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)
    with args.report.open(encoding="utf-8") as stream:
        report = json.load(stream)
    result = evaluate_live_trial(report, load_contract(args.contract))
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())

