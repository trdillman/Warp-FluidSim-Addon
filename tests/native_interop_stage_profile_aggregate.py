"""Aggregate fresh-process native interop stage profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def coefficient_of_variation(values):
    return statistics.pstdev(values) / statistics.fmean(values)


def aggregate_profiles(run_paths):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in run_paths]
    if len(runs) < 3:
        raise ValueError("at least three fresh-process profiles are required")

    packed = [run["workload"]["packed_batch_mean_ms"]["median"] for run in runs]
    no_repack = [run["workload"]["no_repack_batch_mean_ms"]["median"] for run in runs]
    speedups = [run["workload"]["idealized_speedup"] for run in runs]
    pack_stages = [
        run["workload"]["cuda_stage_ms"]["position_pack_ms"]["median"] for run in runs
    ]
    environments = [run["environment"] for run in runs]
    packed_spread = coefficient_of_variation(packed)
    no_repack_spread = coefficient_of_variation(no_repack)
    packed_median = statistics.median(packed)
    pack_median = statistics.median(pack_stages)

    gates = {
        "at_least_three_fresh_processes": len(runs) >= 3,
        "balanced_first_mode": {
            run["workload"]["first_mode"] for run in runs
        } == {"packed", "no_repack"},
        "individual_runs_passed": all(run["passed"] for run in runs),
        "packed_cross_process_spread_within_10_percent": packed_spread <= 0.1,
        "no_repack_cross_process_spread_within_10_percent": no_repack_spread <= 0.1,
        "same_executable": len({env["executable"] for env in environments}) == 1,
        "same_executable_hash": len({
            (env["executable_sha256_before"], env["executable_sha256_after"])
            for env in environments
        }) == 1,
        "same_addon_source": len({env["addon_source"] for env in environments}) == 1,
        "same_process_limits": len({
            json.dumps((env["process_limits_before"], env["process_limits_after"]), sort_keys=True)
            for env in environments
        }) == 1,
        "same_particle_count": {
            run["workload"]["particle_count"] for run in runs
        } == {250_000},
    }
    valid = all(gates.values())
    idealized_speedup = statistics.median(speedups)
    decision = (
        "retain_current_publication"
        if valid
        and idealized_speedup < 2.0
        and pack_median < 1.0
        and pack_median < packed_median * 0.1
        else "measurement_invalid"
        if not valid
        else "further_analysis_required"
    )
    return {
        "passed": valid,
        "decision": decision,
        "run_paths": [str(Path(path).resolve()) for path in run_paths],
        "cross_process_spread_fraction": {
            "packed": packed_spread,
            "no_repack": no_repack_spread,
        },
        "aggregate": {
            "packed_batch_mean_median_ms": packed_median,
            "no_repack_batch_mean_median_ms": statistics.median(no_repack),
            "idealized_speedup_median": idealized_speedup,
            "position_pack_median_ms": pack_median,
        },
        "gates": gates,
    }


def aggregate_draw_profiles(run_paths):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in run_paths]
    if len(runs) < 3:
        raise ValueError("at least three fresh-process profiles are required")

    visible = [run["workload"]["visible_redraw_batch_mean_ms"]["median"] for run in runs]
    hidden = [run["workload"]["hidden_redraw_batch_mean_ms"]["median"] for run in runs]
    publication = [run["workload"]["cuda_publication_ms"]["median"] for run in runs]
    solver = [run["workload"]["solver_gpu_wall_ms"]["median"] for run in runs]
    draw_delta = [visible_ms - hidden_ms for visible_ms, hidden_ms in zip(visible, hidden)]
    environments = [run["environment"] for run in runs]
    visible_spread = coefficient_of_variation(visible)
    hidden_spread = coefficient_of_variation(hidden)

    gates = {
        "at_least_three_fresh_processes": len(runs) >= 3,
        "balanced_first_mode": {
            run["workload"]["first_mode"] for run in runs
        } == {"visible", "hidden"},
        "individual_runs_passed": all(run["passed"] for run in runs),
        "visible_cross_process_spread_within_10_percent": visible_spread <= 0.1,
        "hidden_cross_process_spread_within_10_percent": hidden_spread <= 0.1,
        "positive_pointcloud_draw_delta": all(value > 0.0 for value in draw_delta),
        "same_executable": len({env["executable"] for env in environments}) == 1,
        "same_executable_hash": len({
            (env["executable_sha256_before"], env["executable_sha256_after"])
            for env in environments
        }) == 1,
        "same_addon_source": len({env["addon_source"] for env in environments}) == 1,
        "same_process_limits": len({
            json.dumps((env["process_limits_before"], env["process_limits_after"]), sort_keys=True)
            for env in environments
        }) == 1,
        "same_particle_count": {
            run["workload"]["particle_count"] for run in runs
        } == {600_000},
    }
    valid = all(gates.values())
    hidden_median = statistics.median(hidden)
    draw_delta_median = statistics.median(draw_delta)
    display_dominant_stage = (
        "pointcloud_draw" if draw_delta_median > hidden_median else "viewport_baseline"
    )
    return {
        "passed": valid,
        "decision": "profile_valid" if valid else "measurement_invalid",
        "display_dominant_stage": display_dominant_stage if valid else None,
        "run_paths": [str(Path(path).resolve()) for path in run_paths],
        "cross_process_spread_fraction": {
            "visible": visible_spread,
            "hidden": hidden_spread,
        },
        "aggregate": {
            "visible_redraw_median_ms": statistics.median(visible),
            "hidden_redraw_median_ms": hidden_median,
            "pointcloud_draw_delta_median_ms": draw_delta_median,
            "cuda_publication_median_ms": statistics.median(publication),
            "solver_gpu_wall_median_ms": statistics.median(solver),
        },
        "gates": gates,
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("stage", "draw"), default="stage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        aggregate = aggregate_profiles if args.kind == "stage" else aggregate_draw_profiles
        result = aggregate(args.runs)
    except Exception as exc:  # noqa: BLE001
        result = {"passed": False, "decision": "measurement_invalid", "error": repr(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    prefix = (
        "NATIVE_INTEROP_STAGE_AGGREGATE"
        if args.kind == "stage"
        else "NATIVE_INTEROP_DRAW_AGGREGATE"
    )
    print(prefix + "=" + json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
