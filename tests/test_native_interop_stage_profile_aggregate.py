import json

import native_interop_stage_profile_aggregate as aggregate


def _run(path, first_mode, packed_ms, no_repack_ms):
    payload = {
        "passed": True,
        "environment": {
            "executable": "blender.exe",
            "executable_sha256_before": "candidate-hash",
            "executable_sha256_after": "candidate-hash",
            "addon_source": "addon",
            "process_limits_before": {"priority_class": "0x4000", "affinity_mask": "0xffff"},
            "process_limits_after": {"priority_class": "0x4000", "affinity_mask": "0xffff"},
        },
        "workload": {
            "particle_count": 250_000,
            "first_mode": first_mode,
            "packed_batch_mean_ms": {"median": packed_ms},
            "no_repack_batch_mean_ms": {"median": no_repack_ms},
            "idealized_speedup": packed_ms / no_repack_ms,
            "cuda_stage_ms": {"position_pack_ms": {"median": 0.01}},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregate_accepts_stable_balanced_fresh_processes(tmp_path):
    paths = [
        _run(tmp_path / "run1.json", "packed", 13.0, 13.1),
        _run(tmp_path / "run2.json", "no_repack", 13.2, 13.0),
        _run(tmp_path / "run3.json", "packed", 13.1, 13.2),
    ]

    result = aggregate.aggregate_profiles(paths)

    assert result["passed"]
    assert result["decision"] == "retain_current_publication"


def test_aggregate_rejects_cross_process_spread_above_ten_percent(tmp_path):
    paths = [
        _run(tmp_path / "run1.json", "packed", 10.0, 10.0),
        _run(tmp_path / "run2.json", "no_repack", 10.0, 10.0),
        _run(tmp_path / "run3.json", "packed", 14.0, 14.0),
    ]

    result = aggregate.aggregate_profiles(paths)

    assert not result["passed"]
    assert result["decision"] == "measurement_invalid"
    assert not result["gates"]["packed_cross_process_spread_within_10_percent"]
    assert not result["gates"]["no_repack_cross_process_spread_within_10_percent"]


def _draw_run(path, first_mode, visible_ms, hidden_ms):
    payload = {
        "passed": True,
        "environment": {
            "executable": "blender.exe",
            "executable_sha256_before": "candidate-hash",
            "executable_sha256_after": "candidate-hash",
            "addon_source": "addon",
            "process_limits_before": {"priority_class": "0x4000", "affinity_mask": "0xffff"},
            "process_limits_after": {"priority_class": "0x4000", "affinity_mask": "0xffff"},
        },
        "workload": {
            "particle_count": 600_000,
            "first_mode": first_mode,
            "visible_redraw_batch_mean_ms": {"median": visible_ms},
            "hidden_redraw_batch_mean_ms": {"median": hidden_ms},
            "cuda_publication_ms": {"median": 0.8},
            "solver_gpu_wall_ms": {"median": 25.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_draw_aggregate_attributes_stable_visible_redraw_cost(tmp_path):
    paths = [
        _draw_run(tmp_path / "run1.json", "visible", 14.0, 3.0),
        _draw_run(tmp_path / "run2.json", "hidden", 13.8, 3.1),
        _draw_run(tmp_path / "run3.json", "visible", 14.2, 2.9),
    ]

    result = aggregate.aggregate_draw_profiles(paths)

    assert result["passed"]
    assert result["display_dominant_stage"] == "pointcloud_draw"
    assert result["aggregate"]["pointcloud_draw_delta_median_ms"] == 11.0


def test_draw_aggregate_rejects_unstable_fresh_processes(tmp_path):
    paths = [
        _draw_run(tmp_path / "run1.json", "visible", 10.0, 3.0),
        _draw_run(tmp_path / "run2.json", "hidden", 10.0, 3.0),
        _draw_run(tmp_path / "run3.json", "visible", 15.0, 3.0),
    ]

    result = aggregate.aggregate_draw_profiles(paths)

    assert not result["passed"]
    assert result["decision"] == "measurement_invalid"
    assert not result["gates"]["visible_cross_process_spread_within_10_percent"]
