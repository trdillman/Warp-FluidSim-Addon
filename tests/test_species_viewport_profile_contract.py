"""Contracts for decision-grade dynamic-species viewport receipts."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).parents[1] / "tools" / "profile_species_viewport.py"
_SPEC = importlib.util.spec_from_file_location("profile_species_viewport", _PATH)
_VIEWPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VIEWPORT)


def _case(case, *, solve=10.0, view=2.0, total=13.0, status="supported"):
    return {
        "case": case,
        "status": status,
        "sync_ms": 1.0,
        "solve_ms": solve,
        "view_ms": view,
        "total_ms": total,
        "invariants": {
            "particles": 200_000,
            "active_leaves": 27,
            "active_voxels": 13_824,
            "solver_substeps": 2,
            "surface_cadence": 1,
            "surface_topology": "abc123",
            "active_species": 2 if case == "clear_dye" else 0,
        },
        "presentation": {
            "attribute_name": "fluidsim_color", "attribute_domain": "POINT",
            "cpu_nearest_sample_calls": 0, "corner_color_present": False,
            "display_color_present": False, "launch_inventory": {},
            "readback_inventory": {},
        },
    }


def test_viewport_release_threshold_is_frozen():
    assert dict(_VIEWPORT.VIEWPORT_GATES["clear_dye"]) == {
        "median_fraction": 0.05, "p95_delta_ms": 1.00}
    with pytest.raises(TypeError):
        _VIEWPORT.VIEWPORT_GATES["clear_dye"]["median_fraction"] = 0.2


def test_viewport_receipt_requires_adjacent_pairs_and_matching_invariants():
    rows = [
        _case("disabled", solve=10.0, view=2.0, total=13.0),
        _case("clear_dye", solve=10.2, view=2.05, total=13.25),
        _case("disabled", solve=10.1, view=2.0, total=13.1),
    ]

    report = _VIEWPORT.build_profile_report(rows, production=False)

    assert report["status"] == "diagnostic"
    assert report["metrics"]["view_ms"]["baseline_ms"] == pytest.approx(2.0)
    assert report["metrics"]["total_ms"]["baseline_ms"] == pytest.approx(13.05)
    assert _VIEWPORT.validate_report(report, require_production=False) == []


@pytest.mark.parametrize("field", [
    "particles", "active_leaves", "active_voxels", "solver_substeps",
    "surface_cadence", "surface_topology",
])
def test_viewport_receipt_rejects_compared_case_invariant_mismatch(field):
    rows = [_case("disabled"), _case("clear_dye"), _case("disabled")]
    rows[1]["invariants"][field] = (
        "different" if field == "surface_topology"
        else rows[1]["invariants"][field] + 1)

    report = _VIEWPORT.build_profile_report(rows, production=True)

    assert report["status"] == "fail"
    assert "case_invariants" in _VIEWPORT.validate_report(
        report, require_production=True)


def test_viewport_receipt_rejects_unsynchronized_or_shifted_timing():
    rows = [_case("disabled"), _case("clear_dye"), _case("disabled")]
    rows[1]["total_ms"] = 11.0

    report = _VIEWPORT.build_profile_report(rows, production=True)

    assert report["status"] == "fail"
    assert "timing_invariants" in _VIEWPORT.validate_report(
        report, require_production=True)


def test_viewport_diagnostic_records_unsupported_but_production_rejects_it():
    rows = [
        _case("disabled"),
        _case("clear_dye", status="unsupported"),
        _case("disabled"),
    ]
    report = _VIEWPORT.build_profile_report(rows, production=False)

    assert report["status"] == "diagnostic"
    assert _VIEWPORT.validate_report(report, require_production=False) == []
    assert "unsupported_cases" in _VIEWPORT.validate_report(
        report, require_production=True)


def _strict_provenance():
    return {
        "timestamp_utc": "2026-08-23T00:00:00+00:00",
        "executable": "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe",
        "blender_version": "5.1.0", "python_version": "3.13",
        "warp_version": "1.16.0", "cuda_device": "cuda:0",
        "git_head": "a" * 40, "source_tree_sha256": "b" * 64,
        "profiler_sha256": "c" * 64,
    }


def _strict_viewport_rows(frames):
    rows = []
    presentation = {
        "attribute_name": "fluidsim_color", "attribute_domain": "POINT",
        "cpu_nearest_sample_calls": 0, "corner_color_present": False,
        "display_color_present": False,
        "launch_inventory": {"species_surface_color": 1},
        "readback_inventory": {"surface_result_to_host": 1},
    }
    for _index in range(frames):
        before = _case("disabled", solve=10.0, view=2.0, total=13.0)
        feature = _case("clear_dye", solve=10.2, view=2.05, total=13.25)
        after = _case("disabled", solve=10.1, view=2.0, total=13.1)
        for row in (before, feature, after):
            row["sync_ms"] = 1.0
            row["invariants"]["active_species"] = (
                2 if row["case"] == "clear_dye" else 0)
            row["presentation"] = dict(presentation)
        rows.extend((before, feature, after))
    return rows


def test_strict_viewport_receipt_requires_live_provenance_frames_and_n2():
    frames = _VIEWPORT.PRODUCTION_MINIMUMS["frames"]
    report = _VIEWPORT.build_profile_report(
        _strict_viewport_rows(frames), production=True,
        configuration={"frames": frames, "warmup": 15, "particles": 200_000},
        provenance=_strict_provenance())
    assert _VIEWPORT.validate_report(report, require_production=True) == []

    missing = dict(report)
    missing.pop("provenance")
    assert "provenance" in _VIEWPORT.validate_report(missing)

    short = _VIEWPORT.build_profile_report(
        _strict_viewport_rows(2), production=True,
        configuration={"frames": 2, "warmup": 1, "particles": 200_000},
        provenance=_strict_provenance())
    assert "production_minimums" in _VIEWPORT.validate_report(short)


def test_strict_viewport_rejects_total_gap_and_legacy_color_paths():
    frames = _VIEWPORT.PRODUCTION_MINIMUMS["frames"]
    rows = _strict_viewport_rows(frames)
    rows[1]["total_ms"] += 0.5
    rows[1]["presentation"]["cpu_nearest_sample_calls"] = 1
    rows[1]["presentation"]["attribute_domain"] = "CORNER"
    report = _VIEWPORT.build_profile_report(
        rows, production=True,
        configuration={"frames": frames, "warmup": 15, "particles": 200_000},
        provenance=_strict_provenance())

    errors = _VIEWPORT.validate_report(report, require_production=True)

    assert "timing_invariants" in errors
    assert "presentation_contract" in errors


def test_viewport_main_executes_real_blender_warp_publication_path():
    source = inspect.getsource(_VIEWPORT.main)

    assert "_build_scene_geometry(bpy)" in source
    assert '_new_solver(args, species_enabled=False)' in source
    assert '_new_solver(args, species_enabled=True)' in source
    assert "measure_synchronized_frame(" in source
    assert "output_publication" in source
    assert "_surface_presentation(" in source
    assert "_topology_identity(" in source
    assert "_unsupported_rows" not in source


def test_two_inflow_turbine_particles_are_deterministic_and_balanced():
    first_positions, first_velocities = _VIEWPORT._two_inflow_turbine_particles(20)
    second_positions, second_velocities = _VIEWPORT._two_inflow_turbine_particles(20)

    assert np.array_equal(first_positions, second_positions)
    assert np.array_equal(first_velocities, second_velocities)
    assert len(first_positions[:10]) == len(first_positions[10:])
    assert first_positions[:10, 0].max() < first_positions[10:, 0].min()
