"""Regression contracts for demo-analysis acceptance criteria."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_demo_analyzer():
    path = Path(__file__).parents[1] / "tools" / "analyze_demos.py"
    spec = spec_from_file_location("demo_analyzer", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pre_wall_window_stops_at_first_contact_before_reflection():
    analyzer = _load_demo_analyzer()
    fronts = np.array([1.0, 2.0, 5.6, 5.4, 5.2])

    assert analyzer.first_wall_contact(fronts, wall_x=5.5) == 2


def test_bulk_column_height_ignores_sparse_wall_spray():
    analyzer = _load_demo_analyzer()
    collapsed = np.column_stack((
        np.full(1000, 0.1), np.zeros(1000), np.full(1000, 0.1)))
    spray = np.column_stack((
        np.full(10, 0.1), np.zeros(10), np.full(10, 1.1)))

    assert np.isclose(
        analyzer.bulk_column_height(np.vstack((collapsed, spray))), 0.1)


def test_crown_expansion_window_ends_at_rim_collapse_not_particle_count_peak():
    analyzer = _load_demo_analyzer()
    crown_r = {1: 0.04, 2: 0.11, 3: 0.28, 4: 0.33, 5: 0.32, 6: 0.31}
    # Count continues rising well after radial expansion has stopped.
    crown_n = {1: 500, 2: 1000, 3: 2000, 4: 3000, 5: 4000, 6: 5000}

    assert analyzer.crown_expansion_end(
        crown_r, crown_n, impact_frame=1, frame_numbers=range(1, 7)) == 4


def test_martin_moses_front_position_rejects_value_below_reported_range(monkeypatch):
    analyzer = _load_demo_analyzer()
    captured = []
    a = 2.0 * 9.81

    meta = {
        "gravity": 9.81,
        "column_height": 1.2,
        "column_width": a,
        "fps": 1,
        "domain": [[0.0, 0.0, 0.0], [100.0, 1.0, 1.0]],
        "voxel_size": 0.1,
    }
    frame_positions = np.array([[0.1, 0.0, 0.1], [2.24 * a, 0.0, 0.1]])
    frames = {frame: {"positions": frame_positions} for frame in (1, 2, 3)}
    series = {"whitewater_count": [0, 101, 0]}

    monkeypatch.setattr(analyzer, "load_scene", lambda _name: (meta, series, frames))
    monkeypatch.setattr(analyzer, "common_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyzer,
        "check",
        lambda _scene, name, passed, detail: captured.append((name, passed, detail)),
    )

    analyzer.analyze_dam_break()

    front_check = next(item for item in captured if item[0] == "Martin-Moses front position at T=2")
    assert front_check == (
        "Martin-Moses front position at T=2",
        False,
        "Z(T=2) = 2.24 (experimental 2.5-4.0)",
    )
