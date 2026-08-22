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


def test_column_collapse_uses_pre_reflection_minimum_not_last_frame():
    analyzer = _load_demo_analyzer()
    frames = {
        1: {"positions": np.array([[0.1, 0.0, 1.2], [1.0, 0.0, 0.1]])},
        2: {"positions": np.array([[0.1, 0.0, 0.3], [2.0, 0.0, 0.1]])},
        3: {"positions": np.array([[0.1, 0.0, 0.9], [3.0, 0.0, 0.1]])},
    }

    height = analyzer.left_wall_collapse_height(frames, (1, 2, 3))

    assert height == 0.3
