"""Regression contracts for demo-analysis acceptance criteria."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


def _load_demo_analyzer():
    path = Path(__file__).parents[2] / "tools" / "demos" / "analyze_demos.py"
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


def _dam_break_checks(monkeypatch, history):
    """Run analyze_dam_break on (front x, back-wall column height) per frame."""
    analyzer = _load_demo_analyzer()
    captured = {}
    meta = {
        "gravity": 9.81,
        "column_height": 1.2,
        "column_width": 0.6,
        "fps": 24,
        "domain": [[0.0, -1.0, 0.0], [5.6, 1.0, 1.6]],
        "voxel_size": 0.03,
        "apron_end_x": 2.0,
    }
    frames = {f: {"positions": np.array([[front, 0.0, 0.1], [0.1, 0.0, column]])}
              for f, (front, column) in history.items()}
    series = {"whitewater_count": [0, 101] + [0] * (len(history) - 2)}

    monkeypatch.setattr(analyzer, "load_scene", lambda _name: (meta, series, frames))
    monkeypatch.setattr(analyzer, "common_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        analyzer,
        "check",
        lambda _scene, name, passed, detail: captured.setdefault(name, (passed, detail)),
    )
    analyzer.analyze_dam_break()
    return captured


def test_dam_break_front_phase_ends_where_the_apron_ends(monkeypatch):
    """Once the bore leaves the flat apron it is over; a later low front is terrain."""
    captured = _dam_break_checks(monkeypatch, {
        1: (0.5, 1.0), 2: (1.0, 0.7), 3: (1.5, 0.5), 4: (2.5, 0.1), 5: (1.0, 0.1)})

    assert captured["front advances monotonically"] == (
        True, "min frame-to-frame delta 500.0 mm")
    assert captured["column collapses"][1].startswith(
        "minimum pre-reflection left-wall height 0.50 m")


def test_dam_break_front_check_fails_when_the_apron_window_is_too_short(monkeypatch):
    """A bore that crosses the apron in under three frames proves no advance."""
    captured = _dam_break_checks(monkeypatch, {
        1: (0.5, 1.0), 2: (1.5, 0.5), 3: (2.5, 0.1), 4: (3.0, 0.1)})

    assert captured["front advances monotonically"] == (
        False, "only 2 front(s) before the apron ends (need 3)")


def test_column_collapse_uses_pre_reflection_minimum_not_last_frame():
    analyzer = _load_demo_analyzer()
    frames = {
        1: {"positions": np.array([[0.1, 0.0, 1.2], [1.0, 0.0, 0.1]])},
        2: {"positions": np.array([[0.1, 0.0, 0.3], [2.0, 0.0, 0.1]])},
        3: {"positions": np.array([[0.1, 0.0, 0.9], [3.0, 0.0, 0.1]])},
    }

    height = analyzer.left_wall_collapse_height(frames, (1, 2, 3))

    assert height == 0.3


# --- paddle spin-up acceptance (docs/paddle_spinup_calibration.md) -----------

_PADDLE_CHECK = "angular momentum spins up and decays"
_RING_RADIUS = 0.2
_RING_COUNT = 100


def _paddle_scene(rad_per_s, early_speed, steady_speed, late_speed):
    """A paddle_stir scene whose Lz profile is dialled in exactly.

    Particles sit on a ring of radius _RING_RADIUS about the tank axis with
    purely tangential velocity, so per frame
    Lz = sum(rx*vy - ry*vx) = _RING_COUNT * _RING_RADIUS * speed.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, _RING_COUNT, endpoint=False)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    positions = np.zeros((_RING_COUNT, 3), dtype=np.float64)
    positions[:, :2] = _RING_RADIUS * unit
    positions[:, 2] = 0.05
    tangent = np.stack([-unit[:, 1], unit[:, 0]], axis=1)

    def velocities(speed):
        v = np.zeros((_RING_COUNT, 3), dtype=np.float64)
        v[:, :2] = speed * tangent
        return v

    frames = {}
    for f in range(1, 31):
        if f <= 5:
            speed = early_speed
        elif f >= 26:
            speed = late_speed
        else:
            speed = steady_speed
        frames[f] = {"positions": positions, "velocities": velocities(speed)}

    meta = {
        "voxel_size": 0.012,
        "fps": 24,
        "domain": [[-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]],
        "tank_center": [0.0, 0.0, 0.1],
        "tank_height": 0.4,
        "tank_inner_radius": 0.45,
        "paddle_origin": [0.0, 0.0, 0.0],
        "blade_inner_radius": 0.1,
        "blade_outer_radius": 0.3,
        "blade_z_min": 0.0,
        "blade_z_max": 0.2,
        "blade_thickness": 0.01,
        "blade_count": 3,
        "pool_depth": 0.15,
        "peak_rad_per_s": rad_per_s,
        "rotation_keyframes": [[1, 0.0], [30, rad_per_s * 30.0 / 24.0]],
        "spinup_end_frame": 10,
        "steady_end_frame": 20,
    }
    series = {"frames": sorted(frames), "whitewater_count": [0] * len(frames)}
    return meta, series, frames


def _run_paddle(monkeypatch, scene):
    analyzer = _load_demo_analyzer()
    captured = []
    monkeypatch.setattr(analyzer, "load_scene", lambda _name: scene)
    monkeypatch.setattr(analyzer, "common_checks", lambda *_a, **_k: None)
    monkeypatch.setattr(
        analyzer,
        "check",
        lambda _scene, name, passed, detail: captured.append((name, passed, detail)),
    )
    analyzer.analyze_paddle_stir()
    return next(item for item in captured if item[0] == _PADDLE_CHECK)


def test_paddle_spinup_accepts_shipped_scale(monkeypatch):
    """100 RPM at the shipped settings: ratio ~0.94, circulation ~0.17."""
    _name, passed, detail = _run_paddle(
        monkeypatch,
        _paddle_scene(10.472, early_speed=0.048, steady_speed=0.8, late_speed=0.616),
    )
    assert passed, detail


def test_paddle_spinup_accepts_quarter_speed_scale_variant(monkeypatch):
    """Same physics at 25 RPM. The threshold is nondimensional, so a scene
    with a quarter the drive and a quarter the circulation must still pass."""
    _name, passed, detail = _run_paddle(
        monkeypatch,
        _paddle_scene(2.618, early_speed=0.012, steady_speed=0.2, late_speed=0.154),
    )
    assert passed, detail


def test_paddle_spinup_rejects_undriven_scene_the_ratio_alone_accepts(monkeypatch):
    """The 0.5 RPM regression.

    A scene with effectively no drive has a steady-window mean that is pure
    noise, so the relative margin divides by noise and *explodes* to ~19.6
    instead of collapsing -- the ratio alone accepts it at any threshold
    below 19.6. Only the absolute circulation floor rejects it.
    """
    scene = _paddle_scene(0.0524, early_speed=-2.1e-4, steady_speed=1.13e-5,
                          late_speed=-8.0e-4)
    _name, passed, detail = _run_paddle(monkeypatch, scene)
    assert not passed, detail
    # Prove it is the floor doing the work, not the margin.
    analyzer = _load_demo_analyzer()
    meta, _series, frames = scene
    n = frames[1]["positions"].shape[0]
    steady = n * _RING_RADIUS * 1.13e-5
    early = n * _RING_RADIUS * -2.1e-4
    ratio = (steady - early) / abs(steady)
    assert ratio > analyzer.PADDLE_SPINUP_MOMENTUM_MARGIN
    scale = n * abs(meta["peak_rad_per_s"]) * meta["blade_outer_radius"] ** 2
    assert steady / scale < analyzer.PADDLE_STEADY_CIRCULATION_FLOOR


def test_paddle_spinup_rejects_stationary_paddle(monkeypatch):
    """A stationary paddle has no circulation scale at all; the check must
    fail rather than attempt the division."""
    _name, passed, detail = _run_paddle(
        monkeypatch,
        _paddle_scene(0.0, early_speed=0.001, steady_speed=0.02, late_speed=0.01),
    )
    assert not passed, detail


# --- crown splash height (crown phase only) ----------------------------------

_POOL = 0.05
_CROWN_CENTER = (0.4, 0.4)


def _ring(count, radius, height):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    positions = np.empty((count, 3), dtype=np.float64)
    positions[:, 0] = _CROWN_CENTER[0] + radius * np.cos(angles)
    positions[:, 1] = _CROWN_CENTER[1] + radius * np.sin(angles)
    positions[:, 2] = _POOL + height
    return positions


def test_crown_height_ignores_intact_droplet_and_late_jet_spray(monkeypatch):
    """max_crown_height must come from the crown rim alone.

    The drop reaches the pool at 30 ms and is fully submerged by 56 ms, so
    frame 1 (42 ms) still holds the intact droplet. Frames after the rim
    collapse hold jet breakup spray just outside the 2 cm core exclusion.
    Both sit higher than the rim here, and both must be excluded.
    """
    analyzer = _load_demo_analyzer()
    crown = {2: (300, 0.10, 0.05), 3: (600, 0.12, 0.07),
             4: (1000, 0.14, 0.09), 5: (400, 0.15, 0.06)}
    frames = {1: {"positions": _ring(60, 0.03, 0.12)}}
    for f, (count, radius, height) in crown.items():
        frames[f] = {"positions": _ring(count, radius, height)}
    for f in range(6, 11):
        frames[f] = {"positions": _ring(30, 0.03, 0.30)}
    meta = {"fps": 24, "pool_depth": _POOL,
            "drop_center": [_CROWN_CENTER[0], _CROWN_CENTER[1], 0.2],
            "drop_radius": 0.045, "drop_speed": 3.5}
    series = {"frames": sorted(frames), "whitewater_count": [0] * len(frames)}
    captured = []
    monkeypatch.setattr(analyzer, "load_scene", lambda _name: (meta, series, frames))
    monkeypatch.setattr(analyzer, "common_checks", lambda *_a, **_k: None)
    monkeypatch.setattr(
        analyzer,
        "check",
        lambda _scene, name, passed, detail: captured.append((name, passed, detail)),
    )

    result = analyzer.analyze_crown_splash()

    assert result["peak_frame"] == 4
    assert result["max_crown_height"] == pytest.approx(0.09)
    rise = next(item for item in captured if item[0] == "crown rises above pool")
    assert rise[1], rise[2]
    assert "(frames 2-5)" in rise[2]


def test_crown_phase_height_is_zero_for_an_empty_window():
    analyzer = _load_demo_analyzer()
    assert analyzer.crown_phase_height({1: 0.2, 6: 0.3}, 2, 5) == 0.0
