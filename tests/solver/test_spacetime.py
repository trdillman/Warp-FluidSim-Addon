"""SpaceTime FLIP state: seed, last step size and reference mass m0, without a GPU.

``spacetime`` is host-only (numpy), so everything the solver used to do to its
``_st_*`` attributes runs here against plain stand-ins. The golden values were
recorded from the previous in-solver ``_calibrate_st_reference_mass`` /
``_st_reference_mass`` before they moved, so a drift in the calibration (and so
in every ST bake) fails loudly.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.sim import spacetime
from fluidSim_warp_blender51.sim.spacetime import SpaceTimeState


def _settings(**overrides):
    values = {
        "spacetime_enabled": False,
        "spacetime_seed": 1729,
        "spacetime_reference_mass": 0.0,
        "particles_per_voxel": 8.0,
        "time_scale": 1.0,
        "frame_rate": 24.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _quality(liquid_sampling=8.0, rest_particle_count=8):
    return SimpleNamespace(liquid_sampling=liquid_sampling, rest_particle_count=rest_particle_count)


def _state(settings=None, quality=None):
    return SpaceTimeState(settings or _settings(), quality or _quality())


# --- calibration ------------------------------------------------------------

# (liquid_sampling, seed) -> m0, recorded from the pre-extraction solver method.
CALIBRATION_GOLDENS = {
    (1.0, 1729): 1.0402005308538038,
    (1.0, 9): 0.9209376008170541,
    (1.0, 4294967300): 1.0357899156636945,
    (2.5, 1729): 2.588012563132733,
    (2.5, 9): 2.3546466601469382,
    (2.5, 4294967300): 2.4414544957251634,
    (8.0, 1729): 7.725084631280045,
    (8.0, 9): 8.084321937722693,
    (8.0, 4294967300): 7.809413953575986,
    (12.0, 1729): 12.231675728067868,
    (12.0, 9): 11.754229738745982,
    (12.0, 4294967300): 11.842885444745427,
}


@pytest.mark.parametrize(("sampling", "seed"), sorted(CALIBRATION_GOLDENS))
def test_calibration_reproduces_the_recorded_estimate_exactly(sampling, seed):
    state = _state(_settings(spacetime_seed=seed), _quality(liquid_sampling=sampling))

    assert state.calibrate_reference_mass() == CALIBRATION_GOLDENS[(sampling, seed)]


def test_calibration_is_deterministic_for_a_seed():
    state = _state(_settings(spacetime_seed=77))

    assert state.calibrate_reference_mass() == state.calibrate_reference_mass()


def test_a_manual_reference_mass_is_returned_unchanged():
    state = _state(_settings(spacetime_reference_mass=3.25))

    assert state.calibrate_reference_mass() == 3.25


# --- reference mass ladder ---------------------------------------------------


@pytest.mark.parametrize(
    ("manual", "cached", "rest", "expected"),
    [
        (2.0, 5.0, 8, 2.0),  # a pinned value always wins
        (0.0, 5.0, 8, 5.0),  # then the calibrated one
        (0.0, 0.0, 8, 8.0),  # then the rest particle count
        (0.0, 0.0, 0, 1.0),  # never below one
        (0.0, float("nan"), 7, 7.0),  # a poisoned cache is ignored
        (float("inf"), 5.0, 8, 5.0),  # so is a non-finite pin
    ],
)
def test_reference_mass_falls_back_pin_then_calibration_then_rest_count(
    manual, cached, rest, expected
):
    state = _state(_settings(spacetime_reference_mass=manual), _quality(rest_particle_count=rest))
    state.reference_mass_cached = cached

    assert state.reference_mass() == expected


# --- lifecycle ---------------------------------------------------------------


def test_a_disabled_state_starts_without_calibrating_or_a_previous_step():
    state = _state(_settings(spacetime_enabled=False, spacetime_seed=1729))

    assert (state.seed, state.previous_dt, state.reference_mass_cached) == (1729, 0.0, 0.0)


def test_the_seed_is_masked_to_31_bits():
    assert _state(_settings(spacetime_seed=0xFFFFFFFF + 5)).seed == 4


def test_an_enabled_state_calibrates_once_at_construction():
    state = _state(_settings(spacetime_enabled=True), _quality(liquid_sampling=8.0))

    assert state.reference_mass_cached == CALIBRATION_GOLDENS[(8.0, 1729)]


def test_a_pinned_reference_mass_is_adopted_without_calibrating(monkeypatch):
    def forbidden(self):
        raise AssertionError("a pinned m0 must not be re-estimated")

    monkeypatch.setattr(SpaceTimeState, "calibrate_reference_mass", forbidden)

    state = _state(_settings(spacetime_enabled=True, spacetime_reference_mass=6.5))

    assert state.reference_mass_cached == 6.5


def test_reset_for_run_restores_every_field_and_follows_the_current_settings():
    settings = _settings(spacetime_enabled=True, spacetime_seed=9)
    state = _state(settings, _quality(liquid_sampling=8.0))
    state.advance_seed()
    state.record_step(0.01)
    state.reference_mass_cached = 123.0
    settings.spacetime_seed = 1729

    state.reset_for_run()

    assert state.seed == 1729
    assert state.previous_dt == 0.0
    assert state.reference_mass_cached == CALIBRATION_GOLDENS[(8.0, 1729)]


def test_reset_for_run_does_not_calibrate_when_space_time_is_off(monkeypatch):
    state = _state()
    monkeypatch.setattr(
        SpaceTimeState,
        "calibrate_reference_mass",
        lambda self: (_ for _ in ()).throw(AssertionError("ST is off")),
    )

    state.reset_for_run()

    assert state.reference_mass_cached == 0.0


# --- per-substep --------------------------------------------------------------


def test_the_jitter_stream_is_the_recorded_lcg_and_stays_31_bit():
    state = _state(_settings(spacetime_seed=1729))
    expected = 1729
    for _ in range(5):
        expected = (expected * 1664525 + 1013904223) & 0x7FFFFFFF
        assert state.advance_seed() == expected
        assert state.seed == expected
    assert 0 <= state.seed <= 0x7FFFFFFF


def test_the_first_gather_uses_the_given_step_then_the_previous_one():
    state = _state()

    assert state.gather_dt(0.02) == 0.02

    state.record_step(0.005)

    assert state.gather_dt(0.02) == 0.005
    assert isinstance(state.previous_dt, float)


def test_with_no_step_at_all_a_gather_uses_one_frame_of_time():
    assert _state(_settings(time_scale=2.0, frame_rate=24.0)).gather_dt() == pytest.approx(2.0 / 24.0)
    assert _state(_settings(time_scale=2.0, frame_rate=0.0)).gather_dt(0.0) == 2.0  # rate floors at 1


# --- live density edits -------------------------------------------------------


def _counting_calibration(monkeypatch, value):
    calls = []
    monkeypatch.setattr(
        SpaceTimeState,
        "calibrate_reference_mass",
        lambda self: calls.append(1) or value,
    )
    return calls


def test_a_new_density_scales_one_calibration_instead_of_recomputing(monkeypatch):
    settings = _settings(spacetime_enabled=True, particles_per_voxel=8.0, spacetime_seed=9)
    calls = _counting_calibration(monkeypatch, 3.5)
    state = _state(settings)
    calls.clear()  # construction paid for the calibration

    assert state.rescale_for_density() is True
    assert state.reference_mass_cached == 3.5
    settings.particles_per_voxel = 16.0
    assert state.rescale_for_density() is True
    assert state.reference_mass_cached == 7.0

    assert calls == [1]


def test_a_remembered_calibration_means_no_estimate_at_all(monkeypatch):
    settings = _settings(spacetime_enabled=True, particles_per_voxel=8.0)
    state = _state(settings)
    state.reference_mass_cached = 4.0
    calls = _counting_calibration(monkeypatch, 99.0)

    state.remember_calibration()
    settings.particles_per_voxel = 12.0
    state.rescale_for_density()

    assert calls == []
    assert state.reference_mass_cached == pytest.approx(6.0)


def test_coefficients_are_kept_per_seed(monkeypatch):
    settings = _settings(spacetime_enabled=True, particles_per_voxel=8.0, spacetime_seed=1)
    state = _state(settings)
    state.reference_mass_cached = 4.0
    state.remember_calibration()
    settings.spacetime_seed = 2
    calls = _counting_calibration(monkeypatch, 16.0)

    state.rescale_for_density()

    assert calls == [1]
    assert state.reference_mass_cached == 16.0


@pytest.mark.parametrize(
    "overrides",
    [{"spacetime_enabled": False}, {"spacetime_enabled": True, "spacetime_reference_mass": 5.0}],
)
def test_density_edits_leave_a_pinned_or_disabled_reference_alone(overrides, monkeypatch):
    state = _state(_settings(**overrides))
    state.reference_mass_cached = 4.0
    calls = _counting_calibration(monkeypatch, 99.0)

    state.remember_calibration()
    assert state.rescale_for_density() is False

    assert calls == []
    assert state.reference_mass_cached == 4.0
    assert state._density_coefficient is None


def test_an_unset_calibration_is_not_remembered():
    state = _state(_settings(spacetime_enabled=True))
    state.reference_mass_cached = 0.0

    state.remember_calibration()

    assert state._density_coefficient is None


def test_solvers_with_the_same_seed_but_different_quality_do_not_share_a_coefficient():
    """The remembered coefficient belongs to one state, not to the process."""
    coarse = _state(_settings(spacetime_enabled=True), _quality(liquid_sampling=8.0))
    fine = _state(_settings(spacetime_enabled=True), _quality(liquid_sampling=1.0))

    coarse.remember_calibration()
    assert fine.rescale_for_density() is True

    assert fine.reference_mass_cached == pytest.approx(CALIBRATION_GOLDENS[(1.0, 1729)])


def test_a_seed_edit_discards_the_remembered_coefficient():
    settings = _settings(spacetime_enabled=True)
    state = _state(settings)
    state.remember_calibration()

    settings.spacetime_seed = 9
    state.rescale_for_density()

    assert state.reference_mass_cached == pytest.approx(CALIBRATION_GOLDENS[(8.0, 9)])


# --- checkpoint ----------------------------------------------------------------


def test_the_checkpoint_scalars_keep_their_names_and_types():
    state = _state(_settings(spacetime_enabled=True))
    state.advance_seed()
    state.record_step(0.004)

    scalars = state.checkpoint_scalars()

    assert tuple(scalars) == SpaceTimeState.CHECKPOINT_SCALARS
    assert isinstance(scalars["st_step_seed"], int)
    assert isinstance(scalars["st_previous_dt"], float)
    assert isinstance(scalars["st_reference_mass_cached"], float)


def test_a_checkpoint_round_trips_into_a_fresh_state():
    source = _state(_settings(spacetime_enabled=True))
    source.advance_seed()
    source.advance_seed()
    source.record_step(0.004)
    target = _state(_settings(spacetime_enabled=True, spacetime_seed=5))

    target.restore_checkpoint_scalars(source.checkpoint_scalars())

    assert target.checkpoint_scalars() == source.checkpoint_scalars()


def test_every_exported_scalar_is_one_the_checkpoint_validator_requires():
    source = (Path(spacetime.__file__).parent / "checkpoint_state.py").read_text(encoding="utf-8")
    for name in SpaceTimeState.CHECKPOINT_SCALARS:
        assert f'"{name}"' in source, f"{name} is not a required checkpoint scalar"


@pytest.mark.parametrize("dt", [math.inf, -math.inf, math.nan])
def test_a_non_finite_previous_step_fails_the_timing_check(dt):
    state = _state()
    assert state.timing_is_finite() is True

    state.previous_dt = dt

    assert state.timing_is_finite() is False


# --- the module is host-only ---------------------------------------------------


def test_the_module_needs_neither_warp_nor_blender():
    tree = ast.parse(Path(spacetime.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative (add-on) imports"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "math", "numpy"}
