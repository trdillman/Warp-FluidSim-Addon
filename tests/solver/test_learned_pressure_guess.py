"""Learned pressure initial guess (4.0 lane N, SPEC-learned-pressure-guess).

The feature is gated: ``pressure_learned_guess`` is off by default and hidden,
and no trained weights ship. These tests pin the contract that holds whatever
weights are (or are not) present:

* guess forced off is bitwise identical to the plain zero-guess solve;
* with tiny synthetic fixture weights every solve still meets tolerance;
* missing / mismatched / unreadable weights give a zero guess and a reason,
  decided once at construction (no per-frame warning);
* inference is graph-capturable: the captured graph and the eager launches
  give the same guess bit for bit, and the PCG keeps its CUDA conditional
  graph path while the guess is on;
* the Warp inference matches the numpy reference forward pass.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warp as wp  # noqa: E402

from fluidSim_warp_blender51.sim import learned_guess as lg  # noqa: E402
from fluidSim_warp_blender51.sim import learned_guess_model as lgm  # noqa: E402
from fluidSim_warp_blender51.sim.pressure import PRESSURE_STATUS_CONVERGED  # noqa: E402
import _narrow_band_harness as tank  # noqa: E402

FRAMES = 6


def _fixture_weights(path, *, seed=7, width=4, modes=2, layers=1, hidden=8,
                     schema=lgm.SCHEMA):
    weights = lgm.random_weights(width=width, modes=modes, layers=layers,
                                 proj_hidden=hidden, seed=seed)
    weights["schema"] = np.array(schema)
    lgm.save_weights(path, weights)
    return path


def _run(device, frames=FRAMES, **extra):
    solver = tank.run_tank(device, narrow_band=False, frames=frames, **extra)
    try:
        state = (solver.particle_positions().copy(),
                 solver.pressure_solver.pressure.numpy().copy())
        guess = solver.pressure_solver.learned_guess
        reason = None if guess is None else guess.reason
        message = solver.stats.learned_guess_message
    finally:
        solver.release()
    return state, reason, message


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


# -- contract 1: off == zero guess, bitwise ---------------------------------

def test_missing_weights_is_bitwise_the_zero_guess_path(cuda_device, tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", str(tmp_path / "absent.npz"))
    off, off_reason, off_msg = _run(cuda_device, pressure_learned_guess=False)
    on, on_reason, on_msg = _run(cuda_device, pressure_learned_guess=True)
    assert off_reason is None and off_msg == ""
    assert "missing" in on_reason
    assert on_msg == f"off ({on_reason})"
    assert _same(off, on)


def test_forced_off_with_valid_weights_is_bitwise_the_zero_guess_path(
        cuda_device, tmp_path, monkeypatch):
    path = _fixture_weights(str(tmp_path / "w.npz"))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    off, _, _ = _run(cuda_device, pressure_learned_guess=False)
    monkeypatch.setattr(lg, "FORCE_OFF", True)
    forced, reason, message = _run(cuda_device, pressure_learned_guess=True)
    assert reason == "forced off"
    assert message == "off (forced off)"
    assert _same(off, forced)


# -- contract 2: fixture weights still meet tolerance -----------------------

def test_fixture_weights_converge_every_solve(cuda_device, tmp_path, monkeypatch):
    path = _fixture_weights(str(tmp_path / "w.npz"))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    solver = tank.run_tank(cuda_device, narrow_band=False, frames=0,
                           pressure_learned_guess=True)
    ps = solver.pressure_solver
    statuses, guesses = [], []
    original = ps.solve_pcg

    def recording(voxel_type, max_iterations, tolerance, *args, **kwargs):
        status = original(voxel_type, max_iterations, tolerance, *args, **kwargs)
        statuses.append((ps.last_pressure_status, ps.last_residual,
                         tolerance * np.sqrt(max(1, ps._dim()))))
        guesses.append(float(np.abs(ps.x0.numpy()).max()))
        return status

    ps.solve_pcg = recording
    try:
        assert ps.learned_guess.active, ps.learned_guess.reason
        for _ in range(FRAMES):
            solver.step_frame()
        message = solver.stats.learned_guess_message
    finally:
        solver.release()
    assert message == "on"
    assert statuses
    for status, residual, bound in statuses:
        assert status == PRESSURE_STATUS_CONVERGED
        assert residual <= bound
    # the guess really reached the solver (a non-zero x0 on some solve)
    assert max(guesses) > 0.0


def test_guess_on_is_bitwise_repeatable(cuda_device, tmp_path, monkeypatch):
    path = _fixture_weights(str(tmp_path / "w.npz"))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    a, _, _ = _run(cuda_device, frames=3, pressure_learned_guess=True)
    b, _, _ = _run(cuda_device, frames=3, pressure_learned_guess=True)
    assert _same(a, b)


# -- contract 3: bad weights -> zero guess + reason, decided once ------------

@pytest.mark.parametrize("kind, needle", [
    ("missing", "missing"),
    ("schema", "schema"),
    ("garbage", "unreadable"),
    ("shape", "invalid"),
])
def test_bad_weights_fall_back_silently(cuda_device, tmp_path, monkeypatch, caplog,
                                        kind, needle):
    path = str(tmp_path / "w.npz")
    if kind == "schema":
        _fixture_weights(path, schema="learned_guess_v0")
    elif kind == "garbage":
        with open(path, "wb") as handle:
            handle.write(b"not an npz")
    elif kind == "shape":
        weights = lgm.random_weights(width=4, modes=2, layers=1, proj_hidden=8, seed=1)
        weights["lift_w"] = weights["lift_w"][:, :2]
        lgm.save_weights(path, weights)
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    solver = tank.run_tank(cuda_device, narrow_band=False, frames=0,
                           pressure_learned_guess=True)
    try:
        guess = solver.pressure_solver.learned_guess
        assert not guess.active
        assert needle in guess.reason
        messages = []
        caplog.set_level(logging.DEBUG)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                solver.step_frame()
                messages.append(solver.stats.learned_guess_message)
        assert not caught
        assert not [r for r in caplog.records if "learned" in r.getMessage().lower()]
        assert messages == [f"off ({guess.reason})"] * 3
    finally:
        solver.release()


def test_jacobi_solver_reports_reason(cuda_device, tmp_path, monkeypatch):
    from fluidSim_warp_blender51.sim.enums import ProjectNonDivergentSolver
    path = _fixture_weights(str(tmp_path / "w.npz"))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    _, _, message = _run(cuda_device, frames=1, pressure_learned_guess=True,
                         pressure_solver=int(ProjectNonDivergentSolver.JACOBI))
    assert message.startswith("off (") and "PCG" in message


# -- contract 4: graph capture ----------------------------------------------

def test_inference_graph_matches_eager_and_keeps_pcg_graph(cuda_device, tmp_path, monkeypatch):
    if not wp.get_device(cuda_device).is_cuda:
        pytest.skip("graph capture needs CUDA")
    path = _fixture_weights(str(tmp_path / "w.npz"))
    monkeypatch.setattr(lg, "DEFAULT_WEIGHTS_PATH", path)
    solver = tank.run_tank(cuda_device, narrow_band=False, frames=2,
                           pressure_learned_guess=True)
    try:
        ps = solver.pressure_solver
        guess = ps.learned_guess
        assert guess.active
        assert ps.last_pcg_execution_mode == "cuda_conditional"
        assert guess.graph_launches > 0
        voxel_type = solver.voxel_type.data
        guess.use_graph = True
        assert guess.fill(ps, voxel_type)
        captured = ps.x0.numpy().copy()
        guess.use_graph = False
        ps.x0.zero_()
        assert guess.fill(ps, voxel_type)
        eager = ps.x0.numpy().copy()
        assert np.array_equal(captured, eager)
        assert np.any(eager != 0.0)
    finally:
        solver.release()


# -- Warp inference == numpy reference --------------------------------------

def test_warp_forward_matches_numpy_reference(cuda_device):
    weights = lgm.random_weights(width=6, modes=3, layers=2, proj_hidden=8, seed=3)
    model = lgm.validate_weights(weights)
    dims = (8, 4, 2)
    rng = np.random.default_rng(5)
    feats = rng.standard_normal((lgm.IN_CHANNELS,) + dims[::-1]).astype(np.float32)
    expected = lgm.forward_numpy(model, feats)
    got = lg.forward_warp(model, feats, device=cuda_device)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-4,
                               atol=1e-4 * float(np.abs(expected).max()))


def test_model_rejects_wrong_schema_and_nonfinite():
    weights = lgm.random_weights(width=4, modes=2, layers=1, proj_hidden=4, seed=0)
    bad = dict(weights, schema=np.array("other"))
    with pytest.raises(lgm.WeightsError, match="schema"):
        lgm.validate_weights(bad)
    bad = dict(weights)
    bad["pw_w_0"] = bad["pw_w_0"].copy()
    bad["pw_w_0"][0, 0] = np.nan
    with pytest.raises(lgm.WeightsError, match="invalid"):
        lgm.validate_weights(bad)
