"""Viscosity solve stability validation (test 11)."""
import sys
from collections import OrderedDict
from types import SimpleNamespace

import pytest

wp = pytest.importorskip("warp")

from fluidSim_warp_blender51.sim.constants import LEAF_VOLUME
from fluidSim_warp_blender51.sim import viscosity
from fluidSim_warp_blender51.sim.viscosity import ViscositySolver

from _subset_runner import run_subset


def _fake_viscosity_solver(active_voxels=LEAF_VOLUME, capacity=LEAF_VOLUME * 8):
    solver = object.__new__(ViscositySolver)
    solver.grid = SimpleNamespace(
        active_voxel_count=active_voxels,
        voxel_capacity=capacity,
        leaf_table=object(),
        leaf_coords=object(),
        voxels_dim=object(),
        leaves_dim=object(),
        num_leaves=object(),
        voxel_size=0.1,
    )
    solver.device = "fake-device"
    for name in (
        "rhs", "x", "inv_diag", "r", "z", "p_dir", "ap", "tmp_u", "tmp_v",
        "tmp_w", "counter", "rz", "rz_new", "p_ap", "alpha", "beta",
        "residual", "pcg_stats", "_pcg_p_ap_accum", "_pcg_condition",
        "_pcg_iterations", "_pcg_history", "_pcg_history_pos",
    ):
        setattr(solver, name, object())
    solver._pcg_conditional_graphs = OrderedDict()
    solver._pcg_dot = None
    solver._pcg_native_available = None
    solver._conditional_available = None
    solver._graph_captures = 0
    solver.last_pcg_native_error = ""
    solver.last_pcg_execution_mode = "not_run"
    solver.last_iterations = 0
    solver.last_residual = 0.0
    solver.last_residual_squared = 0.0
    solver.last_viscosity_status = viscosity.VISCOSITY_STATUS_NOT_RUN
    solver.last_viscosity_status_code = viscosity.VISCOSITY_STATUS_CODES[
        viscosity.VISCOSITY_STATUS_NOT_RUN]
    solver.last_viscosity_failure = ""
    solver.last_component_statuses = []
    solver.last_status = ViscositySolver._make_status(
        False, viscosity.VISCOSITY_STATUS_NOT_RUN, 0.0, 0)
    return solver


def test_viscosity_smooth_scales_with_active_voxel_prefix(monkeypatch):
    solver = _fake_viscosity_solver()
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(), ww=object()
    )
    launch_dims = []
    copy_counts = []

    monkeypatch.setattr(
        viscosity.wp,
        "launch",
        lambda _kernel, dim, inputs, device: launch_dims.append(dim),
    )
    monkeypatch.setattr(
        viscosity.wp,
        "copy",
        lambda _dst, _src, count=None: copy_counts.append(count),
    )

    solver.smooth(vel, object(), object(), dt=0.01, iterations=1)

    assert launch_dims == [solver.grid.active_voxel_count]
    assert copy_counts == [solver.grid.active_voxel_count] * 3


def test_viscosity_pcg_uses_active_prefix_for_launches_reductions_and_copy(monkeypatch):
    solver = _fake_viscosity_solver()
    launch_dims = []
    reduction_counts = []
    copy_counts = []

    monkeypatch.setattr(
        viscosity.wp,
        "launch",
        lambda _kernel, dim, inputs, device: launch_dims.append(dim),
    )
    monkeypatch.setattr(
        viscosity.wp,
        "copy",
        lambda _dst, _src, count=None: copy_counts.append(count),
    )
    monkeypatch.setattr(
        viscosity.wp.utils,
        "array_inner",
        lambda _a, _b, out, count=None: reduction_counts.append(count),
    )

    solver._solve_component(
        comp=object(),
        weight=object(),
        viscosity_field=object(),
        voxel_type=object(),
        dt=0.01,
        use_pcg=True,
        iterations=0,
        tolerance=1.0e-4,
    )

    active = solver.grid.active_voxel_count
    assert solver.grid.voxel_capacity not in launch_dims
    assert set(launch_dims) == {1, active}
    assert reduction_counts == [active]
    assert copy_counts == [active]


class _FakeScalarArray:
    """Minimal stand-in for a warp scalar array used by the PCG monitors."""

    def __init__(self, size=1):
        self.values = [0.0] * size

    def numpy(self):
        return list(self.values)

    def zero_(self):
        self.values = [0.0] * len(self.values)


def _host_loop_solver(residuals, *, active_voxels=LEAF_VOLUME,
                      rz=1.0, p_ap=1.0):
    """Fake solver whose reductions replay a scripted residual sequence."""
    solver = _fake_viscosity_solver(active_voxels=active_voxels)
    solver.residual = _FakeScalarArray()
    solver.rz = _FakeScalarArray()
    solver.rz_new = _FakeScalarArray()
    solver.p_ap = _FakeScalarArray()
    solver.pcg_stats = _FakeScalarArray(5)
    solver._scripted = list(residuals)
    solver._rz_value = rz
    solver._p_ap_value = p_ap
    return solver


def _install_host_loop_stubs(monkeypatch, solver):
    def fake_inner(_a, _b, out, count=None):
        if out is solver.residual:
            out.values[0] = (solver._scripted.pop(0) if solver._scripted
                             else 0.0)
        elif out is solver.p_ap:
            out.values[0] = solver._p_ap_value
        else:
            out.values[0] = solver._rz_value

    def fake_launch(kernel, dim, inputs, device):
        if kernel is viscosity.k_pack_pcg_stats:
            residual, rz, p_ap, stats = inputs
            stats.values[0] = residual.values[0]
            stats.values[1] = rz.values[0]
            stats.values[2] = p_ap.values[0]

    monkeypatch.setattr(viscosity.wp.utils, "array_inner", fake_inner)
    monkeypatch.setattr(viscosity.wp, "launch", fake_launch)
    monkeypatch.setattr(viscosity.wp, "copy",
                        lambda _dst, _src, count=None: None)


def _run_host_loop(solver, *, iterations=40, tolerance=1.0e-4, check_every=1):
    n = solver.grid.active_voxel_count
    solver._solve_pcg_host(
        n, [object()] * 5, object(), object(), object(), 0.01,
        check_every, iterations, tolerance * tolerance * max(1, n))
    return solver.last_status


def test_viscosity_default_check_cadence_is_every_iteration():
    """The check is evaluated on device, so it no longer needs amortising."""
    assert viscosity.DEFAULT_VISCOSITY_CHECK_EVERY == 1


def test_viscosity_host_loop_reports_converged_with_residual(monkeypatch):
    solver = _host_loop_solver([1.0, 1.0e-12])
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_CONVERGED
    assert status["converged"] is True
    assert status["iterations"] == 2
    assert solver.last_residual == pytest.approx(1.0e-6)
    assert solver.last_pcg_execution_mode == "host_checked"


def test_viscosity_host_loop_reports_max_iterations(monkeypatch):
    solver = _host_loop_solver([1.0 / (2.0 ** k) for k in range(1, 5)])
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver, iterations=4)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_MAX_ITERATIONS
    assert status["converged"] is False
    assert status["iterations"] == 4
    assert solver.last_viscosity_failure == "max_iterations"


def test_viscosity_host_loop_reports_non_finite_residual(monkeypatch):
    solver = _host_loop_solver([float("nan")])
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_INVALID
    assert solver.last_viscosity_failure == "non_finite_residual"


def test_viscosity_host_loop_reports_singular_system(monkeypatch):
    solver = _host_loop_solver([1.0], p_ap=0.0)
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_SINGULAR
    assert solver.last_viscosity_failure == "singular_system"


def test_viscosity_stagnation_guard_tolerates_slow_but_real_progress(monkeypatch):
    """A consecutive-check test would misfire at the per-iteration cadence.

    Each check improves by only 1%, which a pressure-style consecutive
    comparison survives but which is exactly the regime where a naive port
    would be fragile; over the guard window the residual must fall by more
    than the epsilon, and it does.
    """
    solver = _host_loop_solver([0.99 ** k for k in range(1, 41)])
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver, iterations=40)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_MAX_ITERATIONS


def test_viscosity_stagnation_guard_fires_only_after_the_window(monkeypatch):
    window = viscosity.VISCOSITY_STAGNATION_WINDOW
    solver = _host_loop_solver([1.0] * 40)
    _install_host_loop_stubs(monkeypatch, solver)
    status = _run_host_loop(solver, iterations=40)
    assert status["reason"] == viscosity.VISCOSITY_STATUS_STALLED
    assert status["iterations"] == window + 1
    assert solver.last_viscosity_failure == "stalled"


def test_viscosity_solve_reports_the_worst_component(monkeypatch):
    solver = _fake_viscosity_solver()
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(), ww=object()
    )
    scripted = [
        viscosity.VISCOSITY_STATUS_CONVERGED,
        viscosity.VISCOSITY_STATUS_CONVERGED,
        viscosity.VISCOSITY_STATUS_MAX_ITERATIONS,
    ]

    def fake_component(*_args, **_kwargs):
        return solver._set_viscosity_status(scripted.pop(0), 1.0, 40)

    monkeypatch.setattr(solver, "_solve_component", fake_component)
    status = solver.solve(vel, object(), object(), dt=0.01)

    assert [s["reason"] for s in solver.last_component_statuses] == [
        viscosity.VISCOSITY_STATUS_CONVERGED,
        viscosity.VISCOSITY_STATUS_CONVERGED,
        viscosity.VISCOSITY_STATUS_MAX_ITERATIONS,
    ]
    assert status["reason"] == viscosity.VISCOSITY_STATUS_MAX_ITERATIONS
    assert status["converged"] is False
    assert solver.last_viscosity_status_code == (
        viscosity.VISCOSITY_STATUS_CODES[
            viscosity.VISCOSITY_STATUS_MAX_ITERATIONS])


def test_viscosity_status_codes_are_ordered_by_severity():
    codes = viscosity.VISCOSITY_STATUS_CODES
    assert codes[viscosity.VISCOSITY_STATUS_CONVERGED] < codes[
        viscosity.VISCOSITY_STATUS_MAX_ITERATIONS]
    assert codes[viscosity.VISCOSITY_STATUS_MAX_ITERATIONS] < codes[
        viscosity.VISCOSITY_STATUS_STALLED]
    assert codes[viscosity.VISCOSITY_STATUS_STALLED] < codes[
        viscosity.VISCOSITY_STATUS_SINGULAR]
    assert codes[viscosity.VISCOSITY_STATUS_SINGULAR] < codes[
        viscosity.VISCOSITY_STATUS_INVALID]


def test_viscosity_conditional_path_requires_a_divisible_budget():
    solver = _fake_viscosity_solver()
    # 40 iterations checked every 3 would overshoot the budget.
    assert solver._conditional_supported(40, 3) is False
    assert solver._conditional_supported(0, 1) is False


if __name__ == "__main__":
    sys.exit(run_subset([10], "VISCOSITY"))
