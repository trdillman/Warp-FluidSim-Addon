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
    solver.graph_scope = viscosity.default_viscosity_graph_scope()
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


def test_viscosity_solve_reports_the_slowest_of_equally_healthy_components():
    """When nothing failed, the useful report is the component that worked hardest.

    Severity alone leaves a three-way tie, and ``max`` resolves a tie by
    returning the first element -- always u. Measured on dam_break at a budget
    of 64, that reported 36.4 iterations while w was really taking 41.8, which
    understates how close the budget is to being exhausted.
    """

    solver = _fake_viscosity_solver()
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(),
        ww=object(),
    )
    scripted = [(viscosity.VISCOSITY_STATUS_CONVERGED, 36),
                (viscosity.VISCOSITY_STATUS_CONVERGED, 35),
                (viscosity.VISCOSITY_STATUS_CONVERGED, 42)]

    def fake_component(*_args, **_kwargs):
        reason, iterations = scripted.pop(0)
        return solver._set_viscosity_status(reason, 1.0, iterations)

    solver._solve_component = fake_component
    status = solver.solve(vel, object(), object(), dt=0.01)

    assert status["reason"] == viscosity.VISCOSITY_STATUS_CONVERGED
    assert status["iterations"] == 42, (
        "reported the first converged component instead of the slowest")


def test_viscosity_severity_still_outranks_iteration_count():
    """A failure with few iterations must not be hidden by a slow success."""

    solver = _fake_viscosity_solver()
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(),
        ww=object(),
    )
    scripted = [(viscosity.VISCOSITY_STATUS_CONVERGED, 60),
                (viscosity.VISCOSITY_STATUS_SINGULAR, 3),
                (viscosity.VISCOSITY_STATUS_CONVERGED, 58)]

    def fake_component(*_args, **_kwargs):
        reason, iterations = scripted.pop(0)
        return solver._set_viscosity_status(reason, 1.0, iterations)

    solver._solve_component = fake_component
    status = solver.solve(vel, object(), object(), dt=0.01)

    assert status["reason"] == viscosity.VISCOSITY_STATUS_SINGULAR
    assert status["iterations"] == 3


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



def test_smooth_reports_jacobi_rather_than_inheriting_a_stale_verdict():
    """The Jacobi smoother used to leave the previous PCG status in place.

    It runs a fixed sweep count with no residual test, so anything it reports
    about convergence is borrowed from a solve that is no longer relevant. It
    now names itself instead.
    """

    solver = _fake_viscosity_solver()
    solver._set_viscosity_status(viscosity.VISCOSITY_STATUS_CONVERGED, 0.0, 12)
    vel = SimpleNamespace(
        u=object(), v=object(), w=object(), wu=object(), wv=object(),
        ww=object(),
    )
    calls = []
    module = sys.modules[viscosity.__name__]
    real_launch, real_copy = module.wp.launch, module.wp.copy
    module.wp.launch = lambda *a, **k: calls.append("launch")
    module.wp.copy = lambda *a, **k: None
    try:
        status = solver.smooth(vel, object(), object(), dt=0.01, iterations=4)
    finally:
        module.wp.launch, module.wp.copy = real_launch, real_copy

    assert status["reason"] == viscosity.VISCOSITY_STATUS_JACOBI
    assert status["iterations"] == 4
    assert solver.last_component_statuses == []
    assert len(calls) == 4


def test_solver_records_viscosity_convergence_the_way_it_records_pressure():
    """A solve that exhausts its budget has to be visible in the frame stats.

    ``FluidSolver._pressure`` has always written its iterations and residual
    into ``stats``; ``FluidSolver._viscosity`` discarded the return value, so a
    solve that was cut off every single frame reported nothing at all.
    """

    from fluidSim_warp_blender51.sim.enums import ViscositySolver as VEnum
    from fluidSim_warp_blender51.sim.frame_runtime import FrameStats
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    class _StubSolver:
        def solve(self, *_args, **_kwargs):
            return {
                "converged": False,
                "iterations": 64,
                "reason": viscosity.VISCOSITY_STATUS_MAX_ITERATIONS,
                "residual": 0.25,
            }

    stub = SimpleNamespace(
        settings=SimpleNamespace(
            viscosity_enabled=True,
            viscosity_from_particles=True,
            viscosity_constant=1.0,
            viscosity_solver=int(VEnum.PCG),
            viscosity_iterations=64,
            viscosity_tolerance=1.0e-4,
        ),
        viscosity_solver=_StubSolver(),
        viscosity_field=SimpleNamespace(data=object()),
        voxel_type=SimpleNamespace(data=object()),
        vel=object(),
        stats=FrameStats(),
        device="cpu",
    )
    FluidSolver._viscosity(stub, 0.01)

    assert stub.stats.viscosity_status == (
        viscosity.VISCOSITY_STATUS_MAX_ITERATIONS)
    assert stub.stats.viscosity_converged is False
    assert stub.stats.viscosity_iterations == 64
    assert stub.stats.viscosity_residual == pytest.approx(0.25)

    emitted = stub.stats.as_dict()
    assert emitted["viscosityStatus"] == (
        viscosity.VISCOSITY_STATUS_MAX_ITERATIONS)
    assert emitted["viscosityConverged"] is False
    assert emitted["viscosityIterations"] == 64
    assert emitted["viscosityResidual"] == pytest.approx(0.25)


def test_disabled_viscosity_leaves_the_stats_defaults_alone():
    """No solve ran, so nothing may be reported about one."""

    from fluidSim_warp_blender51.sim.frame_runtime import FrameStats
    from fluidSim_warp_blender51.sim.solver import FluidSolver

    stub = SimpleNamespace(
        settings=SimpleNamespace(viscosity_enabled=False),
        stats=FrameStats(),
    )
    FluidSolver._viscosity(stub, 0.01)

    assert stub.stats.viscosity_status == "DISABLED"
    assert stub.stats.viscosity_iterations == 0


def test_default_iteration_budget_clears_the_measured_requirement():
    """The default has to be able to reach the default tolerance.

    Measured on dam_break at 1e-4 with the ceiling lifted to 256 (39 solves
    per component): u converges in 35-41 iterations, v in 34-39, w in 41-46.
    The old default of 40 sat below w's *minimum*, so w was truncated on every
    solve. This guards the relationship, not the specific number: raising the
    tolerance or changing the preconditioner may lower the requirement, but
    the budget must never drop back beneath it.
    """

    import ast
    import pathlib

    from fluidSim_warp_blender51.sim import constants as C

    measured_worst_case = 46
    assert C.DEFAULT_VISCOSITY_ITERATIONS >= measured_worst_case
    # The conditional-graph path refuses a budget the check cadence cannot
    # divide -- the terminal check has to land exactly on the budget -- so a
    # budget that breaks that relation silently falls back to the host loop.
    assert C.DEFAULT_VISCOSITY_ITERATIONS % (
        viscosity.DEFAULT_VISCOSITY_CHECK_EVERY) == 0
    # The UI default is derived from the constant rather than duplicating it.
    # Checked in the source rather than by import, because reaching the
    # property object needs the bpy stub this module does not install -- and
    # what is being guarded is the absence of a second hard-coded literal,
    # which is a source-level property anyway.
    source = pathlib.Path(
        C.__file__).resolve().parent.parent / "properties.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    defaults = [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "viscosity_iterations"
        for kw in getattr(node.annotation, "keywords", [])
        if kw.arg == "default"
    ]
    assert len(defaults) == 1
    assert isinstance(defaults[0], ast.Attribute), (
        "the UI default is a literal again; it must reference "
        "DEFAULT_VISCOSITY_ITERATIONS")
    assert defaults[0].attr == "DEFAULT_VISCOSITY_ITERATIONS"

if __name__ == "__main__":
    sys.exit(run_subset([10], "VISCOSITY"))


# -- graph scope -------------------------------------------------------------


class _PtrArray:
    """Array stand-in carrying the identity ``_graph_key`` keys on."""

    _counter = 0

    def __init__(self):
        _PtrArray._counter += 1
        self.ptr = _PtrArray._counter


class _FakeGraph:
    """Stand-in for a captured ``wp.Graph``."""


class _CaptureRecorder:
    """Replay the device ops a component solve enqueues, in execution order.

    Under ``loop`` scope most of a solve is issued from the host and only the
    PCG loop is captured; under ``component`` scope nearly all of it is
    recorded into a graph that one ``capture_launch`` replays. Expanding the
    launch of a graph back into the ops recorded inside it puts both scopes
    into the same units, which is what makes them comparable: the paired
    benchmark is only meaningful if the two modes differ in host cost and in
    nothing else.
    """

    def __init__(self):
        self.stack = []
        self.graphs = {}
        self.trace = []

    def op(self, *entry):
        sink = self.graphs[self.stack[-1]] if self.stack else self.trace
        sink.append(entry)

    def scoped_capture(self, device=None):
        return _FakeScopedCapture(self)

    def expand(self, ops=None):
        out = []
        for entry in (self.trace if ops is None else ops):
            if entry[0] == "capture_launch":
                out.extend(self.expand(self.graphs[entry[1]]))
            elif entry[0] == "capture_while":
                out.append(("while", tuple(self.expand(self.graphs[entry[1]]))))
            else:
                out.append(entry)
        return out


class _FakeScopedCapture:
    def __init__(self, recorder):
        self._recorder = recorder
        self.graph = _FakeGraph()

    def __enter__(self):
        self._recorder.graphs[self.graph] = []
        self._recorder.stack.append(self.graph)
        return self

    def __exit__(self, *_exc):
        self._recorder.stack.pop()
        return False


class _RecordingScalarArray(_FakeScalarArray):
    def __init__(self, recorder, name, size=1):
        super().__init__(size)
        self._recorder = recorder
        self._name = name

    def zero_(self):
        super().zero_()
        self._recorder.op("zero", self._name)


def _scope_solver(recorder, scope):
    solver = _fake_viscosity_solver()
    solver.graph_scope = scope
    # the conditional path is available and its reduction is already built,
    # so `_conditional_supported` short-circuits without touching a device
    solver._conditional_available = True
    solver._pcg_native_available = True
    solver._pcg_dot = SimpleNamespace(
        launch=lambda _a, _b, _out, n: recorder.op("dot", n))
    for name in ("rz", "rz_new", "residual", "_pcg_p_ap_accum"):
        setattr(solver, name, _RecordingScalarArray(recorder, name))
    solver.pcg_stats = _FakeScalarArray(5)
    for name in ("leaf_table", "leaf_coords", "num_leaves"):
        setattr(solver.grid, name, _PtrArray())
    return solver


def _install_trace_stubs(monkeypatch, recorder):
    monkeypatch.setattr(
        viscosity.wp, "launch",
        lambda kernel, dim, inputs, device=None: recorder.op(
            "launch", getattr(kernel, "key", kernel), dim))
    monkeypatch.setattr(
        viscosity.wp, "copy",
        lambda _dst, _src, count=None: recorder.op("copy", count))
    monkeypatch.setattr(viscosity.wp, "synchronize_device",
                        lambda _device=None: None)
    monkeypatch.setattr(viscosity.wp, "ScopedCapture", recorder.scoped_capture)
    monkeypatch.setattr(
        viscosity.wp, "capture_while",
        lambda condition, while_body: recorder.op("capture_while", while_body))
    monkeypatch.setattr(
        viscosity.wp, "capture_launch",
        lambda graph: recorder.op("capture_launch", graph))


def _scope_trace(monkeypatch, scope, *, iterations=2, check_every=1):
    """Record the ops one component solve enqueues under ``scope``."""
    recorder = _CaptureRecorder()
    solver = _scope_solver(recorder, scope)
    _install_trace_stubs(monkeypatch, recorder)
    fields = tuple(_PtrArray() for _ in range(4))
    solver._solve_component(*fields, 0.01, True, iterations, 1.0e-4,
                            check_every)
    return solver, recorder, fields


def test_viscosity_graph_scope_default_is_the_whole_component():
    assert (viscosity.default_viscosity_graph_scope()
            == viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)


def test_viscosity_graph_scope_rejects_unknown_modes():
    original = viscosity.default_viscosity_graph_scope()
    try:
        with pytest.raises(ValueError):
            viscosity.set_default_viscosity_graph_scope("whole-frame")
        viscosity.set_default_viscosity_graph_scope(
            viscosity.VISCOSITY_GRAPH_SCOPE_LOOP)
        assert (viscosity.default_viscosity_graph_scope()
                == viscosity.VISCOSITY_GRAPH_SCOPE_LOOP)
    finally:
        viscosity.set_default_viscosity_graph_scope(original)


def test_viscosity_graph_scopes_enqueue_the_same_device_work(monkeypatch):
    """The two scopes must differ in host cost and in nothing else.

    Every claim the paired benchmark can make rests on this: same kernels,
    same launch dimensions, same order, same reductions, same write-back.
    """
    _, loop, _ = _scope_trace(monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_LOOP)
    _, component, _ = _scope_trace(
        monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)
    assert component.expand() == loop.expand()


def test_viscosity_component_scope_issues_one_host_call(monkeypatch):
    """The point of the change: one enqueue per solve instead of sixteen."""
    _, loop, _ = _scope_trace(monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_LOOP)
    _, component, _ = _scope_trace(
        monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)
    assert [op[0] for op in component.trace] == ["capture_launch"]
    assert len(loop.trace) == 16
    # the work itself did not shrink; it moved into the graph
    assert len(component.expand()) == len(loop.expand()) == 16


def test_viscosity_component_scope_writes_back_inside_the_graph(monkeypatch):
    """``comp`` is written last, so an aborted capture cannot corrupt it."""
    _, component, _ = _scope_trace(
        monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)
    ops = component.expand()
    assert ops[-1] == ("copy", LEAF_VOLUME)
    assert ops[-2][:2] == ("launch", "k_viscosity_apply_collision")


def test_viscosity_component_scope_graph_key_separates_components(monkeypatch):
    """The captured body now reads and writes ``comp``, so it must be keyed."""
    solver, _, _ = _scope_trace(
        monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)
    args = (12, _PtrArray(), _PtrArray(), _PtrArray(), 0.01, 1, 40, 1.0e-8)
    comp_u, comp_v = _PtrArray(), _PtrArray()
    key_u = solver._graph_key(
        viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT, *args, comp=comp_u)
    key_v = solver._graph_key(
        viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT, *args, comp=comp_v)
    key_loop = solver._graph_key(
        viscosity.VISCOSITY_GRAPH_SCOPE_LOOP, *args, comp=comp_u)
    assert key_u != key_v, "two components would share one captured graph"
    assert key_u != key_loop, "the two scopes would share one captured graph"


def test_viscosity_component_scope_caches_its_graph(monkeypatch):
    """A repeat solve of the same shape must not recapture."""
    solver, _, fields = _scope_trace(
        monkeypatch, viscosity.VISCOSITY_GRAPH_SCOPE_COMPONENT)
    assert solver._graph_captures == 1
    for _ in range(3):
        solver._solve_component(*fields, 0.01, True, 2, 1.0e-4, 1)
    assert solver._graph_captures == 1
    # a different component is a different shape, and must not reuse it
    solver._solve_component(_PtrArray(), *fields[1:], 0.01, True, 2, 1.0e-4, 1)
    assert solver._graph_captures == 2
