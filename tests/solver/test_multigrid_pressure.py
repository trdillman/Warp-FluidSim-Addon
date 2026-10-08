"""Acceptance tests for multigrid-preconditioned CG (3.0 lane C, spec
``docs/specs/3.0/SPEC-multigrid-pressure.md``).

The interface these tests pin:

* ``PressureSolver.preconditioner`` is a plain attribute, ``"JACOBI"`` (the
  default) or ``"MULTIGRID"``.
* ``PressureSolver._apply_mg_vcycle(r, z)`` applies ONE symmetric V-cycle to
  full-layout float arrays: ``z = M r`` (overwrites ``z``; only the fluid
  unknown entries of ``z`` are defined on return).
* ``_mg_pre_smooth`` / ``_mg_post_smooth`` are read at call time and can be
  overridden per instance (the module constants ``MG_PRE_SMOOTH``,
  ``MG_POST_SMOOTH`` and ``MG_OMEGA`` are the defaults).
* Setup contract used here: construct the solver, set
  ``solver.preconditioner = "MULTIGRID"`` (and any per-instance smoothing
  override), then ``build_rhs`` and ``build_preconditioner(voxel_type)``, and
  only then call ``_apply_mg_vcycle`` / ``solve_pcg``.
* ``solve_pcg(..., check_every=None)``: JACOBI keeps the historical interval
  of 25; MULTIGRID picks its own (5 for ``max_iterations=200``) so a handful
  of V-cycles are not rounded up to 25. An explicit value is always honored.
* Naming convention the source test relies on: every kernel the V-cycle
  launches is named ``k_mg_*`` (shared helpers such as ``k_apply_matrix`` may
  also be launched; they are scanned too). Kernels that reach the iteration
  through recorded launch commands are built in ``_prepare_update_cmds`` /
  ``_prepare_pcg_commands``; the scanner covers those builders as well.
"""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warp as wp  # noqa: E402  (no importorskip: a missing Warp must error)

from fluidSim_warp_blender51.sim import pressure as pressure_module  # noqa: E402
from fluidSim_warp_blender51.sim.constants import (  # noqa: E402
    LEAF_VOLUME,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_NONE,
)
from fluidSim_warp_blender51.sim.enums import ProjectNonDivergentSolver  # noqa: E402
from fluidSim_warp_blender51.sim.fields import MacVelocityField, SparseGrid  # noqa: E402
from fluidSim_warp_blender51.sim.pressure import (  # noqa: E402
    MG_OMEGA,
    MG_POST_SMOOTH,
    MG_PRE_SMOOTH,
    PressureSolver,
)

N = 32                      # 32^3 box = 4x4x4 leaves
SYMMETRY_TOL = 1.0e-6       # brief: |<Mr1,r2>-<r1,Mr2>| / (|Mr1||r2|)
CONVERGE_TOL = 1.0e-5       # brief test 6: PCG tolerance
# The brief asks for relative L2 <= 1e-4 between the MG and Jacobi projected
# velocities "at tolerance 1e-5". Calibration (float64 reference V-cycle on
# this exact problem) shows that is infeasible: Jacobi PCG stopped at 1e-5 is
# itself ~5.6e-4 away from the exact solve, so two correct solvers stopped at
# 1e-5 differ by ~7e-4. The agreement check therefore compares at 1e-6, where
# the same estimate is ~5e-5 (2x margin to the 1e-4 bound).
AGREE_TOL = 1.0e-6
VELOCITY_REL_L2 = 1.0e-4
MIN_ITERATION_RATIO = 3.0   # spec Final target
GARBAGE = 12345.0           # pre-fill for z: a V-cycle must overwrite it

PRESSURE_SOURCE = Path(pressure_module.__file__)


# ---------------------------------------------------------------------------
# Synthetic problem: 32^3 fluid box, rippled free surface (air above), a
# baffle wall and a cube obstacle of VOXEL_COLLISION cells.
# ---------------------------------------------------------------------------

class _Case:
    """One solver wired to the shared problem."""

    def __init__(self, solver, vel, voxel_type, collision_vel):
        self.solver = solver
        self.vel = vel
        self.vtw = voxel_type
        self.cvel = collision_vel


class _Problem:
    def __init__(self, device):
        self.device = device
        grid = SparseGrid(origin=(0, 0, 0), size=(1.6, 1.6, 1.6),
                          voxel_size=0.05, leaf_capacity=64, device=device)
        grid.begin_topology()
        grid.mark_world_box((0, 0, 0), (1.599, 1.599, 1.599))
        nleaf = grid.finalize_topology()
        assert nleaf == 64 and grid.voxel_capacity == N ** 3
        self.grid = grid
        self.full_n = int(grid.active_voxel_count)

        coords = grid.leaf_coords.numpy()[:nleaf]
        slot_of = {tuple(int(v) for v in c): s for s, c in enumerate(coords)}
        x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
        slot = np.empty((N, N, N), dtype=np.int64)
        for ix in range(N // 8):
            for iy in range(N // 8):
                for iz in range(N // 8):
                    slot[ix * 8:(ix + 1) * 8, iy * 8:(iy + 1) * 8,
                         iz * 8:(iz + 1) * 8] = slot_of[(ix, iy, iz)]
        self._flat = slot * LEAF_VOLUME + (x & 7) + 8 * (y & 7) + 64 * (z & 7)
        assert len(np.unique(self._flat)) == N ** 3

        surface = 20 + np.rint(3 * np.sin(x * 0.35) * np.cos(y * 0.27)).astype(int)
        vt = np.full((N, N, N), VOXEL_FLUID, dtype=np.int32)
        vt[z >= surface] = VOXEL_NONE
        vt[(x >= 13) & (x < 16) & (y >= 4) & (y < 28) & (z < 18)] = VOXEL_COLLISION
        vt[(x >= 4) & (x < 9) & (y >= 4) & (y < 9) & (z < 9)] = VOXEL_COLLISION
        self.vt3 = vt
        self.fluid3 = vt == VOXEL_FLUID
        assert int(self.fluid3.sum()) > 15000          # a real pressure problem
        assert int((vt == VOXEL_COLLISION).sum()) > 1000
        assert int((vt == VOXEL_NONE).sum()) > 5000    # free-surface cells exist
        self.fluid_mask = self.to_flat(self.fluid3, bool)
        self.vt_flat = self.to_flat(vt, np.int32)

        rng = np.random.default_rng(11)

        def smooth_field():
            f = np.zeros((N, N, N))
            for _ in range(4):
                k = rng.integers(1, 4, size=3)
                ph = rng.uniform(0, 2 * np.pi, size=3)
                f += rng.normal() * (np.sin(2 * np.pi * k[0] * x / N + ph[0])
                                     * np.sin(2 * np.pi * k[1] * y / N + ph[1])
                                     * np.sin(2 * np.pi * k[2] * z / N + ph[2]))
            return f + 0.05 * rng.normal(size=(N, N, N))

        self.velocity_fields = [smooth_field() for _ in range(3)]

    # -- layout helpers ----------------------------------------------------

    def to_flat(self, a3, dtype):
        out = np.zeros(self.grid.voxel_capacity, dtype=dtype)
        out[self._flat.ravel()] = np.asarray(a3).ravel()
        return out

    def random_unknowns(self, seed):
        """Random float32 vector supported on the fluid unknowns only."""
        rng = np.random.default_rng(seed)
        return self.to_flat(rng.standard_normal((N, N, N)) * self.fluid3, np.float32)

    def smooth_unknowns(self):
        return self.to_flat(self.velocity_fields[0] * self.fluid3, np.float32)

    def spike_unknowns(self):
        """Unit impulse in a fluid cell touching the baffle wall (x == 12)."""
        assert self.vt3[12, 10, 10] == VOXEL_FLUID
        a3 = np.zeros((N, N, N))
        a3[12, 10, 10] = 1.0
        return self.to_flat(a3, np.float32)

    # -- solver factory ------------------------------------------------------

    def new_case(self, preconditioner=None, pre_smooth=None, post_smooth=None):
        """Fresh solver on the shared problem, RHS built, preconditioner built.

        ``preconditioner=None`` leaves the class default untouched. Smoothing
        overrides are set before ``build_preconditioner`` (valid whether the
        implementation reads them at build time or at V-cycle time).
        """
        dev = self.device
        solver = PressureSolver(self.grid, dev)
        if preconditioner is not None:
            solver.preconditioner = preconditioner
        if pre_smooth is not None:
            solver._mg_pre_smooth = pre_smooth
        if post_smooth is not None:
            solver._mg_post_smooth = post_smooth
        vel = MacVelocityField(self.grid)
        for comp, field in zip((vel.u, vel.v, vel.w), self.velocity_fields):
            comp.assign(self.to_flat(field, np.float32))
        for weight in (vel.wu, vel.wv, vel.ww):
            weight.fill_(1.0)
        vtw = wp.array(self.vt_flat, dtype=wp.int32, device=dev)
        cvel = wp.zeros(self.grid.voxel_capacity, dtype=wp.vec3, device=dev)
        counts = wp.zeros(self.grid.voxel_capacity, dtype=wp.int32, device=dev)
        solver.build_rhs(vel, vtw, cvel, counts)
        solver.build_preconditioner(vtw)
        return _Case(solver, vel, vtw, cvel)

    def apply_vcycle(self, case, r_np, z_fill=0.0):
        """Run one uncaptured V-cycle; returns the full-layout z as numpy."""
        r = wp.array(r_np, dtype=float, device=self.device)
        z = wp.array(np.full_like(r_np, z_fill), dtype=float, device=self.device)
        case.solver._apply_mg_vcycle(r, z)
        wp.synchronize_device(self.device)
        return z.numpy().copy()

    def projected_velocity(self, case):
        case.solver.project(case.vel, case.vtw, case.cvel)
        return np.concatenate([
            comp.numpy()[self.fluid_mask].astype(np.float64)
            for comp in (case.vel.u, case.vel.v, case.vel.w)])


@pytest.fixture(scope="module")
def problem(cuda_device):
    return _Problem(cuda_device)


# ---------------------------------------------------------------------------
# float64 host helpers (brief: dot products and norms from .numpy() copies)
# ---------------------------------------------------------------------------

def _dot(a, b):
    return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


def _norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)))


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _symmetry_error(problem, case, seeds=(101, 102)):
    """|<M r1,r2> - <r1,M r2>| / (|M r1| |r2|), on the fluid unknowns."""
    m = problem.fluid_mask
    r1 = problem.random_unknowns(seeds[0])
    r2 = problem.random_unknowns(seeds[1])
    z1 = problem.apply_vcycle(case, r1, z_fill=GARBAGE)[m]
    z2 = problem.apply_vcycle(case, r2, z_fill=GARBAGE)[m]
    assert np.all(np.isfinite(z1)) and np.all(np.isfinite(z2))
    den = _norm(z1) * _norm(r2[m])
    assert den > 0.0, "V-cycle returned zero for a non-zero residual"
    return abs(_dot(z1, r2[m]) - _dot(r1[m], z2)) / den


# ---------------------------------------------------------------------------
# 1-3. SPD preconditioner
# ---------------------------------------------------------------------------

def test_vcycle_is_symmetric(problem):
    case = problem.new_case("MULTIGRID")
    err = _symmetry_error(problem, case)
    print(f"V-cycle symmetry rel err (2 pre / 2 post): {err:.3e}")
    assert err <= SYMMETRY_TOL, (
        f"V-cycle not symmetric: {err:.3e} > {SYMMETRY_TOL:.0e}")


def test_unequal_smoothing_is_detected_as_asymmetric(problem):
    # Positive control first: the harness itself must pass with 2/2, otherwise
    # "1 pre + 3 post fails" would be satisfied by any broken V-cycle.
    control = problem.new_case("MULTIGRID", pre_smooth=2, post_smooth=2)
    control_err = _symmetry_error(problem, control)
    assert control_err <= SYMMETRY_TOL, (
        f"control (2 pre / 2 post) not symmetric: {control_err:.3e}")
    skewed = problem.new_case("MULTIGRID", pre_smooth=1, post_smooth=3)
    skewed_err = _symmetry_error(problem, skewed)
    print(f"V-cycle symmetry rel err (1 pre / 3 post): {skewed_err:.3e}")
    assert skewed_err > SYMMETRY_TOL, (
        f"1 pre / 3 post must FAIL the symmetry check but measured {skewed_err:.3e}")


def test_vcycle_is_positive_definite(problem):
    case = problem.new_case("MULTIGRID")
    m = problem.fluid_mask
    vectors = {
        "random-201": problem.random_unknowns(201),
        "random-202": problem.random_unknowns(202),
        "random-203": problem.random_unknowns(203),
        "smooth": problem.smooth_unknowns(),
        "spike-at-wall": problem.spike_unknowns(),
    }
    for name, r in vectors.items():
        z = problem.apply_vcycle(case, r, z_fill=GARBAGE)
        rayleigh = _dot(z[m], r[m])
        assert np.all(np.isfinite(z[m])), name
        assert rayleigh > 0.0, f"<M r, r> = {rayleigh:.3e} for {name}"


# ---------------------------------------------------------------------------
# 4. CUDA graph capture
# ---------------------------------------------------------------------------

def test_vcycle_capture_replays_bitwise_identical(problem, cuda_device):
    case = problem.new_case("MULTIGRID")
    m = problem.fluid_mask
    r = wp.array(problem.random_unknowns(301), dtype=float, device=cuda_device)
    z = wp.array(np.full(problem.grid.voxel_capacity, GARBAGE, dtype=np.float32),
                 dtype=float, device=cuda_device)

    # Uncaptured reference call (also loads the Warp modules, which cannot be
    # compiled while a capture is open).
    case.solver._apply_mg_vcycle(r, z)
    wp.synchronize_device(cuda_device)
    reference = z.numpy().copy()[m]
    assert np.all(np.isfinite(reference)) and _norm(reference) > 0.0

    with wp.ScopedCapture(device=cuda_device) as capture:
        case.solver._apply_mg_vcycle(r, z)

    for replay in (1, 2):
        z.fill_(GARBAGE)   # a replay must fully rewrite z, not rely on stale data
        wp.capture_launch(capture.graph)
        wp.synchronize_device(cuda_device)
        assert np.array_equal(_bits(z.numpy()[m]), _bits(reference)), (
            f"graph replay {replay} differs from the uncaptured V-cycle")


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------

def test_vcycle_is_bitwise_deterministic(problem):
    m = problem.fluid_mask
    r = problem.random_unknowns(401)
    first = problem.new_case("MULTIGRID")
    second = problem.new_case("MULTIGRID")
    z_first = problem.apply_vcycle(first, r, z_fill=0.0)[m]
    # Different stale contents in z: M r is an overwrite, not an accumulate.
    z_second = problem.apply_vcycle(second, r, z_fill=GARBAGE)[m]
    assert np.all(np.isfinite(z_first)) and _norm(z_first) > 0.0
    assert np.array_equal(_bits(z_first), _bits(z_second)), (
        "two fresh solvers disagree bitwise")
    # And the same solver, repeated.
    for _ in range(3):
        again = problem.apply_vcycle(first, r, z_fill=GARBAGE)[m]
        assert np.array_equal(_bits(z_first), _bits(again)), (
            "repeat call on the same solver is not bitwise reproducible")


# ---------------------------------------------------------------------------
# 6. Convergence on a box with an obstacle
# ---------------------------------------------------------------------------

def _min_iterations(problem, preconditioner, tol, limit=512):
    """Smallest m such that exactly m PCG iterations reach ``tol``.

    ``check_every == max_iterations == m`` runs exactly one batch of m
    iterations and classifies once, so the count is not quantised by the
    production check interval. (A small ``check_every`` is not used for
    Jacobi: the CG residual 2-norm is non-monotone and the stall rule
    ``res_sq >= last * (1 - 1e-6)`` falsely reports "stalled".)
    """
    tol_sq = tol * tol * problem.full_n

    def reaches(m):
        case = problem.new_case(preconditioner)
        try:
            case.solver.solve_pcg(case.vtw, m, tol, check_every=m)
            return case.solver.last_residual_squared < tol_sq
        finally:
            case.solver.release()

    hi = 1
    while not reaches(hi):
        hi *= 2
        assert hi <= limit, f"{preconditioner} did not reach {tol:g} in {limit} iterations"
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid
    return hi


def test_multigrid_pcg_converges_to_tolerance(problem, record_property):
    # A 40-iteration budget is below the ~123 iterations Jacobi needs here, and
    # equals the spec's 3x target (123 / 3 = 41). check_every=5 is odd and
    # divides 40, so the solve must run as a CUDA conditional graph: this also
    # proves the V-cycle is captured, not silently run on a host-batched
    # fallback after a capture failure.
    case = problem.new_case("MULTIGRID")
    status = case.solver.solve_pcg(case.vtw, 40, CONVERGE_TOL, check_every=5)
    tol_sq = CONVERGE_TOL * CONVERGE_TOL * problem.full_n
    record_property("multigrid_natural_iterations", case.solver.last_iterations)
    print(f"MULTIGRID natural: {status['reason']} it={case.solver.last_iterations} "
          f"mode={case.solver.last_pcg_execution_mode} res={case.solver.last_residual:.3e}")
    assert status["converged"], f"MULTIGRID PCG did not converge: {status}"
    assert case.solver.last_residual_squared < tol_sq
    assert case.solver.last_iterations <= 40
    assert case.solver.last_pcg_execution_mode == "cuda_conditional", (
        f"expected the captured conditional graph, got "
        f"{case.solver.last_pcg_execution_mode!r} "
        f"({case.solver.last_pcg_native_error!r})")


def test_multigrid_needs_3x_fewer_iterations_than_jacobi(problem, record_property):
    jacobi = _min_iterations(problem, "JACOBI", CONVERGE_TOL)
    multigrid = _min_iterations(problem, "MULTIGRID", CONVERGE_TOL)
    ratio = jacobi / multigrid
    record_property("jacobi_iterations", jacobi)
    record_property("multigrid_iterations", multigrid)
    record_property("iteration_ratio", round(ratio, 2))
    print(f"iterations to {CONVERGE_TOL:g}: JACOBI {jacobi} vs MULTIGRID {multigrid} "
          f"(ratio {ratio:.2f}x)")
    assert ratio >= MIN_ITERATION_RATIO, (
        f"JACOBI {jacobi} vs MULTIGRID {multigrid} iterations: "
        f"{ratio:.2f}x < {MIN_ITERATION_RATIO}x")


def test_multigrid_velocity_agrees_with_jacobi(problem, record_property):
    # Jacobi uses the production check interval (25, odd, divides 200);
    # MULTIGRID uses 5 because it converges in a handful of iterations.
    jacobi = problem.new_case("JACOBI")
    st_j = jacobi.solver.solve_pcg(jacobi.vtw, 200, AGREE_TOL, check_every=25)
    mg = problem.new_case("MULTIGRID")
    st_m = mg.solver.solve_pcg(mg.vtw, 200, AGREE_TOL, check_every=5)
    assert st_j["converged"] and st_m["converged"], (st_j, st_m)
    # Agreement between two runs is only meaningful if they really used
    # different preconditioners.
    assert mg.solver.last_iterations < jacobi.solver.last_iterations, (
        f"MULTIGRID ran {mg.solver.last_iterations} iterations vs JACOBI "
        f"{jacobi.solver.last_iterations}: the preconditioner was not switched")
    v_j = problem.projected_velocity(jacobi)
    v_m = problem.projected_velocity(mg)
    rel = _norm(v_m - v_j) / _norm(v_j)
    record_property("velocity_rel_l2", rel)
    print(f"projected velocity MG vs JACOBI at tol {AGREE_TOL:g}: rel L2 = {rel:.3e}")
    assert rel <= VELOCITY_REL_L2, f"velocity rel L2 {rel:.3e} > {VELOCITY_REL_L2:.0e}"


# ---------------------------------------------------------------------------
# 6b. Production entry and graph-key isolation
# ---------------------------------------------------------------------------

def _production_settings():
    """The fields ``project_frame`` reads, at the production defaults
    (``DEFAULT_PRESSURE_ITERATIONS`` = 200, ``DEFAULT_PRESSURE_TOLERANCE`` =
    1e-5). 200 is what makes the MULTIGRID interval 5."""
    return SimpleNamespace(
        pressure_solver=int(ProjectNonDivergentSolver.PCG),
        pressure_iterations=200,
        pressure_tolerance=CONVERGE_TOL,
        pressure_threshold=0.0,
        count_stiffness=0.0,
        particles_per_voxel=8.0,
    )


def _project_frame(problem, case):
    counts = wp.zeros(problem.grid.voxel_capacity, dtype=wp.int32,
                      device=problem.device)
    return case.solver.project_frame(
        case.vel, case.vtw, case.cvel, counts, _production_settings(),
        max_gradient=1.0e6)


def test_production_entry_uses_the_multigrid_check_interval(problem):
    """``project_frame`` -> ``solve_pcg`` with NO explicit ``check_every``.

    Every other MULTIGRID test passes ``check_every`` by hand. This one goes
    through the call a real frame makes, so the interval override inside
    ``pressure.py`` is what is under test: with the historical 25 a 13-iteration
    solve would be quantised up to 25 (and the iterations would be ``>= 25``).
    """
    mg = problem.new_case("MULTIGRID")
    status = _project_frame(problem, mg)
    solver = mg.solver
    print(f"project_frame MULTIGRID: it={solver.last_iterations} "
          f"check_every={solver.last_pcg_check_every} "
          f"mode={solver.last_pcg_execution_mode}")
    assert status["converged"], status
    assert solver.last_residual_squared < (
        CONVERGE_TOL * CONVERGE_TOL * problem.full_n)
    assert solver.last_pcg_check_every == 5
    assert 0 < solver.last_iterations < 25, (
        f"{solver.last_iterations} iterations: the MULTIGRID interval did not apply")
    assert solver.last_pcg_execution_mode == "cuda_conditional", (
        f"expected the captured conditional graph, got "
        f"{solver.last_pcg_execution_mode!r} ({solver.last_pcg_native_error!r})")

    # JACOBI through the same entry keeps today's behaviour: interval 25.
    jacobi = problem.new_case("JACOBI")
    status_j = _project_frame(problem, jacobi)
    assert status_j["converged"], status_j
    assert jacobi.solver.last_pcg_check_every == 25
    assert jacobi.solver.last_iterations >= 25
    assert jacobi.solver.last_iterations % 25 == 0   # quantised by the interval


def test_graph_keys_isolate_the_preconditioner_on_one_solver(problem):
    """JACOBI -> MULTIGRID -> JACOBI on ONE solver must never replay a graph
    captured for the other preconditioner.

    A MULTIGRID solve replaying JACOBI's cached graph would silently run the
    Jacobi iteration (the Jacobi iteration count); a JACOBI solve replaying
    the MULTIGRID graph would run V-cycles. Both graph caches key on the
    preconditioner, so this holds for the conditional graphs a solve uses and
    for the host-batched graphs of the fallback path.
    """
    m = problem.fluid_mask

    def fresh_reference(preconditioner):
        ref = problem.new_case(preconditioner)
        ref.solver.solve_pcg(ref.vtw, 200, CONVERGE_TOL)
        out = (ref.solver.last_iterations, ref.solver.pressure.numpy()[m].copy())
        ref.solver.release()
        return out

    jacobi_its, jacobi_p = fresh_reference("JACOBI")
    mg_its, mg_p = fresh_reference("MULTIGRID")
    assert mg_its < jacobi_its, (jacobi_its, mg_its)   # distinguishable

    case = problem.new_case("JACOBI")
    solver = case.solver

    def solve():
        status = solver.solve_pcg(case.vtw, 200, CONVERGE_TOL)
        assert status["converged"], status
        assert solver.last_pcg_execution_mode == "cuda_conditional"
        return solver.last_iterations, solver.pressure.numpy()[m].copy()

    its, p = solve()                               # 1. JACOBI, graph captured
    assert its == jacobi_its and np.array_equal(_bits(p), _bits(jacobi_p))
    assert solver.last_pcg_check_every == 25

    solver.preconditioner = "MULTIGRID"            # 2. same solver, MULTIGRID
    solver.build_preconditioner(case.vtw)
    its, p = solve()
    assert its == mg_its, (
        f"MULTIGRID solve ran {its} iterations on a solver that had a JACOBI "
        f"graph cached; a fresh MULTIGRID solver needs {mg_its} "
        f"(JACOBI needs {jacobi_its})")
    assert np.array_equal(_bits(p), _bits(mg_p))
    assert solver.last_pcg_check_every == 5

    solver.preconditioner = "JACOBI"               # 3. back to JACOBI
    solver.build_preconditioner(case.vtw)
    its, p = solve()
    assert its == jacobi_its
    assert np.array_equal(_bits(p), _bits(jacobi_p)), (
        "JACOBI after a MULTIGRID solve is not bitwise equal to a fresh JACOBI solve")
    assert solver.last_pcg_check_every == 25

    # Both captures coexist, told apart by key: Jacobi keys keep the original
    # (work_n, batch, max_iterations) shape, MULTIGRID keys carry the tail.
    keys = list(solver._pcg_conditional_graphs)
    jacobi_keys = [k for k in keys if len(k) == 3]
    mg_keys = [k for k in keys if len(k) > 3]
    assert [k[1:] for k in jacobi_keys] == [(25, 200)], keys
    assert len(mg_keys) == 1 and mg_keys[0][3] == "MG" and mg_keys[0][1:3] == (5, 200), keys

    # The host-batched cache (fallback path) is keyed the same way.
    full_n = solver._dim()
    work_n = min(full_n, 1 << (solver._build_pressure_index(case.vtw) - 1).bit_length())
    before = set(solver._pcg_graphs)
    seen = {}
    for kind in ("JACOBI", "MULTIGRID"):
        solver.preconditioner = kind
        solver.build_preconditioner(case.vtw)
        solver._prepare_pcg_commands(work_n, case.vtw, solver.pressure)
        solver.rz, solver.rz_new = solver._pcg_rz_a, solver._pcg_rz_b
        solver._launch_pcg_batch(work_n, 2)
        new = set(solver._pcg_graphs) - before - set(seen.values())
        assert len(new) == 1, (kind, new)
        seen[kind] = new.pop()
    wp.synchronize_device(problem.device)
    assert seen["JACOBI"] == (work_n, 2, 0)
    assert seen["MULTIGRID"][:3] == (work_n, 2, 0) and seen["MULTIGRID"][3] == "MG"
    solver.release()


# ---------------------------------------------------------------------------
# 7. Source test: no atomics, no host sync on the per-iteration path
# ---------------------------------------------------------------------------

_ATOMIC_NAMES = frozenset({"atomic_add", "atomic_sub"})
_SYNC_NAMES = frozenset({"numpy", "synchronize", "synchronize_device",
                         "synchronize_stream", "synchronize_event"})


def _is_warp_decorated(fn):
    for dec in fn.decorator_list:
        text = ast.unparse(dec)
        if text.startswith(("wp.kernel", "wp.func")):
            return True
    return False


def _call_name(node):
    """Trailing identifier of a call target (``wp.atomic_add`` -> atomic_add)."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


# Not every kernel of an MG iteration is launched by the V-cycle's own code:
# the x/r update and the compaction of z are recorded launch commands
# (``fast_launch(..., record_cmd=True)``) built once in these methods and
# replayed from ``_record_pcg_iterations`` inside the captured graph. The
# scanner follows them too, so such a kernel cannot hide from the checks.
RECORDED_CMD_BUILDERS = ("_prepare_update_cmds", "_prepare_pcg_commands")
CAPTURED_RECORDERS = ("_record_pcg_iterations",)


def _scan_vcycle(source, entry="_apply_mg_vcycle", cls="PressureSolver",
                 builders=RECORDED_CMD_BUILDERS, recorders=CAPTURED_RECORDERS):
    """Statically scan everything the captured MG iteration can reach.

    ``entry`` and ``recorders`` are the code that runs while a graph is being
    captured; the closure of their ``self.<method>(...)`` calls is scanned for
    host syncs and for kernels. ``builders`` build the recorded launch commands
    those methods replay, so the kernels they name are scanned for atomics too
    (but not for syncs: they run at setup time, not per iteration). A builder
    or recorder that the class does not define is skipped, so a source with no
    recorded commands scans exactly like a plain V-cycle.

    Returns ``(methods, warp_funcs, atomic_hits, sync_hits)``:
    ``methods`` is every scanned method; ``warp_funcs`` is every module-level
    ``@wp.kernel`` / ``@wp.func`` named in them, plus (transitively) those
    named inside other warp functions; ``atomic_hits`` lists
    ``(function, line, name)`` for atomic add/sub calls in ``warp_funcs``;
    ``sync_hits`` lists host-sync calls in the entry/recorder closure.
    """
    tree = ast.parse(source)
    module_warp = {n.name: n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and _is_warp_decorated(n)}
    class_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    assert class_nodes, f"class {cls} not found"
    methods = {n.name: n for n in class_nodes[0].body if isinstance(n, ast.FunctionDef)}
    assert entry in methods, f"{cls}.{entry} not found"

    def closure(roots):
        reached, stack = [], list(roots)
        while stack:
            name = stack.pop()
            if name in reached:
                continue
            reached.append(name)
            for node in ast.walk(methods[name]):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self" and node.func.attr in methods):
                    stack.append(node.func.attr)
        return reached

    captured = closure([entry] + [n for n in recorders if n in methods])
    reached_methods = list(captured)
    for name in closure([n for n in builders if n in methods]):
        if name not in reached_methods:
            reached_methods.append(name)

    warp_funcs, stack = [], []
    for name in reached_methods:
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Name) and node.id in module_warp:
                stack.append(node.id)
    while stack:
        name = stack.pop()
        if name in warp_funcs:
            continue
        warp_funcs.append(name)
        for node in ast.walk(module_warp[name]):
            if isinstance(node, ast.Name) and node.id in module_warp:
                stack.append(node.id)

    atomic_hits = []
    for name in warp_funcs:
        for node in ast.walk(module_warp[name]):
            if isinstance(node, ast.Call) and _call_name(node.func) in _ATOMIC_NAMES:
                atomic_hits.append((name, node.lineno, _call_name(node.func)))
    sync_hits = []
    for name in captured:
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Call) and _call_name(node.func) in _SYNC_NAMES:
                sync_hits.append((name, node.lineno, _call_name(node.func)))
    return reached_methods, warp_funcs, atomic_hits, sync_hits


def _methods_recording_cmds(source, cls="PressureSolver"):
    """Names of the methods that build recorded launch commands
    (a call with a ``record_cmd=`` keyword)."""
    tree = ast.parse(source)
    class_node = next(n for n in tree.body
                      if isinstance(n, ast.ClassDef) and n.name == cls)
    found = []
    for method in class_node.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if (isinstance(node, ast.Call)
                    and any(kw.arg == "record_cmd" for kw in node.keywords)):
                found.append(method.name)
                break
    return found


def test_vcycle_launches_only_atomic_free_kernels_and_never_syncs():
    source = PRESSURE_SOURCE.read_text(encoding="utf-8")
    # The V-cycle's own code launches k_mg_* kernels (entry point alone).
    own_methods, own_funcs, _, _ = _scan_vcycle(source, builders=(), recorders=())
    own_kernels = sorted(n for n in own_funcs if n.startswith("k_mg_"))
    assert own_kernels, (
        "_apply_mg_vcycle (and the self.* helpers it calls: "
        f"{own_methods}) reaches no k_mg_* kernel")

    # The full scan: V-cycle + the captured recorder + the command builders.
    methods, warp_funcs, atomic_hits, sync_hits = _scan_vcycle(source)
    assert not atomic_hits, (
        "atomic add/sub reachable from the V-cycle: "
        + ", ".join(f"{fn}:{line} {name}" for fn, line, name in atomic_hits))
    assert not sync_hits, (
        "host sync reachable from the V-cycle: "
        + ", ".join(f"{fn}:{line} {name}" for fn, line, name in sync_hits))


def test_scan_covers_the_recorded_launch_commands_of_the_mg_iteration():
    """The x/r update and the z compaction are recorded commands, not
    ``fast_launch`` calls inside the V-cycle: the scan must still see them."""
    source = PRESSURE_SOURCE.read_text(encoding="utf-8")
    methods, warp_funcs, _, _ = _scan_vcycle(source)
    for name in RECORDED_CMD_BUILDERS + CAPTURED_RECORDERS:
        assert name in methods, f"{name} missing from the scan: {methods}"
    for kernel in ("k_mg_update_xr_indexed", "k_mg_gather_z"):
        assert kernel in warp_funcs, f"{kernel} is not covered by the scan"
    # The V-cycle alone would not have seen them.
    _, own_funcs, _, _ = _scan_vcycle(source, builders=(), recorders=())
    assert "k_mg_update_xr_indexed" not in own_funcs
    # No other method may build recorded commands without being scanned.
    uncovered = [n for n in _methods_recording_cmds(source) if n not in methods]
    assert not uncovered, (
        f"methods building recorded launch commands but not scanned: {uncovered}; "
        "add them to RECORDED_CMD_BUILDERS")


_SYNTHETIC_BAD = '''
import warp as wp

@wp.func
def helper(a: wp.array(dtype=float), i: int):
    wp.atomic_add(a, i, 1.0)

@wp.kernel
def k_mg_bad(a: wp.array(dtype=float)):
    helper(a, wp.tid())

@wp.kernel
def k_mg_good(a: wp.array(dtype=float)):
    a[wp.tid()] = 1.0

@wp.kernel
def k_unrelated(a: wp.array(dtype=float)):
    wp.atomic_add(a, 0, 1.0)

class PressureSolver:
    def _apply_mg_vcycle(self, r, z):
        self._level(r)
        host = r.numpy()

    def _level(self, r):
        fast_launch(k_mg_bad, dim=1, inputs=[r])
        fast_launch(k_mg_good, dim=1, inputs=[r])
        wp.synchronize_device(self.device)

    def not_reached(self):
        wp.synchronize()
'''

_SYNTHETIC_GOOD = '''
import warp as wp

@wp.kernel
def k_mg_good(a: wp.array(dtype=float)):
    a[wp.tid()] = 1.0

@wp.kernel
def k_unrelated(a: wp.array(dtype=float)):
    wp.atomic_add(a, 0, 1.0)

class PressureSolver:
    def _apply_mg_vcycle(self, r, z):
        self._level(r)

    def _level(self, r):
        fast_launch(k_mg_good, dim=1, inputs=[r])

    def other(self):
        return r.numpy()
'''


def test_scanner_detects_reachable_atomics_and_syncs():
    methods, funcs, atomics, syncs = _scan_vcycle(_SYNTHETIC_BAD)
    assert set(methods) == {"_apply_mg_vcycle", "_level"}
    assert set(funcs) == {"k_mg_bad", "k_mg_good", "helper"}
    assert [(fn, name) for fn, _, name in atomics] == [("helper", "atomic_add")]
    assert sorted(name for _, _, name in syncs) == ["numpy", "synchronize_device"]


def test_scanner_passes_a_clean_vcycle():
    methods, funcs, atomics, syncs = _scan_vcycle(_SYNTHETIC_GOOD)
    assert set(funcs) == {"k_mg_good"}
    assert not atomics and not syncs


# A V-cycle that itself is clean, inside an iteration that also replays two
# recorded commands built elsewhere (as _pcg_update_cmd is). One of those
# kernels has an atomic; the recorder also syncs the host.
_SYNTHETIC_RECORDED = '''
import warp as wp

@wp.kernel
def k_mg_cmd_bad(a: wp.array(dtype=float)):
    wp.atomic_add(a, 0, 1.0)

@wp.kernel
def k_mg_cmd_good(a: wp.array(dtype=float)):
    a[wp.tid()] = 1.0

@wp.kernel
def k_mg_vcycle_good(a: wp.array(dtype=float)):
    a[wp.tid()] = 2.0

@wp.kernel
def k_unrelated(a: wp.array(dtype=float)):
    wp.atomic_add(a, 0, 1.0)

class PressureSolver:
    def _apply_mg_vcycle(self, r, z):
        fast_launch(k_mg_vcycle_good, dim=1, inputs=[r])

    def _prepare_update_cmds(self, n, r):
        self._cmd_bad = fast_launch(
            k_mg_cmd_bad, dim=n, inputs=[r], device=self.device, record_cmd=True)
        self._cmd_good = fast_launch(
            k_mg_cmd_good, dim=n, inputs=[r], device=self.device, record_cmd=True)

    def _prepare_extra_cmds(self, n, r):
        self._cmd_other = fast_launch(
            k_unrelated, dim=n, inputs=[r], device=self.device, record_cmd=True)

    def _record_pcg_iterations(self, n):
        self._apply_mg_vcycle(self.r, self.z)
        self._cmd_bad.launch()
        self._cmd_good.launch()
        wp.synchronize_device(self.device)

    def host_side(self):
        return self.r.numpy()
'''


def test_scanner_follows_recorded_launch_commands_built_elsewhere():
    # The V-cycle alone is clean: the recorded kernels are invisible to it.
    methods, funcs, atomics, syncs = _scan_vcycle(
        _SYNTHETIC_RECORDED, builders=(), recorders=())
    assert set(methods) == {"_apply_mg_vcycle"}
    assert set(funcs) == {"k_mg_vcycle_good"}
    assert not atomics and not syncs

    # With the builder, the atomic in a recorded kernel is found ...
    methods, funcs, atomics, syncs = _scan_vcycle(
        _SYNTHETIC_RECORDED, recorders=())
    assert set(methods) == {"_apply_mg_vcycle", "_prepare_update_cmds"}
    assert set(funcs) == {"k_mg_vcycle_good", "k_mg_cmd_bad", "k_mg_cmd_good"}
    assert [(fn, name) for fn, _, name in atomics] == [("k_mg_cmd_bad", "atomic_add")]
    assert not syncs     # builders run at setup time, they are not synced-checked

    # ... and with the default (builder + recorder) the recorder's host sync is
    # found too, while the unrelated host_side() method stays out of scope.
    methods, funcs, atomics, syncs = _scan_vcycle(_SYNTHETIC_RECORDED)
    assert set(methods) == {"_apply_mg_vcycle", "_prepare_update_cmds",
                            "_record_pcg_iterations"}
    assert [(fn, name) for fn, _, name in atomics] == [("k_mg_cmd_bad", "atomic_add")]
    assert [(fn, name) for fn, _, name in syncs] == [
        ("_record_pcg_iterations", "synchronize_device")]


def test_scanner_flags_command_builders_it_does_not_cover():
    recording = _methods_recording_cmds(_SYNTHETIC_RECORDED)
    assert sorted(recording) == ["_prepare_extra_cmds", "_prepare_update_cmds"]
    methods, *_ = _scan_vcycle(_SYNTHETIC_RECORDED)
    uncovered = [n for n in recording if n not in methods]
    assert uncovered == ["_prepare_extra_cmds"]
    # Naming the extra builder brings its (atomic) kernel into the scan.
    _, funcs, atomics, _ = _scan_vcycle(
        _SYNTHETIC_RECORDED, builders=RECORDED_CMD_BUILDERS + ("_prepare_extra_cmds",))
    assert "k_unrelated" in funcs
    assert ("k_unrelated", "atomic_add") in [(fn, name) for fn, _, name in atomics]


# ---------------------------------------------------------------------------
# 8. JACOBI stays the default and is unaffected by the new attribute
# ---------------------------------------------------------------------------

def test_mg_constants_define_an_spd_smoother():
    assert MG_PRE_SMOOTH == MG_POST_SMOOTH >= 1   # equal counts => symmetric V-cycle
    assert 0.0 < MG_OMEGA <= 1.0                  # damped Jacobi smoother
    assert PressureSolver._mg_pre_smooth == MG_PRE_SMOOTH
    assert PressureSolver._mg_post_smooth == MG_POST_SMOOTH


def test_jacobi_is_default_and_bitwise_unchanged_by_the_attribute(problem):
    assert PressureSolver.preconditioner == "JACOBI"
    m = problem.fluid_mask
    pressures, iterations = [], []
    for preconditioner in (None, "JACOBI"):
        case = problem.new_case(preconditioner)
        status = case.solver.solve_pcg(case.vtw, 200, CONVERGE_TOL, check_every=25)
        assert status["converged"], status
        pressures.append(case.solver.pressure.numpy()[m])
        iterations.append(case.solver.last_iterations)
    assert iterations[0] == iterations[1]
    assert np.array_equal(_bits(pressures[0]), _bits(pressures[1]))
