"""Viscosity (fluid_viscosity family).

Two paths, matching the original solver:

- explicit smoothing (``fluidSimForceViscositySmooth``) — cheap viscosity by
  blending face velocities toward their neighbourhood average, weighted by
  the per-voxel viscosity coefficient.
- implicit solve (Jacobi or PCG) of ``(I - dt*nu*L) u* = u`` per velocity
  component, using the same device-resident PCG machinery style as pressure
  (build RHS, inverse diagonal, apply matrix, alpha/beta/x/r/p updates).

No bpy imports allowed here.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID
from .fields import voxel_coord_of_flat, voxel_flat_index
from .pressure import (
    k_apply_preconditioner,
    k_compute_alpha,
    k_compute_alpha_and_reset,
    k_compute_beta,
    k_pack_pcg_stats,
    k_update_p,
    k_update_xr,
    neighbor_flat,
)
from .registry import register_fn

VISCOSITY_STATUS_NOT_RUN = "not_run"
VISCOSITY_STATUS_SKIPPED = "skipped"
VISCOSITY_STATUS_JACOBI = "jacobi"
VISCOSITY_STATUS_CONVERGED = "converged"
VISCOSITY_STATUS_MAX_ITERATIONS = "max_iterations"
VISCOSITY_STATUS_STALLED = "stalled"
VISCOSITY_STATUS_SINGULAR = "singular"
VISCOSITY_STATUS_INVALID = "invalid"

#: Ordered by severity: ``solve`` reports the component with the highest code.
VISCOSITY_STATUS_CODES = {
    VISCOSITY_STATUS_NOT_RUN: 0,
    VISCOSITY_STATUS_SKIPPED: 1,
    VISCOSITY_STATUS_JACOBI: 2,
    VISCOSITY_STATUS_CONVERGED: 3,
    VISCOSITY_STATUS_MAX_ITERATIONS: 4,
    VISCOSITY_STATUS_STALLED: 5,
    VISCOSITY_STATUS_SINGULAR: 6,
    VISCOSITY_STATUS_INVALID: 7,
}

#: How many convergence checks back the stagnation guard compares against.
#: Pressure compares consecutive checks, which is safe at its ``check_every``
#: of 25 but not at the viscosity cadence of 1: measured over the 120
#: component solves of the dam_break benchmark, a consecutive-check test fires
#: on 26 of them, the earliest at iteration 3 with the residual still ~3e5x
#: above tolerance. Comparing against the check ``VISCOSITY_STAGNATION_WINDOW``
#: back fires on 0 of the 3366 recorded checks (worst observed ratio 0.145).
VISCOSITY_STAGNATION_WINDOW = 10

#: Check convergence every iteration. The check is evaluated on device, so it
#: costs no host stall; the shipped cadence of 10 existed only to amortise a
#: synchronous readback.
DEFAULT_VISCOSITY_CHECK_EVERY = 1

#: How much of a component solve the CUDA conditional graph captures.
#:
#: ``loop`` captures the PCG loop alone and issues the surrounding setup and
#: write-back call by call from the host. ``component`` captures all of it.
#: The distinction is worth a mode flag because the cost it removes is host
#: time, not device time: measured on dam_break with viscosity on, the drain
#: at the end of a component solve was 0.134 ms/frame -- the GPU was already
#: idle -- while the launches bracketing the loop cost 4.911 ms/frame. Priced
#: in isolation on the same kernels at 80k voxels, ``wp.launch`` costs ~28 us
#: per call before any argument marshalling, and the whole bracketing sequence
#: falls from 348 us to 13 us when it is enqueued as one graph.
VISCOSITY_GRAPH_SCOPE_LOOP = "loop"
VISCOSITY_GRAPH_SCOPE_COMPONENT = "component"
VISCOSITY_GRAPH_SCOPES = (VISCOSITY_GRAPH_SCOPE_LOOP,
                          VISCOSITY_GRAPH_SCOPE_COMPONENT)

_default_viscosity_graph_scope = VISCOSITY_GRAPH_SCOPE_COMPONENT


def set_default_viscosity_graph_scope(scope):
    """Set the scope new ViscositySolvers use. Benchmark/diagnostic hook."""

    global _default_viscosity_graph_scope
    if scope not in VISCOSITY_GRAPH_SCOPES:
        raise ValueError(
            "unknown viscosity graph scope %r; expected one of %r"
            % (scope, VISCOSITY_GRAPH_SCOPES))
    _default_viscosity_graph_scope = scope


def default_viscosity_graph_scope():
    return _default_viscosity_graph_scope

# Explicit smoothing path


@wp.kernel
def k_viscosity_smooth(
    u_in: wp.array(dtype=float), v_in: wp.array(dtype=float), w_in: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    viscosity_field: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    dt: float,
    u_out: wp.array(dtype=float), v_out: wp.array(dtype=float), w_out: wp.array(dtype=float),
):
    """fluidSimForceViscositySmooth: explicit diffusion clamp-blended."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    nu = viscosity_field[tid]
    # stability-clamped explicit blend factor
    t = wp.clamp(nu * dt / (voxel_size * voxel_size) * 6.0, 0.0, 1.0)
    su = float(0.0)
    sv = float(0.0)
    sw = float(0.0)
    n = float(0.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            f = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if f >= 0 and (voxel_type[f] & VOXEL_COLLISION) == 0:
                su += u_in[f]
                sv += v_in[f]
                sw += w_in[f]
                n += 1.0
    if n > 0.0:
        inv = 1.0 / n
        u_out[tid] = u_in[tid] + t * (su * inv - u_in[tid]) * wp.where(wu[tid] > 0.0, 1.0, 0.0)
        v_out[tid] = v_in[tid] + t * (sv * inv - v_in[tid]) * wp.where(wv[tid] > 0.0, 1.0, 0.0)
        w_out[tid] = w_in[tid] + t * (sw * inv - w_in[tid]) * wp.where(ww[tid] > 0.0, 1.0, 0.0)
    else:
        u_out[tid] = u_in[tid]
        v_out[tid] = v_in[tid]
        w_out[tid] = w_in[tid]


# Implicit path: (I + c*D - c*L) x = u  with c = dt*nu/h^2 per voxel


@wp.kernel
def k_viscosity_set_solve_scalars(
    dt: float,
    tolerance_squared: float,
    dt_out: wp.array(dtype=float),
    tolerance_squared_out: wp.array(dtype=float),
):
    """Stage one solve's scalars on device.

    The solve kernels read ``dt`` and the squared tolerance from these arrays
    instead of taking them as launch parameters, so a captured graph does not
    bake them in and one graph serves every ``dt`` and tolerance. Both go
    through the same float32 conversion a launch parameter did, so the values
    the kernels see are unchanged.
    """
    dt_out[0] = dt
    tolerance_squared_out[0] = tolerance_squared


@wp.kernel
def k_viscosity_counter_increment(counter: wp.array(dtype=wp.int32)):
    """fluidSimForceViscosityCounterIncrement: solve-pass counter."""
    counter[0] = counter[0] + 1


@wp.kernel
def k_viscosity_build_rhs(
    vel_comp: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    rhs: wp.array(dtype=float),
    x: wp.array(dtype=float),
):
    """fluidSimForceViscosityBuildRhs: rhs = current velocity, x0 = rhs."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    val = float(0.0)
    if weight[tid] > 0.0 and (voxel_type[tid] & VOXEL_COLLISION) == 0:
        val = vel_comp[tid]
    rhs[tid] = val
    x[tid] = val


@wp.kernel
def k_viscosity_compute_inverse_diagonal(
    viscosity_field: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    dt: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
):
    """fluidSimForceViscosityComputeInverseDiagonal."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if weight[tid] <= 0.0 or (voxel_type[tid] & VOXEL_COLLISION) != 0:
        inv_diag[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = viscosity_field[tid] * dt[0] / (voxel_size * voxel_size)
    diag = float(1.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            f = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if f >= 0 and weight[f] > 0.0 and (voxel_type[f] & VOXEL_COLLISION) == 0:
                diag += c
    inv_diag[tid] = 1.0 / diag


@wp.kernel
def k_viscosity_apply_matrix(
    x: wp.array(dtype=float),
    viscosity_field: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    dt: wp.array(dtype=float),
    ax: wp.array(dtype=float),
):
    """fluidSimForceViscosityApplyMatrix: (I + c*(D - adjacency)) x."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if weight[tid] <= 0.0 or (voxel_type[tid] & VOXEL_COLLISION) != 0:
        ax[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    c = viscosity_field[tid] * dt[0] / (voxel_size * voxel_size)
    acc = x[tid]
    for axis in range(3):
        for step in range(-1, 2, 2):
            f = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if f >= 0 and weight[f] > 0.0 and (voxel_type[f] & VOXEL_COLLISION) == 0:
                acc += c * (x[tid] - x[f])
    ax[tid] = acc


@wp.kernel
def k_viscosity_initialize_pcg(
    rhs: wp.array(dtype=float),
    ax: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    r: wp.array(dtype=float),
):
    """fluidSimForceViscosityInitializePcg: r = b - A x0."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    r[tid] = rhs[tid] - ax[tid]


@wp.kernel
def k_viscosity_finalize_initial_residual(
    r: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    z: wp.array(dtype=float),
):
    """fluidSimForceViscosityFinalizeInitialResidual: z = Minv r."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    z[tid] = r[tid] * inv_diag[tid]


@wp.kernel
def k_viscosity_initialize_direction(
    z: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    p: wp.array(dtype=float),
):
    """fluidSimForceViscosityInitializeDirectionFromPreconditioned: p = z."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p[tid] = z[tid]


@wp.kernel
def k_viscosity_apply_collision(
    x: wp.array(dtype=float),
    vel_comp_in: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    vel_comp_out: wp.array(dtype=float),
):
    """fluidSimForceViscosityApplyCollision: write solution back to faces,
    leaving collision faces untouched.

    Each thread reads and writes only its own element, so ``vel_comp_in`` and
    ``vel_comp_out`` may be the same array."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if weight[tid] > 0.0 and (voxel_type[tid] & VOXEL_COLLISION) == 0:
        vel_comp_out[tid] = x[tid]
    else:
        vel_comp_out[tid] = vel_comp_in[tid]


# --- device-side PCG convergence check (ported from PressureSolver) ---------
#
# The shipped loop evaluated its stopping test on the host
# (``float(self.residual.numpy()[0])``), which drains the CUDA stream once per
# check. PressureSolver already evaluates the identical criterion on device
# inside a CUDA conditional graph; these two kernels are the viscosity
# counterparts of ``k_prepare_pcg_conditional_loop`` /
# ``k_finish_pcg_conditional_batch``.


@wp.kernel
def k_viscosity_pcg_beta_and_roll(
    rz_new: wp.array(dtype=float),
    rz: wp.array(dtype=float),
    beta: wp.array(dtype=float),
):
    """beta = rz_new/rz, then roll rz_new into rz and clear the accumulator.

    Identical arithmetic to ``k_compute_beta`` followed by the host loop's
    ``wp.copy(rz, rz_new)``. The extra clear exists because the captured loop
    reduces with an accumulating (atomicAdd) dot product, which requires a
    zeroed target; the host path keeps using the overwriting
    ``wp.utils.array_inner`` and never observes it.
    """
    denom = rz[0]
    if wp.abs(denom) < 1.0e-30:
        beta[0] = 0.0
    else:
        beta[0] = rz_new[0] / denom
    rz[0] = rz_new[0]
    rz_new[0] = 0.0


@wp.kernel
def k_viscosity_prepare_pcg_conditional_loop(
    condition: wp.array(dtype=wp.int32),
    iterations: wp.array(dtype=wp.int32),
    history: wp.array(dtype=float),
    history_pos: wp.array(dtype=wp.int32),
    residual: wp.array(dtype=float),
    window: int,
    stats: wp.array(dtype=float),
):
    """Arm the conditional loop.

    The shipped host loop entered its first iteration unconditionally, so this
    arms unconditionally too: every exit is decided at a check boundary by
    ``k_viscosity_finish_pcg_conditional_batch``.
    """
    condition[0] = 1
    iterations[0] = 0
    history_pos[0] = 0
    for i in range(window):
        history[i] = 1.0e30
    residual[0] = 0.0
    for i in range(5):
        stats[i] = 0.0


@wp.kernel
def k_viscosity_finish_pcg_conditional_batch(
    residual: wp.array(dtype=float),
    rz: wp.array(dtype=float),
    p_ap: wp.array(dtype=float),
    condition: wp.array(dtype=wp.int32),
    iterations: wp.array(dtype=wp.int32),
    history: wp.array(dtype=float),
    history_pos: wp.array(dtype=wp.int32),
    batch: int,
    max_iterations: int,
    tolerance_squared: wp.array(dtype=float),
    window: int,
    stats: wp.array(dtype=float),
):
    """Evaluate the stopping criterion on device and clear the accumulators.

    Exit reasons match ``k_finish_pcg_conditional_batch``: 1 converged,
    2 breakdown/singular, 3 stagnation, 4 non-finite, 5 iteration budget.

    The stagnation test differs from pressure's deliberately. Pressure checks
    every 25 iterations and compares against the immediately preceding check;
    viscosity checks every iteration, and a PCG residual is not monotone from
    one iteration to the next. Measured on the 120 recorded viscosity solves
    of ``artifacts/visc_pcg_residual/visc_pcg_residual_shipped40_all_*.json``,
    a consecutive-iteration test fires on 26 of 120 solves - as early as
    iteration 3, with the residual still 3e5x above tolerance. Comparing
    against the residual ``window`` checks ago fires on none of them (worst
    10-check ratio 0.145), so the guard keeps its purpose without aborting a
    healthy solve.
    """
    res_sq = residual[0]
    rz_value = rz[0]
    p_ap_value = p_ap[0]
    iteration = iterations[0] + batch
    pos = history_pos[0]
    previous = history[pos]
    reason = 0
    keep_going = 0
    if (wp.isnan(res_sq) or wp.isinf(res_sq) or wp.isnan(rz_value)
            or wp.isinf(rz_value) or wp.isnan(p_ap_value)
            or wp.isinf(p_ap_value)):
        reason = 4
    elif res_sq < tolerance_squared[0]:
        reason = 1
    elif (rz_value < -1.0e-20 or wp.abs(rz_value) < 1.0e-30
            or wp.abs(p_ap_value) < 1.0e-30):
        reason = 2
    elif res_sq >= previous * (1.0 - 1.0e-6):
        reason = 3
    elif iteration >= max_iterations:
        reason = 5
    else:
        keep_going = 1
    history[pos] = res_sq
    history_pos[0] = (pos + 1) % window
    condition[0] = keep_going
    iterations[0] = iteration
    stats[0] = res_sq
    stats[1] = rz_value
    stats[2] = p_ap_value
    stats[3] = float(iteration)
    stats[4] = float(reason)
    residual[0] = 0.0


register_fn(k_viscosity_smooth, "fluidSimForceViscositySmooth", strategy="tiled")
register_fn(k_viscosity_counter_increment, "fluidSimForceViscosityCounterIncrement")
register_fn(k_viscosity_build_rhs, "fluidSimForceViscosityBuildRhs")
register_fn(k_viscosity_compute_inverse_diagonal,
            "fluidSimForceViscosityComputeInverseDiagonal")
register_fn(k_viscosity_apply_matrix, "fluidSimForceViscosityApplyMatrix",
            strategy="tiled")
register_fn(k_viscosity_initialize_pcg, "fluidSimForceViscosityInitializePcg")
register_fn(k_viscosity_finalize_initial_residual,
            "fluidSimForceViscosityFinalizeInitialResidual")
register_fn(k_viscosity_initialize_direction,
            "fluidSimForceViscosityInitializeDirectionFromPreconditioned")
register_fn(k_compute_alpha, "fluidSimForceViscosityComputeAlpha")
register_fn(k_update_xr, "fluidSimForceViscosityUpdateXR")
register_fn(k_apply_preconditioner, "fluidSimForceViscosityApplyPreConditioner")
register_fn(k_compute_beta, "fluidSimForceViscosityComputeBeta")
register_fn(k_update_p, "fluidSimForceViscosityUpdateP")
register_fn(k_viscosity_apply_collision, "fluidSimForceViscosityApplyCollision")


# Host-orchestrated driver


class ViscositySolver:
    """Implicit viscosity solve per velocity component (Jacobi or PCG).

    The PCG path evaluates its stopping test on device inside a CUDA
    conditional graph whenever the device supports one, so a convergence check
    costs no host stall; the host-checked loop below is the reference
    implementation and the fallback for CPU / unsupported devices.
    """

    #: Conditional-graph cache capacity. ``dt``, the squared tolerance and the
    #: exact active count are read on device at launch, so a graph is keyed by
    #: shape and topology only: the launch bucket (``_launch_dim``), the
    #: component, and the grid/field buffers. The dam_break viscosity workload
    #: spans under one octave of active voxels, i.e. a handful of buckets per
    #: component. Keying on (active voxels, dt, tolerance) as well used to
    #: re-capture on nearly every other solve.
    _GRAPH_CAPACITY = 32

    #: Launch buckets per octave of active voxels. A captured launch bakes in
    #: its grid size, so graphs launch over the active count rounded up to the
    #: next bucket; every kernel exits past ``num_leaves[0] * LEAF_VOLUME``
    #: and the reductions read that count on device, so the rounding costs
    #: only idle threads (at most 1/4 extra at 4 buckets per octave).
    _LAUNCH_BUCKETS_PER_OCTAVE = 4

    def __init__(self, grid, device):
        self.grid = grid
        self.device = device
        n = grid.voxel_capacity
        self.rhs = wp.zeros(n, dtype=float, device=device)
        self.x = wp.zeros(n, dtype=float, device=device)
        self.inv_diag = wp.zeros(n, dtype=float, device=device)
        self.r = wp.zeros(n, dtype=float, device=device)
        self.z = wp.zeros(n, dtype=float, device=device)
        self.p_dir = wp.zeros(n, dtype=float, device=device)
        self.ap = wp.zeros(n, dtype=float, device=device)
        self.tmp_u = wp.zeros(n, dtype=float, device=device)
        self.tmp_v = wp.zeros(n, dtype=float, device=device)
        self.tmp_w = wp.zeros(n, dtype=float, device=device)
        self.counter = wp.zeros(1, dtype=wp.int32, device=device)
        self.rz = wp.zeros(1, dtype=float, device=device)
        self.rz_new = wp.zeros(1, dtype=float, device=device)
        self.p_ap = wp.zeros(1, dtype=float, device=device)
        self.alpha = wp.zeros(1, dtype=float, device=device)
        self.beta = wp.zeros(1, dtype=float, device=device)
        self.residual = wp.zeros(1, dtype=float, device=device)
        # Per-solve scalars, read on device so captured graphs don't bake them.
        self._solve_dt = wp.zeros(1, dtype=float, device=device)
        self._pcg_tol_sq = wp.zeros(1, dtype=float, device=device)
        # Device-evaluated convergence check (mirrors PressureSolver).
        self.pcg_stats = wp.zeros(5, dtype=float, device=device)
        self._pcg_p_ap_accum = wp.zeros(1, dtype=float, device=device)
        self._pcg_condition = wp.zeros(1, dtype=wp.int32, device=device)
        self._pcg_iterations = wp.zeros(1, dtype=wp.int32, device=device)
        self._pcg_history = wp.zeros(
            VISCOSITY_STAGNATION_WINDOW, dtype=float, device=device)
        self._pcg_history_pos = wp.zeros(1, dtype=wp.int32, device=device)
        self._pcg_conditional_graphs = OrderedDict()
        self._pcg_dot = None
        self._pcg_native_available = None
        self._conditional_available = None
        self._graph_captures = 0
        self.graph_scope = default_viscosity_graph_scope()
        self.last_pcg_native_error = ""
        self.last_pcg_execution_mode = "not_run"
        self.last_iterations = 0
        self.last_residual = 0.0
        self.last_residual_squared = 0.0
        self.last_viscosity_status = VISCOSITY_STATUS_NOT_RUN
        self.last_viscosity_status_code = VISCOSITY_STATUS_CODES[
            VISCOSITY_STATUS_NOT_RUN]
        self.last_viscosity_failure = ""
        self.last_component_statuses = []
        self.last_status = self._make_status(
            False, VISCOSITY_STATUS_NOT_RUN, 0.0, 0)

    # -- status reporting ----------------------------------------------------

    @staticmethod
    def _make_status(converged, reason, residual, iterations):
        return {
            "converged": bool(converged),
            "iterations": int(iterations),
            "reason": reason,
            "residual": float(residual),
        }

    def _set_viscosity_status(self, status, residual_squared, iterations,
                              failure=""):
        """Record one component solve outcome, the way pressure records its own."""
        converged = status in (
            VISCOSITY_STATUS_CONVERGED,
            VISCOSITY_STATUS_JACOBI,
            VISCOSITY_STATUS_SKIPPED,
        )
        res_sq = float(residual_squared)
        if math.isfinite(res_sq):
            residual = math.sqrt(max(0.0, res_sq))
        else:
            res_sq = math.inf
            residual = math.inf
        self.last_viscosity_status = status
        self.last_viscosity_status_code = VISCOSITY_STATUS_CODES.get(
            status, VISCOSITY_STATUS_CODES[VISCOSITY_STATUS_INVALID])
        self.last_viscosity_failure = failure
        self.last_iterations = int(iterations)
        self.last_residual_squared = res_sq
        self.last_residual = residual
        self.last_status = self._make_status(
            converged, status, residual, iterations)
        return self.last_status

    @staticmethod
    def _status_from_reason(reason):
        """Map an exit-reason code written by the device kernel to a status."""
        if reason == 1:
            return VISCOSITY_STATUS_CONVERGED, ""
        if reason == 2:
            return VISCOSITY_STATUS_SINGULAR, "singular_system"
        if reason == 3:
            return VISCOSITY_STATUS_STALLED, "stalled"
        if reason == 4:
            return VISCOSITY_STATUS_INVALID, "non_finite_residual"
        return VISCOSITY_STATUS_MAX_ITERATIONS, "max_iterations"

    # -- device resources ----------------------------------------------------

    def _active_count(self):
        return int(self.grid.active_voxel_count)

    def _ensure_pcg_dot(self):
        """Acquire the allocation-free reduction the captured loop needs.

        ``wp.utils.array_inner`` allocates a scratch buffer, and a child graph
        containing an allocation is rejected by ``wp.capture_while``, which is
        why the conditional path depends on the same native accumulating dot
        product PressureSolver uses.
        """
        if self._pcg_native_available is False:
            return False
        if self._pcg_dot is not None:
            return True
        try:
            from .cuda_reduction import CudaDotProduct

            self._pcg_dot = CudaDotProduct(self.device)
        except Exception as exc:
            self._pcg_native_available = False
            self.last_pcg_native_error = f"{type(exc).__name__}: {exc}"
            return False
        self._pcg_native_available = True
        self.last_pcg_native_error = ""
        return True

    def _conditional_supported(self, iterations, check_every):
        if iterations <= 0 or check_every <= 0:
            return False
        if self._conditional_available is False:
            return False
        if iterations % check_every != 0:
            # The terminal check has to land exactly on the iteration budget,
            # otherwise the captured loop would overshoot it.
            return False
        if self._conditional_available is None:
            try:
                supported = (
                    wp.get_device(self.device).is_cuda
                    and wp.is_conditional_graph_supported()
                )
            except Exception as exc:
                self._conditional_available = False
                self.last_pcg_native_error = f"{type(exc).__name__}: {exc}"
                return False
            self._conditional_available = bool(supported)
            if not supported:
                return False
        return self._ensure_pcg_dot()

    def _launch_dim(self, n):
        """Round the active count up to its launch bucket.

        ``_LAUNCH_BUCKETS_PER_OCTAVE`` buckets per power of two, capped at the
        allocation, so neighbouring active counts share one captured graph.
        """
        n = int(n)
        shift = self._LAUNCH_BUCKETS_PER_OCTAVE.bit_length() - 1
        quantum = 1 << max(0, n.bit_length() - 1 - shift)
        dim = -(-n // quantum) * quantum
        return min(dim, int(self.grid.voxel_capacity))

    def _graph_key(self, scope, launch_dim, weight, viscosity_field, voxel_type,
                   check_every, iterations, comp=None):
        """Everything a captured graph bakes in: shape and buffers only."""
        g = self.grid
        return (
            str(scope), int(launch_dim), float(g.voxel_size),
            int(check_every), int(iterations),
            None if comp is None else comp.ptr,
            weight.ptr, viscosity_field.ptr, voxel_type.ptr,
            g.leaf_table.ptr, g.leaf_coords.ptr, g.num_leaves.ptr,
        )

    def _active_dot(self, a, b, out, dim):
        """Accumulating dot over the active prefix, its length read on device."""
        self._pcg_dot.launch_active(a, b, out, self.grid.num_leaves, dim,
                                    LEAF_VOLUME)

    def _record_pcg_batch(self, n, stencil, viscosity_field, weight,
                          voxel_type, check_every, iterations):
        """Record ``check_every`` PCG iterations plus one device-side check."""
        g = self.grid
        for _ in range(check_every):
            wp.launch(
                k_viscosity_apply_matrix, dim=n,
                inputs=[self.p_dir, viscosity_field, weight, voxel_type,
                        *stencil, g.voxel_size, self._solve_dt, self.ap],
                device=self.device,
            )
            self._active_dot(self.p_dir, self.ap, self._pcg_p_ap_accum, n)
            wp.launch(k_compute_alpha_and_reset, dim=1,
                      inputs=[self.rz, self._pcg_p_ap_accum, self.p_ap,
                              self.alpha],
                      device=self.device)
            wp.launch(k_update_xr, dim=n,
                      inputs=[self.p_dir, self.ap, self.alpha, g.num_leaves,
                              self.x, self.r],
                      device=self.device)
            wp.launch(k_apply_preconditioner, dim=n,
                      inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                      device=self.device)
            self._active_dot(self.r, self.z, self.rz_new, n)
            wp.launch(k_viscosity_pcg_beta_and_roll, dim=1,
                      inputs=[self.rz_new, self.rz, self.beta],
                      device=self.device)
            wp.launch(k_update_p, dim=n,
                      inputs=[self.z, self.beta, g.num_leaves, self.p_dir],
                      device=self.device)
        self._active_dot(self.r, self.r, self.residual, n)
        wp.launch(
            k_viscosity_finish_pcg_conditional_batch, dim=1,
            inputs=[self.residual, self.rz, self.p_ap, self._pcg_condition,
                    self._pcg_iterations, self._pcg_history,
                    self._pcg_history_pos, check_every, iterations,
                    self._pcg_tol_sq,
                    VISCOSITY_STAGNATION_WINDOW, self.pcg_stats],
            device=self.device,
        )

    def _get_pcg_conditional_graph(self, key, comp, n, stencil,
                                   viscosity_field, weight, voxel_type,
                                   check_every, iterations, scope):
        """Capture, or reuse, the graphs for one solve shape.

        Returns ``(entry_graph, body_graph)``. Under ``loop`` scope the entry
        graph is the conditional loop alone and the caller issues the setup,
        preamble and write-back itself; under ``component`` scope the entry
        graph is the whole solve, so one ``capture_launch`` stands in for the
        fourteen host calls that used to bracket the loop.
        """
        graphs = self._pcg_conditional_graphs.get(key)
        if graphs is not None:
            self._pcg_conditional_graphs.move_to_end(key)
            return graphs
        wp.synchronize_device(self.device)
        # `wp.ScopedCapture` does not nest, so the loop body has to be a
        # finished graph before anything that embeds it can be captured.
        with wp.ScopedCapture(device=self.device) as body_capture:
            self._record_pcg_batch(
                n, stencil, viscosity_field, weight, voxel_type,
                check_every, iterations)
        with wp.ScopedCapture(device=self.device) as entry_capture:
            if scope == VISCOSITY_GRAPH_SCOPE_COMPONENT:
                self._record_component_body(
                    comp, weight, viscosity_field, voxel_type, n, stencil,
                    body_capture.graph)
            else:
                wp.capture_while(condition=self._pcg_condition,
                                 while_body=body_capture.graph)
        graphs = (entry_capture.graph, body_capture.graph)
        while len(self._pcg_conditional_graphs) >= self._GRAPH_CAPACITY:
            self._pcg_conditional_graphs.popitem(last=False)
        self._pcg_conditional_graphs[key] = graphs
        self._graph_captures += 1
        return graphs

    def release(self):
        """Release captured PCG work and its native CUDA module lease."""
        self._pcg_conditional_graphs.clear()
        dot = self._pcg_dot
        self._pcg_dot = None
        if dot is not None:
            dot.release()

    def smooth(self, vel, viscosity_field, voxel_type, dt, iterations=1):
        """Run the Jacobi smoother, and say so.

        Returns the same status shape ``solve`` does. It previously
        returned nothing and left ``last_status`` holding whatever the
        previous PCG solve had reported, which made the smoother look
        like a solver that had converged.
        """
        g = self.grid
        n = self._active_count()
        if n <= 0:
            return self._set_viscosity_status(
                VISCOSITY_STATUS_SKIPPED, 0.0, 0)
        for _ in range(iterations):
            wp.launch(
                k_viscosity_smooth, dim=n,
                inputs=[vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww,
                        viscosity_field, voxel_type,
                        g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                        g.num_leaves, g.voxel_size, dt,
                        self.tmp_u, self.tmp_v, self.tmp_w],
                device=self.device,
            )
            wp.copy(vel.u, self.tmp_u, count=n)
            wp.copy(vel.v, self.tmp_v, count=n)
            wp.copy(vel.w, self.tmp_w, count=n)
        # The smoother has no residual test; it runs a fixed sweep count.
        self.last_component_statuses = []
        return self._set_viscosity_status(
            VISCOSITY_STATUS_JACOBI, 0.0, int(iterations))

    # -- recordable launch sequences -----------------------------------------
    #
    # The pieces of a component solve, factored out so the host-issued path and
    # the captured `component` graph emit exactly the same calls in the same
    # order and cannot drift apart. None of them reads back from the device,
    # which is what makes them capturable.

    def _record_solve_scalars(self, dt, tol_sq):
        """Stage ``dt`` and the squared tolerance for the kernels to read."""
        wp.launch(k_viscosity_set_solve_scalars, dim=1,
                  inputs=[dt, tol_sq, self._solve_dt, self._pcg_tol_sq],
                  device=self.device)

    def _record_setup(self, comp, weight, viscosity_field, voxel_type, n,
                      stencil):
        """Build the right-hand side, the initial guess and the preconditioner."""
        g = self.grid
        wp.launch(k_viscosity_counter_increment, dim=1, inputs=[self.counter],
                  device=self.device)
        wp.launch(
            k_viscosity_build_rhs, dim=n,
            inputs=[comp, weight, voxel_type, g.num_leaves, self.rhs, self.x],
            device=self.device,
        )
        wp.launch(
            k_viscosity_compute_inverse_diagonal, dim=n,
            inputs=[viscosity_field, weight, voxel_type, *stencil,
                    g.voxel_size, self._solve_dt, self.inv_diag],
            device=self.device,
        )

    def _record_pcg_initialization(self, weight, viscosity_field, voxel_type,
                                   n, stencil):
        """r0 from the initial residual, z0 = Minv r0, p0 = z0."""
        g = self.grid
        wp.launch(
            k_viscosity_apply_matrix, dim=n,
            inputs=[self.x, viscosity_field, weight, voxel_type,
                    *stencil, g.voxel_size, self._solve_dt, self.ap],
            device=self.device,
        )
        wp.launch(
            k_viscosity_initialize_pcg, dim=n,
            inputs=[self.rhs, self.ap, self.inv_diag, g.num_leaves, self.r],
            device=self.device,
        )
        wp.launch(
            k_viscosity_finalize_initial_residual, dim=n,
            inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
            device=self.device,
        )
        wp.launch(
            k_viscosity_initialize_direction, dim=n,
            inputs=[self.z, g.num_leaves, self.p_dir],
            device=self.device,
        )

    def _record_conditional_preamble(self, n):
        """Seed rz and arm the device-side loop guard."""
        # The captured dot products accumulate with atomicAdd, so every target
        # has to start at zero.
        self._pcg_p_ap_accum.zero_()
        self.rz_new.zero_()
        self.rz.zero_()
        self.residual.zero_()
        self._active_dot(self.r, self.z, self.rz, n)
        wp.launch(
            k_viscosity_prepare_pcg_conditional_loop, dim=1,
            inputs=[self._pcg_condition, self._pcg_iterations,
                    self._pcg_history, self._pcg_history_pos, self.residual,
                    VISCOSITY_STAGNATION_WINDOW, self.pcg_stats],
            device=self.device,
        )

    def _record_writeback(self, comp, weight, voxel_type, n):
        """Write the solution back into the velocity component."""
        g = self.grid
        # write back with collision guard, in place: a separate copy would
        # have to bake the exact count into a captured graph
        wp.launch(
            k_viscosity_apply_collision, dim=n,
            inputs=[self.x, comp, weight, voxel_type, g.num_leaves, comp],
            device=self.device,
        )

    def _record_component_body(self, comp, weight, viscosity_field, voxel_type,
                               n, stencil, loop_body):
        """The entire component solve, minus the closing status readback."""
        self._record_setup(comp, weight, viscosity_field, voxel_type, n,
                           stencil)
        self._record_pcg_initialization(weight, viscosity_field, voxel_type,
                                        n, stencil)
        self._record_conditional_preamble(n)
        wp.capture_while(condition=self._pcg_condition, while_body=loop_body)
        self._record_writeback(comp, weight, voxel_type, n)

    def _solve_component(self, comp, weight, viscosity_field, voxel_type, dt,
                         use_pcg, iterations, tolerance,
                         check_every=DEFAULT_VISCOSITY_CHECK_EVERY):
        g = self.grid
        n = self._active_count()
        if n <= 0:
            return self._set_viscosity_status(VISCOSITY_STATUS_SKIPPED, 0.0, 0)
        stencil = [g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim, g.num_leaves]
        tol_sq = 0.0
        conditional = False
        if use_pcg:
            check_every = max(1, int(check_every))
            tol_sq = tolerance * tolerance * max(1, n)
            conditional = self._conditional_supported(iterations, check_every)
        # Graphs launch over the bucket so one serves every nearby active
        # count. Every kernel exits past the active prefix, so the host-issued
        # launches around them use the same dimension and both scopes stay
        # launch-for-launch identical; the host loop keeps the exact count.
        dim = self._launch_dim(n) if conditional else n
        self._record_solve_scalars(dt, tol_sq)
        if use_pcg:
            if (self.graph_scope == VISCOSITY_GRAPH_SCOPE_COMPONENT
                    and conditional):
                try:
                    if self._solve_pcg_conditional(
                            comp, dim, stencil, viscosity_field, weight,
                            voxel_type, check_every, iterations,
                            VISCOSITY_GRAPH_SCOPE_COMPONENT):
                        # Setup and write-back were enqueued with the loop, so
                        # the solve is already complete.
                        return self.last_status
                except Exception as exc:
                    # A capture failure must never take the simulation down.
                    # `comp` is only written at the very end of the captured
                    # body, so an aborted capture leaves it untouched and the
                    # host-issued path below can redo the solve from scratch.
                    self._conditional_available = False
                    self.last_pcg_native_error = f"{type(exc).__name__}: {exc}"
                    self._pcg_conditional_graphs.clear()
        self._record_setup(comp, weight, viscosity_field, voxel_type, dim,
                           stencil)
        if not use_pcg:
            # Jacobi: x_{k+1} = Minv * (b + (Minv^-1 - A) x_k) implemented via
            # residual form x += Minv*(b - A x)
            for _ in range(iterations):
                wp.launch(
                    k_viscosity_apply_matrix, dim=n,
                    inputs=[self.x, viscosity_field, weight, voxel_type,
                            *stencil, g.voxel_size, self._solve_dt, self.ap],
                    device=self.device,
                )
                wp.launch(
                    k_viscosity_initialize_pcg, dim=n,
                    inputs=[self.rhs, self.ap, self.inv_diag, g.num_leaves, self.r],
                    device=self.device,
                )
                wp.launch(
                    k_viscosity_finalize_initial_residual, dim=n,
                    inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                    device=self.device,
                )
                # x += z (damped Jacobi via preconditioned residual)
                wp.launch(
                    k_axpy_inplace, dim=n,
                    inputs=[self.z, 1.0, g.num_leaves, self.x],
                    device=self.device,
                )
            self.last_pcg_execution_mode = "jacobi"
            self._set_viscosity_status(
                VISCOSITY_STATUS_JACOBI, 0.0, iterations)
        else:
            self._record_pcg_initialization(
                weight, viscosity_field, voxel_type, dim, stencil)
            solved = False
            if conditional and self._conditional_available is not False:
                try:
                    solved = self._solve_pcg_conditional(
                        comp, dim, stencil, viscosity_field, weight,
                        voxel_type, check_every, iterations,
                        VISCOSITY_GRAPH_SCOPE_LOOP)
                except Exception as exc:
                    # A capture failure must never take the simulation down;
                    # drop to the host-checked loop for the rest of the run.
                    self._conditional_available = False
                    self.last_pcg_native_error = f"{type(exc).__name__}: {exc}"
                    self._pcg_conditional_graphs.clear()
                    solved = False
            if not solved:
                self._solve_pcg_host(
                    n, stencil, viscosity_field, weight, voxel_type,
                    check_every, iterations, tol_sq)
        self._record_writeback(comp, weight, voxel_type, dim)
        return self.last_status

    def _solve_pcg_conditional(self, comp, dim, stencil, viscosity_field,
                               weight, voxel_type, check_every,
                               iterations, scope):
        """Run the PCG loop as one CUDA conditional graph.

        The stopping test runs on device, so no iteration costs a host stall:
        exactly one readback (``pcg_stats``) happens per component solve,
        instead of one per convergence check. Under ``component`` scope the
        setup, preamble and write-back ride along inside the same graph, and
        the caller must not issue them itself.

        ``dim`` is the launch bucket. ``dt``, the tolerance and the exact
        active count are already on device (``_record_solve_scalars``,
        ``grid.num_leaves``), so the graph does not depend on them.
        """
        key = self._graph_key(scope, dim, weight, viscosity_field, voxel_type,
                              check_every, iterations, comp=comp)
        graphs = self._get_pcg_conditional_graph(
            key, comp, dim, stencil, viscosity_field, weight, voxel_type,
            check_every, iterations, scope)
        if scope != VISCOSITY_GRAPH_SCOPE_COMPONENT:
            self._record_conditional_preamble(dim)
        wp.capture_launch(graphs[0])
        stats = self.pcg_stats.numpy()
        res_sq = float(stats[0])
        it = int(round(float(stats[3])))
        reason = int(round(float(stats[4])))
        status, failure = self._status_from_reason(reason)
        self.last_pcg_execution_mode = "cuda_conditional"
        self._set_viscosity_status(status, res_sq, it, failure)
        return True

    def _solve_pcg_host(self, n, stencil, viscosity_field, weight, voxel_type,
                        check_every, iterations, tol_sq):
        """Host-checked PCG loop: the reference implementation.

        Identical arithmetic to the captured loop, with the criterion
        evaluated after a synchronous readback instead of on device.
        """
        g = self.grid
        self.last_pcg_execution_mode = "host_checked"
        wp.utils.array_inner(self.r, self.z, out=self.rz, count=n)
        history = []
        status = VISCOSITY_STATUS_MAX_ITERATIONS
        failure = "max_iterations"
        res_sq = 0.0
        it = 0
        while it < iterations:
            wp.launch(
                k_viscosity_apply_matrix, dim=n,
                inputs=[self.p_dir, viscosity_field, weight, voxel_type,
                        *stencil, g.voxel_size, self._solve_dt, self.ap],
                device=self.device,
            )
            wp.utils.array_inner(self.p_dir, self.ap, out=self.p_ap, count=n)
            wp.launch(k_compute_alpha, dim=1,
                      inputs=[self.rz, self.p_ap, self.alpha], device=self.device)
            wp.launch(k_update_xr, dim=n,
                      inputs=[self.p_dir, self.ap, self.alpha, g.num_leaves,
                              self.x, self.r],
                      device=self.device)
            wp.launch(k_apply_preconditioner, dim=n,
                      inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                      device=self.device)
            wp.utils.array_inner(self.r, self.z, out=self.rz_new, count=n)
            wp.launch(k_compute_beta, dim=1,
                      inputs=[self.rz_new, self.rz, self.beta], device=self.device)
            wp.launch(k_update_p, dim=n,
                      inputs=[self.z, self.beta, g.num_leaves, self.p_dir],
                      device=self.device)
            wp.copy(self.rz, self.rz_new, count=1)
            it += 1
            if it % check_every == 0 or it == iterations:
                wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
                wp.launch(k_pack_pcg_stats, dim=1,
                          inputs=[self.residual, self.rz, self.p_ap,
                                  self.pcg_stats],
                          device=self.device)
                stats = self.pcg_stats.numpy()
                res_sq = float(stats[0])
                rz_value = float(stats[1])
                p_ap_value = float(stats[2])
                if len(history) >= VISCOSITY_STAGNATION_WINDOW:
                    previous = history[-VISCOSITY_STAGNATION_WINDOW]
                else:
                    previous = math.inf
                history.append(res_sq)
                if not (math.isfinite(res_sq) and math.isfinite(rz_value)
                        and math.isfinite(p_ap_value)):
                    status = VISCOSITY_STATUS_INVALID
                    failure = "non_finite_residual"
                elif res_sq < tol_sq:
                    status = VISCOSITY_STATUS_CONVERGED
                    failure = ""
                elif (rz_value < -1.0e-20 or abs(rz_value) < 1.0e-30
                        or abs(p_ap_value) < 1.0e-30):
                    status = VISCOSITY_STATUS_SINGULAR
                    failure = "singular_system"
                elif res_sq >= previous * (1.0 - 1.0e-6):
                    status = VISCOSITY_STATUS_STALLED
                    failure = "stalled"
                elif it >= iterations:
                    status = VISCOSITY_STATUS_MAX_ITERATIONS
                    failure = "max_iterations"
                else:
                    continue
                break
        self._set_viscosity_status(status, res_sq, it, failure)

    def solve(self, vel, viscosity_field, voxel_type, dt, use_pcg=True,
              iterations=40, tolerance=1.0e-4,
              check_every=DEFAULT_VISCOSITY_CHECK_EVERY):
        statuses = []
        for comp, weight in ((vel.u, vel.wu), (vel.v, vel.wv), (vel.w, vel.ww)):
            statuses.append(self._solve_component(
                comp, weight, viscosity_field, voxel_type, dt,
                use_pcg, iterations, tolerance, check_every))
        self.last_component_statuses = statuses
        if statuses:
            # Report the worst component so a silently non-converged solve is
            # observable; status codes are ordered by severity. Iterations
            # break ties, because when all three converge the severity alone
            # would report whichever component happened to be first (u), and
            # the interesting one is the slowest -- it is the component that
            # would be truncated first if the budget were lowered.
            worst = max(
                statuses,
                key=lambda s: (
                    VISCOSITY_STATUS_CODES.get(
                        s["reason"],
                        VISCOSITY_STATUS_CODES[VISCOSITY_STATUS_INVALID]),
                    s["iterations"],
                ),
            )
            self.last_status = worst
            self.last_viscosity_status = worst["reason"]
            self.last_viscosity_status_code = VISCOSITY_STATUS_CODES.get(
                worst["reason"],
                VISCOSITY_STATUS_CODES[VISCOSITY_STATUS_INVALID])
        return self.last_status


@wp.kernel
def k_axpy_inplace(
    src: wp.array(dtype=float),
    scale: float,
    num_leaves: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    dst[tid] = dst[tid] + scale * src[tid]
