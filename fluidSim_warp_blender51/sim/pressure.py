"""Pressure projection (fluid_pressure_projection family).

Solves the variable-topology Poisson system on fluid voxels of the sparse
grid, with free-surface Dirichlet and solid Neumann conditions:

    A q = -div(u) * h^2        (q = p * dt / rho, folded units)
    u_face -= (q_hi - q_lo) / h

Paths preserved from the original kernel inventory:
- divergence build (+ particle-count pressure anti-clumping source)
- Jacobi iterations (all unroll-width variants collapse to one kernel)
- PCG: inverse-diagonal preconditioner, apply-matrix, alpha/beta/x/r/p updates
  with device-resident reductions
- electrostatic leaf aggregates: a single-level coarse per-leaf potential
  solve (``coarse_initial_guess``) that can seed the PCG initial guess via
  ``solve_pcg(..., use_coarse_guess=True)``. This is real, reachable code,
  but ``project_frame`` (the only production caller) never passes
  ``use_coarse_guess=True``, so it is currently exercised only by tests, not
  by the shipped pipeline. Its leaf reductions are gathers (one thread per
  leaf, fixed voxel order), not atomics.
- multigrid-preconditioned CG (``PressureSolver.preconditioner =
  "MULTIGRID"``): one symmetric V-cycle replaces the Jacobi preconditioner.
  Four levels per leaf (8^3, 4^3, 2^3 and 1 cell per leaf), damped-Jacobi
  smoothing with equal pre/post counts, restriction = 0.25 x the gathered sum
  of the eight children, piecewise-constant prolongation masked to fluid. Every
  kernel is an atomic-free gather with a fixed summation order, so the
  preconditioner is bitwise deterministic and records inside the existing
  CUDA graph capture. The registered
  ``fluidSimProjectNonDivergentSampleDown``/
  ``fluidSimProjectNonDivergentExpandVelocityIntoCollision`` kernels still
  have no call sites; the V-cycle uses the ``k_mg_*`` kernels instead.
- pressure threshold + scale, velocity expansion into collision

No bpy imports allowed here.
"""

from __future__ import annotations

import math
from collections import OrderedDict, namedtuple

import warp as wp
from .fastlaunch import launch as fast_launch

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID
from .enums import ProjectNonDivergentSolver
from .fields import (
    leaf_linear_index,
    voxel_coord_of_flat,
    voxel_flat_index,
)
from .registry import register_fn

PRESSURE_STATUS_NOT_RUN = "not_run"
PRESSURE_STATUS_JACOBI = "jacobi"
PRESSURE_STATUS_CONVERGED = "converged"
PRESSURE_STATUS_MAX_ITERATIONS = "max_iterations"
PRESSURE_STATUS_STALLED = "stalled"
PRESSURE_STATUS_SINGULAR = "singular"
PRESSURE_STATUS_INVALID = "invalid"

PRESSURE_STATUS_CODES = {
    PRESSURE_STATUS_NOT_RUN: 0,
    PRESSURE_STATUS_JACOBI: 1,
    PRESSURE_STATUS_CONVERGED: 2,
    PRESSURE_STATUS_MAX_ITERATIONS: 3,
    PRESSURE_STATUS_STALLED: 4,
    PRESSURE_STATUS_SINGULAR: 5,
    PRESSURE_STATUS_INVALID: 6,
}

# A PCG solve is "stalled" only when this many iterations pass without the
# residual improving on the best value seen in the solve. The CG residual
# 2-norm is not monotone: on a resting Jacobi column it rises ~1.3x by
# iteration 25 and then converges. Comparing each check to the previous one
# stopped those solves at iteration 25 and projected an unconverged pressure,
# sinking the column ~25% in 30 frames (docs/JACOBI_COLUMN_DIAGNOSIS_20261010.md).
PCG_STALL_WINDOW = 100

# Multigrid-preconditioned CG (lane C). The PCG preconditioner is selected by
# the plain attribute ``PressureSolver.preconditioner``. A symmetric V-cycle
# needs equal pre/post smoothing counts and the same smoother on both sides,
# so these three constants are the single source of truth for the V-cycle.
PRECONDITIONER_JACOBI = "JACOBI"
PRECONDITIONER_MULTIGRID = "MULTIGRID"
MG_PRE_SMOOTH = 2
MG_POST_SMOOTH = 2
MG_OMEGA = 0.67
# Hierarchy: level l (0 = the pressure grid) has leaf side 8 >> l voxels, so
# 512, 64, 8 and 1 cells per leaf. The coarsest level is solved by a fixed
# number of damped-Jacobi sweeps from a zero guess. Four sweeps (4.0; was 16)
# keep the preconditioner good enough for the same 4-6 iterations: each sweep
# is a latency-bound graph node, and 16 cost ~0.6 ms/solve on dam_break.
MG_LEVELS = 4
MG_FINE_SHIFT = 3
MG_COARSE_SWEEPS = 4
# Check interval a MULTIGRID solve picks for itself when the caller passes
# none: it must be odd and divide ``max_iterations`` to keep the conditional
# graph (largest first). Jacobi keeps the caller's value / 25 exactly.
# 4.0 checks every iteration: MULTIGRID converges in 4-6, so a coarser
# interval ran up to 4 V-cycles past the tolerance (dam_break +1.09x vs 3.2.1).
MG_CHECK_EVERY_CANDIDATES = (1,)
# A MULTIGRID solve stops at half the configured tolerance. Stopping at the
# tolerance itself (7 V-cycles on chromatic_vortex, where 3.2.1 ran on to 10)
# visibly lowered that demo's liquid level over a full run (mean surface
# volume -2.9%); half the tolerance costs one V-cycle and restores it.
MG_TOLERANCE_SCALE = 0.5
PCG_DEFAULT_CHECK_EVERY = 25
# The residual is in right-hand-side units, -divergence * h^2, so a fixed
# tolerance allows divergence ~ tolerance / h^2: 25x more at 2 mm than at
# 1 cm. At 2 mm the initial residual already passed and the solve ran 0-1
# iterations, so any force (surface tension first) compressed the liquid.
# Finer grids keep the divergence a 1 cm grid allows; 1 cm and coarser grids
# (every shipped demo) keep the tolerance unchanged.
PRESSURE_TOLERANCE_REFERENCE_VOXEL = 0.01


def effective_pressure_tolerance(tolerance: float, voxel_size: float) -> float:
    """The PCG stop tolerance for ``tolerance`` on a grid of ``voxel_size``."""

    ratio = float(voxel_size) / PRESSURE_TOLERANCE_REFERENCE_VOXEL
    if not math.isfinite(ratio) or ratio <= 0.0:
        return float(tolerance)
    return float(tolerance) * min(1.0, ratio * ratio)


@wp.func
def finite_pressure_value(value: float) -> float:
    if wp.isnan(value) or wp.isinf(value):
        return 0.0
    return value


@wp.func
def bounded_pressure_gradient(p_hi: float, p_lo: float, inv_h: float, limit: float) -> float:
    g = (p_hi - p_lo) * inv_h
    if wp.isnan(g) or wp.isinf(g):
        g = 0.0
    return wp.clamp(g, -limit, limit)


# Divergence + matrix helpers


@wp.func
def neighbor_flat(
    flat: wp.int32,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    axis: wp.int32,
    step: wp.int32,
) -> wp.int32:
    local = ijk[axis] & 7
    if local + step >= 0 and local + step < 8:
        stride = wp.int32(1)
        if axis == 1:
            stride = 8
        elif axis == 2:
            stride = 64
        return flat + step * stride
    o = wp.vec3i(ijk[0], ijk[1], ijk[2])
    if axis == 0:
        o = wp.vec3i(ijk[0] + step, ijk[1], ijk[2])
    elif axis == 1:
        o = wp.vec3i(ijk[0], ijk[1] + step, ijk[2])
    else:
        o = wp.vec3i(ijk[0], ijk[1], ijk[2] + step)
    return voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)


@wp.kernel
def k_mark_pressure_unknowns(
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    flags: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    flags[tid] = 0
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) != 0 and (vt & VOXEL_COLLISION) == 0:
        flags[tid] = 1


@wp.kernel
def k_scatter_pressure_unknowns(
    flags: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    size: int,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= size:
        return
    flag = flags[tid]
    if flag != 0:
        indices[offsets[tid]] = tid
    if tid == size - 1:
        count[0] = offsets[tid] + flag


@wp.kernel
def k_compute_divergence(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    collision_vel: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    rhs: wp.array(dtype=float),
    divergence: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentComputeDivergence.

    rhs = -div(u) * h^2 with solid faces replaced by collider velocity.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        rhs[tid] = 0.0
        divergence[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    div = float(0.0)
    # low faces stored at this voxel; high faces at +axis neighbor
    for axis in range(3):
        hi = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, 1)
        lo_n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, -1)
        # A missing neighbour (domain edge or unallocated leaf) is a solid
        # wall to k_project, which zeroes that face; count it as zero here
        # too, or pressure compensates for a flux projection then deletes.
        # Branch rather than wp.where: both wp.where operands are evaluated,
        # which read index -1.
        f_lo = float(0.0)
        f_hi = float(0.0)
        if axis == 0:
            if lo_n >= 0:
                f_lo = u[tid]
            if hi >= 0:
                f_hi = u[hi]
        elif axis == 1:
            if lo_n >= 0:
                f_lo = v[tid]
            if hi >= 0:
                f_hi = v[hi]
        else:
            if lo_n >= 0:
                f_lo = w[tid]
            if hi >= 0:
                f_hi = w[hi]
        # solid neighbors: face takes collider velocity component
        if lo_n >= 0 and (voxel_type[lo_n] & VOXEL_COLLISION) != 0:
            f_lo = collision_vel[lo_n][axis]
        if hi >= 0 and (voxel_type[hi] & VOXEL_COLLISION) != 0:
            f_hi = collision_vel[hi][axis]
        div += f_hi - f_lo
    div = div / voxel_size
    divergence[tid] = div
    rhs[tid] = -div * voxel_size * voxel_size


@wp.kernel
def k_particle_count_to_pressure_rhs(
    voxel_counts: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    rest_count: float,
    stiffness: float,
    voxel_size: float,
    rhs: wp.array(dtype=float),
):
    """fluidSimParticleToCountPressure + fluidSimParticleCountToPressure:
    volume-preservation source that pushes overfull voxels apart."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        return
    if stiffness <= 0.0 or wp.isnan(stiffness) or wp.isinf(stiffness):
        return
    if rest_count <= 0.0 or wp.isnan(rest_count) or wp.isinf(rest_count):
        return
    if voxel_size <= 0.0 or wp.isnan(voxel_size) or wp.isinf(voxel_size):
        return

    over = float(voxel_counts[tid]) - rest_count
    if over > 0.0:
        safe_over = wp.min(over, rest_count)
        source = stiffness * safe_over * voxel_size * voxel_size
        if wp.isnan(source) or wp.isinf(source):
            return
        base = rhs[tid]
        if wp.isnan(base) or wp.isinf(base):
            base = 0.0
        rhs[tid] = base + source


@wp.kernel
def k_compute_inverse_diagonal(
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentComputeInverseDiagonal8/16/32 and the
    collision preconditioner: diag = #non-solid neighbors."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        inv_diag[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    diag = float(0.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            solid = wp.int32(0)
            if n >= 0 and (voxel_type[n] & VOXEL_COLLISION) != 0:
                solid = 1
            if n < 0:
                solid = 1  # unallocated outside == solid domain wall
            if solid == 0:
                diag += 1.0
    if diag > 0.0:
        inv_diag[tid] = 1.0 / diag
    else:
        inv_diag[tid] = 0.0


@wp.kernel
def k_apply_matrix(
    x: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    ax: wp.array(dtype=float),
    compact_x: wp.array(dtype=float),
    compact_ax: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentApplyMatrix8/16/32: Ax for the 7-point
    Poisson stencil restricted to fluid voxels."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        compact_x[work_tid] = 0.0
        compact_ax[work_tid] = 0.0
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0 or inv_diag[tid] == 0.0:
        ax[tid] = 0.0
        compact_x[work_tid] = 0.0
        compact_ax[work_tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    diag = 1.0 / inv_diag[tid]
    acc = diag * x[tid]
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if n >= 0:
                nvt = voxel_type[n]
                if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                    acc -= x[n]
    ax[tid] = acc
    compact_x[work_tid] = x[tid]
    compact_ax[work_tid] = acc


@wp.kernel
def k_jacobi_iteration(
    x_in: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentJacobi{2,4,8,16,32}_(original|optimizedUnroll):
    one damped-free Jacobi sweep; the width variants were OpenCL unroll
    factors and collapse into this single Warp kernel."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0 or inv_diag[tid] == 0.0:
        x_out[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    acc = rhs[tid]
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if n >= 0:
                nvt = voxel_type[n]
                if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                    acc += x_in[n]
    x_out[tid] = acc * inv_diag[tid]



# PCG scalar update kernels (device-resident alpha/beta)


@wp.kernel
def k_zero_pcg_vectors(
    x: wp.array(dtype=float), r: wp.array(dtype=float),
    z: wp.array(dtype=float), p: wp.array(dtype=float),
    ap: wp.array(dtype=float), x0: wp.array(dtype=float),
    zero_x0: int,
):
    i = wp.tid()
    x[i] = 0.0
    r[i] = 0.0
    z[i] = 0.0
    p[i] = 0.0
    ap[i] = 0.0
    if zero_x0 != 0:
        x0[i] = 0.0


@wp.kernel
def k_initialize_pcg(
    rhs: wp.array(dtype=float),
    x0: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    use_index: int,
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
    z: wp.array(dtype=float),
    p: wp.array(dtype=float),
    compact_r: wp.array(dtype=float),
    compact_z: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentInitializePcg: x=x0, r=b-Ax0 (x0 usually 0
    so r=b here; when a coarse initial guess is used the caller pre-applies
    the matrix), z=Minv r, p=z."""
    work_tid = wp.tid()
    tid = work_tid
    if use_index != 0:
        if work_tid >= pressure_count[0]:
            compact_r[work_tid] = 0.0
            compact_z[work_tid] = 0.0
            return
        tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    x[tid] = x0[tid]
    ri = rhs[tid]
    r[tid] = ri
    zi = ri * inv_diag[tid]
    z[tid] = zi
    p[tid] = zi
    if use_index != 0:
        compact_r[work_tid] = ri
        compact_z[work_tid] = zi


@wp.kernel
def k_subtract_ax_from_r(
    ax: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    r: wp.array(dtype=float),
    z: wp.array(dtype=float),
    p: wp.array(dtype=float),
    compact_r: wp.array(dtype=float),
    compact_z: wp.array(dtype=float),
):
    """Complete r = b - A x0 for a nonzero initial guess, refresh z and p."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        compact_r[work_tid] = 0.0
        compact_z[work_tid] = 0.0
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ri = r[tid] - ax[tid]
    r[tid] = ri
    zi = ri * inv_diag[tid]
    z[tid] = zi
    p[tid] = zi
    compact_r[work_tid] = ri
    compact_z[work_tid] = zi


@wp.kernel
def k_compute_alpha(
    rz: wp.array(dtype=float),
    p_ap: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentComputeAlpha."""
    denom = p_ap[0]
    if wp.abs(denom) < 1.0e-30:
        alpha[0] = 0.0
    else:
        alpha[0] = rz[0] / denom


@wp.kernel
def k_update_xr(
    p: wp.array(dtype=float),
    ap: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentUpdateXR."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    a = alpha[0]
    x[tid] = x[tid] + a * p[tid]
    r[tid] = r[tid] - a * ap[tid]


@wp.kernel
def k_apply_preconditioner(
    r: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    z: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentApplyPreConditioner (Jacobi/collision)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    z[tid] = r[tid] * inv_diag[tid]


@wp.kernel
def k_update_xr_preconditioned(
    p: wp.array(dtype=float),
    ap: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
    z: wp.array(dtype=float),
    compact_r: wp.array(dtype=float),
    compact_z: wp.array(dtype=float),
):
    """Update PCG state and apply the Jacobi preconditioner in one pass."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        compact_r[work_tid] = 0.0
        compact_z[work_tid] = 0.0
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    a = alpha[0]
    x[tid] = x[tid] + a * p[tid]
    ri = r[tid] - a * ap[tid]
    r[tid] = ri
    z[tid] = ri * inv_diag[tid]
    compact_r[work_tid] = ri
    compact_z[work_tid] = z[tid]


@wp.kernel
def k_compute_beta(
    rz_new: wp.array(dtype=float),
    rz_old: wp.array(dtype=float),
    beta: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentComputeBeta."""
    denom = rz_old[0]
    if wp.abs(denom) < 1.0e-30:
        beta[0] = 0.0
    else:
        beta[0] = rz_new[0] / denom


@wp.kernel
def k_compute_alpha_and_reset(
    rz: wp.array(dtype=float),
    p_ap_accum: wp.array(dtype=float),
    p_ap: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
):
    denom = p_ap_accum[0]
    p_ap[0] = denom
    p_ap_accum[0] = 0.0
    if wp.abs(denom) < 1.0e-30:
        alpha[0] = 0.0
    else:
        alpha[0] = rz[0] / denom


@wp.kernel
def k_compute_beta_and_reset_old(
    rz_new: wp.array(dtype=float),
    rz_old: wp.array(dtype=float),
    beta: wp.array(dtype=float),
):
    denom = rz_old[0]
    if wp.abs(denom) < 1.0e-30:
        beta[0] = 0.0
    else:
        beta[0] = rz_new[0] / denom
    rz_old[0] = 0.0


@wp.kernel
def k_pack_pcg_stats(
    residual: wp.array(dtype=float),
    rz: wp.array(dtype=float),
    p_ap: wp.array(dtype=float),
    stats: wp.array(dtype=float),
):
    """Pack PCG monitor scalars into one array for a single host readback."""
    stats[0] = residual[0]
    stats[1] = rz[0]
    stats[2] = p_ap[0]


@wp.kernel
def k_prepare_pcg_conditional_loop(
    residual: wp.array(dtype=float),
    rz: wp.array(dtype=float),
    condition: wp.array(dtype=wp.int32),
    iterations: wp.array(dtype=wp.int32),
    last_residual: wp.array(dtype=float),
    tolerance_squared: float,
    loop_tolerance_squared: wp.array(dtype=float),
    stats: wp.array(dtype=float),
):
    res_sq = residual[0]
    keep_going = 1
    reason = 0
    # the captured loop reads the tolerance from here, so one capture serves
    # every tolerance instead of one per active voxel count
    loop_tolerance_squared[0] = tolerance_squared
    if wp.isnan(res_sq) or wp.isinf(res_sq):
        keep_going = 0
        reason = 4
    elif res_sq < tolerance_squared:
        keep_going = 0
        reason = 1
    condition[0] = keep_going
    iterations[0] = 0
    # [0] best residual of this solve, [1] the iteration that reached it
    last_residual[0] = res_sq
    last_residual[1] = 0.0
    stats[0] = res_sq
    stats[1] = rz[0]
    stats[2] = 0.0
    stats[3] = 0.0
    stats[4] = float(reason)
    residual[0] = 0.0


@wp.kernel
def k_finish_pcg_conditional_batch(
    residual: wp.array(dtype=float),
    rz_current: wp.array(dtype=float),
    rz_canonical: wp.array(dtype=float),
    p_ap: wp.array(dtype=float),
    condition: wp.array(dtype=wp.int32),
    iterations: wp.array(dtype=wp.int32),
    last_residual: wp.array(dtype=float),
    batch: int,
    max_iterations: int,
    stall_window: int,
    tolerance_squared: wp.array(dtype=float),
    stats: wp.array(dtype=float),
):
    res_sq = residual[0]
    rz_value = rz_current[0]
    p_ap_value = p_ap[0]
    iteration = iterations[0] + batch
    reason = 0
    keep_going = 0
    if (wp.isnan(res_sq) or wp.isinf(res_sq) or wp.isnan(rz_value)
            or wp.isinf(rz_value) or wp.isnan(p_ap_value) or wp.isinf(p_ap_value)):
        reason = 4
    elif res_sq < tolerance_squared[0]:
        reason = 1
    elif rz_value < -1.0e-20 or wp.abs(rz_value) < 1.0e-30 or wp.abs(p_ap_value) < 1.0e-30:
        reason = 2
    else:
        if res_sq < last_residual[0] * (1.0 - 1.0e-6):
            last_residual[0] = res_sq
            last_residual[1] = float(iteration)
        if iteration - int(last_residual[1]) >= stall_window:
            reason = 3
        elif iteration >= max_iterations:
            reason = 5
        else:
            keep_going = 1
    condition[0] = keep_going
    iterations[0] = iteration
    stats[0] = res_sq
    stats[1] = rz_value
    stats[2] = p_ap_value
    stats[3] = float(iteration)
    stats[4] = float(reason)
    rz_canonical[0] = rz_value
    rz_current[0] = 0.0
    residual[0] = 0.0


@wp.kernel
def k_update_p(
    z: wp.array(dtype=float),
    beta: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    p: wp.array(dtype=float),
):
    """Shared full-grid PCG direction update (also used by viscosity)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p[tid] = z[tid] + beta[0] * p[tid]


@wp.kernel
def k_update_p_indexed(
    z: wp.array(dtype=float),
    beta: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    p: wp.array(dtype=float),
):
    """Compact pressure-only PCG direction update."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p[tid] = z[tid] + beta[0] * p[tid]


# Electrostatic leaf aggregates (coarse long-range solve / multigrid)


@wp.kernel
def k_build_leaf_aggregates(
    rhs: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    leaf_rhs: wp.array(dtype=float),
    leaf_fluid: wp.array(dtype=wp.int32),
):
    """fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates8/16/32.

    A gather, one thread per leaf: the leaf's voxels are summed in a fixed
    order into ``leaf_rhs`` / ``leaf_fluid`` (added onto their current
    contents, like the former per-voxel atomics). Atomic float adds made the
    sum order, and so the coarse guess, nondeterministic.
    """
    slot = wp.tid()
    if slot >= num_leaves[0]:
        return
    base = slot * LEAF_VOLUME
    total = float(0.0)
    count = wp.int32(0)
    for local in range(LEAF_VOLUME):
        vt = voxel_type[base + local]
        if (vt & VOXEL_FLUID) != 0 and (vt & VOXEL_COLLISION) == 0:
            total += rhs[base + local]
            count += 1
    leaf_rhs[slot] = leaf_rhs[slot] + total
    leaf_fluid[slot] = leaf_fluid[slot] + count


@wp.kernel
def k_solve_leaf_potential_jacobi(
    leaf_rhs: wp.array(dtype=float),
    leaf_fluid: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    pot_in: wp.array(dtype=float),
    pot_out: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentElectrostaticSolvePotential8/16/32:
    Jacobi sweep of the leaf-scale Poisson problem (coarse level).
    The coarse cell size is 8h; scaling is folded into the prolongation."""
    slot = wp.tid()
    if slot >= num_leaves[0]:
        return
    if leaf_fluid[slot] == 0:
        pot_out[slot] = 0.0
        return
    lc = leaf_coords[slot]
    acc = leaf_rhs[slot] / float(LEAF_VOLUME) * 64.0  # rhs scaled to coarse h
    diag = float(0.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            o = wp.vec3i(lc[0], lc[1], lc[2])
            if axis == 0:
                o = wp.vec3i(lc[0] + step, lc[1], lc[2])
            elif axis == 1:
                o = wp.vec3i(lc[0], lc[1] + step, lc[2])
            else:
                o = wp.vec3i(lc[0], lc[1], lc[2] + step)
            if o[0] < 0 or o[1] < 0 or o[2] < 0:
                continue
            if o[0] >= leaves_dim[0] or o[1] >= leaves_dim[1] or o[2] >= leaves_dim[2]:
                continue
            nslot = leaf_table[o[0] + leaves_dim[0] * (o[1] + leaves_dim[1] * o[2])]
            diag += 1.0
            if nslot >= 0 and leaf_fluid[nslot] > 0:
                acc += pot_in[nslot]
    if diag > 0.0:
        pot_out[slot] = acc / diag
    else:
        pot_out[slot] = 0.0


@wp.kernel
def k_sample_up_leaf_potential(
    pot: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentSampleUp: prolong the coarse potential into
    the fine unknown as the initial guess."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        x[tid] = 0.0
        return
    x[tid] = pot[tid // LEAF_VOLUME]


@wp.kernel
def k_sample_down_residual(
    r: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    leaf_rhs: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentSampleDown: restrict fine residual to the
    leaf level.

    A gather, one thread per leaf in a fixed voxel order (see
    ``k_build_leaf_aggregates``); added onto ``leaf_rhs``.
    """
    slot = wp.tid()
    if slot >= num_leaves[0]:
        return
    base = slot * LEAF_VOLUME
    total = float(0.0)
    for local in range(LEAF_VOLUME):
        vt = voxel_type[base + local]
        if (vt & VOXEL_FLUID) != 0 and (vt & VOXEL_COLLISION) == 0:
            total += r[base + local]
    leaf_rhs[slot] = leaf_rhs[slot] + total


# Multigrid V-cycle preconditioner (every kernel is an atomic-free gather)
#
# Level l keeps the leaf layout of the pressure grid with the leaf side halved
# per level: ``cells = slot << (3 * shift) | k << (2 * shift) | j << shift | i``
# with ``shift`` = 3, 2, 1, 0 for l = 0..3. Every kernel guards on the device
# side ``num_leaves`` so launch dims can be fixed from the leaf capacity.


@wp.func
def mg_unknown(vt: wp.int32) -> wp.int32:
    """1 for a pressure unknown (fluid, not collision), else 0."""
    if (vt & VOXEL_FLUID) != 0:
        if (vt & VOXEL_COLLISION) == 0:
            return 1
    return 0


@wp.func
def mg_cell_neighbor(
    cell: wp.int32,
    shift: wp.int32,
    axis: wp.int32,
    step: wp.int32,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
) -> wp.int32:
    """Neighbouring cell on ``axis`` at a multigrid level, or -1 when it is
    outside the domain or in an unallocated leaf (a solid wall)."""
    cell_shift = 3 * shift
    mask = (1 << shift) - 1
    local = cell & ((1 << cell_shift) - 1)
    stride = 1 << (axis * shift)
    c = (local >> (axis * shift)) & mask
    nc = c + step
    if nc >= 0 and nc <= mask:
        return cell + step * stride
    slot = cell >> cell_shift
    lc = leaf_coords[slot]
    o = wp.vec3i(lc[0], lc[1], lc[2])
    if axis == 0:
        o = wp.vec3i(lc[0] + step, lc[1], lc[2])
    elif axis == 1:
        o = wp.vec3i(lc[0], lc[1] + step, lc[2])
    else:
        o = wp.vec3i(lc[0], lc[1], lc[2] + step)
    if o[0] < 0 or o[1] < 0 or o[2] < 0:
        return wp.int32(-1)
    if o[0] >= leaves_dim[0] or o[1] >= leaves_dim[1] or o[2] >= leaves_dim[2]:
        return wp.int32(-1)
    nslot = leaf_table[leaf_linear_index(o, leaves_dim)]
    if nslot < 0:
        return wp.int32(-1)
    return (nslot << cell_shift) + local + ((nc & mask) - c) * stride


@wp.func
def mg_apply_a(
    x: wp.array(dtype=float),
    vtype: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    cell: wp.int32,
    shift: wp.int32,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
) -> float:
    """(A x)[cell] for an unknown with ``inv_diag != 0``; same arithmetic as
    ``k_apply_matrix`` (diag * x minus the unknown neighbours)."""
    acc = (1.0 / inv_diag[cell]) * x[cell]
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = mg_cell_neighbor(cell, shift, axis, step, leaf_table, leaf_coords, leaves_dim)
            if n >= 0:
                if mg_unknown(vtype[n]) == 1:
                    acc -= x[n]
    return acc


@wp.kernel
def k_mg_build_types(
    src_type: wp.array(dtype=wp.int32),
    dst_shift: int,
    num_leaves: wp.array(dtype=wp.int32),
    dst_type: wp.array(dtype=wp.int32),
):
    """Coarse cell type from its eight children (one level finer, same leaf):
    fluid if any child is a pressure unknown, solid only if every child is
    solid, otherwise air."""
    tid = wp.tid()
    cell_shift = 3 * dst_shift
    if tid >= (num_leaves[0] << cell_shift):
        return
    slot = tid >> cell_shift
    local = tid & ((1 << cell_shift) - 1)
    mask = (1 << dst_shift) - 1
    ci = local & mask
    cj = (local >> dst_shift) & mask
    ck = local >> (2 * dst_shift)
    src_shift = dst_shift + 1
    base = slot << (3 * src_shift)
    any_unknown = wp.int32(0)
    all_solid = wp.int32(1)
    for c in range(2):
        for b in range(2):
            for a in range(2):
                child = (base + (2 * ci + a) + ((2 * cj + b) << src_shift)
                         + ((2 * ck + c) << (2 * src_shift)))
                vt = src_type[child]
                if mg_unknown(vt) == 1:
                    any_unknown = 1
                if (vt & VOXEL_COLLISION) == 0:
                    all_solid = 0
    if any_unknown == 1:
        dst_type[tid] = VOXEL_FLUID
    elif all_solid == 1:
        dst_type[tid] = VOXEL_COLLISION
    else:
        dst_type[tid] = 0


@wp.kernel
def k_mg_inverse_diagonal(
    vtype: wp.array(dtype=wp.int32),
    shift: int,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
):
    """1 / (#non-solid neighbours) for unknowns, 0 elsewhere. Air neighbours
    count (free-surface Dirichlet); solid, unallocated and out-of-domain
    neighbours do not (Neumann), as in ``k_compute_inverse_diagonal``."""
    tid = wp.tid()
    if tid >= (num_leaves[0] << (3 * shift)):
        return
    if mg_unknown(vtype[tid]) == 0:
        inv_diag[tid] = 0.0
        return
    diag = float(0.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = mg_cell_neighbor(tid, shift, axis, step, leaf_table, leaf_coords, leaves_dim)
            if n >= 0:
                if (vtype[n] & VOXEL_COLLISION) == 0:
                    diag += 1.0
    if diag > 0.0:
        inv_diag[tid] = 1.0 / diag
    else:
        inv_diag[tid] = 0.0


@wp.func
def mg_work_cell(
    tid: wp.int32,
    shift: wp.int32,
    use_idx: wp.int32,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
) -> wp.int32:
    """Cell a work thread owns at a level, or -1 when it owns none.

    ``use_idx != 0`` walks a compact list of active cells (the pressure
    solver's unknown index at the finest level, where only a few percent of
    the allocated cells are unknowns); otherwise one thread per cell."""
    if use_idx != 0:
        if tid >= count[0]:
            return -1
        cell = indices[tid]
        if cell >= (num_leaves[0] << (3 * shift)):
            return -1
        return cell
    if tid >= (num_leaves[0] << (3 * shift)):
        return -1
    return tid


@wp.kernel
def k_mg_smooth_zero(
    b: wp.array(dtype=float),
    vtype: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    omega: float,
    num_leaves: wp.array(dtype=wp.int32),
    shift: int,
    use_idx: int,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=float),
):
    """First damped-Jacobi sweep from x = 0: ``x = omega * b / diag``."""
    tid = mg_work_cell(wp.tid(), shift, use_idx, indices, count, num_leaves)
    if tid < 0:
        return
    inv = inv_diag[tid]
    if mg_unknown(vtype[tid]) == 0 or inv == 0.0:
        x_out[tid] = 0.0
        return
    x_out[tid] = omega * inv * b[tid]


@wp.kernel
def k_mg_smooth(
    x_in: wp.array(dtype=float),
    b: wp.array(dtype=float),
    vtype: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    omega: float,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    shift: int,
    use_idx: int,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=float),
):
    """One damped-Jacobi sweep: ``x_out = x_in + omega * (b - A x_in) / diag``."""
    tid = mg_work_cell(wp.tid(), shift, use_idx, indices, count, num_leaves)
    if tid < 0:
        return
    inv = inv_diag[tid]
    if mg_unknown(vtype[tid]) == 0 or inv == 0.0:
        x_out[tid] = 0.0
        return
    ax = mg_apply_a(x_in, vtype, inv_diag, tid, shift, leaf_table, leaf_coords, leaves_dim)
    x_out[tid] = x_in[tid] + omega * inv * (b[tid] - ax)


@wp.kernel
def k_mg_residual(
    x: wp.array(dtype=float),
    b: wp.array(dtype=float),
    vtype: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    shift: int,
    use_idx: int,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    res_out: wp.array(dtype=float),
):
    """``res_out = b - A x`` at active unknowns (``k_mg_restrict_sum`` reads
    only those, so other cells need no clearing). Kept apart from the
    restriction so the stencil is evaluated once per fine cell by many threads
    instead of eight times by one thread."""
    tid = mg_work_cell(wp.tid(), shift, use_idx, indices, count, num_leaves)
    if tid < 0:
        return
    if mg_unknown(vtype[tid]) == 0 or inv_diag[tid] == 0.0:
        res_out[tid] = 0.0
        return
    ax = mg_apply_a(x, vtype, inv_diag, tid, shift, leaf_table, leaf_coords, leaves_dim)
    res_out[tid] = b[tid] - ax


@wp.kernel
def k_mg_restrict_sum(
    res: wp.array(dtype=float),
    fine_vtype: wp.array(dtype=wp.int32),
    fine_inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    fine_shift: int,
    coarse_b: wp.array(dtype=float),
):
    """``coarse_b = 0.25 * sum over the 8 children of res``: a gather, one
    thread per coarse cell, children visited in a fixed order. Only active
    fine unknowns contribute."""
    tid = wp.tid()
    coarse_shift = fine_shift - 1
    cell_shift = 3 * coarse_shift
    if tid >= (num_leaves[0] << cell_shift):
        return
    slot = tid >> cell_shift
    local = tid & ((1 << cell_shift) - 1)
    mask = (1 << coarse_shift) - 1
    ci = local & mask
    cj = (local >> coarse_shift) & mask
    ck = local >> (2 * coarse_shift)
    base = slot << (3 * fine_shift)
    acc = float(0.0)
    for c in range(2):
        for bb in range(2):
            for a in range(2):
                child = (base + (2 * ci + a) + ((2 * cj + bb) << fine_shift)
                         + ((2 * ck + c) << (2 * fine_shift)))
                if mg_unknown(fine_vtype[child]) == 1:
                    if fine_inv_diag[child] != 0.0:
                        acc += res[child]
    coarse_b[tid] = 0.25 * acc


@wp.func
def mg_prolonged(
    x: wp.array(dtype=float),
    e: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    cell: wp.int32,
    shift: wp.int32,
) -> float:
    """``(x + P e)[cell]``: piecewise-constant prolongation from the parent
    coarse cell, applied where ``inv_diag != 0`` (active unknowns) only."""
    xv = x[cell]
    if inv_diag[cell] != 0.0:
        d = shift - 1
        cell_shift = 3 * shift
        slot = cell >> cell_shift
        local = cell & ((1 << cell_shift) - 1)
        mask = (1 << shift) - 1
        i = local & mask
        j = (local >> shift) & mask
        k = local >> (2 * shift)
        parent = (slot << (3 * d)) + (i >> 1) + ((j >> 1) << d) + ((k >> 1) << (2 * d))
        xv = xv + e[parent]
    return xv


@wp.kernel
def k_mg_smooth_prolong(
    x_in: wp.array(dtype=float),
    e: wp.array(dtype=float),
    b: wp.array(dtype=float),
    vtype: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    omega: float,
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    shift: int,
    use_idx: int,
    indices: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    x_out: wp.array(dtype=float),
):
    """First post-smoothing sweep with the coarse-grid correction folded in:
    one damped-Jacobi sweep from ``x_in + P e``, which is never stored. Same
    arithmetic as a separate prolongation followed by ``k_mg_smooth``."""
    tid = mg_work_cell(wp.tid(), shift, use_idx, indices, count, num_leaves)
    if tid < 0:
        return
    inv = inv_diag[tid]
    if mg_unknown(vtype[tid]) == 0 or inv == 0.0:
        x_out[tid] = 0.0
        return
    xc = mg_prolonged(x_in, e, inv_diag, tid, shift)
    acc = (1.0 / inv) * xc
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = mg_cell_neighbor(tid, shift, axis, step, leaf_table, leaf_coords, leaves_dim)
            if n >= 0:
                if mg_unknown(vtype[n]) == 1:
                    acc -= mg_prolonged(x_in, e, inv_diag, n, shift)
    x_out[tid] = xc + omega * inv * (b[tid] - acc)


@wp.kernel
def k_mg_update_xr_indexed(
    p: wp.array(dtype=float),
    ap: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
    compact_r: wp.array(dtype=float),
):
    """``x += alpha p``, ``r -= alpha Ap`` over the compact unknowns. Like
    ``k_update_xr_preconditioned`` without the fused Jacobi ``z``: the V-cycle
    produces ``z`` afterwards."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        compact_r[work_tid] = 0.0
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    a = alpha[0]
    x[tid] = x[tid] + a * p[tid]
    ri = r[tid] - a * ap[tid]
    r[tid] = ri
    compact_r[work_tid] = ri


@wp.kernel
def k_mg_gather_z(
    z: wp.array(dtype=float),
    pressure_indices: wp.array(dtype=wp.int32),
    pressure_count: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    set_p: int,
    p: wp.array(dtype=float),
    compact_z: wp.array(dtype=float),
):
    """Compact the V-cycle output for the r.z reduction (zero beyond the
    unknown count); with ``set_p`` also start the search direction ``p = z``."""
    work_tid = wp.tid()
    if work_tid >= pressure_count[0]:
        compact_z[work_tid] = 0.0
        return
    tid = pressure_indices[work_tid]
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    zi = z[tid]
    compact_z[work_tid] = zi
    if set_p != 0:
        p[tid] = zi


# Projection + post ops


@wp.func
def solid_face_weight(weight: float, here_solid: bool, n_solid: wp.int32) -> float:
    """Sampling weight for a face that touches solid.

    A face with solid on exactly one side is a wall: it keeps the collider
    velocity and a valid weight, so fluid cannot flow through it. A face with
    solid on both sides lies inside the collider. It is not a fluid face, so
    it gets weight 0 and interpolation skips it; otherwise fluid beside a
    wall samples those pinned tangential faces and is held against the wall
    (no-slip) instead of sliding along it (free-slip).
    """
    if here_solid and n_solid != 0:
        return 0.0
    return wp.max(weight, 1.0e-6)


@wp.kernel
def k_project(
    x: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    max_gradient: float,
    collision_vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentProject: subtract the pressure gradient on
    faces between (fluid|air) voxels; solid faces take collider velocity, and
    faces inside a collider are invalidated (see solid_face_weight)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    vt = voxel_type[tid]
    inv_h = 1.0 / voxel_size
    here_fluid = (vt & VOXEL_FLUID) != 0 and (vt & VOXEL_COLLISION) == 0
    here_solid = (vt & VOXEL_COLLISION) != 0
    limit = wp.max(max_gradient, 1.0e-6)
    x_here = float(0.0)
    if here_fluid:
        x_here = finite_pressure_value(x[tid])
    for axis in range(3):
        n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, -1)
        n_fluid = wp.int32(0)
        n_solid = wp.int32(0)
        x_n = float(0.0)
        if n >= 0:
            nvt = voxel_type[n]
            if (nvt & VOXEL_COLLISION) != 0:
                n_solid = 1
            elif (nvt & VOXEL_FLUID) != 0:
                n_fluid = 1
                x_n = finite_pressure_value(x[n])
        else:
            n_solid = 1  # domain wall
        if here_solid or n_solid != 0:
            # face on a solid boundary: prescribe collider velocity
            vel_c = wp.vec3(0.0, 0.0, 0.0)
            if here_solid:
                vel_c = collision_vel[tid]
            elif n >= 0:
                vel_c = collision_vel[n]
            if axis == 0:
                u[tid] = vel_c[0]
                wu[tid] = solid_face_weight(wu[tid], here_solid, n_solid)
            elif axis == 1:
                v[tid] = vel_c[1]
                wv[tid] = solid_face_weight(wv[tid], here_solid, n_solid)
            else:
                w[tid] = vel_c[2]
                ww[tid] = solid_face_weight(ww[tid], here_solid, n_solid)
        elif here_fluid or n_fluid != 0:
            g = bounded_pressure_gradient(x_here, x_n, inv_h, limit)
            if axis == 0:
                u[tid] = u[tid] - g
            elif axis == 1:
                v[tid] = v[tid] - g
            else:
                w[tid] = w[tid] - g



@wp.kernel
def k_expand_velocity_into_collision(
    u_in: wp.array(dtype=float), v_in: wp.array(dtype=float), w_in: wp.array(dtype=float),
    wu_in: wp.array(dtype=float), wv_in: wp.array(dtype=float), ww_in: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    u_out: wp.array(dtype=float), v_out: wp.array(dtype=float), w_out: wp.array(dtype=float),
    wu_out: wp.array(dtype=float), wv_out: wp.array(dtype=float), ww_out: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentExpandVelocityIntoCollision: flood valid
    velocity one ring into empty/collision voxels for clean sampling."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    su = float(0.0)
    sv = float(0.0)
    sw = float(0.0)
    nu = float(0.0)
    nv = float(0.0)
    nw = float(0.0)
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                o = wp.vec3i(ijk[0] + dx, ijk[1] + dy, ijk[2] + dz)
                flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
                if flat < 0:
                    continue
                if wu_in[flat] > 1.0e-7:
                    su += u_in[flat]
                    nu += 1.0
                if wv_in[flat] > 1.0e-7:
                    sv += v_in[flat]
                    nv += 1.0
                if ww_in[flat] > 1.0e-7:
                    sw += w_in[flat]
                    nw += 1.0
    if wu_in[tid] > 1.0e-7:
        u_out[tid] = u_in[tid]
        wu_out[tid] = wu_in[tid]
    elif nu > 0.0:
        u_out[tid] = su / nu
        wu_out[tid] = 1.0e-6
    else:
        u_out[tid] = u_in[tid]
        wu_out[tid] = wu_in[tid]
    if wv_in[tid] > 1.0e-7:
        v_out[tid] = v_in[tid]
        wv_out[tid] = wv_in[tid]
    elif nv > 0.0:
        v_out[tid] = sv / nv
        wv_out[tid] = 1.0e-6
    else:
        v_out[tid] = v_in[tid]
        wv_out[tid] = wv_in[tid]
    if ww_in[tid] > 1.0e-7:
        w_out[tid] = w_in[tid]
        ww_out[tid] = ww_in[tid]
    elif nw > 0.0:
        w_out[tid] = sw / nw
        ww_out[tid] = 1.0e-6
    else:
        w_out[tid] = w_in[tid]
        ww_out[tid] = ww_in[tid]


@wp.kernel
def k_pressure_threshold_compute(
    x: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    out_max: wp.array(dtype=float),
):
    """fluidSimPressureThresholdCompute: max |pressure| over active voxels."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    v = x[tid]
    if not wp.isnan(v) and not wp.isinf(v):
        wp.atomic_max(out_max, 0, wp.abs(v))


@wp.kernel
def k_pressure_threshold_apply(
    max_val: wp.array(dtype=float),
    relative_threshold: float,
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
):
    """fluidSimPressureThresholdApply: zero small pressures."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return

    v = x[tid]
    if wp.isnan(v) or wp.isinf(v):
        x[tid] = 0.0
        return

    threshold = relative_threshold * max_val[0]
    if threshold <= 0.0 or wp.isnan(threshold) or wp.isinf(threshold):
        return
    if wp.abs(v) < threshold:
        x[tid] = 0.0


@wp.kernel
def k_pressure_scale(
    scale: float,
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
):
    """fluidSimPressureScale."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    x[tid] = x[tid] * scale


# Fixed partial-sum count for ``measure_residual``: fixed, so the reduction
# order (and therefore the reported residual) is the same on every call.
RESIDUAL_PARTIALS = 65536


@wp.kernel
def k_residual_partial_squares(
    rhs: wp.array(dtype=float),
    ap: wp.array(dtype=float),
    n: int,
    stride: int,
    partials: wp.array(dtype=wp.float64),
):
    """Strided partial sums of ``(rhs - ap)^2`` for the residual norm.

    The difference is formed in float32, exactly as the old host readback
    did; only the accumulation is widened to float64.
    """
    tid = wp.tid()
    acc = wp.float64(0.0)
    i = tid
    while i < n:
        d = wp.float64(rhs[i] - ap[i])
        acc += d * d
        i += stride
    partials[tid] = acc



register_fn(k_compute_divergence,
            "fluidSimProjectNonDivergentComputeDivergence",
            "fluidSimProjectNonDivergentComputeDivergence_old")
register_fn(k_particle_count_to_pressure_rhs,
            "fluidSimParticleToCountPressure", "fluidSimParticleCountToPressure")
register_fn(k_compute_inverse_diagonal,
            "fluidSimProjectNonDivergentPreConditionerNew_internal",
            "fluidSimProjectNonDivergentCollisionPreConditionerNew",
            "fluidSimProjectNonDivergentComputeInverseDiagonal8",
            "fluidSimProjectNonDivergentComputeInverseDiagonal16",
            "fluidSimProjectNonDivergentComputeInverseDiagonal32",
            strategy="flat",
            notes="workgroup-width variants collapse to one kernel")
register_fn(k_apply_matrix,
            "fluidSimProjectNonDivergentApplyMatrix8",
            "fluidSimProjectNonDivergentApplyMatrix16",
            "fluidSimProjectNonDivergentApplyMatrix32",
            strategy="tiled")
for _w in (2, 4, 8, 16, 32):
    register_fn(k_jacobi_iteration,
                f"fluidSimProjectNonDivergentJacobi{_w}_original",
                f"fluidSimProjectNonDivergentJacobi{_w}_optimizedUnroll",
                strategy="tiled",
                notes="unroll-width variants collapse to one kernel")
register_fn(k_initialize_pcg, "fluidSimProjectNonDivergentInitializePcg")
register_fn(k_compute_alpha, "fluidSimProjectNonDivergentComputeAlpha")
register_fn(k_update_xr, "fluidSimProjectNonDivergentUpdateXR")
register_fn(k_apply_preconditioner, "fluidSimProjectNonDivergentApplyPreConditioner")
register_fn(k_compute_beta, "fluidSimProjectNonDivergentComputeBeta")
register_fn(k_compute_alpha_and_reset,
            "fluidSimProjectNonDivergentComputeAlphaAndReset")
register_fn(k_compute_beta_and_reset_old,
            "fluidSimProjectNonDivergentComputeBetaAndResetOld")
register_fn(k_prepare_pcg_conditional_loop,
            "fluidSimProjectNonDivergentPrepareConditionalLoop")
register_fn(k_finish_pcg_conditional_batch,
            "fluidSimProjectNonDivergentFinishConditionalBatch")
register_fn(k_update_p_indexed, "fluidSimProjectNonDivergentUpdateP")
register_fn(k_build_leaf_aggregates,
            "fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates8",
            "fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates16",
            "fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates32",
            strategy="tiled")
register_fn(k_solve_leaf_potential_jacobi,
            "fluidSimProjectNonDivergentElectrostaticSolvePotential8",
            "fluidSimProjectNonDivergentElectrostaticSolvePotential16",
            "fluidSimProjectNonDivergentElectrostaticSolvePotential32",
            strategy="tiled")
register_fn(k_sample_up_leaf_potential, "fluidSimProjectNonDivergentSampleUp")
register_fn(k_sample_down_residual, "fluidSimProjectNonDivergentSampleDown",
            notes="Leaf-level gather reduction (no atomics); no call sites. "
                  "The multigrid V-cycle uses the k_mg_* kernels.")
register_fn(k_project, "fluidSimProjectNonDivergentProject")
register_fn(k_expand_velocity_into_collision,
            "fluidSimProjectNonDivergentExpandVelocityIntoCollision",
            notes="No call sites in this module.")
register_fn(k_pressure_threshold_compute, "fluidSimPressureThresholdCompute")
register_fn(k_pressure_threshold_apply, "fluidSimPressureThresholdApply")
register_fn(k_pressure_scale, "fluidSimPressureScale")


# Host-orchestrated solver driver (GPU-resident state)


class _MGView(namedtuple(
        "_MGView", "shift vtype inv_diag b x t dim use_idx indices count")):
    """One level of a V-cycle: cell types, inverse diagonal, right-hand side,
    solution, and the ping-pong temporary. Level 0 aliases the caller's
    arrays; coarse levels alias ``_MGLevel`` storage. ``dim`` is the launch
    dim of the level's work kernels; with ``use_idx`` they walk the compact
    cell list ``indices`` (device count ``count``) instead of every cell."""

    __slots__ = ()


class _MGLevel:
    """Device storage for one coarse multigrid level (levels 1..MG_LEVELS-1).

    ``shift`` is log2 of the level's leaf side (2, 1, 0 for 4^3, 2^3, 1 cell
    per leaf), so a level holds ``capacity << (3 * shift)`` cells of each of
    five arrays: cell type, inverse diagonal, rhs, solution and temporary.
    Allocated once and never resized: captured graphs bake the pointers.
    """

    __slots__ = ("shift", "vtype", "inv_diag", "b", "x", "t")

    def __init__(self, shift, leaf_capacity, device):
        n = max(1, leaf_capacity << (3 * shift))
        self.shift = shift
        self.vtype = wp.zeros(n, dtype=wp.int32, device=device)
        self.inv_diag = wp.zeros(n, dtype=float, device=device)
        self.b = wp.zeros(n, dtype=float, device=device)
        self.x = wp.zeros(n, dtype=float, device=device)
        self.t = wp.zeros(n, dtype=float, device=device)

    def nbytes(self):
        return sum(
            a.size * wp.types.type_size_in_bytes(a.dtype)
            for a in (self.vtype, self.inv_diag, self.b, self.x, self.t))


class PressureSolver:
    """Owns pressure work buffers and runs Jacobi or PCG projections."""

    # Captured CUDA graphs kept alive at once, evicted least-recently-used.
    _PCG_GRAPH_CAPACITY = 8

    # PCG preconditioner kind: "JACOBI" (default, fused into the PCG kernels)
    # or "MULTIGRID" (symmetric V-cycle, see ``_apply_mg_vcycle``). Plain
    # attribute on purpose: no RNA / SolverSettings wiring in this lane.
    preconditioner = PRECONDITIONER_JACOBI
    # Class-level smoothing counts; tests override them per instance.
    _mg_pre_smooth = MG_PRE_SMOOTH
    _mg_post_smooth = MG_POST_SMOOTH
    # Multigrid state. Class-level defaults so a solver built without
    # ``__init__`` (release-path tests) still reads a coherent "no MG" state.
    _mg_levels = None          # [None, level1, level2, level3] once allocated
    _mg_voxel_type = None      # level-0 voxel_type the hierarchy was built for
    _mg_leaf_n = 0             # launch-dim leaf count (pow2, <= grid.capacity)
    _pcg_cmd_kind = None       # preconditioner the recorded PCG commands serve
    _pcg_mg_gather_cmd = None  # compacts the V-cycle output for the r.z dot
    last_pcg_check_every = 0   # check interval the last solve_pcg ran with
    _residual_partials = None  # float64 scratch for measure_residual, lazy
    # Optional learned initial guess (sim.learned_guess.LearnedGuess). Set by
    # FluidSolver only when ``pressure_learned_guess`` is on; None otherwise,
    # and an inactive guess leaves every solve on the zero-guess path.
    learned_guess = None

    def __del__(self):
        # Cyclic validation/benchmark owners may reach GC without an explicit
        # release. Never let their native lease outlive the captured graphs.
        try:
            self.release()
        except Exception:
            pass

    def __init__(self, grid, device):
        self.grid = grid
        self.device = device
        n = grid.voxel_capacity
        self.rhs = wp.zeros(n, dtype=float, device=device)
        self.divergence = wp.zeros(n, dtype=float, device=device)
        self.pressure = wp.zeros(n, dtype=float, device=device)
        self.pressure_tmp = wp.zeros(n, dtype=float, device=device)
        self.inv_diag = wp.zeros(n, dtype=float, device=device)
        self.r = wp.zeros(n, dtype=float, device=device)
        self.z = wp.zeros(n, dtype=float, device=device)
        self.p_dir = wp.zeros(n, dtype=float, device=device)
        self.ap = wp.zeros(n, dtype=float, device=device)
        self.x0 = wp.zeros(n, dtype=float, device=device)
        self._pressure_flags = wp.zeros(n, dtype=wp.int32, device=device)
        self._pressure_offsets = wp.zeros(n, dtype=wp.int32, device=device)
        self.pressure_indices = wp.zeros(n, dtype=wp.int32, device=device)
        self._pressure_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.pressure_count = 0
        self._pcg_compact_p = wp.zeros(n, dtype=float, device=device)
        self._pcg_compact_ap = wp.zeros(n, dtype=float, device=device)
        self._pcg_compact_r = wp.zeros(n, dtype=float, device=device)
        self._pcg_compact_z = wp.zeros(n, dtype=float, device=device)
        # coarse (leaf) level
        self.leaf_rhs = wp.zeros(grid.capacity, dtype=float, device=device)
        self.leaf_fluid = wp.zeros(grid.capacity, dtype=wp.int32, device=device)
        self.leaf_pot = wp.zeros(grid.capacity, dtype=float, device=device)
        self.leaf_pot_tmp = wp.zeros(grid.capacity, dtype=float, device=device)
        # device scalars
        self.rz = wp.zeros(1, dtype=float, device=device)
        self.rz_new = wp.zeros(1, dtype=float, device=device)
        self.p_ap = wp.zeros(1, dtype=float, device=device)
        self._pcg_p_ap_accum = wp.zeros(1, dtype=float, device=device)
        self.alpha = wp.zeros(1, dtype=float, device=device)
        self.beta = wp.zeros(1, dtype=float, device=device)
        self.residual = wp.zeros(1, dtype=float, device=device)
        self.max_pressure = wp.zeros(1, dtype=float, device=device)
        # packed [residual^2, r.z, p.Ap] monitor scalars: one readback per
        # convergence checkpoint instead of three per PCG iteration.
        self.pcg_stats = wp.zeros(5, dtype=float, device=device)
        self._pcg_condition = wp.zeros(1, dtype=wp.int32, device=device)
        self._pcg_iterations = wp.zeros(1, dtype=wp.int32, device=device)
        self._pcg_last_checked_res = wp.zeros(2, dtype=float, device=device)
        self._pcg_tol_sq = wp.zeros(1, dtype=float, device=device)
        self._pcg_voxel_type = None
        self._pcg_cmd_kind = None
        self._pcg_mg_gather_cmd = None
        self._mg_levels = None
        self._mg_voxel_type = None
        self._mg_leaf_n = 0
        self._mg_work_n = 0
        self._mg_graphs = OrderedDict()
        self.last_pcg_check_every = 0
        self._pcg_apply_cmd = None
        self._pcg_update_cmd = None
        self._pcg_update_p_cmd = None
        self._pcg_rz_a = self.rz
        self._pcg_rz_b = self.rz_new
        self._pcg_alpha_cmds = None
        self._pcg_beta_cmds = None
        self._pcg_pack_cmds = None
        self._pcg_graphs = OrderedDict()
        self._pcg_conditional_graphs = OrderedDict()
        self._pcg_dot = None
        self._pcg_native_available = None
        self.last_pcg_native_error = ""
        self.last_pcg_execution_mode = "host_batched"
        self.last_iterations = 0
        self.last_residual = 0.0
        self.last_residual_squared = 0.0
        self.last_pressure_status = PRESSURE_STATUS_NOT_RUN
        self.last_pressure_status_code = PRESSURE_STATUS_CODES[PRESSURE_STATUS_NOT_RUN]
        self.last_pcg_failure = ""
        self.last_pressure_valid = False
        self.last_status = self._make_status(
            False, PRESSURE_STATUS_NOT_RUN, 0.0, 0)

    # -- helpers -------------------------------------------------------------

    def _dim(self):
        return max(1, self.grid.active_voxel_count)

    def _leaf_dim(self):
        return max(1, self.grid.active_leaf_count)

    def _stencil_args(self):
        g = self.grid
        return [g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim, g.num_leaves]

    def _build_pressure_index(self, voxel_type):
        n = self._dim()
        fast_launch(
            k_mark_pressure_unknowns, dim=n,
            inputs=[voxel_type, self.grid.num_leaves, self._pressure_flags],
            device=self.device,
        )
        wp.utils.array_scan(
            self._pressure_flags[:n], self._pressure_offsets[:n], inclusive=False)
        fast_launch(
            k_scatter_pressure_unknowns, dim=n,
            inputs=[self._pressure_flags, self._pressure_offsets, n,
                    self.pressure_indices, self._pressure_count],
            device=self.device,
        )
        self.pressure_count = int(self._pressure_count.numpy()[0])
        return self.pressure_count

    def _zero_pcg_vectors(self, n, x, zero_x0=False):
        """Clear the PCG working set over the launch range [0, n).

        Both PCG solvers launch and reduce over ``n`` derived from the host
        leaf mirror, while the kernels themselves are guarded by the
        device-side ``num_leaves``. Any slot the device guard skips is never
        written by the initializer, so without this it would carry the
        previous frame's values straight into the r.z / p.Ap reductions.
        """
        fast_launch(
            k_zero_pcg_vectors, dim=n,
            inputs=[x, self.r, self.z, self.p_dir, self.ap, self.x0,
                    1 if zero_x0 else 0],
            device=self.device,
        )

    def _persistent_inner_product(self, a, b, out, count):
        out.zero_()
        if self._ensure_pcg_dot():
            self._pcg_dot.launch(a, b, out, count)
        else:
            wp.utils.array_inner(a, b, out=out, count=count)

    def _ensure_pcg_dot(self):
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

    def _release_native_pcg_dot(self):
        """Drop captured graphs before releasing their native kernel lease."""
        dot = self._pcg_dot
        self._pcg_dot = None
        self._pcg_graphs.clear()
        self._pcg_conditional_graphs.clear()
        self._mg_graphs.clear()
        if dot is not None:
            dot.release()

    def release(self):
        """Release captured PCG work and its native CUDA module lease."""
        self._release_native_pcg_dot()
        if self.learned_guess is not None:
            self.learned_guess.release()
        self._pcg_voxel_type = None
        self._pcg_cmd_kind = None
        self._pcg_apply_cmd = None
        self._pcg_update_cmd = None
        self._pcg_update_p_cmd = None
        self._pcg_mg_gather_cmd = None
        self._pcg_alpha_cmds = None
        self._pcg_beta_cmds = None
        self._pcg_pack_cmds = None

    def _pcg_kind(self):
        """The validated preconditioner kind (``JACOBI`` or ``MULTIGRID``)."""
        kind = self.preconditioner
        if kind not in (PRECONDITIONER_JACOBI, PRECONDITIONER_MULTIGRID):
            raise ValueError(
                f"unknown PCG preconditioner {kind!r}; expected "
                f"{PRECONDITIONER_JACOBI!r} or {PRECONDITIONER_MULTIGRID!r}")
        return kind

    def _pcg_graph_key_tail(self):
        """Key suffix that keeps preconditioner variants apart in both graph
        caches. Empty for Jacobi, so Jacobi keys stay ``(n, batch, x)``.

        A MULTIGRID capture bakes the V-cycle's launch dims (``_mg_leaf_n``)
        and its smoothing counts into the graph, so all of them belong in the
        key; a Jacobi graph can then never be replayed for a MULTIGRID solve
        or the other way round.
        """
        if self._pcg_cmd_kind == PRECONDITIONER_MULTIGRID:
            return ("MG", self._mg_leaf_n, self._mg_pre_smooth,
                    self._mg_post_smooth, MG_COARSE_SWEEPS)
        return ()

    def _set_pcg_cmd_dims(self, work_n):
        self._pcg_apply_cmd.set_dim(work_n)
        self._pcg_update_cmd.set_dim(work_n)
        self._pcg_update_p_cmd.set_dim(work_n)
        if self._pcg_mg_gather_cmd is not None:
            self._pcg_mg_gather_cmd.set_dim(work_n)

    def _record_pcg_iterations(self, work_n, batch, start_parity):
        parity = start_parity
        for _ in range(batch):
            self._pcg_apply_cmd.launch()
            if self._pcg_dot is None:
                wp.utils.array_inner(
                    self._pcg_compact_p, self._pcg_compact_ap,
                    out=self._pcg_p_ap_accum, count=work_n)
            else:
                self._pcg_dot.launch(
                    self._pcg_compact_p, self._pcg_compact_ap,
                    self._pcg_p_ap_accum, work_n)
            self._pcg_alpha_cmds[parity].launch()
            self._pcg_update_cmd.launch()
            if self._pcg_cmd_kind == PRECONDITIONER_MULTIGRID:
                # z = M^-1 r (symmetric V-cycle), then compact it for r.z
                self._apply_mg_vcycle(self.r, self.z)
                self._pcg_mg_gather_cmd.launch()
            rz_out = self._pcg_rz_b if parity == 0 else self._pcg_rz_a
            if self._pcg_dot is None:
                wp.utils.array_inner(
                    self._pcg_compact_r, self._pcg_compact_z,
                    out=rz_out, count=work_n)
            else:
                self._pcg_dot.launch(
                    self._pcg_compact_r, self._pcg_compact_z, rz_out, work_n)
            self._pcg_beta_cmds[parity].launch()
            self._pcg_update_p_cmd.launch()
            parity ^= 1
        return parity

    def _launch_pcg_batch(self, work_n, batch):
        """Launch an unchanged fixed-coefficient PCG batch as one CUDA graph."""
        start_parity = 0 if self.rz is self._pcg_rz_a else 1
        key = (work_n, batch, start_parity) + self._pcg_graph_key_tail()
        graph = self._pcg_graphs.get(key)
        if graph is not None:
            self._pcg_graphs.move_to_end(key)
        else:
            self._ensure_pcg_dot()
            # Evict one LRU entry. Flushing the whole map instead forced every
            # surviving key to re-capture (and re-pay the synchronize below)
            # every time the tail-iteration batch minted a fresh key.
            if len(self._pcg_graphs) >= self._PCG_GRAPH_CAPACITY:
                self._pcg_graphs.popitem(last=False)
            self._set_pcg_cmd_dims(work_n)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._record_pcg_iterations(work_n, batch, start_parity)
            graph = capture.graph
            self._pcg_graphs[key] = graph
        wp.capture_launch(graph)
        if batch & 1:
            self.rz, self.rz_new = self.rz_new, self.rz

    def _get_pcg_conditional_graphs(self, work_n, batch, max_iterations):
        self._set_pcg_cmd_dims(work_n)
        # No tolerance here: it scales with the active voxel count, so keying
        # on it re-captured on almost every solve. The loop reads it from
        # _pcg_tol_sq, which k_prepare_pcg_conditional_loop writes per solve.
        key = (work_n, batch, max_iterations) + self._pcg_graph_key_tail()
        graphs = self._pcg_conditional_graphs.get(key)
        if graphs is not None:
            self._pcg_conditional_graphs.move_to_end(key)
        else:
            # Evict one LRU entry, as _launch_pcg_batch does, so a cold key
            # does not force every surviving shape to re-capture.
            if len(self._pcg_conditional_graphs) >= self._PCG_GRAPH_CAPACITY:
                self._pcg_conditional_graphs.popitem(last=False)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as body_capture:
                parity = self._record_pcg_iterations(work_n, batch, 0)
                if parity != 1:
                    raise RuntimeError("Conditional PCG batches must have odd parity")
                self._pcg_dot.launch(
                    self._pcg_compact_r, self._pcg_compact_r,
                    self.residual, work_n)
                fast_launch(
                    k_finish_pcg_conditional_batch,
                    dim=1,
                    inputs=[self.residual, self._pcg_rz_b, self._pcg_rz_a,
                            self.p_ap, self._pcg_condition,
                            self._pcg_iterations, self._pcg_last_checked_res,
                            batch, max_iterations,
                            self._pcg_stall_window(batch), self._pcg_tol_sq,
                            self.pcg_stats],
                    device=self.device,
                )
            with wp.ScopedCapture(device=self.device) as loop_capture:
                wp.capture_while(
                    condition=self._pcg_condition,
                    while_body=body_capture.graph,
                )
            graphs = (loop_capture.graph, body_capture.graph)
            self._pcg_conditional_graphs[key] = graphs
        return graphs

    def _launch_pcg_conditional(self, work_n, graphs, tol_sq):
        if self.rz is self._pcg_rz_b:
            wp.copy(self._pcg_rz_a, self._pcg_rz_b, count=1)
        self._pcg_rz_b.zero_()
        self._pcg_p_ap_accum.zero_()
        self.rz = self._pcg_rz_a
        self.rz_new = self._pcg_rz_b
        fast_launch(
            k_prepare_pcg_conditional_loop,
            dim=1,
            inputs=[self.residual, self.rz, self._pcg_condition,
                    self._pcg_iterations, self._pcg_last_checked_res,
                    tol_sq, self._pcg_tol_sq, self.pcg_stats],
            device=self.device,
        )
        wp.capture_launch(graphs[0])
        self.rz = self._pcg_rz_a
        self.rz_new = self._pcg_rz_b
        return self.pcg_stats.numpy()

    def _prepare_update_cmds(self, n, kind, x):
        """Record the preconditioner-specific part of the PCG iteration.

        JACOBI fuses ``z = D^-1 r`` into the x/r update. MULTIGRID updates x
        and r only; the V-cycle and a compaction of its output follow in
        ``_record_pcg_iterations``.
        """
        g = self.grid
        if kind == PRECONDITIONER_MULTIGRID:
            self._pcg_update_cmd = fast_launch(
                k_mg_update_xr_indexed, dim=n,
                inputs=[self.p_dir, self.ap, self.alpha,
                        self.pressure_indices, self._pressure_count,
                        g.num_leaves, x, self.r, self._pcg_compact_r],
                device=self.device, record_cmd=True,
            )
            self._pcg_mg_gather_cmd = fast_launch(
                k_mg_gather_z, dim=n,
                inputs=[self.z, self.pressure_indices, self._pressure_count,
                        g.num_leaves, 0, self.p_dir, self._pcg_compact_z],
                device=self.device, record_cmd=True,
            )
        else:
            self._pcg_update_cmd = fast_launch(
                k_update_xr_preconditioned, dim=n,
                inputs=[self.p_dir, self.ap, self.alpha, self.inv_diag,
                        self.pressure_indices, self._pressure_count,
                        g.num_leaves, x, self.r, self.z,
                        self._pcg_compact_r, self._pcg_compact_z],
                device=self.device, record_cmd=True,
            )
            self._pcg_mg_gather_cmd = None
        self._pcg_cmd_kind = kind

    def _prepare_pcg_commands(self, n, voxel_type, x):
        kind = self._pcg_kind()
        same_voxel_type = self._pcg_voxel_type is voxel_type
        if same_voxel_type and self._pcg_cmd_kind == kind:
            return
        if kind == PRECONDITIONER_MULTIGRID and (
                self._mg_voxel_type is not voxel_type):
            raise RuntimeError(
                "MULTIGRID preconditioner is not built for this voxel_type; "
                "call build_preconditioner(voxel_type) after selecting it")
        g = self.grid
        if same_voxel_type:
            # Only the preconditioner changed: the shared commands, and every
            # captured graph (their keys carry the kind), stay valid.
            self._prepare_update_cmds(n, kind, x)
            return
        self._pcg_voxel_type = voxel_type
        self._pcg_apply_cmd = fast_launch(
            k_apply_matrix, dim=n,
            inputs=[self.p_dir, self.pressure_indices, self._pressure_count,
                    voxel_type, self.inv_diag,
                    *self._stencil_args(), self.ap,
                    self._pcg_compact_p, self._pcg_compact_ap],
            device=self.device, record_cmd=True,
        )
        self._prepare_update_cmds(n, kind, x)
        self._pcg_update_p_cmd = fast_launch(
            k_update_p_indexed, dim=n,
            inputs=[self.z, self.beta, self.pressure_indices,
                    self._pressure_count, g.num_leaves, self.p_dir],
            device=self.device, record_cmd=True,
        )
        self._pcg_alpha_cmds = (
            fast_launch(k_compute_alpha_and_reset, dim=1,
                      inputs=[self._pcg_rz_a, self._pcg_p_ap_accum,
                              self.p_ap, self.alpha],
                      device=self.device, record_cmd=True),
            fast_launch(k_compute_alpha_and_reset, dim=1,
                      inputs=[self._pcg_rz_b, self._pcg_p_ap_accum,
                              self.p_ap, self.alpha],
                      device=self.device, record_cmd=True),
        )
        self._pcg_beta_cmds = (
            fast_launch(k_compute_beta_and_reset_old, dim=1,
                      inputs=[self._pcg_rz_b, self._pcg_rz_a, self.beta],
                      device=self.device, record_cmd=True),
            fast_launch(k_compute_beta_and_reset_old, dim=1,
                      inputs=[self._pcg_rz_a, self._pcg_rz_b, self.beta],
                      device=self.device, record_cmd=True),
        )
        self._pcg_pack_cmds = (
            fast_launch(k_pack_pcg_stats, dim=1,
                      inputs=[self.residual, self._pcg_rz_a, self.p_ap,
                              self.pcg_stats],
                      device=self.device, record_cmd=True),
            fast_launch(k_pack_pcg_stats, dim=1,
                      inputs=[self.residual, self._pcg_rz_b, self.p_ap,
                              self.pcg_stats],
                      device=self.device, record_cmd=True),
        )
        self._pcg_graphs.clear()
        self._pcg_conditional_graphs.clear()

    # -- multigrid V-cycle ---------------------------------------------------

    def _ensure_mg_levels(self):
        """Allocate the coarse levels once; captured graphs bake their
        pointers, so they are never reallocated or resized."""
        if self._mg_levels is None:
            capacity = self.grid.capacity
            self._mg_levels = [None] + [
                _MGLevel(MG_FINE_SHIFT - level, capacity, self.device)
                for level in range(1, MG_LEVELS)
            ]
        return self._mg_levels

    def mg_extra_bytes_per_leaf(self):
        """Device bytes per leaf slot held by levels 1.. (level 0 reuses the
        pressure solver's own arrays)."""
        levels = self._ensure_mg_levels()
        total = sum(lv.nbytes() for lv in levels[1:])
        return total / float(self.grid.capacity)

    def _mg_dim(self, shift):
        """Launch dim of a level: fixed from the leaf capacity bucket, with the
        device-side ``num_leaves`` guard doing the exact clipping."""
        return self._mg_leaf_n << (3 * shift)

    def _build_mg_hierarchy(self, voxel_type):
        """(Re)build the coarse cell types and inverse diagonals for
        ``voxel_type``. Level 0 is the pressure solver's own ``voxel_type`` and
        ``inv_diag``; levels 1.. are derived from it, one level at a time.

        The six launches are replayed from a captured graph once the launch
        dims (a power-of-two leaf bucket) repeat: they only depend on device
        data, so a rebuild per pressure solve is one graph launch, not six
        host-side launches."""
        g = self.grid
        self._ensure_mg_levels()
        if self._mg_voxel_type is not voxel_type:
            self._mg_graphs.clear()
        self._mg_voxel_type = voxel_type
        self._mg_leaf_n = min(
            g.capacity, 1 << (max(1, g.active_leaf_count) - 1).bit_length())
        self._launch_mg_cached(("hier", self._mg_leaf_n),
                               lambda: self._record_mg_hierarchy(voxel_type))

    def _record_mg_hierarchy(self, voxel_type):
        g = self.grid
        src = voxel_type
        for level in range(1, MG_LEVELS):
            lv = self._mg_levels[level]
            fast_launch(
                k_mg_build_types, dim=self._mg_dim(lv.shift),
                inputs=[src, lv.shift, g.num_leaves, lv.vtype],
                device=self.device,
            )
            fast_launch(
                k_mg_inverse_diagonal, dim=self._mg_dim(lv.shift),
                inputs=[lv.vtype, lv.shift, g.leaf_table, g.leaf_coords,
                        g.leaves_dim, g.num_leaves, lv.inv_diag],
                device=self.device,
            )
            src = lv.vtype

    def _launch_mg_cached(self, key, record):
        """Run ``record`` (a fixed launch sequence) as a cached CUDA graph.

        The first call per key runs eagerly (loading the kernels); the second
        captures and replays; later calls only replay. CPU devices and a
        failed capture fall back to the eager launches."""
        graphs = self._mg_graphs
        entry = graphs.get(key)
        if entry is not None and entry is not True and entry is not False:
            graphs.move_to_end(key)
            wp.capture_launch(entry)
            return
        if (entry is False or not wp.get_device(self.device).is_cuda):
            record()
            return
        if entry is None:
            # Modules are loaded by the eager pass; capturing cold kernels
            # would race the module load.
            if len(graphs) >= self._PCG_GRAPH_CAPACITY:
                graphs.popitem(last=False)
            graphs[key] = True
            record()
            return
        try:
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                record()
        except Exception:
            graphs[key] = False
            record()
            return
        graphs[key] = capture.graph
        wp.capture_launch(capture.graph)

    def _mg_level_views(self, r, z):
        """Per-level ``_MGView``s for one V-cycle. Level 0 solves ``A z = r``
        on the caller's arrays with ``pressure_tmp`` as its ping-pong partner,
        so the fine level costs no extra memory."""
        levels = self._mg_levels
        g = self.grid
        # Inside the PCG start graph the fine level walks the compact unknown
        # list (``_mg_work_n`` launch threads); a direct call walks all cells.
        work_n = self._mg_work_n
        views = [_MGView(MG_FINE_SHIFT, self._mg_voxel_type, self.inv_diag,
                         r, z, self.pressure_tmp,
                         work_n if work_n else self._mg_dim(MG_FINE_SHIFT),
                         1 if work_n else 0,
                         self.pressure_indices, self._pressure_count)]
        for lv in levels[1:]:
            views.append(_MGView(lv.shift, lv.vtype, lv.inv_diag, lv.b, lv.x,
                                 lv.t, self._mg_dim(lv.shift), 0,
                                 self.pressure_indices, self._pressure_count))
        return views

    def _mg_sweeps(self, lv, bufs, done, count):
        """Run ``count`` damped-Jacobi sweeps on level ``lv``, ping-ponging
        between ``bufs``. ``done`` sweeps already ran; sweep ``k`` (1-based)
        writes ``bufs[(k - 1) & 1]``, and the very first one starts from a
        zero guess. Returns the new sweep count."""
        g = self.grid
        for _ in range(count):
            if done == 0:
                fast_launch(
                    k_mg_smooth_zero, dim=lv.dim,
                    inputs=[lv.b, lv.vtype, lv.inv_diag, MG_OMEGA,
                            g.num_leaves, lv.shift, lv.use_idx, lv.indices,
                            lv.count, bufs[0]],
                    device=self.device,
                )
            else:
                fast_launch(
                    k_mg_smooth, dim=lv.dim,
                    inputs=[bufs[(done - 1) & 1], lv.b, lv.vtype, lv.inv_diag,
                            MG_OMEGA, g.leaf_table, g.leaf_coords,
                            g.leaves_dim, g.num_leaves, lv.shift, lv.use_idx,
                            lv.indices, lv.count, bufs[done & 1]],
                    device=self.device,
                )
            done += 1
        return done

    def _mg_vcycle_level(self, views, level):
        """Recursive symmetric V-cycle on ``views[level]``; the result lands in
        that level's ``x``. The sweeps ping-pong between ``x`` and ``t`` and
        the first buffer is chosen so the last sweep writes ``x``."""
        g = self.grid
        lv = views[level]
        if level == MG_LEVELS - 1:
            sweeps = MG_COARSE_SWEEPS
            bufs = (lv.x, lv.t) if sweeps & 1 else (lv.t, lv.x)
            self._mg_sweeps(lv, bufs, 0, sweeps)
            return
        pre = self._mg_pre_smooth
        post = self._mg_post_smooth
        bufs = (lv.x, lv.t) if (pre + post) & 1 else (lv.t, lv.x)
        done = self._mg_sweeps(lv, bufs, 0, pre)
        cur = bufs[(done - 1) & 1]
        # The other ping-pong buffer is free until the first post sweep, so
        # it holds the fine residual between the two restriction kernels.
        res = bufs[done & 1]
        coarse = views[level + 1]
        fast_launch(
            k_mg_residual, dim=lv.dim,
            inputs=[cur, lv.b, lv.vtype, lv.inv_diag, g.leaf_table,
                    g.leaf_coords, g.leaves_dim, g.num_leaves, lv.shift,
                    lv.use_idx, lv.indices, lv.count, res],
            device=self.device,
        )
        fast_launch(
            k_mg_restrict_sum, dim=coarse.dim,
            inputs=[res, lv.vtype, lv.inv_diag, g.num_leaves, lv.shift,
                    coarse.b],
            device=self.device,
        )
        self._mg_vcycle_level(views, level + 1)
        # First post sweep applies the coarse correction on the fly.
        fast_launch(
            k_mg_smooth_prolong, dim=lv.dim,
            inputs=[cur, coarse.x, lv.b, lv.vtype, lv.inv_diag, MG_OMEGA,
                    g.leaf_table, g.leaf_coords, g.leaves_dim, g.num_leaves,
                    lv.shift, lv.use_idx, lv.indices, lv.count,
                    bufs[done & 1]],
            device=self.device,
        )
        self._mg_sweeps(lv, bufs, done + 1, post - 1)

    def _record_mg_start(self, work_n):
        g = self.grid
        self._mg_work_n = work_n
        try:
            self._apply_mg_vcycle(self.r, self.z)
        finally:
            self._mg_work_n = 0
        fast_launch(
            k_mg_gather_z, dim=work_n,
            inputs=[self.z, self.pressure_indices, self._pressure_count,
                    g.num_leaves, 1, self.p_dir, self._pcg_compact_z],
            device=self.device,
        )

    def _apply_mg_vcycle(self, r, z):
        """Apply one symmetric V-cycle: ``z = M^-1 r`` on full-layout arrays.

        ``r`` and ``z`` are full-layout float arrays (``grid.voxel_capacity``);
        ``r`` is read only at pressure unknowns and ``z`` is overwritten (not
        accumulated) at the active voxels, so no zeroing is needed. The whole
        call is a fixed launch sequence with no atomics, no host readback and
        no synchronization, which makes it CUDA-graph capturable and bitwise
        deterministic. Call ``build_preconditioner(voxel_type)`` with
        ``preconditioner = "MULTIGRID"`` first.
        """
        if self._mg_levels is None or self._mg_voxel_type is None:
            raise RuntimeError(
                "multigrid hierarchy is not built; call build_preconditioner "
                "with preconditioner='MULTIGRID' first")
        if self._mg_pre_smooth < 1 or self._mg_post_smooth < 1:
            raise ValueError(
                "multigrid needs at least one pre and one post smoothing "
                f"sweep (got {self._mg_pre_smooth}/{self._mg_post_smooth})")
        views = self._mg_level_views(r, z)
        self._mg_vcycle_level(views, 0)

    def _make_status(
        self, converged, reason, residual, iterations,
        project_velocity=None,
    ):
        """Build status with convergence and projection safety separated."""
        if project_velocity is None:
            project_velocity = converged
        return {
            "converged": bool(converged),
            "iterations": int(iterations),
            "reason": reason,
            "residual": float(residual),
            "project_velocity": bool(project_velocity),
        }

    def _set_pressure_status(self, status, failure="", pressure_valid=False):
        self.last_pressure_status = status
        self.last_pressure_status_code = PRESSURE_STATUS_CODES.get(
            status, PRESSURE_STATUS_CODES[PRESSURE_STATUS_INVALID])
        self.last_pcg_failure = failure
        self.last_pressure_valid = pressure_valid
        converged = status in (
            PRESSURE_STATUS_CONVERGED,
            PRESSURE_STATUS_JACOBI,
        )
        self.last_status = self._make_status(
            converged, failure or status, self.last_residual,
            self.last_iterations, project_velocity=pressure_valid)

    def should_project_pressure(self):
        """True when the current pressure field is safe to project."""
        return self.last_pressure_valid

    def project_frame(self, velocity, voxel_type, collision_velocity,
                      voxel_counts, settings, *, max_gradient,
                      pressure_source=None):
        """Build, solve, and safely apply one pressure projection."""
        resolved_quality = getattr(settings, "resolved_quality", None)
        rest_count = (resolved_quality.rest_particle_count
                      if resolved_quality is not None
                      else settings.particles_per_voxel)
        self.build_rhs(
            velocity, voxel_type, collision_velocity, voxel_counts,
            rest_count=rest_count,
            count_stiffness=settings.count_stiffness,
        )
        self.build_preconditioner(voxel_type)
        if int(settings.pressure_solver) == int(ProjectNonDivergentSolver.JACOBI):
            self.solve_jacobi(voxel_type, settings.pressure_iterations)
        else:
            guess = self.learned_guess
            if guess is not None and not guess.active:
                guess = None
            self.solve_pcg(
                voxel_type, settings.pressure_iterations,
                effective_pressure_tolerance(
                    settings.pressure_tolerance, self.grid.voxel_size),
                initial_guess=guess)
            if guess is not None:
                guess.observe(self, voxel_type)
        # Original FIELD semantics blend into the existing pressure field.
        # The callback is deliberately after solve (which initializes pressure)
        # and before threshold/projection (which consumes it).
        if pressure_source is not None:
            pressure_source()
        if not self.should_project_pressure():
            return self.last_status
        if settings.pressure_threshold > 0.0:
            self.apply_threshold(settings.pressure_threshold)
        self.project(
            velocity, voxel_type, collision_velocity,
            max_gradient=max_gradient)
        return self.last_status

    def build_rhs(self, vel, voxel_type, collision_vel, voxel_counts,
                  rest_count=8.0, count_stiffness=0.0):
        g = self.grid
        fast_launch(
            k_compute_divergence, dim=self._dim(),
            inputs=[vel.u, vel.v, vel.w, voxel_type, collision_vel,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, g.voxel_size, self.rhs, self.divergence],
            device=self.device,
        )
        if count_stiffness > 0.0:
            fast_launch(
                k_particle_count_to_pressure_rhs, dim=self._dim(),
                inputs=[voxel_counts, voxel_type, g.num_leaves,
                        rest_count, count_stiffness, g.voxel_size, self.rhs],
                device=self.device,
            )

    def build_preconditioner(self, voxel_type):
        g = self.grid
        kind = self._pcg_kind()
        fast_launch(
            k_compute_inverse_diagonal, dim=self._dim(),
            inputs=[voxel_type, g.leaf_table, g.leaf_coords, g.voxels_dim,
                    g.leaves_dim, g.num_leaves, self.inv_diag],
            device=self.device,
        )
        if kind == PRECONDITIONER_MULTIGRID:
            self._build_mg_hierarchy(voxel_type)

    def coarse_initial_guess(self, voxel_type, iterations=32):
        """Electrostatic leaf-aggregate long-range guess (multigrid path)."""
        g = self.grid
        n_leaf = self._leaf_dim()
        self.leaf_rhs[:n_leaf].zero_()
        self.leaf_fluid[:n_leaf].zero_()
        self.leaf_pot[:n_leaf].zero_()
        self.leaf_pot_tmp[:n_leaf].zero_()
        fast_launch(
            k_build_leaf_aggregates, dim=self._leaf_dim(),
            inputs=[self.rhs, voxel_type, g.num_leaves, self.leaf_rhs, self.leaf_fluid],
            device=self.device,
        )
        src, dst = self.leaf_pot, self.leaf_pot_tmp
        for _ in range(iterations):
            fast_launch(
                k_solve_leaf_potential_jacobi, dim=self._leaf_dim(),
                inputs=[self.leaf_rhs, self.leaf_fluid, g.leaf_table,
                        g.leaf_coords, g.leaves_dim, g.num_leaves, src, dst],
                device=self.device,
            )
            src, dst = dst, src
        fast_launch(
            k_sample_up_leaf_potential, dim=self._dim(),
            inputs=[src, voxel_type, g.num_leaves, self.x0],
            device=self.device,
        )

    def solve_jacobi(self, voxel_type, iterations):
        g = self.grid
        n = self._dim()
        src, dst = self.pressure, self.pressure_tmp
        src[:n].zero_()
        for _ in range(iterations):
            fast_launch(
                k_jacobi_iteration, dim=n,
                inputs=[src, self.rhs, voxel_type, self.inv_diag,
                        g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                        g.num_leaves, dst],
                device=self.device,
            )
            src, dst = dst, src
        if src is not self.pressure:
            wp.copy(self.pressure, src, count=n)
        self.last_iterations = iterations
        # Jacobi always runs its fixed iteration budget and always projects
        # the result (unlike PCG's adaptive stop/fail-closed logic, there is
        # no separate "give up early" path to guard here). But the previous
        # hardcoded last_residual = -1.0 actively hid whether that fixed
        # budget was actually enough: diagnostics/UI/validation could not
        # distinguish a well-converged Jacobi solve from a wildly divergent
        # one. Compute the real residual via the same operator PCG/validation
        # already use (one host readback, matching measure_residual()).
        residual = self.measure_residual(voxel_type)
        self.last_residual = residual if math.isfinite(residual) else math.inf
        self.last_residual_squared = (
            residual * residual if math.isfinite(residual) else math.inf)
        self._set_pressure_status(
            PRESSURE_STATUS_JACOBI, PRESSURE_STATUS_JACOBI, True)
        return self.last_status

    @staticmethod
    def _pcg_stall_window(check_every):
        """Iterations without a new best residual that count as a stall.

        Never shorter than one check interval, so a stall always needs at
        least one checkpoint that failed to improve on the best residual.
        """
        return max(PCG_STALL_WINDOW, int(check_every))

    def _resolve_check_every(self, check_every, max_iterations):
        """Residual-check interval of one ``solve_pcg``.

        An explicit caller value is honored as is. With none, JACOBI keeps the
        historical 25. MULTIGRID converges in a handful of iterations, so any
        coarser interval overshoots the tolerance by whole V-cycles; it picks
        the largest candidate that is odd and divides ``max_iterations`` (the
        CUDA conditional graph needs both) -- 1 since 4.0, which always fits.
        """
        if check_every is not None:
            return max(1, int(check_every))
        if self._pcg_kind() == PRECONDITIONER_MULTIGRID:
            for candidate in MG_CHECK_EVERY_CANDIDATES:
                if max_iterations > 0 and max_iterations % candidate == 0:
                    return candidate
        return PCG_DEFAULT_CHECK_EVERY

    def solve_pcg(self, voxel_type, max_iterations, tolerance, use_coarse_guess=False,
                  check_every=None, initial_guess=None):
        """Solve A x = rhs on the fluid unknowns by (multigrid-)PCG.

        ``x0`` is zero unless ``use_coarse_guess`` (electrostatic leaf guess)
        or ``initial_guess`` (an object whose ``fill(self, voxel_type)``
        writes ``self.x0`` and returns True) supplies one. When neither does,
        the launches are exactly the historical zero-guess sequence, so the
        result is bitwise unchanged.
        """
        self.last_pcg_execution_mode = "not_run"
        g = self.grid
        full_n = self._dim()
        pressure_count = self._build_pressure_index(voxel_type)
        x = self.pressure
        if pressure_count <= 0:
            x[:full_n].zero_()
            wp.utils.array_inner(
                self.rhs, self.rhs, out=self.residual, count=full_n)
            res_sq = float(self.residual.numpy()[0])
            self.last_iterations = 0
            self.last_residual_squared = res_sq
            self.last_residual = (
                math.sqrt(max(0.0, res_sq)) if math.isfinite(res_sq) else math.inf)
            tol_sq = tolerance * tolerance * max(1, full_n)
            if not math.isfinite(res_sq):
                self._set_pressure_status(
                    PRESSURE_STATUS_INVALID, "non_finite_residual", False)
            elif res_sq < tol_sq:
                self._set_pressure_status(
                    PRESSURE_STATUS_CONVERGED, PRESSURE_STATUS_CONVERGED, True)
            else:
                self._set_pressure_status(
                    PRESSURE_STATUS_SINGULAR, "singular_system", False)
            return self.last_status
        work_n = min(full_n, 1 << (pressure_count - 1).bit_length())
        self._prepare_pcg_commands(work_n, voxel_type, x)
        guessed = bool(use_coarse_guess)
        if guessed:
            self.coarse_initial_guess(voxel_type)
        elif initial_guess is not None:
            guessed = bool(initial_guess.fill(self, voxel_type))
        self._zero_pcg_vectors(full_n, x, zero_x0=not guessed)
        fast_launch(
            k_initialize_pcg, dim=work_n,
            inputs=[self.rhs, self.x0, self.inv_diag, self.pressure_indices,
                    self._pressure_count, 1,
                    g.num_leaves,
                    x, self.r, self.z, self.p_dir,
                    self._pcg_compact_r, self._pcg_compact_z],
            device=self.device,
        )
        if guessed:
            fast_launch(
                k_apply_matrix, dim=work_n,
                inputs=[x, self.pressure_indices, self._pressure_count,
                        voxel_type, self.inv_diag,
                        *self._stencil_args(), self.ap,
                        self._pcg_compact_p, self._pcg_compact_ap],
                device=self.device,
            )
            fast_launch(
                k_subtract_ax_from_r, dim=work_n,
                inputs=[self.ap, self.inv_diag, self.pressure_indices,
                        self._pressure_count,
                        g.num_leaves,
                        self.r, self.z, self.p_dir,
                        self._pcg_compact_r, self._pcg_compact_z],
                device=self.device,
            )
        if self._pcg_cmd_kind == PRECONDITIONER_MULTIGRID:
            # z0 = M^-1 r0 replaces the Jacobi z the initializer wrote, and
            # p0 = z0 (the compaction also feeds the r.z reduction below).
            self._launch_mg_cached(
                ("start", work_n) + self._pcg_graph_key_tail(),
                lambda: self._record_mg_start(work_n))
        check_every = self._resolve_check_every(check_every, max_iterations)
        self.last_pcg_check_every = check_every
        if self._pcg_cmd_kind == PRECONDITIONER_MULTIGRID:
            tolerance = tolerance * MG_TOLERANCE_SCALE
        tol_sq = tolerance * tolerance * max(1, full_n)
        status = PRESSURE_STATUS_MAX_ITERATIONS
        failure = "max_iterations"
        pressure_valid = False
        wp.utils.array_inner(
            self._pcg_compact_r, self._pcg_compact_z,
            out=self.rz, count=work_n)
        wp.utils.array_inner(
            self._pcg_compact_r, self._pcg_compact_r,
            out=self.residual, count=work_n)
        best_res = 0.0
        best_it = 0
        stall_window = self._pcg_stall_window(check_every)
        res_sq = 0.0
        it = 0
        use_conditional = (
            max_iterations > 0
            and wp.get_device(self.device).is_cuda
            and wp.is_conditional_graph_supported()
            and check_every % 2 == 1
            and max_iterations % check_every == 0
        )
        conditional_graphs = None
        if use_conditional and self._ensure_pcg_dot():
            try:
                conditional_graphs = self._get_pcg_conditional_graphs(
                    work_n, check_every, max_iterations)
            except Exception as exc:
                self._release_native_pcg_dot()
                self._pcg_native_available = False
                self.last_pcg_native_error = f"{type(exc).__name__}: {exc}"
        use_conditional = conditional_graphs is not None
        if use_conditional:
            self.last_pcg_execution_mode = "cuda_conditional"
            stats = self._launch_pcg_conditional(work_n, conditional_graphs, tol_sq)
            res_sq = float(stats[0])
            it = int(round(float(stats[3])))
            reason = int(round(float(stats[4])))
            if reason == 1:
                status = PRESSURE_STATUS_CONVERGED
                failure = PRESSURE_STATUS_CONVERGED
                pressure_valid = True
            elif reason == 2:
                status = PRESSURE_STATUS_SINGULAR
                failure = "singular_system"
            elif reason == 3:
                status = PRESSURE_STATUS_STALLED
                failure = "stalled"
            elif reason == 4:
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
            else:
                status = PRESSURE_STATUS_MAX_ITERATIONS
                failure = "max_iterations"
        else:
            self.last_pcg_execution_mode = "host_batched"
            best_res = float(self.residual.numpy()[0])
            res_sq = best_res
            if not math.isfinite(best_res):
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
            elif best_res < tol_sq:
                status = PRESSURE_STATUS_CONVERGED
                failure = PRESSURE_STATUS_CONVERGED
                pressure_valid = True
        # Batched device-resident iterations: all recurrence scalars (alpha,
        # beta) stay on the GPU. The scalar update kernels clamp to zero on
        # tiny denominators, so a breakdown mid-batch freezes x/r in place
        # and is classified at the next checkpoint from one packed readback
        # instead of three host syncs per iteration.
        while (not use_conditional and it < max_iterations
               and status == PRESSURE_STATUS_MAX_ITERATIONS):
            batch = min(check_every, max_iterations - it)
            self._launch_pcg_batch(work_n, batch)
            it += batch
            # single checkpoint readback: [residual^2, r.z, p.Ap]
            wp.utils.array_inner(
                self._pcg_compact_r, self._pcg_compact_r,
                out=self.residual, count=work_n)
            parity = 0 if self.rz is self._pcg_rz_a else 1
            self._pcg_pack_cmds[parity].launch()
            stats = self.pcg_stats.numpy()
            res_sq = float(stats[0])
            rz_cur = float(stats[1])
            p_ap_cur = float(stats[2])
            if (not math.isfinite(res_sq) or not math.isfinite(rz_cur)
                    or not math.isfinite(p_ap_cur)):
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
            elif res_sq < tol_sq:
                status = PRESSURE_STATUS_CONVERGED
                failure = PRESSURE_STATUS_CONVERGED
                pressure_valid = True
            elif (rz_cur < -1.0e-20 or abs(rz_cur) < 1.0e-30
                    or abs(p_ap_cur) < 1.0e-30):
                status = PRESSURE_STATUS_SINGULAR
                failure = "singular_system"
            else:
                # mirrors k_finish_pcg_conditional_batch
                if res_sq < best_res * (1.0 - 1.0e-6):
                    best_res = res_sq
                    best_it = it
                if it - best_it >= stall_window:
                    status = PRESSURE_STATUS_STALLED
                    failure = "stalled"
        self.last_iterations = it
        if not math.isfinite(res_sq):
            status = PRESSURE_STATUS_INVALID
            failure = "non_finite_residual"
            res_sq = math.inf
        self.last_residual_squared = res_sq
        self.last_residual = math.sqrt(max(0.0, res_sq)) if math.isfinite(res_sq) else math.inf
        pressure_valid = pressure_valid or (
            math.isfinite(res_sq)
            and status in (PRESSURE_STATUS_MAX_ITERATIONS,
                           PRESSURE_STATUS_STALLED)
        )
        if not pressure_valid:
            self.pressure[:full_n].zero_()
        self._set_pressure_status(status, failure, pressure_valid)
        return self.last_status

    def measure_residual(self, voxel_type):
        """L2 norm of (rhs - A*pressure) for the current pressure solution.

        Used by validation/diagnostics. The norm is reduced on the device and
        only the scalar is read back; the rhs and A*p arrays stay resident.
        """
        full_n = self._dim()
        work_n = self._build_pressure_index(voxel_type)
        self.ap[:full_n].zero_()
        if work_n:
            fast_launch(
                k_apply_matrix, dim=work_n,
                inputs=[self.pressure, self.pressure_indices,
                        self._pressure_count, voxel_type, self.inv_diag,
                        *self._stencil_args(), self.ap,
                        self._pcg_compact_p, self._pcg_compact_ap],
                device=self.device,
            )
        partials = min(RESIDUAL_PARTIALS, full_n)
        if self._residual_partials is None:
            self._residual_partials = wp.zeros(
                RESIDUAL_PARTIALS, dtype=wp.float64, device=self.device)
        fast_launch(
            k_residual_partial_squares, dim=partials,
            inputs=[self.rhs, self.ap, full_n, partials,
                    self._residual_partials],
            device=self.device,
        )
        squared = wp.utils.array_sum(self._residual_partials[:partials])
        return math.sqrt(float(squared))

    def apply_threshold(self, relative_threshold):
        if relative_threshold <= 0.0:
            return
        g = self.grid
        self.max_pressure.zero_()
        fast_launch(
            k_pressure_threshold_compute, dim=self._dim(),
            inputs=[self.pressure, g.num_leaves, self.max_pressure],
            device=self.device,
        )
        fast_launch(
            k_pressure_threshold_apply, dim=self._dim(),
            inputs=[self.max_pressure, relative_threshold, g.num_leaves, self.pressure],
            device=self.device,
        )

    def apply_scale(self, scale):
        if scale == 1.0:
            return
        fast_launch(
            k_pressure_scale, dim=self._dim(),
            inputs=[scale, self.grid.num_leaves, self.pressure],
            device=self.device,
        )


    def project(self, vel, voxel_type, collision_vel, max_gradient=1.0e6):
        g = self.grid
        fast_launch(
            k_project, dim=self._dim(),
            inputs=[self.pressure, voxel_type, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, g.voxel_size,
                    max_gradient,
                    collision_vel,
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
            device=self.device,
        )
