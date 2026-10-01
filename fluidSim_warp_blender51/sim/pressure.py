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
  by the shipped pipeline. There is no restriction/prolongation V-cycle: the
  registered ``fluidSimProjectNonDivergentSampleDown``/
  ``fluidSimProjectNonDivergentExpandVelocityIntoCollision`` kernels have no
  call sites anywhere in this module and do not implement one.
- pressure threshold + scale, velocity expansion into collision

No bpy imports allowed here.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import warp as wp
from .fastlaunch import launch as fast_launch

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID
from .enums import ProjectNonDivergentSolver
from .fields import (
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



@wp.func
def st_phase_from_mass(weight_sum: float, reference_mass: float, eta_phi: float) -> float:
    denom = wp.max(reference_mass * wp.max(eta_phi, 1.0e-6), 1.0e-8)
    x = weight_sum / denom
    if x <= 0.0:
        return 0.0
    return wp.min(wp.sqrt(x), 1.0)


@wp.func
def st_face_beta(
    axis: wp.int32,
    flat: wp.int32,
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
) -> float:
    weight = wu[flat]
    if axis == 1:
        weight = wv[flat]
    elif axis == 2:
        weight = ww[flat]
    phi = st_phase_from_mass(weight, reference_mass, eta_phi)
    rho = rho_liquid * phi + rho_gas * (1.0 - phi)
    return 1.0 / wp.max(rho, wp.max(eps_rho, 1.0e-8))


@wp.func
def st_beta_for_cell_face(
    tid: wp.int32,
    ijk: wp.vec3i,
    axis: wp.int32,
    step: wp.int32,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
) -> float:
    face_flat = tid
    if step > 0:
        face_ijk = wp.vec3i(ijk[0], ijk[1], ijk[2])
        if axis == 0:
            face_ijk = wp.vec3i(ijk[0] + 1, ijk[1], ijk[2])
        elif axis == 1:
            face_ijk = wp.vec3i(ijk[0], ijk[1] + 1, ijk[2])
        else:
            face_ijk = wp.vec3i(ijk[0], ijk[1], ijk[2] + 1)
        face_flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, face_ijk)
    if face_flat < 0:
        return 0.0
    return st_face_beta(axis, face_flat, wu, wv, ww, reference_mass, eta_phi, rho_liquid, rho_gas, eps_rho)

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
        f_lo = float(0.0)
        f_hi = float(0.0)
        if axis == 0:
            f_lo = u[tid]
            f_hi = wp.where(hi >= 0, u[hi], 0.0)
        elif axis == 1:
            f_lo = v[tid]
            f_hi = wp.where(hi >= 0, v[hi], 0.0)
        else:
            f_lo = w[tid]
            f_hi = wp.where(hi >= 0, w[hi], 0.0)
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




# ST-FLIP variable-coefficient pressure matrix from space-time accumulators


@wp.kernel
def k_compute_inverse_diagonal_st(
    voxel_type: wp.array(dtype=wp.int32),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
    inv_diag: wp.array(dtype=float),
):
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
                solid = 1
            if solid == 0:
                diag += st_beta_for_cell_face(
                    tid, ijk, axis, step, leaf_table, voxels_dim, leaves_dim,
                    wu, wv, ww, reference_mass, eta_phi, rho_liquid, rho_gas, eps_rho)
    if diag > 0.0:
        inv_diag[tid] = 1.0 / diag
    else:
        inv_diag[tid] = 0.0


@wp.kernel
def k_apply_matrix_st(
    x: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
    ax: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0 or inv_diag[tid] == 0.0:
        ax[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    acc = float(0.0)
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = neighbor_flat(tid, leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            solid = wp.int32(0)
            if n >= 0 and (voxel_type[n] & VOXEL_COLLISION) != 0:
                solid = 1
            if n < 0:
                solid = 1
            if solid == 0:
                beta = st_beta_for_cell_face(
                    tid, ijk, axis, step, leaf_table, voxels_dim, leaves_dim,
                    wu, wv, ww, reference_mass, eta_phi, rho_liquid, rho_gas, eps_rho)
                acc += beta * x[tid]
                if n >= 0:
                    nvt = voxel_type[n]
                    if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                        acc -= beta * x[n]
    ax[tid] = acc


@wp.kernel
def k_jacobi_iteration_st(
    x_in: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
    x_out: wp.array(dtype=float),
):
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
            solid = wp.int32(0)
            if n >= 0 and (voxel_type[n] & VOXEL_COLLISION) != 0:
                solid = 1
            if n < 0:
                solid = 1
            if solid == 0 and n >= 0:
                nvt = voxel_type[n]
                if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                    beta = st_beta_for_cell_face(
                        tid, ijk, axis, step, leaf_table, voxels_dim, leaves_dim,
                        wu, wv, ww, reference_mass, eta_phi, rho_liquid, rho_gas, eps_rho)
                    acc += beta * x_in[n]
    x_out[tid] = acc * inv_diag[tid]

# PCG scalar update kernels (device-resident alpha/beta)


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
    stats: wp.array(dtype=float),
):
    res_sq = residual[0]
    keep_going = 1
    reason = 0
    if wp.isnan(res_sq) or wp.isinf(res_sq):
        keep_going = 0
        reason = 4
    elif res_sq < tolerance_squared:
        keep_going = 0
        reason = 1
    condition[0] = keep_going
    iterations[0] = 0
    last_residual[0] = res_sq
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
    tolerance_squared: float,
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
    elif res_sq < tolerance_squared:
        reason = 1
    elif rz_value < -1.0e-20 or wp.abs(rz_value) < 1.0e-30 or wp.abs(p_ap_value) < 1.0e-30:
        reason = 2
    elif res_sq >= last_residual[0] * (1.0 - 1.0e-6):
        reason = 3
    elif iteration >= max_iterations:
        reason = 5
    else:
        keep_going = 1
        last_residual[0] = res_sq
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
    """fluidSimProjectNonDivergentElectrostaticBuildLeafAggregates8/16/32."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        return
    leaf = tid // LEAF_VOLUME
    wp.atomic_add(leaf_rhs, leaf, rhs[tid])
    wp.atomic_add(leaf_fluid, leaf, 1)


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
    leaf level (used by the V-cycle path)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        return
    wp.atomic_add(leaf_rhs, tid // LEAF_VOLUME, r[tid])


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
def k_project_st(
    x: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    wu_mass: wp.array(dtype=float),
    wv_mass: wp.array(dtype=float),
    ww_mass: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    max_gradient: float,
    reference_mass: float,
    eta_phi: float,
    rho_liquid: float,
    rho_gas: float,
    eps_rho: float,
    collision_vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """Variable-coefficient ST-FLIP projection.

    Uses beta_f = 1/max(rho(phi_f), eps_rho), with phi_f read from the same
    face P2G accumulators used for velocity normalization.  This is the folded
    q=p*dt discretization of Eq. (14)-(16) in the add-on's pressure units.
    """
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
            n_solid = 1
        if here_solid or n_solid != 0:
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
            beta = st_face_beta(axis, tid, wu_mass, wv_mass, ww_mass,
                                reference_mass, eta_phi,
                                rho_liquid, rho_gas, eps_rho)
            g = bounded_pressure_gradient(beta * x_here, beta * x_n, inv_h, limit)
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
register_fn(k_compute_inverse_diagonal_st,
            "fluidSimProjectNonDivergentPreConditionerSpatiotemporal")
register_fn(k_apply_matrix_st,
            "fluidSimProjectNonDivergentApplyMatrixSpatiotemporal")
for _w in (2, 4, 8, 16, 32):
    register_fn(k_jacobi_iteration,
                f"fluidSimProjectNonDivergentJacobi{_w}_original",
                f"fluidSimProjectNonDivergentJacobi{_w}_optimizedUnroll",
                strategy="tiled",
                notes="unroll-width variants collapse to one kernel")
register_fn(k_jacobi_iteration_st,
            "fluidSimProjectNonDivergentJacobiSpatiotemporal",
            strategy="tiled")
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
            notes="No call sites; no restriction/prolongation V-cycle is "
                  "wired into this module despite the kernel name.")
register_fn(k_project, "fluidSimProjectNonDivergentProject")
register_fn(k_project_st, "fluidSimProjectNonDivergentProjectSpatiotemporal")
register_fn(k_expand_velocity_into_collision,
            "fluidSimProjectNonDivergentExpandVelocityIntoCollision",
            notes="No call sites in this module.")
register_fn(k_pressure_threshold_compute, "fluidSimPressureThresholdCompute")
register_fn(k_pressure_threshold_apply, "fluidSimPressureThresholdApply")
register_fn(k_pressure_scale, "fluidSimPressureScale")


# Host-orchestrated solver driver (GPU-resident state)


class PressureSolver:
    """Owns pressure work buffers and runs Jacobi or PCG projections."""

    # Captured CUDA graphs kept alive at once, evicted least-recently-used.
    _PCG_GRAPH_CAPACITY = 8

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
        self._pcg_last_checked_res = wp.zeros(1, dtype=float, device=device)
        self._pcg_voxel_type = None
        self._pcg_apply_cmd = None
        self._pcg_update_cmd = None
        self._pcg_update_p_cmd = None
        self._pcg_rz_a = self.rz
        self._pcg_rz_b = self.rz_new
        self._pcg_alpha_cmds = None
        self._pcg_beta_cmds = None
        self._pcg_pack_cmds = None
        self._pcg_graphs = OrderedDict()
        self._pcg_conditional_graphs = {}
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

    def _zero_pcg_vectors(self, n, x):
        """Clear the PCG working set over the launch range [0, n).

        Both PCG solvers launch and reduce over ``n`` derived from the host
        leaf mirror, while the kernels themselves are guarded by the
        device-side ``num_leaves``. Any slot the device guard skips is never
        written by the initializer, so without this it would carry the
        previous frame's values straight into the r.z / p.Ap reductions.
        """
        for array in (x, self.r, self.z, self.p_dir, self.ap):
            array[:n].zero_()

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
        if dot is not None:
            dot.release()

    def release(self):
        """Release captured PCG work and its native CUDA module lease."""
        self._release_native_pcg_dot()
        self._pcg_voxel_type = None
        self._pcg_apply_cmd = None
        self._pcg_update_cmd = None
        self._pcg_update_p_cmd = None
        self._pcg_alpha_cmds = None
        self._pcg_beta_cmds = None
        self._pcg_pack_cmds = None

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
        key = (work_n, batch, start_parity)
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
            self._pcg_apply_cmd.set_dim(work_n)
            self._pcg_update_cmd.set_dim(work_n)
            self._pcg_update_p_cmd.set_dim(work_n)
            wp.synchronize_device(self.device)
            with wp.ScopedCapture(device=self.device) as capture:
                self._record_pcg_iterations(work_n, batch, start_parity)
            graph = capture.graph
            self._pcg_graphs[key] = graph
        wp.capture_launch(graph)
        if batch & 1:
            self.rz, self.rz_new = self.rz_new, self.rz

    def _get_pcg_conditional_graphs(self, work_n, batch, max_iterations, tol_sq):
        self._pcg_apply_cmd.set_dim(work_n)
        self._pcg_update_cmd.set_dim(work_n)
        self._pcg_update_p_cmd.set_dim(work_n)
        key = (work_n, batch, max_iterations, float(tol_sq))
        graphs = self._pcg_conditional_graphs.get(key)
        if graphs is None:
            if len(self._pcg_conditional_graphs) >= 8:
                self._pcg_conditional_graphs.clear()
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
                            batch, max_iterations, tol_sq, self.pcg_stats],
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
                    tol_sq, self.pcg_stats],
            device=self.device,
        )
        wp.capture_launch(graphs[0])
        self.rz = self._pcg_rz_a
        self.rz_new = self._pcg_rz_b
        return self.pcg_stats.numpy()

    def _prepare_pcg_commands(self, n, voxel_type, x):
        if self._pcg_voxel_type is voxel_type:
            return
        g = self.grid
        self._pcg_voxel_type = voxel_type
        self._pcg_apply_cmd = fast_launch(
            k_apply_matrix, dim=n,
            inputs=[self.p_dir, self.pressure_indices, self._pressure_count,
                    voxel_type, self.inv_diag,
                    *self._stencil_args(), self.ap,
                    self._pcg_compact_p, self._pcg_compact_ap],
            device=self.device, record_cmd=True,
        )
        self._pcg_update_cmd = fast_launch(
            k_update_xr_preconditioned, dim=n,
            inputs=[self.p_dir, self.ap, self.alpha, self.inv_diag,
                    self.pressure_indices, self._pressure_count,
                    g.num_leaves, x, self.r, self.z,
                    self._pcg_compact_r, self._pcg_compact_z],
            device=self.device, record_cmd=True,
        )
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
                      reference_mass=None, pressure_source=None):
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
        st_projection = bool(
            settings.spacetime_enabled
            and settings.spacetime_variable_projection)
        if reference_mass is None:
            configured_mass = float(settings.spacetime_reference_mass)
            reference_mass = (
                configured_mass
                if configured_mass > 0.0 and math.isfinite(configured_mass)
                else max(1.0, float(rest_count)))
        st_args = {
            "reference_mass": reference_mass,
            "eta_phi": settings.spacetime_phase_eta,
            "rho_liquid": settings.spacetime_liquid_density,
            "rho_gas": settings.spacetime_gas_density,
            "eps_rho": settings.spacetime_min_density,
        }
        if st_projection:
            self.build_preconditioner_st(voxel_type, velocity, **st_args)
        else:
            self.build_preconditioner(voxel_type)
        if int(settings.pressure_solver) == int(ProjectNonDivergentSolver.JACOBI):
            if st_projection:
                self.solve_jacobi_st(
                    voxel_type, settings.pressure_iterations, velocity,
                    **st_args)
            else:
                self.solve_jacobi(voxel_type, settings.pressure_iterations)
        elif st_projection:
            self.solve_pcg_st(
                voxel_type, settings.pressure_iterations,
                settings.pressure_tolerance, velocity, **st_args)
        else:
            self.solve_pcg(
                voxel_type, settings.pressure_iterations,
                settings.pressure_tolerance)
        # Original FIELD semantics blend into the existing pressure field.
        # The callback is deliberately after solve (which initializes pressure)
        # and before threshold/projection (which consumes it).
        if pressure_source is not None:
            pressure_source()
        if not self.should_project_pressure():
            return self.last_status
        if settings.pressure_threshold > 0.0:
            self.apply_threshold(settings.pressure_threshold)
        if st_projection:
            self.project_st(
                velocity, voxel_type, collision_velocity,
                max_gradient=max_gradient, **st_args)
        else:
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
        fast_launch(
            k_compute_inverse_diagonal, dim=self._dim(),
            inputs=[voxel_type, g.leaf_table, g.leaf_coords, g.voxels_dim,
                    g.leaves_dim, g.num_leaves, self.inv_diag],
            device=self.device,
        )



    def build_preconditioner_st(self, voxel_type, vel, reference_mass=8.0,
                                eta_phi=0.5, rho_liquid=1.0,
                                rho_gas=0.0, eps_rho=0.05):
        g = self.grid
        fast_launch(
            k_compute_inverse_diagonal_st, dim=self._dim(),
            inputs=[voxel_type, vel.wu, vel.wv, vel.ww,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, reference_mass, eta_phi,
                    rho_liquid, rho_gas, eps_rho, self.inv_diag],
            device=self.device,
        )

    def solve_jacobi_st(self, voxel_type, iterations, vel, reference_mass=8.0,
                        eta_phi=0.5, rho_liquid=1.0, rho_gas=0.0,
                        eps_rho=0.05):
        g = self.grid
        n = self._dim()
        src, dst = self.pressure, self.pressure_tmp
        src[:n].zero_()
        for _ in range(iterations):
            fast_launch(
                k_jacobi_iteration_st, dim=n,
                inputs=[src, self.rhs, voxel_type, self.inv_diag,
                        vel.wu, vel.wv, vel.ww,
                        g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                        g.num_leaves, reference_mass, eta_phi,
                        rho_liquid, rho_gas, eps_rho, dst],
                device=self.device,
            )
            src, dst = dst, src
        if src is not self.pressure:
            wp.copy(self.pressure, src, count=n)
        self.last_iterations = iterations
        # Same fix as solve_jacobi: report the real residual instead of a
        # -1.0 placeholder, measured against the ST variable-coefficient
        # operator this solve actually used (measure_residual() applies the
        # constant-coefficient one).
        residual = self._measure_residual_st(
            voxel_type, vel, reference_mass, eta_phi, rho_liquid, rho_gas,
            eps_rho)
        self.last_residual = residual if math.isfinite(residual) else math.inf
        self.last_residual_squared = (
            residual * residual if math.isfinite(residual) else math.inf)
        self._set_pressure_status(
            PRESSURE_STATUS_JACOBI, PRESSURE_STATUS_JACOBI, True)
        return self.last_status

    def _measure_residual_st(self, voxel_type, vel, reference_mass, eta_phi,
                             rho_liquid, rho_gas, eps_rho):
        """L2 norm of (rhs - A_st*pressure); one readback, like measure_residual()."""
        import numpy as _np

        g = self.grid
        n = self._dim()
        self.ap[:n].zero_()
        fast_launch(
            k_apply_matrix_st, dim=n,
            inputs=[self.pressure, voxel_type, self.inv_diag,
                    vel.wu, vel.wv, vel.ww,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, reference_mass, eta_phi,
                    rho_liquid, rho_gas, eps_rho, self.ap],
            device=self.device,
        )
        r = self.rhs[:n].numpy() - self.ap[:n].numpy()
        return float(_np.linalg.norm(r))

    def solve_pcg_st(self, voxel_type, max_iterations, tolerance, vel,
                     reference_mass=8.0, eta_phi=0.5, rho_liquid=1.0,
                     rho_gas=0.0, eps_rho=0.05, check_every=25):
        """PCG solve for the ST-FLIP variable-coefficient pressure matrix."""
        g = self.grid
        n = self._dim()
        x = self.pressure
        self.x0[:n].zero_()
        self._zero_pcg_vectors(n, x)
        fast_launch(
            k_initialize_pcg, dim=n,
            inputs=[self.rhs, self.x0, self.inv_diag,
                    self.pressure_indices, self._pressure_count, 0,
                    g.num_leaves,
                    x, self.r, self.z, self.p_dir,
                    self._pcg_compact_r, self._pcg_compact_z],
            device=self.device,
        )
        check_every = max(1, int(check_every))
        tol_sq = tolerance * tolerance * max(1, n)
        status = PRESSURE_STATUS_MAX_ITERATIONS
        failure = "max_iterations"
        pressure_valid = False
        wp.utils.array_inner(self.r, self.z, out=self.rz, count=n)
        wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
        last_checked_res = float(self.residual.numpy()[0])
        res_sq = last_checked_res
        it = 0
        if not math.isfinite(last_checked_res):
            status = PRESSURE_STATUS_INVALID
            failure = "non_finite_residual"
        elif last_checked_res < tol_sq:
            status = PRESSURE_STATUS_CONVERGED
            failure = PRESSURE_STATUS_CONVERGED
            pressure_valid = True
        while it < max_iterations and status == PRESSURE_STATUS_MAX_ITERATIONS:
            batch = min(check_every, max_iterations - it)
            for _ in range(batch):
                fast_launch(
                    k_apply_matrix_st, dim=n,
                    inputs=[self.p_dir, voxel_type, self.inv_diag,
                            vel.wu, vel.wv, vel.ww,
                            g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                            g.num_leaves, reference_mass, eta_phi,
                            rho_liquid, rho_gas, eps_rho, self.ap],
                    device=self.device,
                )
                wp.utils.array_inner(self.p_dir, self.ap, out=self.p_ap, count=n)
                fast_launch(k_compute_alpha, dim=1,
                          inputs=[self.rz, self.p_ap, self.alpha],
                          device=self.device)
                fast_launch(
                    k_update_xr, dim=n,
                    inputs=[self.p_dir, self.ap, self.alpha, g.num_leaves,
                            x, self.r],
                    device=self.device,
                )
                fast_launch(
                    k_apply_preconditioner, dim=n,
                    inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                    device=self.device,
                )
                wp.utils.array_inner(self.r, self.z, out=self.rz_new, count=n)
                fast_launch(k_compute_beta, dim=1,
                          inputs=[self.rz_new, self.rz, self.beta],
                          device=self.device)
                fast_launch(
                    k_update_p, dim=n,
                    inputs=[self.z, self.beta, g.num_leaves, self.p_dir],
                    device=self.device,
                )
                wp.copy(self.rz, self.rz_new)
                it += 1
            wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
            fast_launch(k_pack_pcg_stats, dim=1,
                      inputs=[self.residual, self.rz, self.p_ap,
                              self.pcg_stats],
                      device=self.device)
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
            elif res_sq >= last_checked_res * (1.0 - 1.0e-6):
                status = PRESSURE_STATUS_STALLED
                failure = "stalled"
            else:
                last_checked_res = res_sq
        self.last_iterations = it
        if not math.isfinite(res_sq):
            status = PRESSURE_STATUS_INVALID
            failure = "non_finite_residual"
            res_sq = math.inf
        self.last_residual_squared = res_sq
        self.last_residual = math.sqrt(max(0.0, res_sq)) if math.isfinite(res_sq) else math.inf
        # A finite stalled/iteration-limited solution is useful best effort:
        # projecting it reduces divergence. The previous fail-open path
        # discarded all progress, zeroed pressure, and advected the completely
        # unprojected velocity field. Singular/non-finite systems remain
        # fail-closed because their pressure field is not trustworthy.
        pressure_valid = pressure_valid or (
            math.isfinite(res_sq)
            and status in (PRESSURE_STATUS_MAX_ITERATIONS,
                           PRESSURE_STATUS_STALLED)
        )
        if not pressure_valid:
            self.pressure[:n].zero_()
        self._set_pressure_status(status, failure, pressure_valid)
        return self.last_status

    def coarse_initial_guess(self, voxel_type, iterations=32):
        """Electrostatic leaf-aggregate long-range guess (multigrid path)."""
        g = self.grid
        n_leaf = self._leaf_dim()
        self.leaf_rhs[:n_leaf].zero_()
        self.leaf_fluid[:n_leaf].zero_()
        self.leaf_pot[:n_leaf].zero_()
        self.leaf_pot_tmp[:n_leaf].zero_()
        fast_launch(
            k_build_leaf_aggregates, dim=self._dim(),
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

    def solve_pcg(self, voxel_type, max_iterations, tolerance, use_coarse_guess=False,
                  check_every=25):
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
        if use_coarse_guess:
            self.coarse_initial_guess(voxel_type)
        else:
            self.x0[:full_n].zero_()
        self._zero_pcg_vectors(full_n, x)
        fast_launch(
            k_initialize_pcg, dim=work_n,
            inputs=[self.rhs, self.x0, self.inv_diag, self.pressure_indices,
                    self._pressure_count, 1,
                    g.num_leaves,
                    x, self.r, self.z, self.p_dir,
                    self._pcg_compact_r, self._pcg_compact_z],
            device=self.device,
        )
        if use_coarse_guess:
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
        check_every = max(1, int(check_every))
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
        last_checked_res = 0.0
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
                    work_n, check_every, max_iterations, tol_sq)
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
            last_checked_res = float(self.residual.numpy()[0])
            res_sq = last_checked_res
            if not math.isfinite(last_checked_res):
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
            elif last_checked_res < tol_sq:
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
            elif res_sq >= last_checked_res * (1.0 - 1.0e-6):
                status = PRESSURE_STATUS_STALLED
                failure = "stalled"
            else:
                last_checked_res = res_sq
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

        Used by validation/diagnostics; performs one readback.
        """
        import numpy as _np

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
        r = self.rhs[:full_n].numpy() - self.ap[:full_n].numpy()
        return float(_np.linalg.norm(r))

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



    def project_st(self, vel, voxel_type, collision_vel, max_gradient=1.0e6,
                   reference_mass=8.0, eta_phi=0.5, rho_liquid=1.0,
                   rho_gas=0.0, eps_rho=0.05):
        g = self.grid
        fast_launch(
            k_project_st, dim=self._dim(),
            inputs=[self.pressure, voxel_type, vel.wu, vel.wv, vel.ww,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, g.voxel_size, max_gradient,
                    reference_mass, eta_phi, rho_liquid, rho_gas, eps_rho,
                    collision_vel,
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
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
