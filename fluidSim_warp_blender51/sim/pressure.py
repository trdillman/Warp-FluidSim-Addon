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
- electrostatic leaf aggregates: coarse per-leaf potential solve used as the
  long-range initial guess (multigrid Off/VCycle/HalfVCycle)
- pressure threshold + scale, velocity expansion into collision

No bpy imports allowed here.
"""

from __future__ import annotations

import math

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID
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

# ---------------------------------------------------------------------------
# Divergence + matrix helpers
# ---------------------------------------------------------------------------


@wp.func
def neighbor_flat(
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    axis: wp.int32,
    step: wp.int32,
) -> wp.int32:
    o = wp.vec3i(ijk[0], ijk[1], ijk[2])
    if axis == 0:
        o = wp.vec3i(ijk[0] + step, ijk[1], ijk[2])
    elif axis == 1:
        o = wp.vec3i(ijk[0], ijk[1] + step, ijk[2])
    else:
        o = wp.vec3i(ijk[0], ijk[1], ijk[2] + step)
    return voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)


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
        hi = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, 1)
        lo_n = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, -1)
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
            n = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
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
    voxel_type: wp.array(dtype=wp.int32),
    inv_diag: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    ax: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentApplyMatrix8/16/32: Ax for the 7-point
    Poisson stencil restricted to fluid voxels."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0 or inv_diag[tid] == 0.0:
        ax[tid] = 0.0
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    diag = 1.0 / inv_diag[tid]
    acc = diag * x[tid]
    for axis in range(3):
        for step in range(-1, 2, 2):
            n = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if n >= 0:
                nvt = voxel_type[n]
                if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                    acc -= x[n]
    ax[tid] = acc


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
            n = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, step)
            if n >= 0:
                nvt = voxel_type[n]
                if (nvt & VOXEL_FLUID) != 0 and (nvt & VOXEL_COLLISION) == 0:
                    acc += x_in[n]
    x_out[tid] = acc * inv_diag[tid]


# ---------------------------------------------------------------------------
# PCG scalar update kernels (device-resident alpha/beta)
# ---------------------------------------------------------------------------


@wp.kernel
def k_initialize_pcg(
    rhs: wp.array(dtype=float),
    x0: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
    z: wp.array(dtype=float),
    p: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentInitializePcg: x=x0, r=b-Ax0 (x0 usually 0
    so r=b here; when a coarse initial guess is used the caller pre-applies
    the matrix), z=Minv r, p=z."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    x[tid] = x0[tid]
    ri = rhs[tid]
    r[tid] = ri
    zi = ri * inv_diag[tid]
    z[tid] = zi
    p[tid] = zi


@wp.kernel
def k_subtract_ax_from_r(
    ax: wp.array(dtype=float),
    inv_diag: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    r: wp.array(dtype=float),
    z: wp.array(dtype=float),
    p: wp.array(dtype=float),
):
    """Complete r = b - A x0 for a nonzero initial guess, refresh z and p."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ri = r[tid] - ax[tid]
    r[tid] = ri
    zi = ri * inv_diag[tid]
    z[tid] = zi
    p[tid] = zi


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
def k_update_p(
    z: wp.array(dtype=float),
    beta: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    p: wp.array(dtype=float),
):
    """fluidSimProjectNonDivergentUpdateP."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p[tid] = z[tid] + beta[0] * p[tid]


# ---------------------------------------------------------------------------
# Electrostatic leaf aggregates (coarse long-range solve / multigrid)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Projection + post ops
# ---------------------------------------------------------------------------


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
    faces between (fluid|air) voxels; solid faces take collider velocity."""
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
        n = neighbor_flat(leaf_table, voxels_dim, leaves_dim, ijk, axis, -1)
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
                wu[tid] = wp.max(wu[tid], 1.0e-6)
            elif axis == 1:
                v[tid] = vel_c[1]
                wv[tid] = wp.max(wv[tid], 1.0e-6)
            else:
                w[tid] = vel_c[2]
                ww[tid] = wp.max(ww[tid], 1.0e-6)
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
register_fn(k_update_p, "fluidSimProjectNonDivergentUpdateP")
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
register_fn(k_sample_down_residual, "fluidSimProjectNonDivergentSampleDown")
register_fn(k_project, "fluidSimProjectNonDivergentProject")
register_fn(k_expand_velocity_into_collision,
            "fluidSimProjectNonDivergentExpandVelocityIntoCollision")
register_fn(k_pressure_threshold_compute, "fluidSimPressureThresholdCompute")
register_fn(k_pressure_threshold_apply, "fluidSimPressureThresholdApply")
register_fn(k_pressure_scale, "fluidSimPressureScale")


# ---------------------------------------------------------------------------
# Host-orchestrated solver driver (GPU-resident state)
# ---------------------------------------------------------------------------


class PressureSolver:
    """Owns pressure work buffers and runs Jacobi or PCG projections."""

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
        # coarse (leaf) level
        self.leaf_rhs = wp.zeros(grid.capacity, dtype=float, device=device)
        self.leaf_fluid = wp.zeros(grid.capacity, dtype=wp.int32, device=device)
        self.leaf_pot = wp.zeros(grid.capacity, dtype=float, device=device)
        self.leaf_pot_tmp = wp.zeros(grid.capacity, dtype=float, device=device)
        # device scalars
        self.rz = wp.zeros(1, dtype=float, device=device)
        self.rz_new = wp.zeros(1, dtype=float, device=device)
        self.p_ap = wp.zeros(1, dtype=float, device=device)
        self.alpha = wp.zeros(1, dtype=float, device=device)
        self.beta = wp.zeros(1, dtype=float, device=device)
        self.residual = wp.zeros(1, dtype=float, device=device)
        self.max_pressure = wp.zeros(1, dtype=float, device=device)
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
        return self.grid.voxel_capacity

    def _stencil_args(self):
        g = self.grid
        return [g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim, g.num_leaves]

    def _make_status(self, converged, reason, residual, iterations):
        return {
            "converged": bool(converged),
            "iterations": int(iterations),
            "reason": reason,
            "residual": float(residual),
            "project_velocity": bool(converged),
        }

    def _set_pressure_status(self, status, failure="", pressure_valid=False):
        self.last_pressure_status = status
        self.last_pressure_status_code = PRESSURE_STATUS_CODES.get(
            status, PRESSURE_STATUS_CODES[PRESSURE_STATUS_INVALID])
        self.last_pcg_failure = failure
        self.last_pressure_valid = pressure_valid
        self.last_status = self._make_status(
            pressure_valid, failure or status, self.last_residual,
            self.last_iterations)

    def should_project_pressure(self):
        """True when the current pressure field is safe to project."""
        return self.last_pressure_valid

    def build_rhs(self, vel, voxel_type, collision_vel, voxel_counts,
                  rest_count=8.0, count_stiffness=0.0):
        g = self.grid
        wp.launch(
            k_compute_divergence, dim=self._dim(),
            inputs=[vel.u, vel.v, vel.w, voxel_type, collision_vel,
                    g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                    g.num_leaves, g.voxel_size, self.rhs, self.divergence],
            device=self.device,
        )
        if count_stiffness > 0.0:
            wp.launch(
                k_particle_count_to_pressure_rhs, dim=self._dim(),
                inputs=[voxel_counts, voxel_type, g.num_leaves,
                        rest_count, count_stiffness, g.voxel_size, self.rhs],
                device=self.device,
            )

    def build_preconditioner(self, voxel_type):
        g = self.grid
        wp.launch(
            k_compute_inverse_diagonal, dim=self._dim(),
            inputs=[voxel_type, g.leaf_table, g.leaf_coords, g.voxels_dim,
                    g.leaves_dim, g.num_leaves, self.inv_diag],
            device=self.device,
        )

    def coarse_initial_guess(self, voxel_type, iterations=32):
        """Electrostatic leaf-aggregate long-range guess (multigrid path)."""
        g = self.grid
        self.leaf_rhs.zero_()
        self.leaf_fluid.zero_()
        self.leaf_pot.zero_()
        self.leaf_pot_tmp.zero_()
        wp.launch(
            k_build_leaf_aggregates, dim=self._dim(),
            inputs=[self.rhs, voxel_type, g.num_leaves, self.leaf_rhs, self.leaf_fluid],
            device=self.device,
        )
        src, dst = self.leaf_pot, self.leaf_pot_tmp
        for _ in range(iterations):
            wp.launch(
                k_solve_leaf_potential_jacobi, dim=g.capacity,
                inputs=[self.leaf_rhs, self.leaf_fluid, g.leaf_table,
                        g.leaf_coords, g.leaves_dim, g.num_leaves, src, dst],
                device=self.device,
            )
            src, dst = dst, src
        wp.launch(
            k_sample_up_leaf_potential, dim=self._dim(),
            inputs=[src, voxel_type, g.num_leaves, self.x0],
            device=self.device,
        )

    def solve_jacobi(self, voxel_type, iterations):
        g = self.grid
        src, dst = self.pressure, self.pressure_tmp
        src.zero_()
        for _ in range(iterations):
            wp.launch(
                k_jacobi_iteration, dim=self._dim(),
                inputs=[src, self.rhs, voxel_type, self.inv_diag,
                        g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim,
                        g.num_leaves, dst],
                device=self.device,
            )
            src, dst = dst, src
        if src is not self.pressure:
            wp.copy(self.pressure, src)
        self.last_iterations = iterations
        self.last_residual = -1.0
        self.last_residual_squared = -1.0
        self._set_pressure_status(
            PRESSURE_STATUS_JACOBI, PRESSURE_STATUS_JACOBI, True)
        return self.last_status

    def solve_pcg(self, voxel_type, max_iterations, tolerance, use_coarse_guess=False,
                  check_every=25):
        g = self.grid
        n = self._dim()
        x = self.pressure
        if use_coarse_guess:
            self.coarse_initial_guess(voxel_type)
        else:
            self.x0.zero_()
        wp.launch(
            k_initialize_pcg, dim=n,
            inputs=[self.rhs, self.x0, self.inv_diag, g.num_leaves,
                    x, self.r, self.z, self.p_dir],
            device=self.device,
        )
        if use_coarse_guess:
            wp.launch(
                k_apply_matrix, dim=n,
                inputs=[x, voxel_type, self.inv_diag, *self._stencil_args(), self.ap],
                device=self.device,
            )
            wp.launch(
                k_subtract_ax_from_r, dim=n,
                inputs=[self.ap, self.inv_diag, g.num_leaves,
                        self.r, self.z, self.p_dir],
                device=self.device,
            )
        check_every = max(1, int(check_every))
        tol_sq = tolerance * tolerance * max(1, n)
        status = PRESSURE_STATUS_MAX_ITERATIONS
        failure = "max_iterations"
        pressure_valid = False
        wp.utils.array_inner(self.r, self.z, out=self.rz)
        wp.utils.array_inner(self.r, self.r, out=self.residual)
        last_checked_res = float(self.residual.numpy()[0])
        it = 0
        if not math.isfinite(last_checked_res):
            status = PRESSURE_STATUS_INVALID
            failure = "non_finite_residual"
        elif last_checked_res < tol_sq:
            status = PRESSURE_STATUS_CONVERGED
            failure = PRESSURE_STATUS_CONVERGED
            pressure_valid = True
        while it < max_iterations:
            if status != PRESSURE_STATUS_MAX_ITERATIONS:
                break
            wp.launch(
                k_apply_matrix, dim=n,
                inputs=[self.p_dir, voxel_type, self.inv_diag,
                        *self._stencil_args(), self.ap],
                device=self.device,
            )
            wp.utils.array_inner(self.p_dir, self.ap, out=self.p_ap)
            denom = float(self.p_ap.numpy()[0])
            rz_old = float(self.rz.numpy()[0])
            if not math.isfinite(denom) or not math.isfinite(rz_old):
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
                break
            if abs(denom) < 1.0e-30 or abs(rz_old) < 1.0e-30:
                status = PRESSURE_STATUS_SINGULAR
                failure = "singular_system"
                break
            wp.launch(k_compute_alpha, dim=1,
                      inputs=[self.rz, self.p_ap, self.alpha], device=self.device)
            wp.launch(
                k_update_xr, dim=n,
                inputs=[self.p_dir, self.ap, self.alpha, g.num_leaves, x, self.r],
                device=self.device,
            )
            wp.launch(
                k_apply_preconditioner, dim=n,
                inputs=[self.r, self.inv_diag, g.num_leaves, self.z],
                device=self.device,
            )
            wp.utils.array_inner(self.r, self.z, out=self.rz_new)
            rz_new = float(self.rz_new.numpy()[0])
            if not math.isfinite(rz_new):
                status = PRESSURE_STATUS_INVALID
                failure = "non_finite_residual"
                break
            if rz_new < -1.0e-20:
                status = PRESSURE_STATUS_SINGULAR
                failure = "singular_system"
                break
            wp.launch(k_compute_beta, dim=1,
                      inputs=[self.rz_new, self.rz, self.beta], device=self.device)
            wp.launch(
                k_update_p, dim=n,
                inputs=[self.z, self.beta, g.num_leaves, self.p_dir],
                device=self.device,
            )
            wp.copy(self.rz, self.rz_new)
            it += 1
            if it % check_every == 0 or it == max_iterations:
                wp.utils.array_inner(self.r, self.r, out=self.residual)
                res_sq = float(self.residual.numpy()[0])
                if not math.isfinite(res_sq):
                    status = PRESSURE_STATUS_INVALID
                    failure = "non_finite_residual"
                    break
                if res_sq < tol_sq:
                    status = PRESSURE_STATUS_CONVERGED
                    failure = PRESSURE_STATUS_CONVERGED
                    pressure_valid = True
                    break
                if res_sq >= last_checked_res * (1.0 - 1.0e-6):
                    status = PRESSURE_STATUS_STALLED
                    failure = "stalled"
                    break
                last_checked_res = res_sq
        self.last_iterations = it
        wp.utils.array_inner(self.r, self.r, out=self.residual)
        res_sq = float(self.residual.numpy()[0])
        if not math.isfinite(res_sq):
            status = PRESSURE_STATUS_INVALID
            failure = "non_finite_residual"
            res_sq = math.inf
        self.last_residual_squared = res_sq
        self.last_residual = math.sqrt(max(0.0, res_sq)) if math.isfinite(res_sq) else math.inf
        if status != PRESSURE_STATUS_CONVERGED:
            self.pressure.zero_()
        self._set_pressure_status(status, failure, pressure_valid)
        return self.last_status

    def measure_residual(self, voxel_type):
        """L2 norm of (rhs - A*pressure) for the current pressure solution.

        Used by validation/diagnostics; performs one readback.
        """
        import numpy as _np

        wp.launch(
            k_apply_matrix, dim=self._dim(),
            inputs=[self.pressure, voxel_type, self.inv_diag,
                    *self._stencil_args(), self.ap],
            device=self.device,
        )
        r = self.rhs.numpy() - self.ap.numpy()
        return float(_np.linalg.norm(r))

    def apply_threshold(self, relative_threshold):
        if relative_threshold <= 0.0:
            return
        g = self.grid
        self.max_pressure.zero_()
        wp.launch(
            k_pressure_threshold_compute, dim=self._dim(),
            inputs=[self.pressure, g.num_leaves, self.max_pressure],
            device=self.device,
        )
        wp.launch(
            k_pressure_threshold_apply, dim=self._dim(),
            inputs=[self.max_pressure, relative_threshold, g.num_leaves, self.pressure],
            device=self.device,
        )

    def apply_scale(self, scale):
        if scale == 1.0:
            return
        wp.launch(
            k_pressure_scale, dim=self._dim(),
            inputs=[scale, self.grid.num_leaves, self.pressure],
            device=self.device,
        )

    def project(self, vel, voxel_type, collision_vel, max_gradient=1.0e6):
        g = self.grid
        wp.launch(
            k_project, dim=self._dim(),
            inputs=[self.pressure, voxel_type, g.leaf_table, g.leaf_coords,
                    g.voxels_dim, g.leaves_dim, g.num_leaves, g.voxel_size,
                    max_gradient,
                    collision_vel,
                    vel.u, vel.v, vel.w, vel.wu, vel.wv, vel.ww],
            device=self.device,
        )
