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

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import LEAF_VOLUME, VOXEL_COLLISION, VOXEL_FLUID
from .fields import voxel_coord_of_flat, voxel_flat_index
from .pressure import (
    k_apply_preconditioner,
    k_compute_alpha,
    k_compute_beta,
    k_update_p,
    k_update_xr,
    neighbor_flat,
)
from .registry import register_fn

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
    dt: float,
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
    c = viscosity_field[tid] * dt / (voxel_size * voxel_size)
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
    dt: float,
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
    c = viscosity_field[tid] * dt / (voxel_size * voxel_size)
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
    leaving collision faces untouched."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if weight[tid] > 0.0 and (voxel_type[tid] & VOXEL_COLLISION) == 0:
        vel_comp_out[tid] = x[tid]
    else:
        vel_comp_out[tid] = vel_comp_in[tid]


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
    """Implicit viscosity solve per velocity component (Jacobi or PCG)."""

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
        self.last_iterations = 0

    def _active_count(self):
        return int(self.grid.active_voxel_count)

    def smooth(self, vel, viscosity_field, voxel_type, dt, iterations=1):
        g = self.grid
        n = self._active_count()
        if n <= 0:
            return
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

    def _solve_component(self, comp, weight, viscosity_field, voxel_type, dt,
                         use_pcg, iterations, tolerance):
        g = self.grid
        n = self._active_count()
        if n <= 0:
            self.last_iterations = 0
            return
        stencil = [g.leaf_table, g.leaf_coords, g.voxels_dim, g.leaves_dim, g.num_leaves]
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
                    g.voxel_size, dt, self.inv_diag],
            device=self.device,
        )
        if not use_pcg:
            # Jacobi: x_{k+1} = Minv * (b + (Minv^-1 - A) x_k) implemented via
            # residual form x += Minv*(b - A x)
            for _ in range(iterations):
                wp.launch(
                    k_viscosity_apply_matrix, dim=n,
                    inputs=[self.x, viscosity_field, weight, voxel_type,
                            *stencil, g.voxel_size, dt, self.ap],
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
            self.last_iterations = iterations
        else:
            wp.launch(
                k_viscosity_apply_matrix, dim=n,
                inputs=[self.x, viscosity_field, weight, voxel_type,
                        *stencil, g.voxel_size, dt, self.ap],
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
            wp.utils.array_inner(self.r, self.z, out=self.rz, count=n)
            it = 0
            while it < iterations:
                wp.launch(
                    k_viscosity_apply_matrix, dim=n,
                    inputs=[self.p_dir, viscosity_field, weight, voxel_type,
                            *stencil, g.voxel_size, dt, self.ap],
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
                if it % 10 == 0 or it == iterations:
                    wp.utils.array_inner(self.r, self.r, out=self.residual, count=n)
                    if float(self.residual.numpy()[0]) < tolerance * tolerance * max(1, n):
                        break
            self.last_iterations = it
        # write back with collision guard
        wp.launch(
            k_viscosity_apply_collision, dim=n,
            inputs=[self.x, comp, weight, voxel_type, g.num_leaves, self.tmp_u],
            device=self.device,
        )
        wp.copy(comp, self.tmp_u, count=n)

    def solve(self, vel, viscosity_field, voxel_type, dt, use_pcg=True,
              iterations=40, tolerance=1.0e-4):
        self._solve_component(vel.u, vel.wu, viscosity_field, voxel_type, dt,
                              use_pcg, iterations, tolerance)
        self._solve_component(vel.v, vel.wv, viscosity_field, voxel_type, dt,
                              use_pcg, iterations, tolerance)
        self._solve_component(vel.w, vel.ww, viscosity_field, voxel_type, dt,
                              use_pcg, iterations, tolerance)


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
