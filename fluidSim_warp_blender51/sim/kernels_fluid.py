"""FluidSim liquid kernels: PIC/FLIP transfer, advection, forces, particle
indexing/sorting/relaxation and lifecycle (``fluidSim*`` family).

Original source namespace was ``paradigm*``; the registry maintains aliases.
No bpy imports allowed here.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import (
    DEFAULT_RELAX_BIN_SAMPLE_LIMIT,
    LEAF_VOLUME,
    VOXEL_ACTIVE,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_SINK,
    VOXEL_SOURCE_REGION_TRACKING,
)
from .fields import (
    leaf_linear_index,
    leaf_coord_of_voxel,
    sample_mac_velocity,
    sample_scalar_trilinear,
    sample_vec3_trilinear,
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_floor,
    voxel_in_domain,
    world_to_voxel,
)
from .registry import register, register_fn

# ---------------------------------------------------------------------------
# Particle -> grid (P2G)
# ---------------------------------------------------------------------------


@wp.kernel
def k_particle_set_grid_data(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
):
    """fluidSimParticleSetGridData: staggered trilinear velocity scatter."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    vel = velocities[tid]
    for axis in range(3):
        off = wp.vec3(0.5, 0.5, 0.5)
        if axis == 0:
            off = wp.vec3(0.0, 0.5, 0.5)
        elif axis == 1:
            off = wp.vec3(0.5, 0.0, 0.5)
        else:
            off = wp.vec3(0.5, 0.5, 0.0)
        q = vp - off
        base = voxel_floor(q)
        f = q - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
        comp = vel[axis]
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                    if flat >= 0:
                        wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                        wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                        wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                        wgt = wx * wy * wz
                        if axis == 0:
                            wp.atomic_add(u, flat, wgt * comp)
                            wp.atomic_add(wu, flat, wgt)
                        elif axis == 1:
                            wp.atomic_add(v, flat, wgt * comp)
                            wp.atomic_add(wv, flat, wgt)
                        else:
                            wp.atomic_add(w, flat, wgt * comp)
                            wp.atomic_add(ww, flat, wgt)


@wp.kernel
def k_normalize_mac(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if wu[tid] > 1.0e-8:
        u[tid] = u[tid] / wu[tid]
    if wv[tid] > 1.0e-8:
        v[tid] = v[tid] / wv[tid]
    if ww[tid] > 1.0e-8:
        w[tid] = w[tid] / ww[tid]


@wp.kernel
def k_particle_to_count(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
):
    """fluidSimParticleToCount: per-voxel particle occupancy count."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ijk = voxel_floor(vp)
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat >= 0:
        wp.atomic_add(voxel_counts, flat, 1)


@wp.kernel
def k_particle_to_grid_active(
    voxel_counts: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
):
    """fluidSimParticleToGridActive: mark fluid voxels from occupancy."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if voxel_counts[tid] > 0 and (vt & VOXEL_COLLISION) == 0:
        voxel_type[tid] = vt | VOXEL_FLUID | VOXEL_ACTIVE
    else:
        voxel_type[tid] = vt & (~VOXEL_FLUID)




# ---------------------------------------------------------------------------
# Spatiotemporal FLIP (ST-FLIP) P2G / phase helpers
# ---------------------------------------------------------------------------


@wp.func
def _st_poly6_1d(r: float) -> float:
    q = 1.0 - r * r
    if q <= 0.0:
        return 0.0
    return q * q * q


@wp.func
def _st_spatial_weight_1d(r: float) -> float:
    # Eq. (18), separable 1-D poly6 normalized over [-1, 1].
    return (35.0 / 32.0) * _st_poly6_1d(r)


@wp.func
def _st_temporal_weight(tau: float) -> float:
    # Eq. (19), one-sided half-step-shifted poly6 on tau <= 1/2.
    if tau > 0.5:
        return 0.0
    return (35.0 / 16.0) * _st_poly6_1d(tau - 0.5)


@wp.func
def _st_phase_from_mass(weight_sum: float, reference_mass: float, eta_phi: float) -> float:
    denom = wp.max(reference_mass * wp.max(eta_phi, 1.0e-6), 1.0e-8)
    x = weight_sum / denom
    if x <= 0.0:
        return 0.0
    return wp.min(wp.sqrt(x), 1.0)


@wp.kernel
def k_particle_set_grid_data_spacetime(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    time_offsets: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    previous_dt: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
):
    """ST-FLIP 4D->3D P2G scatter.

    Particle velocity/mass are deposited to MAC faces with the separable
    spatial poly6 kernel W_S (Eq. 18) multiplied by the one-sided temporal
    kernel W_T(-delta_t / previous_dt) (Eq. 19).  The face accumulators are
    intentionally left as raw weighted mass sums; velocity normalization still
    divides momentum by mass, while the raw sums become the space-time phase
    field used by projection.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    safe_prev_dt = wp.max(wp.abs(previous_dt), 1.0e-8)
    tau = -time_offsets[tid] / safe_prev_dt
    wt = _st_temporal_weight(tau)
    if wt <= 0.0:
        return

    vp = world_to_voxel(positions[tid], origin, voxel_size)
    vel = velocities[tid]
    for axis in range(3):
        off = wp.vec3(0.5, 0.5, 0.5)
        if axis == 0:
            off = wp.vec3(0.0, 0.5, 0.5)
        elif axis == 1:
            off = wp.vec3(0.5, 0.0, 0.5)
        else:
            off = wp.vec3(0.5, 0.5, 0.0)
        q = vp - off
        base = voxel_floor(q)
        comp = vel[axis]
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    ijk = wp.vec3i(base[0] + dx, base[1] + dy, base[2] + dz)
                    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
                    if flat >= 0:
                        rx = q[0] - float(ijk[0])
                        ry = q[1] - float(ijk[1])
                        rz = q[2] - float(ijk[2])
                        wgt = wt * _st_spatial_weight_1d(rx) * _st_spatial_weight_1d(ry) * _st_spatial_weight_1d(rz)
                        if axis == 0:
                            wp.atomic_add(u, flat, wgt * comp)
                            wp.atomic_add(wu, flat, wgt)
                        elif axis == 1:
                            wp.atomic_add(v, flat, wgt * comp)
                            wp.atomic_add(wv, flat, wgt)
                        else:
                            wp.atomic_add(w, flat, wgt * comp)
                            wp.atomic_add(ww, flat, wgt)


@wp.kernel
def k_spacetime_weights_to_phase_active(
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    reference_mass: float,
    eta_phi: float,
    active_threshold: float,
    phase_cell: wp.array(dtype=float),
):
    """Collapse face P2G accumulators into a cell-centered phase mask.

    The field is C(m/(eta*m0)) with C(x)=min(sqrt(x), 1), following Eq. (13).
    Free-surface cells below active_threshold are removed from the pressure
    system as p=0 Dirichlet air cells; source-region fluid bits are preserved.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_COLLISION) != 0:
        phase_cell[tid] = 0.0
        voxel_type[tid] = vt & (~VOXEL_FLUID)
        return

    ijk = voxel_coord_of_flat(leaf_coords, tid)
    accum = _st_phase_from_mass(wu[tid], reference_mass, eta_phi)
    accum += _st_phase_from_mass(wv[tid], reference_mass, eta_phi)
    accum += _st_phase_from_mass(ww[tid], reference_mass, eta_phi)
    hi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]))
    if hi >= 0:
        accum += _st_phase_from_mass(wu[hi], reference_mass, eta_phi)
    hi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]))
    if hi >= 0:
        accum += _st_phase_from_mass(wv[hi], reference_mass, eta_phi)
    hi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] + 1))
    if hi >= 0:
        accum += _st_phase_from_mass(ww[hi], reference_mass, eta_phi)
    phi = accum / 6.0
    phase_cell[tid] = phi

    if phi >= active_threshold or (vt & VOXEL_SOURCE_REGION_TRACKING) != 0:
        voxel_type[tid] = vt | VOXEL_FLUID | VOXEL_ACTIVE
    else:
        voxel_type[tid] = vt & (~VOXEL_FLUID)


register_fn(k_particle_set_grid_data, "fluidSimParticleSetGridData")
register_fn(k_particle_set_grid_data_spacetime, "fluidSimParticleSetGridDataSpatiotemporal")
register_fn(k_spacetime_weights_to_phase_active, "fluidSimSpacetimeWeightsToPhaseActive")
register_fn(k_normalize_mac, "fluidSimWhiteWaterSetGridDataNormalize_internal")
register_fn(k_particle_to_count, "fluidSimParticleToCount")
register_fn(k_particle_to_grid_active, "fluidSimParticleToGridActive")


# ---------------------------------------------------------------------------
# Grid -> particle (G2P): PIC/FLIP blend
# ---------------------------------------------------------------------------


@wp.kernel
def k_apply_collider_material_to_grid(
    voxel_type: wp.array(dtype=wp.int32),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    impact_u: wp.array(dtype=float),
    impact_v: wp.array(dtype=float),
    impact_w: wp.array(dtype=float),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Apply collider slip and bounce to fluid cells beside solid voxels."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    for axis in range(3):
        lo_offset = wp.vec3i(0, 0, 0)
        hi_offset = wp.vec3i(0, 0, 0)
        if axis == 0:
            lo_offset = wp.vec3i(-1, 0, 0)
            hi_offset = wp.vec3i(1, 0, 0)
        elif axis == 1:
            lo_offset = wp.vec3i(0, -1, 0)
            hi_offset = wp.vec3i(0, 1, 0)
        else:
            lo_offset = wp.vec3i(0, 0, -1)
            hi_offset = wp.vec3i(0, 0, 1)
        lo = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk + lo_offset)
        hi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk + hi_offset)
        lo_solid = False
        hi_solid = False
        if lo >= 0:
            lo_solid = collision_sdf[lo] < 0.0
        if hi >= 0:
            hi_solid = collision_sdf[hi] < 0.0
        if lo_solid == hi_solid:
            continue

        # MAC components live on lower-axis faces. A solid lower neighbor
        # contacts this voxel's face; a solid upper neighbor contacts the
        # upper voxel's lower face. Only modify that normal face component.
        face = tid
        normal_sign = float(1.0)
        if hi_solid:
            face = hi
            normal_sign = -1.0
        vcol_axis = collision_vel[face][axis]
        restitution = wp.clamp(collision_restitution[face], 0.0, 1.0)
        current_axis = float(0.0)
        impact_axis = float(0.0)
        if axis == 0:
            current_axis = u[face]
            impact_axis = impact_u[face]
        elif axis == 1:
            current_axis = v[face]
            impact_axis = impact_v[face]
        else:
            current_axis = w[face]
            impact_axis = impact_w[face]
        current_n = (current_axis - vcol_axis) * normal_sign
        impact_n = (impact_axis - vcol_axis) * normal_sign
        if impact_n < 0.0:
            result_n = wp.max(current_n, -impact_n * restitution)
            result_axis = vcol_axis + normal_sign * result_n
            if axis == 0:
                u[face] = result_axis
            elif axis == 1:
                v[face] = result_axis
            else:
                w[face] = result_axis


register_fn(k_apply_collider_material_to_grid,
            "fluidSimApplyColliderMaterialToGrid")


@wp.kernel
def k_staggered_field_to_particle(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    u_old: wp.array(dtype=float),
    v_old: wp.array(dtype=float),
    w_old: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    sample_type: wp.int32,  # ParticleSampleType: 0 PIC, 1 FLIP
    flip_ratio: float,
    velocities: wp.array(dtype=wp.vec3),
):
    """fluidSimStaggeredVectorFieldToParticle: PIC/FLIP velocity update.

    ``u_old`` holds the pre-force/projection grid velocity so the FLIP path
    adds the grid delta; the PIC path replaces with the new grid velocity.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    v_new = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp)
    if sample_type == 0:
        velocities[tid] = v_new
    else:
        v_old_s = sample_mac_velocity(
            u_old, v_old, w_old, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, vp
        )
        v_flip = velocities[tid] + (v_new - v_old_s)
        velocities[tid] = flip_ratio * v_flip + (1.0 - flip_ratio) * v_new


register_fn(k_staggered_field_to_particle, "fluidSimStaggeredVectorFieldToParticle")


# ---------------------------------------------------------------------------
# Advection (fluid_advection)
# ---------------------------------------------------------------------------


@wp.func
def _grid_vel(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i, leaves_dim: wp.vec3i,
    origin: wp.vec3, voxel_size: float, p: wp.vec3,
) -> wp.vec3:
    return sample_mac_velocity(
        u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim,
        world_to_voxel(p, origin, voxel_size),
    )


@wp.func
def trace_position(
    p0: wp.vec3,
    vel_p: wp.vec3,
    dt: float,
    trace_mode: wp.int32,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i, leaves_dim: wp.vec3i,
    origin: wp.vec3, voxel_size: float,
) -> wp.vec3:
    """TraceMode semantics: None/SingleSample/MidPoint/ModifiedMidPoint/RK3."""
    if trace_mode == 0:  # None: particle velocity only
        return p0 + vel_p * dt
    if trace_mode == 1:  # SingleSample
        v1 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, p0)
        return p0 + v1 * dt
    if trace_mode == 2:  # MidPoint
        v1 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, p0)
        pm = p0 + v1 * (0.5 * dt)
        v2 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, pm)
        return p0 + v2 * dt
    if trace_mode == 3:  # ModifiedMidPoint: blend particle & midpoint grid velocity
        pm = p0 + vel_p * (0.5 * dt)
        v2 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, pm)
        return p0 + (0.5 * (vel_p + v2)) * dt
    # RK3 (Ralston)
    v1 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, p0)
    p1 = p0 + v1 * (0.5 * dt)
    v2 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, p1)
    p2 = p0 + v2 * (0.75 * dt)
    v3 = _grid_vel(u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim, origin, voxel_size, p2)
    return p0 + (v1 * (2.0 / 9.0) + v2 * (3.0 / 9.0) + v3 * (4.0 / 9.0)) * dt


@wp.func
def collision_sdf_gradient(
    vp: wp.vec3,
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
) -> wp.vec3:
    eps = 0.5
    gx = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp + wp.vec3(eps, 0.0, 0.0), 1.0e9) - sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp - wp.vec3(eps, 0.0, 0.0), 1.0e9)
    gy = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp + wp.vec3(0.0, eps, 0.0), 1.0e9) - sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp - wp.vec3(0.0, eps, 0.0), 1.0e9)
    gz = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp + wp.vec3(0.0, 0.0, eps), 1.0e9) - sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp - wp.vec3(0.0, 0.0, eps), 1.0e9)
    return wp.vec3(gx, gy, gz)


@wp.func
def resolve_collision(
    p: wp.vec3,
    vel: wp.vec3,
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
):
    """Push a particle out of the collision SDF and reflect velocity."""
    vp = world_to_voxel(p, origin, voxel_size)
    phi = sample_scalar_trilinear(
        collision_sdf, leaf_table, voxels_dim, leaves_dim, vp, 1.0e9
    )
    p_out = p
    v_out = vel
    if phi < 0.0:
        n = collision_sdf_gradient(
            vp, collision_sdf, leaf_table, voxels_dim, leaves_dim)
        ln = wp.length(n)
        if ln > 1.0e-8:
            n = n / ln
            # Central differences span one voxel, so ln / voxel_size is the
            # world-space gradient magnitude. Do not assume a perfect SDF.
            p_out = p - n * phi * (voxel_size / ln) * 1.001
            vcol = sample_vec3_trilinear(
                collision_vel, leaf_table, voxels_dim, leaves_dim,
                vp, wp.vec3(0.0))
            friction = sample_scalar_trilinear(
                collision_friction, leaf_table, voxels_dim, leaves_dim,
                vp, 0.0)
            restitution = sample_scalar_trilinear(
                collision_restitution, leaf_table, voxels_dim, leaves_dim,
                vp, 0.0)
            vrel = v_out - vcol
            vn = wp.dot(vrel, n)
            if vn < 0.0:
                vt = vrel - n * vn
                v_out = vcol + vt * (1.0 - friction) - n * vn * restitution
    return p_out, v_out


@wp.func
def resolve_collision_segment(
    p0: wp.vec3,
    p1: wp.vec3,
    vel: wp.vec3,
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
):
    """Resolve the first outside-to-inside crossing and retain clearance."""
    vp0 = world_to_voxel(p0, origin, voxel_size)
    vp1 = world_to_voxel(p1, origin, voxel_size)
    phi0 = sample_scalar_trilinear(
        collision_sdf, leaf_table, voxels_dim, leaves_dim, vp0, 1.0e9)
    hit = p0
    hit_found = phi0 < 0.0
    outside = p0
    inside = p0
    outside_t = float(0.0)
    inside_t = float(0.0)
    segment_length = wp.length(p1 - p0)
    sample_count = wp.int32(wp.ceil(segment_length / (0.5 * voxel_size)))
    sample_count = wp.max(1, wp.min(sample_count, 16))
    for sample_index in range(1, 17):
        if sample_index <= sample_count and not hit_found:
            t = float(sample_index) / float(sample_count)
            sample = p0 + (p1 - p0) * t
            sample_phi = sample_scalar_trilinear(
                collision_sdf, leaf_table, voxels_dim, leaves_dim,
                world_to_voxel(sample, origin, voxel_size), 1.0e9)
            if sample_phi < 0.0:
                inside = sample
                inside_t = t
                hit_found = True
            else:
                outside = sample
                outside_t = t
    if not hit_found:
        return p1, vel
    if phi0 >= 0.0:
        for _iteration in range(8):
            middle = 0.5 * (outside + inside)
            middle_phi = sample_scalar_trilinear(
                collision_sdf, leaf_table, voxels_dim, leaves_dim,
                world_to_voxel(middle, origin, voxel_size), 1.0e9)
            if middle_phi < 0.0:
                inside = middle
                inside_t = 0.5 * (outside_t + inside_t)
            else:
                outside = middle
                outside_t = 0.5 * (outside_t + inside_t)
        hit = inside

    p_out, v_out = resolve_collision(
        hit, vel, collision_sdf, collision_vel,
        collision_friction, collision_restitution,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size)
    vp = world_to_voxel(p_out, origin, voxel_size)
    n = collision_sdf_gradient(
        vp, collision_sdf, leaf_table, voxels_dim, leaves_dim)
    n_len = wp.length(n)
    if n_len > 1.0e-8:
        p_out = p_out + n * ((0.05 * voxel_size) / n_len)
    if phi0 >= 0.0:
        remaining_dt = dt * (1.0 - inside_t)
        candidate = p_out + v_out * remaining_dt
        p_out, v_out = resolve_collision(
            candidate, v_out, collision_sdf, collision_vel,
            collision_friction, collision_restitution,
            leaf_table, voxels_dim, leaves_dim, origin, voxel_size)
    return p_out, v_out


@wp.kernel
def k_advect_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    trace_mode: wp.int32,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    open_boundaries: wp.int32,
    alive: wp.array(dtype=wp.int32),
):
    """fluidSimAdvectParticles with TraceMode + collision + domain handling."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    if alive[tid] == 0:
        return
    p0 = positions[tid]
    vel = velocities[tid]
    p1 = trace_position(
        p0, vel, dt, trace_mode,
        u, v, w, wu, wv, ww,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size,
    )
    p1, vel = resolve_collision_segment(
        p0, p1, vel, collision_sdf, collision_vel,
        collision_friction, collision_restitution,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size, dt,
    )
    # domain boundary
    margin = 1.0e-4
    for a in range(3):
        lo = domain_min[a] + margin
        hi = domain_max[a] - margin
        if p1[a] < lo:
            if open_boundaries != 0:
                alive[tid] = 0
            else:
                if a == 0:
                    p1 = wp.vec3(lo, p1[1], p1[2])
                    vel = wp.vec3(wp.max(vel[0], 0.0), vel[1], vel[2])
                elif a == 1:
                    p1 = wp.vec3(p1[0], lo, p1[2])
                    vel = wp.vec3(vel[0], wp.max(vel[1], 0.0), vel[2])
                else:
                    p1 = wp.vec3(p1[0], p1[1], lo)
                    vel = wp.vec3(vel[0], vel[1], wp.max(vel[2], 0.0))
        if p1[a] > hi:
            if open_boundaries != 0:
                alive[tid] = 0
            else:
                if a == 0:
                    p1 = wp.vec3(hi, p1[1], p1[2])
                    vel = wp.vec3(wp.min(vel[0], 0.0), vel[1], vel[2])
                elif a == 1:
                    p1 = wp.vec3(p1[0], hi, p1[2])
                    vel = wp.vec3(vel[0], wp.min(vel[1], 0.0), vel[2])
                else:
                    p1 = wp.vec3(p1[0], p1[1], hi)
                    vel = wp.vec3(vel[0], vel[1], wp.min(vel[2], 0.0))
    positions[tid] = p1
    velocities[tid] = vel




@wp.func
def _st_smoothstep01(x: float) -> float:
    s = wp.clamp(x, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


@wp.kernel
def k_advect_particles_spacetime(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    time_offsets: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    trace_mode: wp.int32,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    open_boundaries: wp.int32,
    step_seed: wp.int32,
    jitter_strength: float,
    adaptive_jitter: wp.int32,
    local_cfl_limit: float,
    max_local_substeps: wp.int32,
    alive: wp.array(dtype=wp.int32),
):
    """ST-FLIP stochastic staggered-time advection with residual carryover.

    Implements Eq. (10)-(12): each particle carries delta_t = t_global-t_p.
    The actual advection interval is clamp(dt + delta_t + gamma*xi*dt, 0, 2dt),
    and the residual is updated to dt + delta_t - dt_act to prevent time drift.
    The trajectory is internally sub-stepped to a local CFL bound for collision
    robustness at high global CFL.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    if alive[tid] == 0:
        return

    vel = velocities[tid]
    local_cfl = wp.length(vel) * wp.abs(dt) / wp.max(voxel_size, 1.0e-8)
    gamma = wp.max(jitter_strength, 0.0)
    if adaptive_jitter != 0:
        gamma = gamma * _st_smoothstep01(local_cfl)

    state = wp.rand_init(step_seed, tid)
    xi = wp.randf(state) - 0.5
    residual = time_offsets[tid]
    dt_act = wp.clamp(dt + residual + gamma * xi * dt, 0.0, 2.0 * dt)
    time_offsets[tid] = dt + residual - dt_act

    nsteps = wp.int32(1)
    if local_cfl_limit > 1.0e-6:
        est = wp.length(vel) * wp.abs(dt_act) / (wp.max(voxel_size, 1.0e-8) * local_cfl_limit)
        nsteps = wp.int32(wp.ceil(est))
        if nsteps < 1:
            nsteps = 1
    max_steps = max_local_substeps
    if max_steps < 1:
        max_steps = 1
    if nsteps > max_steps:
        nsteps = max_steps
    sub_dt = dt_act / float(nsteps)

    pcur = positions[tid]
    vcur = vel
    i = wp.int32(0)
    while i < nsteps:
        pnext = trace_position(
            pcur, vcur, sub_dt, trace_mode,
            u, v, w, wu, wv, ww,
            leaf_table, voxels_dim, leaves_dim, origin, voxel_size,
        )
        pnext, vcur = resolve_collision_segment(
            pcur, pnext, vcur, collision_sdf, collision_vel,
            collision_friction, collision_restitution,
            leaf_table, voxels_dim, leaves_dim, origin, voxel_size, sub_dt,
        )
        margin = 1.0e-4
        for a in range(3):
            lo = domain_min[a] + margin
            hi = domain_max[a] - margin
            if pnext[a] < lo:
                if open_boundaries != 0:
                    alive[tid] = 0
                    positions[tid] = pnext
                    velocities[tid] = vcur
                    return
                else:
                    if a == 0:
                        pnext = wp.vec3(lo, pnext[1], pnext[2])
                        vcur = wp.vec3(wp.max(vcur[0], 0.0), vcur[1], vcur[2])
                    elif a == 1:
                        pnext = wp.vec3(pnext[0], lo, pnext[2])
                        vcur = wp.vec3(vcur[0], wp.max(vcur[1], 0.0), vcur[2])
                    else:
                        pnext = wp.vec3(pnext[0], pnext[1], lo)
                        vcur = wp.vec3(vcur[0], vcur[1], wp.max(vcur[2], 0.0))
            if pnext[a] > hi:
                if open_boundaries != 0:
                    alive[tid] = 0
                    positions[tid] = pnext
                    velocities[tid] = vcur
                    return
                else:
                    if a == 0:
                        pnext = wp.vec3(hi, pnext[1], pnext[2])
                        vcur = wp.vec3(wp.min(vcur[0], 0.0), vcur[1], vcur[2])
                    elif a == 1:
                        pnext = wp.vec3(pnext[0], hi, pnext[2])
                        vcur = wp.vec3(vcur[0], wp.min(vcur[1], 0.0), vcur[2])
                    else:
                        pnext = wp.vec3(pnext[0], pnext[1], hi)
                        vcur = wp.vec3(vcur[0], vcur[1], wp.min(vcur[2], 0.0))
        pcur = pnext
        i = i + 1

    positions[tid] = pcur
    velocities[tid] = vcur


@wp.kernel
def k_resync_particle_positions_spacetime(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    time_offsets: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    trace_mode: wp.int32,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    render_positions: wp.array(dtype=wp.vec3),
):
    """Temporary render/export re-synchronization (Algorithm 1, lines 31-34).

    The simulation state remains at each particle's jittered sample time; this
    kernel predicts an aligned position at the global frame time by advecting
    a copy by the residual delta_t.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    p0 = positions[tid]
    vel = velocities[tid]
    dt_resync = time_offsets[tid]
    p1 = trace_position(
        p0, vel, dt_resync, trace_mode,
        u, v, w, wu, wv, ww,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size,
    )
    p1, vel = resolve_collision_segment(
        p0, p1, vel, collision_sdf, collision_vel,
        collision_friction, collision_restitution,
        leaf_table, voxels_dim, leaves_dim, origin, voxel_size, dt_resync,
    )
    margin = 1.0e-4
    p1 = wp.vec3(
        wp.clamp(p1[0], domain_min[0] + margin, domain_max[0] - margin),
        wp.clamp(p1[1], domain_min[1] + margin, domain_max[1] - margin),
        wp.clamp(p1[2], domain_min[2] + margin, domain_max[2] - margin),
    )
    render_positions[tid] = p1


register_fn(k_advect_particles, "fluidSimAdvectParticles")
register_fn(k_advect_particles_spacetime, "fluidSimAdvectParticlesSpatiotemporal")
register_fn(k_resync_particle_positions_spacetime, "fluidSimResyncParticlePositionsSpatiotemporal")


# ---------------------------------------------------------------------------
# Forces (fluid_forces_attraction)
# ---------------------------------------------------------------------------


@wp.kernel
def k_forces_gravity(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    gravity: wp.vec3,
    dt: float,
):
    """fluidSimForcesGravity: gravity on valid staggered faces."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if wu[tid] > 0.0:
        u[tid] = u[tid] + gravity[0] * dt
    if wv[tid] > 0.0:
        v[tid] = v[tid] + gravity[1] * dt
    if ww[tid] > 0.0:
        w[tid] = w[tid] + gravity[2] * dt


@wp.kernel
def k_forces_directional(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    force: wp.vec3,
    dt: float,
):
    """fluidSimForcesDirectionalForce."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if wu[tid] > 0.0:
        u[tid] = u[tid] + force[0] * dt
    if wv[tid] > 0.0:
        v[tid] = v[tid] + force[1] * dt
    if ww[tid] > 0.0:
        w[tid] = w[tid] + force[2] * dt


@wp.kernel
def k_forces_omnidirectional(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    origin: wp.vec3,
    voxel_size: float,
    center: wp.vec3,
    strength: float,  # >0 repel, <0 attract
    falloff_radius: float,
    dt: float,
):
    """fluidSimForcesOmnidirectionalForce: radial force around a point."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    # face positions per axis
    for axis in range(3):
        fp = origin + wp.vec3(
            (float(ijk[0]) + wp.where(axis == 0, 0.0, 0.5)) * voxel_size,
            (float(ijk[1]) + wp.where(axis == 1, 0.0, 0.5)) * voxel_size,
            (float(ijk[2]) + wp.where(axis == 2, 0.0, 0.5)) * voxel_size,
        )
        d = fp - center
        r = wp.length(d)
        if r < 1.0e-6 or r > falloff_radius:
            continue
        f = strength * (1.0 - r / falloff_radius) / r
        if axis == 0 and wu[tid] > 0.0:
            u[tid] = u[tid] + d[0] * f * dt
        elif axis == 1 and wv[tid] > 0.0:
            v[tid] = v[tid] + d[1] * f * dt
        elif axis == 2 and ww[tid] > 0.0:
            w[tid] = w[tid] + d[2] * f * dt


@wp.kernel
def k_forces_speed_limit(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    max_speed: float,
    limit_behavior: wp.int32,  # LimitBehavior: 0 Maintain(clamp) 1 Percent 2 Acceleration
    percent: float,
    dt: float,
):
    """fluidSimForcesSpeedLimit with LimitBehavior semantics."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    xp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0] + 1, ijk[1], ijk[2]))
    yp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1] + 1, ijk[2]))
    zp = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, wp.vec3i(ijk[0], ijk[1], ijk[2] + 1))
    cx = 0.5 * (u[tid] + wp.where(xp >= 0, u[xp], u[tid]))
    cy = 0.5 * (v[tid] + wp.where(yp >= 0, v[yp], v[tid]))
    cz = 0.5 * (w[tid] + wp.where(zp >= 0, w[zp], w[tid]))
    speed = wp.length(wp.vec3(cx, cy, cz))
    if speed <= max_speed or speed < 1.0e-9:
        return
    scale = float(1.0)
    if limit_behavior == 0:      # Maintain: hard clamp to max speed
        scale = max_speed / speed
    elif limit_behavior == 1:    # Percent: reduce excess by percent per step
        target = max_speed + (speed - max_speed) * wp.clamp(1.0 - percent, 0.0, 1.0)
        scale = target / speed
    else:                        # Acceleration: decelerate at fixed rate
        target = wp.max(max_speed, speed - percent * dt)
        scale = target / speed
    if wu[tid] > 0.0:
        u[tid] = u[tid] * scale
    if wv[tid] > 0.0:
        v[tid] = v[tid] * scale
    if ww[tid] > 0.0:
        w[tid] = w[tid] * scale


@wp.kernel
def k_forces_diffusion(
    u_in: wp.array(dtype=float), v_in: wp.array(dtype=float), w_in: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    amount: float,  # 0..1 blend to 6-neighbour average
    u_out: wp.array(dtype=float), v_out: wp.array(dtype=float), w_out: wp.array(dtype=float),
):
    """fluidSimForcesDiffusion: explicit velocity smoothing."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    su = float(0.0)
    sv = float(0.0)
    sw = float(0.0)
    n = float(0.0)
    for a in range(3):
        for s in range(-1, 2, 2):
            o = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if a == 0:
                o = wp.vec3i(ijk[0] + s, ijk[1], ijk[2])
            elif a == 1:
                o = wp.vec3i(ijk[0], ijk[1] + s, ijk[2])
            else:
                o = wp.vec3i(ijk[0], ijk[1], ijk[2] + s)
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
            if flat >= 0:
                su += u_in[flat]
                sv += v_in[flat]
                sw += w_in[flat]
                n += 1.0
    if n > 0.0:
        inv = 1.0 / n
        u_out[tid] = u_in[tid] + amount * (su * inv - u_in[tid]) * wp.where(wu[tid] > 0.0, 1.0, 0.0)
        v_out[tid] = v_in[tid] + amount * (sv * inv - v_in[tid]) * wp.where(wv[tid] > 0.0, 1.0, 0.0)
        w_out[tid] = w_in[tid] + amount * (sw * inv - w_in[tid]) * wp.where(ww[tid] > 0.0, 1.0, 0.0)
    else:
        u_out[tid] = u_in[tid]
        v_out[tid] = v_in[tid]
        w_out[tid] = w_in[tid]


@wp.kernel
def k_dissipate_particle_attribute(
    attr: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    rate: float,
    target: float,
    dt: float,
):
    """fluidSimForcesDissipateParticleAttribute: exponential decay to target."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    k = wp.exp(-rate * dt)
    attr[tid] = target + (attr[tid] - target) * k


register_fn(k_forces_gravity, "fluidSimForcesGravity")
register_fn(k_forces_directional, "fluidSimForcesDirectionalForce")
register_fn(k_forces_omnidirectional, "fluidSimForcesOmnidirectionalForce")
register_fn(k_forces_speed_limit, "fluidSimForcesSpeedLimit")
register_fn(k_forces_diffusion, "fluidSimForcesDiffusion")
register_fn(k_dissipate_particle_attribute, "fluidSimForcesDissipateParticleAttribute")


# ---------------------------------------------------------------------------
# Attraction (mask fields steering fluid)
# ---------------------------------------------------------------------------


@wp.kernel
def k_attract_mask_to_field(
    mask_src: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    gain: float,
    field: wp.array(dtype=float),
):
    """fluidSimAttractMaskToField: accumulate attraction mask into field."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    field[tid] = field[tid] + gain * mask_src[tid]


@wp.kernel
def k_attract_smooth_field(
    src: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    dst: wp.array(dtype=float),
):
    """fluidSimAttractSmoothField: 6-neighbour smoothing pass."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    acc = src[tid]
    n = float(1.0)
    for a in range(3):
        for s in range(-1, 2, 2):
            o = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if a == 0:
                o = wp.vec3i(ijk[0] + s, ijk[1], ijk[2])
            elif a == 1:
                o = wp.vec3i(ijk[0], ijk[1] + s, ijk[2])
            else:
                o = wp.vec3i(ijk[0], ijk[1], ijk[2] + s)
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
            if flat >= 0:
                acc += src[flat]
                n += 1.0
    dst[tid] = acc / n


@wp.kernel
def k_attract_direction(
    attract_field: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    dt: float,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """fluidSimAttractDirection: accelerate along attraction-field gradient."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    inv2h = 1.0 / (2.0 * voxel_size)
    c = attract_field[tid]
    for axis in range(3):
        lo = wp.vec3i(ijk[0] - wp.where(axis == 0, 1, 0), ijk[1] - wp.where(axis == 1, 1, 0), ijk[2] - wp.where(axis == 2, 1, 0))
        hi = wp.vec3i(ijk[0] + wp.where(axis == 0, 1, 0), ijk[1] + wp.where(axis == 1, 1, 0), ijk[2] + wp.where(axis == 2, 1, 0))
        flo = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, lo)
        fhi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, hi)
        g = (wp.where(fhi >= 0, attract_field[fhi], c) - wp.where(flo >= 0, attract_field[flo], c)) * inv2h
        dv = strength * g * dt
        if axis == 0 and wu[tid] > 0.0:
            u[tid] = u[tid] + dv
        elif axis == 1 and wv[tid] > 0.0:
            v[tid] = v[tid] + dv
        elif axis == 2 and ww[tid] > 0.0:
            w[tid] = w[tid] + dv


@wp.kernel
def k_attract_sdf_gradient_direction(
    sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_size: float,
    strength: float,
    max_influence_dist: float,
    dt: float,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """fluidSimAttractSdfGradientDirection: pull fluid along -grad(SDF)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    phi = sdf[tid]
    if phi > max_influence_dist:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    inv2h = 1.0 / (2.0 * voxel_size)
    for axis in range(3):
        lo = wp.vec3i(ijk[0] - wp.where(axis == 0, 1, 0), ijk[1] - wp.where(axis == 1, 1, 0), ijk[2] - wp.where(axis == 2, 1, 0))
        hi = wp.vec3i(ijk[0] + wp.where(axis == 0, 1, 0), ijk[1] + wp.where(axis == 1, 1, 0), ijk[2] + wp.where(axis == 2, 1, 0))
        flo = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, lo)
        fhi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, hi)
        g = (wp.where(fhi >= 0, sdf[fhi], phi) - wp.where(flo >= 0, sdf[flo], phi)) * inv2h
        dv = -strength * g * dt  # descend the SDF
        if axis == 0 and wu[tid] > 0.0:
            u[tid] = u[tid] + dv
        elif axis == 1 and wv[tid] > 0.0:
            v[tid] = v[tid] + dv
        elif axis == 2 and ww[tid] > 0.0:
            w[tid] = w[tid] + dv


register_fn(k_attract_mask_to_field, "fluidSimAttractMaskToField")
register_fn(k_attract_smooth_field, "fluidSimAttractSmoothField")
register_fn(k_attract_direction, "fluidSimAttractDirection")
register_fn(k_attract_sdf_gradient_direction, "fluidSimAttractSdfGradientDirection")


# ---------------------------------------------------------------------------
# Particle indexing / sorting (fluid_particle_indexing_sorting)
# ---------------------------------------------------------------------------


@wp.kernel
def k_particle_to_voxel_count_and_update_index(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
    particle_voxel: wp.array(dtype=wp.int32),
):
    """fluidSimParticleToVoxelCountAndUpdateVoxelIndex (+UpdateParticleVoxelIndex)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ijk = voxel_floor(vp)
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    particle_voxel[tid] = flat
    if flat >= 0:
        wp.atomic_add(voxel_counts, flat, 1)


@wp.kernel
def k_scatter_particles_to_bins(
    particle_voxel: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
):
    """fluidSimParticleToVoxelParticleArray: bin scatter using start offsets."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    voxel = particle_voxel[tid]
    if voxel < 0:
        return
    slot = starts[voxel] + wp.atomic_add(cursor, voxel, 1)
    indices[slot] = tid


def voxel_count_to_starts(counts: wp.array, starts: wp.array):
    """fluidSimVoxelCountToVoxelParticleArrayStart: exclusive prefix scan."""
    wp.utils.array_scan(counts, starts, inclusive=False)


register_fn(
    k_particle_to_voxel_count_and_update_index,
    "fluidSimParticleToVoxelCountAndUpdateVoxelIndex",
    "fluidSimUpdateParticleVoxelIndex",
)
register_fn(voxel_count_to_starts, "fluidSimVoxelCountToVoxelParticleArrayStart",
            strategy="host_orchestrated", notes="warp device scan")
register_fn(k_scatter_particles_to_bins, "fluidSimParticleToVoxelParticleArray",
            "fluidSimVoxelParticleArraySort",
            notes="atomic in-bin order; original in-bin sort not required for correctness")


# leaf-level binning (fluidSimParticleToLeafCount etc.)


@wp.kernel
def k_particle_to_leaf_count(
    particle_voxel: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    leaf_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    voxel = particle_voxel[tid]
    if voxel < 0:
        return
    wp.atomic_add(leaf_counts, voxel // LEAF_VOLUME, 1)


@wp.kernel
def k_scatter_particles_to_leaf_bins(
    particle_voxel: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    leaf_starts: wp.array(dtype=wp.int32),
    leaf_cursor: wp.array(dtype=wp.int32),
    leaf_indices: wp.array(dtype=wp.int32),
):
    """fluidSimSortParticlesLeafArrayMutex analogue: atomic cursor per leaf
    (mutex loops collapse to hardware atomics under CUDA)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    voxel = particle_voxel[tid]
    if voxel < 0:
        return
    leaf = voxel // LEAF_VOLUME
    slot = leaf_starts[leaf] + wp.atomic_add(leaf_cursor, leaf, 1)
    leaf_indices[slot] = tid


def leaf_count_to_starts(counts: wp.array, starts: wp.array):
    wp.utils.array_scan(counts, starts, inclusive=False)


register_fn(k_particle_to_leaf_count, "fluidSimParticleToLeafCount")
register_fn(leaf_count_to_starts, "fluidSimParticleLeafCountToLeafArrayStart",
            strategy="host_orchestrated", notes="warp device scan")
register_fn(k_scatter_particles_to_leaf_bins, "fluidSimSortParticlesLeafArrayMutex",
            "fluidSimSortParticlesRandomMutex",
            notes="atomic scatter replaces mutex spin loops")


@wp.kernel
def k_leaf_sort_voxel_type_tracked(
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    leaf_fluid_count: wp.array(dtype=wp.int32),
):
    """fluidSimLeafSortVoxelTypeTracked: per-leaf fluid voxel census used to
    order/track active leaves."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if (voxel_type[tid] & VOXEL_FLUID) != 0:
        wp.atomic_add(leaf_fluid_count, tid // LEAF_VOLUME, 1)


register_fn(k_leaf_sort_voxel_type_tracked, "fluidSimLeafSortVoxelTypeTracked")


# ---------------------------------------------------------------------------
# Lifecycle (fluid_particles_lifecycle_transfer, fluid_misc)
# ---------------------------------------------------------------------------


@wp.kernel
def k_particle_age(
    age: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    dt: float,
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    age[tid] = age[tid] + dt


@wp.kernel
def k_particle_lifespan(
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    use_lifespan: wp.int32,
    alive: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    if use_lifespan != 0 and life[tid] > 0.0 and age[tid] >= life[tid]:
        alive[tid] = 0


@wp.kernel
def k_particle_in_sink_check(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    alive: wp.array(dtype=wp.int32),
):
    """fluidSimParticleInSinkCheck: kill particles inside sink voxels."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ijk = voxel_floor(vp)
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat >= 0 and (voxel_type[flat] & VOXEL_SINK) != 0:
        alive[tid] = 0


@wp.kernel
def k_particles_relax(
    positions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    particle_voxel: wp.array(dtype=wp.int32),
    bin_counts: wp.array(dtype=wp.int32),
    bin_starts: wp.array(dtype=wp.int32),
    bin_indices: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    separation: float,
    strength: float,
    relax_mode: wp.int32,  # RelaxMode: 0 LocalVoxel, 1 NeighborVoxels
    collision_sdf: wp.array(dtype=float),
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    positions_out: wp.array(dtype=wp.vec3),
):
    """fluidSimParticlesRelax: push-apart within voxel bins."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    p = positions[tid]
    voxel = particle_voxel[tid]
    if voxel < 0:
        positions_out[tid] = p
        return
    push = wp.vec3(0.0, 0.0, 0.0)
    ijk = voxel_coord_of_flat(leaf_coords, voxel)
    r = wp.where(relax_mode == 1, 1, 0)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                o = wp.vec3i(ijk[0] + dx, ijk[1] + dy, ijk[2] + dz)
                nflat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
                if nflat < 0:
                    continue
                start = bin_starts[nflat]
                num = bin_counts[nflat]
                limit = num
                if limit > DEFAULT_RELAX_BIN_SAMPLE_LIMIT:
                    limit = DEFAULT_RELAX_BIN_SAMPLE_LIMIT
                stride = 1
                if num > limit:
                    stride = (num + limit - 1) // limit
                for sample in range(limit):
                    j = bin_indices[start + sample * stride]
                    if j == tid:
                        continue
                    d = p - positions[j]
                    dist = wp.length(d)
                    if dist < separation and dist > 1.0e-9:
                        push += d * ((separation - dist) / (separation * dist))
    # Cap the aggregate push so dense clusters cannot eject particles.
    push_len = wp.length(push)
    max_push = 1.0
    if push_len > max_push:
        push = push * (max_push / push_len)
    p_new = p + push * (strength * separation)
    # If the relaxed position enters a collider, project it out along the SDF
    # gradient instead of reverting to p, which may already be inside.
    vp = world_to_voxel(p_new, origin, voxel_size)
    phi = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp, 1.0e9)
    if phi < 0.0:
        gx = sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp + wp.vec3(1.0, 0.0, 0.0), 1.0e9) - sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp - wp.vec3(1.0, 0.0, 0.0), 1.0e9)
        gy = sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp + wp.vec3(0.0, 1.0, 0.0), 1.0e9) - sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp - wp.vec3(0.0, 1.0, 0.0), 1.0e9)
        gz = sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp + wp.vec3(0.0, 0.0, 1.0), 1.0e9) - sample_scalar_trilinear(
            collision_sdf, leaf_table, voxels_dim, leaves_dim,
            vp - wp.vec3(0.0, 0.0, 1.0), 1.0e9)
        n = wp.vec3(gx, gy, gz)
        ln = wp.length(n)
        if ln > 1.0e-8:
            n = n * (1.0 / ln)
            p_new = p_new - n * phi * 1.01
    # keep inside the domain (same margin as advection)
    margin = 1.0e-4
    p_new = wp.vec3(
        wp.clamp(p_new[0], domain_min[0] + margin, domain_max[0] - margin),
        wp.clamp(p_new[1], domain_min[1] + margin, domain_max[1] - margin),
        wp.clamp(p_new[2], domain_min[2] + margin, domain_max[2] - margin),
    )
    positions_out[tid] = p_new


@wp.kernel
def k_expand_fluid_mask(
    voxel_type_in: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    num_leaves: wp.array(dtype=wp.int32),
    voxel_type_out: wp.array(dtype=wp.int32),
):
    """fluidSimExpandFluidMask: dilate the fluid bit by one voxel."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type_in[tid]
    voxel_type_out[tid] = vt
    if (vt & VOXEL_FLUID) != 0 or (vt & VOXEL_COLLISION) != 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    for a in range(3):
        for s in range(-1, 2, 2):
            o = wp.vec3i(ijk[0], ijk[1], ijk[2])
            if a == 0:
                o = wp.vec3i(ijk[0] + s, ijk[1], ijk[2])
            elif a == 1:
                o = wp.vec3i(ijk[0], ijk[1] + s, ijk[2])
            else:
                o = wp.vec3i(ijk[0], ijk[1], ijk[2] + s)
            flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, o)
            if flat >= 0 and (voxel_type_in[flat] & VOXEL_FLUID) != 0:
                voxel_type_out[tid] = vt | VOXEL_FLUID
                return


@wp.kernel
def k_clear_just_initialized_tracked(
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
):
    """fluidSimClearJustInitializedVoxelTypeTracked /
    fluidSimResetSourceShapeTrackedVoxelType: clear tracking bits."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    voxel_type[tid] = voxel_type[tid] & (~VOXEL_SOURCE_REGION_TRACKING)


@wp.kernel
def k_fluid_max_active_value(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    out_max: wp.array(dtype=float),
):
    """fluidSimComputeMaxActiveValue: max fluid-voxel velocity magnitude
    (drives CFL substep selection)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if (voxel_type[tid] & VOXEL_FLUID) == 0:
        return
    m = wp.sqrt(u[tid] * u[tid] + v[tid] * v[tid] + w[tid] * w[tid])
    wp.atomic_max(out_max, 0, m)


register_fn(k_particle_age, "fluidSimParticleAge")
register_fn(k_particle_lifespan, "fluidSimParticleLifespan")
register_fn(k_particle_in_sink_check, "fluidSimParticleInSinkCheck")
register_fn(k_particles_relax, "fluidSimParticlesRelax")
register_fn(k_expand_fluid_mask, "fluidSimExpandFluidMask")
register_fn(k_clear_just_initialized_tracked,
            "fluidSimClearJustInitializedVoxelTypeTracked",
            "fluidSimResetSourceShapeTrackedVoxelType")
register_fn(k_fluid_max_active_value, "fluidSimComputeMaxActiveValue")
