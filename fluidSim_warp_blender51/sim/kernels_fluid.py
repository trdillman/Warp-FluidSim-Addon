"""FluidSim liquid kernels: PIC/FLIP transfer, advection, forces, particle
indexing/sorting/relaxation and lifecycle (``fluidSim*`` family).

No bpy imports allowed here.
"""

from __future__ import annotations

import warp as wp
from .fastlaunch import launch as fast_launch

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

# Particle -> grid (P2G)


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
def k_gather_grid_data(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
    bin_starts: wp.array(dtype=wp.int32),
    bin_indices: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
):
    """Deterministic P2G: the transpose of :func:`k_particle_set_grid_data`.

    One thread owns one active voxel and pulls every contribution addressed to
    it, instead of every particle pushing into shared accumulators with
    ``wp.atomic_add``.  Float addition is not associative, so the scatter's sum
    depends on thread arrival order and no two runs of the same bake agree
    bitwise; the gather sums in a fixed order and is reproducible.

    The neighbourhood is exact rather than conservative.  MAC face offsets have
    components in {0, 0.5}, so ``base = floor(vp - off)`` is either ``c`` or
    ``c - 1`` per component for a particle binned at ``c = floor(vp)``, and the
    scatter's targets are ``base + d`` for ``d`` in {0, 1}.  Every target
    therefore lies within one voxel of ``c``, and sweeping the 3x3x3 bin
    neighbourhood visits every particle that can reach this voxel -- no more,
    no fewer.  Each term is recomputed with the identical expressions the
    scatter uses, so only the summation order differs.

    One semantic difference: a particle whose own voxel is outside the domain
    or unallocated is absent from the bins and so contributes nothing here,
    where the scatter would still deposit into any allocated neighbour.  The
    same particles are already excluded from ``voxel_counts`` and from
    relaxation, and topology is rebuilt from the live particle set before every
    transfer, so the set is empty in practice.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    su = float(0.0)
    sv = float(0.0)
    sw = float(0.0)
    swu = float(0.0)
    swv = float(0.0)
    sww = float(0.0)
    for nz in range(-1, 2):
        for ny in range(-1, 2):
            for nx in range(-1, 2):
                nflat = voxel_flat_index(
                    leaf_table, voxels_dim, leaves_dim,
                    wp.vec3i(ijk[0] + nx, ijk[1] + ny, ijk[2] + nz),
                )
                if nflat >= 0:
                    start = bin_starts[nflat]
                    for s in range(voxel_counts[nflat]):
                        p = bin_indices[start + s]
                        vp = world_to_voxel(positions[p], origin, voxel_size)
                        vel = velocities[p]
                        # The three MAC faces are unrolled by hand so each
                        # one carries its own constant offset; a loop over
                        # the axis index would have to select the offset
                        # vector and the velocity component at run time.
                        q = vp - wp.vec3(0.0, 0.5, 0.5)
                        base = voxel_floor(q)
                        dx = ijk[0] - base[0]
                        dy = ijk[1] - base[1]
                        dz = ijk[2] - base[2]
                        ok = dx >= 0 and dx <= 1 and dy >= 0 and dy <= 1
                        if ok and dz >= 0 and dz <= 1:
                            f = q - wp.vec3(
                                float(base[0]), float(base[1]), float(base[2]))
                            wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                            wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                            wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                            wgt = wx * wy * wz
                            su += wgt * vel[0]
                            swu += wgt
                        q = vp - wp.vec3(0.5, 0.0, 0.5)
                        base = voxel_floor(q)
                        dx = ijk[0] - base[0]
                        dy = ijk[1] - base[1]
                        dz = ijk[2] - base[2]
                        ok = dx >= 0 and dx <= 1 and dy >= 0 and dy <= 1
                        if ok and dz >= 0 and dz <= 1:
                            f = q - wp.vec3(
                                float(base[0]), float(base[1]), float(base[2]))
                            wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                            wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                            wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                            wgt = wx * wy * wz
                            sv += wgt * vel[1]
                            swv += wgt
                        q = vp - wp.vec3(0.5, 0.5, 0.0)
                        base = voxel_floor(q)
                        dx = ijk[0] - base[0]
                        dy = ijk[1] - base[1]
                        dz = ijk[2] - base[2]
                        ok = dx >= 0 and dx <= 1 and dy >= 0 and dy <= 1
                        if ok and dz >= 0 and dz <= 1:
                            f = q - wp.vec3(
                                float(base[0]), float(base[1]), float(base[2]))
                            wx = wp.where(dx == 1, f[0], 1.0 - f[0])
                            wy = wp.where(dy == 1, f[1], 1.0 - f[1])
                            wz = wp.where(dz == 1, f[2], 1.0 - f[2])
                            wgt = wx * wy * wz
                            sw += wgt * vel[2]
                            sww += wgt
    u[tid] = su
    v[tid] = sv
    w[tid] = sw
    wu[tid] = swu
    wv[tid] = swv
    ww[tid] = sww


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
    """Mark occupied voxels fluid/active without clearing existing bits.

    Additive like the original ``paradigmParticleToGridActive``. Grid state is
    reset once per substep by ``FluidSolver._clear_grid_fields``; this kernel
    must not erase bits established by another phase.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if voxel_counts[tid] > 0 and (vt & VOXEL_COLLISION) == 0:
        voxel_type[tid] = vt | VOXEL_FLUID | VOXEL_ACTIVE


register_fn(k_particle_set_grid_data, "fluidSimParticleSetGridData",
            notes="faithful scatter reference; the solver runs the bitwise-equal "
                  "gather in fluidSimParticleSetGridData_gather")
register_fn(k_gather_grid_data, "fluidSimParticleSetGridData_gather",
            notes="voxel-per-thread transpose of the scatter over the sorted "
                  "bins: same terms, fixed summation order, no float atomics")
register_fn(k_normalize_mac, "fluidSimWhiteWaterSetGridDataNormalize_internal")
register_fn(k_particle_to_count, "fluidSimParticleToCount")
register_fn(k_particle_to_grid_active, "fluidSimParticleToGridActive")


# Grid -> particle (G2P): PIC/FLIP blend


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
    friction_time_ratio: float,
    impact_u: wp.array(dtype=float),
    impact_v: wp.array(dtype=float),
    impact_w: wp.array(dtype=float),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Apply collider bounce and friction to faces beside solid voxels.

    Restitution acts on the normal face between a fluid voxel and a solid
    neighbour. Friction acts on the tangential faces of a non-solid voxel
    with a solid neighbour along another axis: it blends the face velocity
    toward the collider velocity (0 keeps free slip, 1 gives no-slip).
    friction_time_ratio is dt / COLLIDER_FRICTION_REFERENCE_TIME, so the
    slip kept per substep is (1 - friction) ** friction_time_ratio.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & VOXEL_COLLISION) != 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    lo_idx = wp.vec3i(-1, -1, -1)
    hi_idx = wp.vec3i(-1, -1, -1)
    lo_solid = wp.vec3i(0, 0, 0)
    hi_solid = wp.vec3i(0, 0, 0)
    for axis in range(3):
        offset = wp.vec3i(0, 0, 0)
        if axis == 0:
            offset = wp.vec3i(1, 0, 0)
        elif axis == 1:
            offset = wp.vec3i(0, 1, 0)
        else:
            offset = wp.vec3i(0, 0, 1)
        lo = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk - offset)
        hi = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk + offset)
        lo_idx[axis] = lo
        hi_idx[axis] = hi
        if lo >= 0:
            if collision_sdf[lo] < 0.0:
                lo_solid[axis] = 1
        if hi >= 0:
            if collision_sdf[hi] < 0.0:
                hi_solid[axis] = 1

    if (vt & VOXEL_FLUID) != 0:
        for axis in range(3):
            if lo_solid[axis] == hi_solid[axis]:
                continue

            # MAC components live on lower-axis faces. A solid lower neighbor
            # contacts this voxel's face; a solid upper neighbor contacts the
            # upper voxel's lower face. Only modify that normal face component.
            face = tid
            normal_sign = float(1.0)
            if hi_solid[axis] != 0:
                face = hi_idx[axis]
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

    # Friction on this voxel's own lower faces. Skip normal contact faces
    # (solid lower neighbour; the projection pins those) and faces on the
    # grid edge. A face is tangential to a wall when this voxel has a solid
    # neighbour along another axis; the most frictional such neighbour sets
    # the blend. Friction 0 writes nothing, so free-slip stays bit-exact.
    for b in range(3):
        if lo_idx[b] < 0 or lo_solid[b] != 0:
            continue
        friction = float(0.0)
        vcol = float(0.0)
        for a in range(3):
            if a == b:
                continue
            if lo_solid[a] != 0:
                f_lo = collision_friction[lo_idx[a]]
                if f_lo > friction:
                    friction = f_lo
                    vcol = collision_vel[lo_idx[a]][b]
            if hi_solid[a] != 0:
                f_hi = collision_friction[hi_idx[a]]
                if f_hi > friction:
                    friction = f_hi
                    vcol = collision_vel[hi_idx[a]][b]
        if friction > 0.0:
            keep = wp.pow(1.0 - wp.min(friction, 1.0), friction_time_ratio)
            if b == 0:
                u[tid] = vcol + (u[tid] - vcol) * keep
            elif b == 1:
                v[tid] = vcol + (v[tid] - vcol) * keep
            else:
                w[tid] = vcol + (w[tid] - vcol) * keep

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


# Advection (fluid_advection)


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
    """Collider SDF change across one voxel, per axis.

    Central differences at +/-0.5 voxel. A sample with no allocated voxel
    under it returns the 1e9 fallback; using it would swamp that component
    and point the normal at a fixed diagonal (REVIEW-017). Such a sample is
    replaced by a one-sided difference against the centre over the same
    one-voxel scale, and an axis with no valid sample on either side
    contributes nothing.
    """
    eps = 0.5
    g = wp.vec3(0.0, 0.0, 0.0)
    for axis in range(3):
        offset = wp.vec3(0.0, 0.0, 0.0)
        offset[axis] = eps
        hi = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp + offset, 1.0e9)
        lo = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp - offset, 1.0e9)
        hi_ok = hi < 5.0e8
        lo_ok = lo < 5.0e8
        if hi_ok and lo_ok:
            g[axis] = hi - lo
        elif hi_ok or lo_ok:
            centre = sample_scalar_trilinear(collision_sdf, leaf_table, voxels_dim, leaves_dim, vp, 1.0e9)
            if centre < 5.0e8:
                if hi_ok:
                    g[axis] = 2.0 * (hi - centre)
                else:
                    g[axis] = 2.0 * (centre - lo)
    return g


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
            # world-space gradient magnitude. Do not assume a perfect SDF,
            # but near a medial axis / thin feature (e.g. a paddle blade a
            # couple of voxels thick) that magnitude can collapse toward
            # zero even though phi itself is a modest, valid penetration
            # depth. voxel_size / ln then blows up and teleports the
            # particle. Clamp the correction to a bounded multiple of the
            # raw penetration depth: exact for a proper unit-gradient SDF
            # (factor 1), still corrects real gradient-magnitude error, but
            # can't diverge from a degenerate estimate.
            grad_correction = wp.clamp(voxel_size / ln, 0.25, 4.0)
            p_out = p - n * phi * grad_correction * 1.001
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


register_fn(k_advect_particles, "fluidSimAdvectParticles")


# Forces (fluid_forces_attraction)


@wp.struct
class ForceFieldProxy:
    force_type: int
    shape: int
    texture_mode: int
    flags: int
    seed: int
    center: wp.vec3
    direction: wp.vec3
    source_velocity: wp.vec3
    guide_p0: wp.vec3
    guide_p1: wp.vec3
    guide_p2: wp.vec3
    guide_p3: wp.vec3
    guide_count: int
    strength: float
    min_distance: float
    max_distance: float
    falloff_power: float
    flow: float
    noise: float
    size: float
    rest_length: float
    damping: float
    linear_drag: float
    quadratic_drag: float
    guide_minimum: float
    guide_free: float
    guide_clump: float


@wp.func
def force_proxy_noise(point: wp.vec3, size: float, seed: int, time: float):
    """Small deterministic vector field used by turbulence/texture proxies."""
    scale = 1.0 / wp.max(wp.abs(size), 1.0e-3)
    phase = float(seed) * 0.173 + time * 0.37
    p = point * scale + wp.vec3(phase, phase * 1.37, phase * 1.91)
    return wp.vec3(
        wp.sin(p[1] * 1.7 + p[2] * 0.6) * wp.cos(p[0] * 0.7),
        wp.sin(p[2] * 1.5 + p[0] * 0.5) * wp.cos(p[1] * 0.9),
        wp.sin(p[0] * 1.3 + p[1] * 0.8) * wp.cos(p[2] * 1.1),
    )


@wp.func
def force_proxy_attenuation(
    distance: float, min_distance: float, max_distance: float, power: float,
):
    if max_distance > 0.0 and distance > max_distance:
        return 0.0
    if power <= 0.0 or distance <= min_distance:
        return 1.0
    return wp.pow(1.0 + distance - min_distance, -power)


@wp.func
def force_field_proxy_acceleration(
    field: ForceFieldProxy, point: wp.vec3, velocity: wp.vec3, time: float,
):
    rel_center = point - field.center
    rel = rel_center
    axial = wp.dot(rel_center, field.direction)
    if ((field.force_type == 3 or field.force_type == 4)
            and field.shape != 0):
        rel = rel_center - field.direction * axial
    elif field.shape == 1:  # Line
        rel = rel_center - field.direction * axial
    elif field.shape == 2:  # Plane
        rel = field.direction * axial
    distance = wp.length(rel)
    attenuation = force_proxy_attenuation(
        distance, field.min_distance, field.max_distance,
        field.falloff_power)
    if attenuation <= 0.0:
        return wp.vec3(0.0)
    magnitude = field.strength * attenuation
    if field.noise > 0.0:
        magnitude += field.noise * attenuation * force_proxy_noise(
            point, wp.max(field.size, 1.0), field.seed, time)[0]

    if field.force_type == 1:  # Blender Wind
        force = field.direction * magnitude
    elif field.force_type == 2 or field.force_type == 6:  # Force / Charge
        force = wp.vec3(0.0)
        if distance > 1.0e-8:
            force = rel * (magnitude / distance)
    elif field.force_type == 3:  # Vortex
        tangent = wp.cross(field.direction, rel)
        tangent_length = wp.length(tangent)
        force = wp.vec3(0.0)
        if tangent_length > 1.0e-8:
            force = tangent * (magnitude / tangent_length)
    elif field.force_type == 4:  # Magnetic: q(v x B), unit fluid charge
        magnetic = field.direction
        if field.shape == 0 or field.shape == 1:
            magnetic = wp.cross(field.direction, rel)
        magnetic_length = wp.length(magnetic)
        force = wp.vec3(0.0)
        if magnetic_length > 1.0e-8:
            force = wp.cross(
                velocity, magnetic * (magnitude / magnetic_length))
    elif field.force_type == 5:  # Harmonic spring
        force = -velocity * (
            field.damping * 2.0 * wp.sqrt(wp.abs(field.strength)) * attenuation)
        if distance > 1.0e-8:
            force += rel * (
                -field.strength * (distance - field.rest_length)
                * attenuation / distance)
    elif field.force_type == 7:  # Lennard-Jones fluid-scale analogue
        force = wp.vec3(0.0)
        if distance > 1.0e-6:
            ratio = field.size / distance
            ratio2 = ratio * ratio
            ratio6 = ratio2 * ratio2 * ratio2
            factor = wp.clamp(
                -ratio6 * (1.0 - ratio6) / distance, -2.0, 2.0)
            force = rel * (field.strength * factor * attenuation)
    elif field.force_type == 8:  # Texture: procedural GPU vector/grad/curl proxy
        coordinate = point
        if (field.flags & 1) == 0:
            coordinate = point - field.center
        p = coordinate / wp.max(wp.abs(field.size), 1.0e-3)
        if field.texture_mode == 1:  # Gradient
            phase = float(field.seed) * 0.173
            force = wp.vec3(
                wp.cos(p[0] + phase) * wp.cos(p[1]) * wp.sin(p[2]),
                -wp.sin(p[0] + phase) * wp.sin(p[1]) * wp.sin(p[2]),
                wp.sin(p[0] + phase) * wp.cos(p[1]) * wp.cos(p[2]),
            ) * magnitude
        elif field.texture_mode == 2:  # Curl-like divergence-free field
            force = wp.vec3(
                wp.cos(p[1]) - wp.sin(p[2] + float(field.seed)),
                wp.cos(p[2]) - wp.sin(p[0] + float(field.seed)),
                wp.cos(p[0]) - wp.sin(p[1] + float(field.seed)),
            ) * magnitude
        else:
            force = force_proxy_noise(p, 1.0, field.seed, 0.0) * magnitude
    elif field.force_type == 9:  # Curve Guide, four-point polyline proxy
        closest = field.guide_p0
        tangent = field.direction
        best_distance_sq = wp.dot(point - closest, point - closest)

        ab = field.guide_p1 - field.guide_p0
        denom = wp.dot(ab, ab)
        if field.guide_count > 1 and denom > 1.0e-10:
            t = wp.clamp(wp.dot(point - field.guide_p0, ab) / denom, 0.0, 1.0)
            candidate = field.guide_p0 + ab * t
            candidate_distance_sq = wp.dot(point - candidate, point - candidate)
            if candidate_distance_sq < best_distance_sq:
                closest = candidate
                tangent = wp.normalize(ab)
                best_distance_sq = candidate_distance_sq

        ab = field.guide_p2 - field.guide_p1
        denom = wp.dot(ab, ab)
        if field.guide_count > 2 and denom > 1.0e-10:
            t = wp.clamp(wp.dot(point - field.guide_p1, ab) / denom, 0.0, 1.0)
            candidate = field.guide_p1 + ab * t
            candidate_distance_sq = wp.dot(point - candidate, point - candidate)
            if candidate_distance_sq < best_distance_sq:
                closest = candidate
                tangent = wp.normalize(ab)
                best_distance_sq = candidate_distance_sq

        ab = field.guide_p3 - field.guide_p2
        denom = wp.dot(ab, ab)
        if field.guide_count > 3 and denom > 1.0e-10:
            t = wp.clamp(wp.dot(point - field.guide_p2, ab) / denom, 0.0, 1.0)
            candidate = field.guide_p2 + ab * t
            candidate_distance_sq = wp.dot(point - candidate, point - candidate)
            if candidate_distance_sq < best_distance_sq:
                closest = candidate
                tangent = wp.normalize(ab)

        guide_speed = wp.max(wp.abs(field.strength), field.guide_minimum)
        if field.strength < 0.0:
            guide_speed = -guide_speed
        coupling = wp.clamp(1.0 - field.guide_free, 0.0, 1.0)
        force = ((tangent * guide_speed - velocity) * coupling
                 + (closest - point) * wp.abs(field.strength)
                 * (1.0 + field.guide_clump)) * attenuation
    elif field.force_type == 10:  # Boid goal proxy for state-less fluid
        force = wp.vec3(0.0)
        if distance > 1.0e-8:
            force = -rel * (magnitude / distance)
    elif field.force_type == 11:  # Turbulence
        coordinate = point
        if (field.flags & 1) == 0:
            coordinate = point - field.center
        force = force_proxy_noise(
            coordinate, wp.max(field.size, 1.0e-3), field.seed, 0.0,
        ) * magnitude
    elif field.force_type == 12:  # Linear + quadratic drag
        speed = wp.length(velocity)
        force = -velocity * attenuation * (
            field.linear_drag + field.quadratic_drag * speed)
    elif field.force_type == 13:  # Fluid-flow target velocity proxy
        force = (field.source_velocity * field.strength
                 - velocity * field.flow) * attenuation
    else:
        force = wp.vec3(0.0)

    if (field.force_type != 5 and field.force_type != 12
            and field.force_type != 13 and field.flow != 0.0):
        force -= velocity * (field.flow * attenuation)
    return force


@wp.kernel
def k_forces_proxy_fields(
    fields: wp.array(dtype=ForceFieldProxy),
    num_fields: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    time: float,
    dt: float,
):
    """Apply native Blender force-field proxies to valid staggered faces."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    velocity = wp.vec3(u[tid], v[tid], w[tid])
    for axis in range(3):
        point = origin + wp.vec3(
            (float(ijk[0]) + wp.where(axis == 0, 0.0, 0.5)) * voxel_size,
            (float(ijk[1]) + wp.where(axis == 1, 0.0, 0.5)) * voxel_size,
            (float(ijk[2]) + wp.where(axis == 2, 0.0, 0.5)) * voxel_size,
        )
        acceleration = wp.vec3(0.0)
        for field_index in range(num_fields):
            acceleration += force_field_proxy_acceleration(
                fields[field_index], point, velocity, time)
        if axis == 0 and wu[tid] > 0.0:
            u[tid] = u[tid] + acceleration[0] * dt
        elif axis == 1 and wv[tid] > 0.0:
            v[tid] = v[tid] + acceleration[1] * dt
        elif axis == 2 and ww[tid] > 0.0:
            w[tid] = w[tid] + acceleration[2] * dt


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
    u_out: wp.array(dtype=float), v_out: wp.array(dtype=float), w_out: wp.array(dtype=float),
):
    """fluidSimForcesSpeedLimit with LimitBehavior semantics.

    Reads +x/+y/+z neighbor faces to estimate cell-centered speed, so it must
    write scaled results to separate output buffers rather than in place:
    scaling ``u[tid]`` while a neighboring thread still needs to read the
    original ``u[tid]`` as its own ``u[xp]`` is a data race whose outcome
    depends on launch scheduling (every other velocity-mutating kernel in
    this module double-buffers for the same reason, e.g. ``k_forces_diffusion``).
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    u_out[tid] = u[tid]
    v_out[tid] = v[tid]
    w_out[tid] = w[tid]
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
        u_out[tid] = u[tid] * scale
    if wv[tid] > 0.0:
        v_out[tid] = v[tid] * scale
    if ww[tid] > 0.0:
        w_out[tid] = w[tid] * scale


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
register_fn(k_forces_proxy_fields, "fluidSimForcesBlenderFieldProxies")
register_fn(k_forces_directional, "fluidSimForcesDirectionalForce")
register_fn(k_forces_omnidirectional, "fluidSimForcesOmnidirectionalForce")
register_fn(k_forces_speed_limit, "fluidSimForcesSpeedLimit")
register_fn(k_forces_diffusion, "fluidSimForcesDiffusion")
register_fn(k_dissipate_particle_attribute, "fluidSimForcesDissipateParticleAttribute")


# Attraction (mask fields steering fluid)


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


# Particle indexing / sorting (fluid_particle_indexing_sorting)


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
    particle_arrival: wp.array(dtype=wp.int32),
):
    """fluidSimParticleToVoxelCountAndUpdateVoxelIndex (+UpdateParticleVoxelIndex).

    ``particle_arrival`` receives the particle's arrival slot within its voxel
    (the counter value before its own increment).  It is a unique slot in
    ``[0, voxel_counts[voxel])`` but depends on thread arrival order, so it is
    only used as a scratch position that ``k_rank_bin_particles`` reorders.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    ijk = voxel_floor(vp)
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    particle_voxel[tid] = flat
    if flat >= 0:
        particle_arrival[tid] = wp.atomic_add(voxel_counts, flat, 1)


@wp.kernel
def k_scatter_particles_to_arrival_slots(
    particle_voxel: wp.array(dtype=wp.int32),
    particle_arrival: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    unordered: wp.array(dtype=wp.int32),
):
    """Place each particle at its arrival slot: bins complete, order arbitrary."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    voxel = particle_voxel[tid]
    if voxel < 0:
        return
    unordered[starts[voxel] + particle_arrival[tid]] = tid


@wp.kernel
def k_rank_bin_particles(
    particle_voxel: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    voxel_counts: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    unordered: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
):
    """Move each particle to its rank by particle index within its bin.

    The rank is the number of bin members with a smaller particle index, which
    depends only on bin membership, never on the arrival order the unordered
    pass recorded.  Cost is one read per bin member per particle; bins hold a
    handful of particles (particles-per-voxel), so this is a short loop.
    """
    tid = wp.tid()
    if tid >= count[0]:
        return
    voxel = particle_voxel[tid]
    if voxel < 0:
        return
    start = starts[voxel]
    n = voxel_counts[voxel]
    rank = int(0)
    for k in range(n):
        if unordered[start + k] < tid:
            rank += 1
    indices[start + rank] = tid


def scatter_particles_to_bins(particle_voxel, particle_arrival, count, np_dim,
                              voxel_counts, starts, unordered, indices):
    """fluidSimParticleToVoxelParticleArray + fluidSimVoxelParticleArraySort.

    Two launches: scatter every particle to the arrival slot the count kernel
    reserved, then rank each bin by particle index.  The result is ascending
    particle index within each bin -- identical to a stable sort by voxel, the
    in-bin ordering the original performed as a separate sort pass, and the
    property the P2G gather and the relaxation sub-sampling need in order to
    be reproducible.  A stable radix sort gives the same array but costs about
    twice the host launch overhead per call, which dominates at bake sizes.
    """
    device = indices.device
    fast_launch(
        k_scatter_particles_to_arrival_slots, dim=np_dim,
        inputs=[particle_voxel, particle_arrival, count, starts],
        outputs=[unordered], device=device,
    )
    fast_launch(
        k_rank_bin_particles, dim=np_dim,
        inputs=[particle_voxel, count, voxel_counts, starts, unordered],
        outputs=[indices], device=device,
    )


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
register_fn(scatter_particles_to_bins, "fluidSimParticleToVoxelParticleArray",
            "fluidSimVoxelParticleArraySort",
            strategy="host_orchestrated",
            notes="arrival scatter + rank: ascending particle index within each bin")


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


# Lifecycle (fluid_particles_lifecycle_transfer, fluid_misc)


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
                for sample in range(limit):
                    j = bin_indices[start + (sample * num) // limit]
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
