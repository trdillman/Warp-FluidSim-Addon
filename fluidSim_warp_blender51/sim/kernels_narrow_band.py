"""Narrow-band FLIP kernels (lane J, see SPEC-narrow-band-particles.md).

Every kernel here is launched only while ``narrow_band_enabled`` is on, so
importing this module changes nothing for a simulation with the mode off.

``phi`` is the persistent grid liquid level set (negative = liquid, metres,
clamped to ``[-limit, +limit]``).  ``cls`` is the per-voxel class: 0 air,
1 band (``-band < phi < 0``), 2 interior (``phi <= -band``).

No bpy imports allowed here.
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

from .constants import (
    LEAF_VOLUME,
    MAX_PARTICLES_PER_VOXEL,
    VOXEL_ACTIVE,
    VOXEL_COLLISION,
    VOXEL_FLUID,
    VOXEL_SINK,
)
from .fields import (
    leaf_coord_of_voxel,
    leaf_linear_index,
    sample_mac_component,
    sample_mac_velocity,
    sample_scalar_trilinear,
    voxel_center_world,
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_floor,
    voxel_in_domain,
    world_to_voxel,
)
from .kernels_fluid import trace_position

# Depth sentinel for the inward Eikonal sweep: far larger than any real depth.
EIKONAL_FAR = 1.0e4
# trace_position mode 2 is MidPoint, the RK2 member of TraceMode.
TRACE_MIDPOINT = 2


# Topology: carry phi across a rebuild and keep interior leaves alive


@wp.kernel
def k_nb_remap_leaf_field_default(
    old_data: wp.array(dtype=float),
    old_leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    default_value: float,
    new_data: wp.array(dtype=float),
):
    """``k_remap_leaf_field`` with an explicit value for brand-new leaves."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    leaf = tid // LEAF_VOLUME
    local = tid - leaf * LEAF_VOLUME
    lc = leaf_coords[leaf]
    dense = (lc[2] * leaves_dim[1] + lc[1]) * leaves_dim[0] + lc[0]
    old_slot = old_leaf_table[dense]
    if old_slot < 0:
        new_data[tid] = default_value
        return
    new_data[tid] = old_data[old_slot * LEAF_VOLUME + local]


@wp.kernel
def k_nb_mark_leaves_from_phi(
    phi: wp.array(dtype=float),
    old_leaf_coords: wp.array(dtype=wp.vec3i),
    old_num_leaves: wp.array(dtype=wp.int32),
    leaves_dim: wp.vec3i,
    leaf_flags: wp.array(dtype=wp.int32),
):
    """Flag every previous leaf that holds liquid, so the interior survives
    a rebuild that sees no interior particles."""
    tid = wp.tid()
    if tid >= old_num_leaves[0] * LEAF_VOLUME:
        return
    if phi[tid] >= 0.0:
        return
    lc = old_leaf_coords[tid // LEAF_VOLUME]
    leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1


# Advection (semi-Lagrangian, RK2 backtrace)


@wp.kernel
def k_nb_advect_scalar(
    phi: wp.array(dtype=float),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    limit: float,
    phi_out: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if collision_sdf[tid] < 0.0:
        phi_out[tid] = limit
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    x = voxel_center_world(ijk, origin, voxel_size)
    back = trace_position(
        x, wp.vec3(0.0, 0.0, 0.0), -dt, TRACE_MIDPOINT,
        u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim,
        origin, voxel_size)
    value = sample_scalar_trilinear(
        phi, leaf_table, voxels_dim, leaves_dim,
        world_to_voxel(back, origin, voxel_size), limit)
    phi_out[tid] = wp.clamp(value, -limit, limit)


@wp.kernel
def k_nb_advect_velocity(
    nb_u: wp.array(dtype=float), nb_v: wp.array(dtype=float),
    nb_w: wp.array(dtype=float),
    ones: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    out_u: wp.array(dtype=float), out_v: wp.array(dtype=float),
    out_w: wp.array(dtype=float),
):
    """Advect the persistent MAC velocity by itself (every face valid)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    base = wp.vec3(float(ijk[0]), float(ijk[1]), float(ijk[2]))
    for axis in range(3):
        off = wp.vec3(0.5, 0.5, 0.5)
        if axis == 0:
            off = wp.vec3(0.0, 0.5, 0.5)
        elif axis == 1:
            off = wp.vec3(0.5, 0.0, 0.5)
        else:
            off = wp.vec3(0.5, 0.5, 0.0)
        x = origin + (base + off) * voxel_size
        back = trace_position(
            x, wp.vec3(0.0, 0.0, 0.0), -dt, TRACE_MIDPOINT,
            nb_u, nb_v, nb_w, ones, ones, ones,
            leaf_table, voxels_dim, leaves_dim, origin, voxel_size)
        vp = world_to_voxel(back, origin, voxel_size)
        if axis == 0:
            out_u[tid] = sample_mac_component(
                nb_u, ones, leaf_table, voxels_dim, leaves_dim, vp, 0)
        elif axis == 1:
            out_v[tid] = sample_mac_component(
                nb_v, ones, leaf_table, voxels_dim, leaves_dim, vp, 1)
        else:
            out_w[tid] = sample_mac_component(
                nb_w, ones, leaf_table, voxels_dim, leaves_dim, vp, 2)


# Level-set maintenance: union with the particle splat, then inward Eikonal


@wp.kernel
def k_nb_union_band(
    phi_adv: wp.array(dtype=float),
    splat: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    pin_depth: float,
    limit: float,
    phi_out: wp.array(dtype=float),
):
    """Re-derive the near-surface level set from the particles.

    Deeper than ``pin_depth`` the advected value is kept (no particle
    witnesses it).  Elsewhere the splat decides: positive splat is air, and a
    negative splat is liquid of unknown depth (``-limit``) for the Eikonal
    sweep to refine.  A sink voxel is air at any depth: the sink kills the
    particles that would witness liquid there, and keeping the deep advected
    value left a drain at the bottom of a tank full forever (B34).
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if collision_sdf[tid] < 0.0 or (voxel_type[tid] & VOXEL_SINK) != 0:
        phi_out[tid] = limit
        return
    p = phi_adv[tid]
    s = splat[tid]
    if p < -pin_depth:
        phi_out[tid] = p
    elif s < 0.0:
        phi_out[tid] = -limit
    else:
        phi_out[tid] = wp.min(s, limit)


@wp.func
def eikonal_depth(a0_in: float, a1_in: float, a2_in: float, h: float) -> float:
    """Godunov upwind solve for the depth of a cell from its axis minima."""
    a0 = float(a0_in)
    a1 = float(a1_in)
    a2 = float(a2_in)
    t = float(0.0)
    if a0 > a1:
        t = a0
        a0 = a1
        a1 = t
    if a1 > a2:
        t = a1
        a1 = a2
        a2 = t
    if a0 > a1:
        t = a0
        a0 = a1
        a1 = t
    t = a0 + h
    if t > a1:
        diff = a1 - a0
        t = 0.5 * (a0 + a1 + wp.sqrt(wp.max(2.0 * h * h - diff * diff, 0.0)))
        if t > a2:
            s = a0 + a1 + a2
            q = a0 * a0 + a1 * a1 + a2 * a2
            t = (s + wp.sqrt(wp.max(s * s - 3.0 * (q - h * h), 0.0))) / 3.0
    return t


@wp.func
def _nb_neighbor_depth(
    phi: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    axis: int,
    sign: int,
    h: float,
) -> float:
    n = ijk
    n[axis] = n[axis] + sign
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n)
    if flat < 0:
        return EIKONAL_FAR
    if collision_sdf[flat] < 0.0:
        return EIKONAL_FAR
    return wp.max(-phi[flat], -h)


@wp.kernel
def k_nb_eikonal_inward(
    phi_in: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    voxel_size: float,
    limit: float,
    phi_out: wp.array(dtype=float),
):
    """One Jacobi sweep of the inward distance, liquid cells only.

    Air and solid cells are fixed; walls and unallocated cells are ignored (a
    wall is not a free surface), so a column resting on a floor is deep, not
    shallow.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p = phi_in[tid]
    if p >= 0.0:
        phi_out[tid] = p
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    ax = wp.min(
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 0, -1, voxel_size),
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 0, 1, voxel_size))
    ay = wp.min(
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 1, -1, voxel_size),
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 1, 1, voxel_size))
    az = wp.min(
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 2, -1, voxel_size),
        _nb_neighbor_depth(phi_in, collision_sdf, leaf_table, voxels_dim,
                           leaves_dim, ijk, 2, 1, voxel_size))
    d = wp.max(wp.min(-p, eikonal_depth(ax, ay, az, voxel_size)), 1.0e-3 * voxel_size)
    phi_out[tid] = -wp.min(d, limit)


@wp.kernel
def k_nb_set_fluid_from_phi(
    phi: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
):
    """Additive fluid mask for liquid the particles no longer witness.

    Same precedence as sourcing (collision > sink > fluid): a sink voxel is
    never made fluid.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if phi[tid] < 0.0 and (vt & (VOXEL_COLLISION | VOXEL_SINK)) == 0:
        voxel_type[tid] = vt | VOXEL_FLUID | VOXEL_ACTIVE


@wp.kernel
def k_nb_classify(
    phi: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    band: float,
    cls: wp.array(dtype=wp.int32),
    liquid_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p = phi[tid]
    c = wp.int32(0)
    if p < -band:
        c = 2
    elif p < 0.0:
        c = 1
    cls[tid] = c
    if c != 0:
        wp.atomic_add(liquid_count, 0, 1)


# Interior velocity


@wp.func
def _nb_face_wants_fill(
    phi: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    ijk: wp.vec3i,
    axis: int,
    here_liquid: bool,
) -> bool:
    n_ijk = ijk
    n_ijk[axis] = n_ijk[axis] - 1
    n = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, n_ijk)
    if n < 0:
        return False
    if (voxel_type[n] & VOXEL_COLLISION) != 0:
        return False
    return here_liquid or phi[n] < 0.0


@wp.kernel
def k_nb_fill_interior_velocity(
    phi: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    nb_u: wp.array(dtype=float), nb_v: wp.array(dtype=float),
    nb_w: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
):
    """Give liquid faces that no particle touched the persisted velocity.

    A face qualifies when it has no particle weight, both cells are free of
    colliders, the neighbour exists, and either cell is liquid.  Particle
    faces (the band) are left alone.  The weight becomes 1 so gravity, the
    sampler and the pressure projection treat the face as valid.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if (voxel_type[tid] & VOXEL_COLLISION) != 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    here_liquid = phi[tid] < 0.0
    for axis in range(3):
        if _nb_face_wants_fill(phi, voxel_type, leaf_table, voxels_dim, leaves_dim, ijk, axis, here_liquid):
            if axis == 0:
                if wu[tid] <= 1.0e-8:
                    u[tid] = nb_u[tid]
                    wu[tid] = 1.0
            elif axis == 1:
                if wv[tid] <= 1.0e-8:
                    v[tid] = nb_v[tid]
                    wv[tid] = 1.0
            else:
                if ww[tid] <= 1.0e-8:
                    w[tid] = nb_w[tid]
                    ww[tid] = 1.0


# Kill and reseed


@wp.kernel
def k_nb_kill_interior(
    position: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    kill_depth: float,
    alive: wp.array(dtype=wp.int32),
    killed: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    ijk = voxel_floor(world_to_voxel(position[tid], origin, voxel_size))
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat >= 0 and phi[flat] < -kill_depth and alive[tid] != 0:
        alive[tid] = 0
        wp.atomic_add(killed, 0, 1)


@wp.kernel
def k_nb_reseed_demand(
    cls: wp.array(dtype=wp.int32),
    phi: wp.array(dtype=float),
    collision_sdf: wp.array(dtype=float),
    voxel_counts: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
    min_depth: float,
    trigger: int,
    fill: int,
    demand: wp.array(dtype=wp.int32),
):
    """Particles a band cell is short of.

    Only cells at least ``min_depth`` below the surface are topped up: the
    surface layer itself is owned by the particles that define it, so
    reseeding there would inflate the volume.

    Only a cell with fewer than ``trigger`` particles (an empty cell, as when
    the band moves into ground the interior kill had cleared) is filled, and
    only to ``fill``.  Topping every sparse cell up to the full density is a
    ratchet: nothing ever removes the surplus where the flow converges, so a
    spreading liquid whose full-band particles dilute would be re-densified
    until the count passed the full band's (dam_break: 50k, 237k, then the
    600k cap against a constant 256k).
    """
    tid = wp.tid()
    demand[tid] = 0
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if cls[tid] != 1 or phi[tid] > -min_depth or collision_sdf[tid] < 0.0:
        return
    have = voxel_counts[tid]
    if have < trigger:
        demand[tid] = wp.max(fill - have, 0)


@wp.kernel
def k_nb_reseed_reserve(
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    n: int,
    capacity: int,
    owed: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
    nb_overflow: wp.array(dtype=wp.int32),
):
    """Reserve the reseed block.

    A request clipped by the buffer raises the overflow flag. ``owed`` counts
    the particles the band killed and has not yet replaced: reseeding moves
    liquid mass from the interior to the surface, it never creates any, so it
    stops there without raising the flag.
    """
    base = count[0]
    total = offsets[n - 1] + demand[n - 1]
    room = wp.max(0, capacity - base)
    admitted = wp.min(wp.min(total, room), wp.max(0, owed[0]))
    slot_base[0] = base
    slot_limit[0] = admitted
    if total > room:
        nb_overflow[0] = 1
    count[0] = base + admitted
    owed[0] = owed[0] - admitted


@wp.kernel
def k_nb_reseed_commit(
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    seed: int,
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
):
    """Emit the demanded particles; slots follow voxel order, not arrival."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    emit_n = demand[tid]
    if emit_n <= 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, voxel_size)
    local = offsets[tid]
    for k in range(emit_n):
        if local + k >= slot_limit[0]:
            return
        slot = slot_base[0] + local + k
        state = wp.rand_init(seed, tid * MAX_PARTICLES_PER_VOXEL + k)
        p = center + wp.vec3(
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
        )
        position[slot] = p
        velocity[slot] = sample_mac_velocity(
            u, v, w, wu, wv, ww, leaf_table, voxels_dim, leaves_dim,
            world_to_voxel(p, origin, voxel_size))
        age[slot] = 0.0
        life[slot] = 1.0e30
        temperature[slot] = 0.0
        viscosity[slot] = 0.0
        density[slot] = 1.0
        time_offset[slot] = 0.0
        alive[slot] = 1


# Union SDF for every consumer of the liquid surface


@wp.kernel
def k_nb_union_into_solver_sdf(
    sdf: wp.array(dtype=float),
    phi: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    depth: float,
):
    """``sdf = min(splat, phi)`` where phi is a full cell below the surface."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    p = phi[tid]
    if p < -depth:
        sdf[tid] = wp.min(sdf[tid], p)


@wp.kernel
def k_nb_union_into_render_sdf(
    render_sdf: wp.array(dtype=float),
    render_leaf_coords: wp.array(dtype=wp.vec3i),
    render_num_leaves: wp.array(dtype=wp.int32),
    render_origin: wp.vec3,
    render_voxel_size: float,
    phi: wp.array(dtype=float),
    sim_leaf_table: wp.array(dtype=wp.int32),
    sim_voxels_dim: wp.vec3i,
    sim_leaves_dim: wp.vec3i,
    sim_origin: wp.vec3,
    sim_voxel_size: float,
    depth: float,
):
    """The same union on a render grid that differs from the sim grid."""
    tid = wp.tid()
    if tid >= render_num_leaves[0] * LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(render_leaf_coords, tid)
    x = voxel_center_world(ijk, render_origin, render_voxel_size)
    value = sample_scalar_trilinear(
        phi, sim_leaf_table, sim_voxels_dim, sim_leaves_dim,
        world_to_voxel(x, sim_origin, sim_voxel_size), 1.0e30)
    if value < -depth:
        render_sdf[tid] = wp.min(render_sdf[tid], value)


@wp.kernel
def k_nb_mark_render_leaves(
    phi: wp.array(dtype=float),
    sim_leaf_coords: wp.array(dtype=wp.vec3i),
    sim_num_leaves: wp.array(dtype=wp.int32),
    sim_origin: wp.vec3,
    sim_voxel_size: float,
    depth: float,
    render_origin: wp.vec3,
    render_voxel_size: float,
    render_voxels_dim: wp.vec3i,
    render_leaves_dim: wp.vec3i,
    render_leaf_flags: wp.array(dtype=wp.int32),
):
    """Allocate render leaves under interior liquid (no particles mark them)."""
    tid = wp.tid()
    if tid >= sim_num_leaves[0] * LEAF_VOLUME:
        return
    if phi[tid] >= -depth:
        return
    ijk = voxel_coord_of_flat(sim_leaf_coords, tid)
    centre = voxel_center_world(ijk, sim_origin, sim_voxel_size)
    reach = 0.499 * sim_voxel_size
    for corner in range(8):
        sx = wp.where((corner & 1) != 0, reach, -reach)
        sy = wp.where((corner & 2) != 0, reach, -reach)
        sz = wp.where((corner & 4) != 0, reach, -reach)
        r = voxel_floor(world_to_voxel(
            centre + wp.vec3(sx, sy, sz), render_origin, render_voxel_size))
        if voxel_in_domain(r, render_voxels_dim):
            render_leaf_flags[leaf_linear_index(
                leaf_coord_of_voxel(r), render_leaves_dim)] = 1


# Source top-up guard


@wp.kernel
def k_nb_adjust_interior_counts(
    phi: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    depth: float,
    delta: int,
    counts: wp.array(dtype=wp.int32),
):
    """Add ``delta`` to the particle count of every interior voxel.

    Called with ``+CREDIT`` before sourcing and ``-CREDIT`` after, so the
    top-up passes (which emit up to ``emit_rate - have``) see the particle-free
    interior as already full and do not refill it.  The count is restored
    exactly before P2G reads it.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    if phi[tid] < -depth:
        counts[tid] = counts[tid] + delta


@wp.kernel
def k_nb_store_velocity(
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    out_u: wp.array(dtype=float), out_v: wp.array(dtype=float),
    out_w: wp.array(dtype=float),
):
    """Persist the projected MAC velocity (the grid is cleared every substep)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    out_u[tid] = u[tid]
    out_v[tid] = v[tid]
    out_w[tid] = w[tid]
