"""Deterministic fluid -> rigid-body force gathers (4.0 two-way coupling).

docs/specs/4.0/SPEC-rigid-coupling.md, trap 2: no ``wp.atomic_add`` on the
bake path. Every per-body sum here is a two-level segmented gather in a fixed
order: a chunk kernel writes one partial per (body, chunk) with a sequential
loop over its slice, and ``k_rigid_reduce`` adds the partials of each body in
chunk order on one thread. The result is bitwise repeatable for a fixed launch.

Sums are carried in float64: the net force on a floating body is the small
difference of large opposing face pressures, which float32 sums over
thousands of faces would round away.

No bpy imports allowed here.
"""

from __future__ import annotations

import warp as wp

from .constants import (
    LEAF_DIM,
    LEAF_MASK,
    LEAF_SHIFT,
    LEAF_VOLUME,
    VOXEL_COLLISION,
    VOXEL_FLUID,
)
from .fields import (
    sample_mac_velocity,
    voxel_flat_index,
    world_to_voxel,
)
from .pressure import neighbor_flat

wp.set_module_options({"enable_backward": False})


# Packed per-body row (``body_state``), device resident: uploaded when the
# bodies are attached, then advanced in place by ``k_rigid_integrate``.
# rank, centre of mass (3), rotation (9, row major), linear velocity (3),
# angular velocity (3), drag coefficient ``0.5 Cd rho``, the voxel box the
# face gather walks: lowest voxel (3) and extent in voxels (3), and the
# orientation quaternion (w x y z) the rotation is derived from.
BODY_RANK = 0
BODY_COM = 1
BODY_ROT = 4
BODY_VEL = 13
BODY_OMEGA = 16
BODY_COEFF = 19
BODY_BOX_LO = 20
BODY_BOX_DIM = 23
BODY_QUAT = 26
BODY_WIDTH = 30

# Per-body constants (``body_const``), uploaded with the state: density,
# mesh volume, bounding radius, body-frame inertia per unit density (9, row
# major), axis locks (loc x y z, rot x y z; 1 = locked), and the slice of the
# concatenated body-frame mesh points used for wall contact.
CONST_DENSITY = 0
CONST_VOLUME = 1
CONST_RADIUS = 2
CONST_INERTIA = 3
CONST_LOCK = 12
CONST_POINT_FIRST = 18
CONST_POINT_COUNT = 19
CONST_WIDTH = 20


@wp.func
def _finite(value: float) -> float:
    if wp.isnan(value) or wp.isinf(value):
        return 0.0
    return value


@wp.func
def _body_vec(state: wp.array2d(dtype=wp.float64), body: int, at: int) -> wp.vec3d:
    return wp.vec3d(state[body, at], state[body, at + 1], state[body, at + 2])


@wp.func
def _body_rot(state: wp.array2d(dtype=wp.float64), body: int) -> wp.mat33d:
    return wp.mat33d(
        state[body, 4], state[body, 5], state[body, 6],
        state[body, 7], state[body, 8], state[body, 9],
        state[body, 10], state[body, 11], state[body, 12])


@wp.kernel
def k_rigid_face_chunks(
    q: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    owner: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    liquid_density: float,
    q_offset: float,
    body_state: wp.array2d(dtype=wp.float64),
    chunks_per_body: wp.int32,
    pressure_valid: wp.int32,
    out_impulse: wp.array(dtype=wp.vec3d),
    out_angular: wp.array(dtype=wp.vec3d),
    out_volume: wp.array(dtype=wp.float64),
):
    """Pressure impulse on the solid faces of each body that border fluid.

    The projection subtracts ``(q_here - q_n) / h`` with ``q = p dt / rho``, so
    over one substep a face of area ``h^2`` with fluid pressure ``p`` on it
    pushes the solid by ``-rho q h^2 d`` (``d``: unit step from the solid
    voxel into the fluid voxel). Faces to air carry zero pressure; faces to
    another solid or the domain wall carry none.

    Two first-order grid offsets are removed so the body floats where the
    particles say the liquid is. ``q`` is extrapolated linearly from the two
    fluid voxel centres outside the face to the face itself (a voxel-centre
    read makes every body one voxel thicker to the pressure). ``q_offset``
    (``|g| dt delta``) is subtracted from every wetted face: the projection
    pins ``p = 0`` at the first air-voxel centre, on average ``delta`` (one
    voxel) above the particle interface. On a fully wetted body the constant
    cancels; on a floating body it removes the extra waterplane load.
    ``out_volume`` is the body's rasterized solid volume; the host rescales
    the pressure impulse by true / rasterized volume (voxelization error).

    Each body's partials split its voxel box (``BODY_BOX_*``, x fastest), a
    conservative bound on every voxel it owns, so the cost follows the body's
    size rather than the grid's. ``pressure_valid == 0`` (an unprojected
    substep has no pressure) matches no voxel: zero impulse and volume.
    """
    tid = wp.tid()
    body = tid // chunks_per_body
    part = tid - body * chunks_per_body
    rank = int(body_state[body, 0])
    if pressure_valid == 0:
        rank = -2
    com = _body_vec(body_state, body, 1)
    lo = wp.vec3i(int(body_state[body, 20]), int(body_state[body, 21]),
                  int(body_state[body, 22]))
    dx = int(body_state[body, 23])
    dy = int(body_state[body, 24])
    dz = int(body_state[body, 25])
    cells = dx * dy * dz
    chunk = (cells + chunks_per_body - 1) // chunks_per_body
    start = part * chunk
    stop = wp.min(start + chunk, cells)
    h = wp.float64(voxel_size)
    scale = wp.float64(liquid_density) * h * h
    half = wp.float64(0.5)
    ox = wp.float64(origin[0])
    oy = wp.float64(origin[1])
    oz = wp.float64(origin[2])
    impulse = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    angular = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    volume = wp.float64(0.0)
    for cell in range(start, stop):
        ijk = wp.vec3i(lo[0] + cell % dx, lo[1] + (cell // dx) % dy,
                       lo[2] + cell // (dx * dy))
        flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
        if flat < 0:
            continue
        if owner[flat] != rank:
            continue
        if (voxel_type[flat] & VOXEL_COLLISION) == 0:
            continue
        volume = volume + h * h * h
        centre = wp.vec3d(
            ox + (wp.float64(ijk[0]) + half) * h,
            oy + (wp.float64(ijk[1]) + half) * h,
            oz + (wp.float64(ijk[2]) + half) * h)
        for axis in range(3):
            for side in range(2):
                step = 2 * side - 1
                n = neighbor_flat(flat, leaf_table, voxels_dim, leaves_dim,
                                  ijk, axis, step)
                if n < 0:
                    continue
                nvt = voxel_type[n]
                if (nvt & VOXEL_COLLISION) != 0 or (nvt & VOXEL_FLUID) == 0:
                    continue
                q_face = wp.float64(_finite(q[n]))
                ijk_n = wp.vec3i(ijk[0], ijk[1], ijk[2])
                ijk_n[axis] = ijk_n[axis] + step
                n2 = neighbor_flat(n, leaf_table, voxels_dim, leaves_dim,
                                   ijk_n, axis, step)
                if n2 >= 0:
                    n2vt = voxel_type[n2]
                    if (n2vt & VOXEL_COLLISION) == 0 and (n2vt & VOXEL_FLUID) != 0:
                        q_face = (wp.float64(1.5) * q_face
                                  - half * wp.float64(_finite(q[n2])))
                q_face = q_face - wp.float64(q_offset)
                magnitude = -scale * q_face * wp.float64(step)
                j = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
                face = centre
                if axis == 0:
                    j = wp.vec3d(magnitude, wp.float64(0.0), wp.float64(0.0))
                    face = centre + wp.vec3d(half * h * wp.float64(step), wp.float64(0.0), wp.float64(0.0))
                elif axis == 1:
                    j = wp.vec3d(wp.float64(0.0), magnitude, wp.float64(0.0))
                    face = centre + wp.vec3d(wp.float64(0.0), half * h * wp.float64(step), wp.float64(0.0))
                else:
                    j = wp.vec3d(wp.float64(0.0), wp.float64(0.0), magnitude)
                    face = centre + wp.vec3d(wp.float64(0.0), wp.float64(0.0), half * h * wp.float64(step))
                impulse = impulse + j
                angular = angular + wp.cross(face - com, j)
    out_impulse[tid] = impulse
    out_angular[tid] = angular
    out_volume[tid] = volume


@wp.kernel
def k_rigid_drag_chunks(
    sample_points: wp.array(dtype=wp.vec3d),
    sample_normals: wp.array(dtype=wp.vec3d),
    sample_valid: wp.array(dtype=wp.int32),
    chunk_body: wp.array(dtype=wp.int32),
    body_state: wp.array2d(dtype=wp.float64),
    u: wp.array(dtype=float), v: wp.array(dtype=float), w: wp.array(dtype=float),
    wu: wp.array(dtype=float), wv: wp.array(dtype=float), ww: wp.array(dtype=float),
    voxel_type: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    chunk: wp.int32,
    out_impulse: wp.array(dtype=wp.vec3d),
    out_angular: wp.array(dtype=wp.vec3d),
    out_stiffness: wp.array(dtype=wp.float64),
):
    """Quadratic drag ``0.5 Cd rho a |r| r`` from the relative velocity one
    voxel outside each wetted boundary sample. ``body_drag_coeff`` holds
    ``0.5 Cd rho``; each sample normal's length is its patch area ``a``.
    ``out_stiffness`` sums ``d|j| / d|r|`` so the host can apply the drag
    implicitly (a light body under explicit drag overshoots and diverges)."""
    tid = wp.tid()
    body = chunk_body[tid]
    pos = _body_vec(body_state, body, 1)
    rot = _body_rot(body_state, body)
    vel = _body_vec(body_state, body, 13)
    omega = _body_vec(body_state, body, 16)
    coeff = body_state[body, 19]
    h = wp.float64(voxel_size)
    impulse = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    angular = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    stiffness = wp.float64(0.0)
    start = tid * chunk
    for i in range(start, start + chunk):
        if sample_valid[i] == 0:
            continue
        arm = rot * sample_points[i]
        area_normal = rot * sample_normals[i]
        area = wp.length(area_normal)
        if area <= wp.float64(0.0):
            continue
        normal = area_normal / area
        p = pos + arm
        probe = p + normal * h
        probe32 = wp.vec3(float(probe[0]), float(probe[1]), float(probe[2]))
        vp = world_to_voxel(probe32, origin, voxel_size)
        cell = wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])),
                        int(wp.floor(vp[2])))
        flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, cell)
        if flat < 0:
            continue
        vt = voxel_type[flat]
        if (vt & VOXEL_FLUID) == 0 or (vt & VOXEL_COLLISION) != 0:
            continue
        fluid = sample_mac_velocity(u, v, w, wu, wv, ww, leaf_table,
                                    voxels_dim, leaves_dim, vp)
        solid = vel + wp.cross(omega, arm)
        rel = wp.vec3d(wp.float64(_finite(fluid[0])),
                       wp.float64(_finite(fluid[1])),
                       wp.float64(_finite(fluid[2]))) - solid
        k = coeff * area * wp.length(rel) * wp.float64(dt)
        j = rel * k
        impulse = impulse + j
        angular = angular + wp.cross(arm, j)
        stiffness = stiffness + wp.float64(2.0) * k
    out_impulse[tid] = impulse
    out_angular[tid] = angular
    out_stiffness[tid] = stiffness


@wp.kernel
def k_rigid_reduce(
    face_impulse: wp.array(dtype=wp.vec3d),
    face_angular: wp.array(dtype=wp.vec3d),
    face_volume: wp.array(dtype=wp.float64),
    face_chunks: wp.int32,
    drag_impulse: wp.array(dtype=wp.vec3d),
    drag_angular: wp.array(dtype=wp.vec3d),
    drag_stiffness: wp.array(dtype=wp.float64),
    drag_first: wp.array(dtype=wp.int32),
    drag_count: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.float64, ndim=2),
):
    """One thread per body: add its partials in chunk order.

    Row layout: pressure impulse (3), pressure angular impulse (3), drag
    impulse (3), drag angular impulse (3), rasterized volume, drag stiffness.
    """
    body = wp.tid()
    zero = wp.float64(0.0)
    impulse = wp.vec3d(zero, zero, zero)
    angular = wp.vec3d(zero, zero, zero)
    volume = zero
    base = body * face_chunks
    for c in range(face_chunks):
        impulse = impulse + face_impulse[base + c]
        angular = angular + face_angular[base + c]
        volume = volume + face_volume[base + c]
    d_impulse = wp.vec3d(zero, zero, zero)
    d_angular = wp.vec3d(zero, zero, zero)
    stiffness = zero
    first = drag_first[body]
    for c in range(drag_count[body]):
        d_impulse = d_impulse + drag_impulse[first + c]
        d_angular = d_angular + drag_angular[first + c]
        stiffness = stiffness + drag_stiffness[first + c]
    for k in range(3):
        out[body, k] = impulse[k]
        out[body, 3 + k] = angular[k]
        out[body, 6 + k] = d_impulse[k]
        out[body, 9 + k] = d_angular[k]
    out[body, 12] = volume
    out[body, 13] = stiffness


@wp.func
def _quat_rot(qw: wp.float64, qx: wp.float64, qy: wp.float64,
              qz: wp.float64) -> wp.mat33d:
    one = wp.float64(1.0)
    two = wp.float64(2.0)
    return wp.mat33d(
        one - two * (qy * qy + qz * qz), two * (qx * qy - qw * qz), two * (qx * qz + qw * qy),
        two * (qx * qy + qw * qz), one - two * (qx * qx + qz * qz), two * (qy * qz - qw * qx),
        two * (qx * qz - qw * qy), two * (qy * qz + qw * qx), one - two * (qx * qx + qy * qy))


@wp.func
def _row_finite(impulses: wp.array2d(dtype=wp.float64), b: int) -> int:
    for k in range(14):
        value = impulses[b, k]
        if wp.isnan(value) or wp.isinf(value):
            return 0
    return 1


@wp.func
def _row_vec(impulses: wp.array2d(dtype=wp.float64), b: int, at: int,
             finite: int) -> wp.vec3d:
    zero = wp.float64(0.0)
    if finite == 0:
        return wp.vec3d(zero, zero, zero)
    return wp.vec3d(impulses[b, at], impulses[b, at + 1], impulses[b, at + 2])


@wp.kernel
def k_rigid_integrate(
    impulses: wp.array2d(dtype=wp.float64),
    body_const: wp.array2d(dtype=wp.float64),
    body_points: wp.array(dtype=wp.vec3d),
    gravity: wp.vec3d,
    dt: wp.float64,
    dv_max: wp.float64,
    domain_lo: wp.vec3d,
    domain_hi: wp.vec3d,
    origin: wp.vec3,
    voxel_size: float,
    voxels_dim: wp.vec3i,
    box_margin: wp.int32,
    body_state: wp.array2d(dtype=wp.float64),
):
    """Advance one body by one substep from its gathered fluid impulses.

    One thread per body, in float64: the device port of
    ``RigidCoupling._integrate`` followed by ``_contain`` (the host versions
    stay as the reference the tests compare against). Semi-implicit Euler with
    implicit drag, the net per-substep velocity clamp, the gyroscopic term and
    axis locks; then inelastic contact with the domain walls. Writes the new
    pose, velocities and face-gather box back into ``body_state`` so the next
    substep needs no host round trip.
    """
    b = wp.tid()
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    half = wp.float64(0.5)
    finite = _row_finite(impulses, b)
    rasterized = zero
    stiffness = zero
    if finite == 1:
        rasterized = impulses[b, 12]
        stiffness = wp.max(impulses[b, 13], zero)
    density = body_const[b, 0]
    volume = body_const[b, 1]
    radius = body_const[b, 2]
    mass = density * volume
    # Pressure: rescale by true / rasterized volume (Archimedes on the mesh),
    # bounded for a body half outside the grid.
    scale = one
    if rasterized > zero:
        scale = volume / rasterized
    scale = wp.min(wp.max(scale, half), wp.float64(2.0))
    impulse = _row_vec(impulses, b, 0, finite) * scale
    angular = _row_vec(impulses, b, 3, finite) * scale
    # Drag, applied implicitly: dv = J / (m + K) for stiffness K.
    damping = mass / (mass + stiffness)
    impulse = impulse + _row_vec(impulses, b, 6, finite) * damping
    angular = angular + _row_vec(impulses, b, 9, finite) * damping
    # Clamp the net per-substep impulse (fluid + weight) to m * dv_max.
    dv = impulse / mass + gravity * dt
    size = wp.length(dv)
    if size > dv_max:
        dv = dv * (dv_max / size)
    lin = _body_vec(body_state, b, 13) + dv
    qw = body_state[b, 26]
    qx = body_state[b, 27]
    qy = body_state[b, 28]
    qz = body_state[b, 29]
    rot = _quat_rot(qw, qx, qy, qz)
    unit = wp.mat33d(
        body_const[b, 3], body_const[b, 4], body_const[b, 5],
        body_const[b, 6], body_const[b, 7], body_const[b, 8],
        body_const[b, 9], body_const[b, 10], body_const[b, 11])
    inertia = (rot * unit * wp.transpose(rot)) * density
    omega = _body_vec(body_state, b, 16)
    torque = angular - wp.cross(omega, inertia * omega) * dt
    if wp.determinant(inertia) != zero:
        omega = omega + wp.inverse(inertia) * torque
    for axis in range(3):
        if body_const[b, 12 + axis] != zero:
            lin[axis] = zero
        if body_const[b, 15 + axis] != zero:
            omega[axis] = zero
    speed = wp.length(lin)
    if speed > dv_max:
        lin = lin * (dv_max / speed)
    spin = wp.length(omega) * wp.max(radius, wp.float64(1.0e-6))
    if spin > dv_max:
        omega = omega * (dv_max / spin)
    pos = _body_vec(body_state, b, 1) + lin * dt
    if omega[0] != zero or omega[1] != zero or omega[2] != zero:
        # q += dt / 2 * (0, omega) q, renormalized.
        dw = -omega[0] * qx - omega[1] * qy - omega[2] * qz
        dx = omega[0] * qw + omega[1] * qz - omega[2] * qy
        dy = -omega[0] * qz + omega[1] * qw + omega[2] * qx
        dz = omega[0] * qy - omega[1] * qx + omega[2] * qw
        qw = qw + half * dt * dw
        qx = qx + half * dt * dx
        qy = qy + half * dt * dy
        qz = qz + half * dt * dz
        norm = wp.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw = qw / norm
        qx = qx / norm
        qy = qy / norm
        qz = qz / norm
    rot = _quat_rot(qw, qx, qy, qz)
    # Inelastic contact with the domain walls.
    first = int(body_const[b, 18])
    count = int(body_const[b, 19])
    if count > 0:
        low = rot * body_points[first]
        high = low
        for i in range(first + 1, first + count):
            w = rot * body_points[i]
            low = wp.min(low, w)
            high = wp.max(high, w)
        low = pos + low
        high = pos + high
        for axis in range(3):
            if low[axis] < domain_lo[axis]:
                pos[axis] = pos[axis] + (domain_lo[axis] - low[axis])
                lin[axis] = wp.max(lin[axis], zero)
            elif high[axis] > domain_hi[axis]:
                pos[axis] = pos[axis] - (high[axis] - domain_hi[axis])
                lin[axis] = wp.min(lin[axis], zero)
    h = wp.float64(voxel_size)
    for k in range(3):
        body_state[b, 1 + k] = pos[k]
        body_state[b, 13 + k] = lin[k]
        body_state[b, 16 + k] = omega[k]
        # Face-gather box: lowest voxel of the bounding sphere less the
        # margin, clamped so the (fixed) extent stays inside the grid.
        extent = int(body_state[b, 23 + k])
        lo = int(wp.floor((pos[k] - radius - wp.float64(origin[k])) / h)) - box_margin
        lo = wp.min(wp.max(lo, 0), voxels_dim[k] - extent)
        body_state[b, 20 + k] = wp.float64(lo)
    for r in range(3):
        for c in range(3):
            body_state[b, 4 + 3 * r + c] = rot[r, c]
    body_state[b, 26] = qw
    body_state[b, 27] = qx
    body_state[b, 28] = qy
    body_state[b, 29] = qz


@wp.kernel
def k_rasterize_rigid_ranked(
    local_mesh: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    band: float,
    body_state: wp.array2d(dtype=wp.float64),
    body: wp.int32,
    friction: float,
    restitution: float,
    rank: wp.int32,
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    collision_friction: wp.array(dtype=float),
    collision_restitution: wp.array(dtype=float),
    owner: wp.array(dtype=wp.int32),
):
    """``k_rasterize_collider_material_ranked`` for a Dynamic collider.

    The body's mesh stays in its body frame (centred on the centre of mass,
    built once); each voxel centre is mapped into that frame with the pose in
    ``body_state``. Distance is rotation invariant, so this equals the world
    query without re-posing and refitting the mesh every substep, and the
    pose never leaves the device.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * LEAF_VOLUME:
        return
    lc = leaf_coords[tid // LEAF_VOLUME]
    local = tid - (tid // LEAF_VOLUME) * LEAF_VOLUME
    li = local & LEAF_MASK
    lj = (local >> LEAF_SHIFT) & LEAF_MASK
    lk = local >> (2 * LEAF_SHIFT)
    ijk = wp.vec3i(lc[0] * LEAF_DIM + li, lc[1] * LEAF_DIM + lj, lc[2] * LEAF_DIM + lk)
    p = origin + wp.vec3(
        (float(ijk[0]) + 0.5) * voxel_size,
        (float(ijk[1]) + 0.5) * voxel_size,
        (float(ijk[2]) + 0.5) * voxel_size,
    )
    arm = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2])) - _body_vec(
        body_state, body, 1)
    pl = wp.transpose(_body_rot(body_state, body)) * arm
    pb = wp.vec3(float(pl[0]), float(pl[1]), float(pl[2]))
    q = wp.mesh_query_point_sign_normal(local_mesh, pb, band)
    if not q.result:
        return
    cp = wp.mesh_eval_position(local_mesh, q.face, q.u, q.v)
    distance = wp.length(pb - cp) * q.sign
    current = collision_sdf[tid]
    if distance < current or (distance == current and rank < owner[tid]):
        v = _body_vec(body_state, body, 13) + wp.cross(_body_vec(body_state, body, 16), arm)
        collision_sdf[tid] = distance
        collision_vel[tid] = wp.vec3(float(v[0]), float(v[1]), float(v[2]))
        collision_friction[tid] = friction
        collision_restitution[tid] = restitution
        owner[tid] = rank
