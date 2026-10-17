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
# major), axis locks (loc x y z, rot x y z; 1 = locked), the slice of the
# concatenated body-frame mesh points used for wall contact, the slice of the
# concatenated contact points (mesh points plus boundary samples) and the
# body's own friction and restitution.
CONST_DENSITY = 0
CONST_VOLUME = 1
CONST_RADIUS = 2
CONST_INERTIA = 3
CONST_LOCK = 12
CONST_POINT_FIRST = 18
CONST_POINT_COUNT = 19
CONST_CONTACT_FIRST = 20
CONST_CONTACT_COUNT = 21
CONST_FRICTION = 22
CONST_RESTITUTION = 23
CONST_WIDTH = 24

# One collider-vertex contact partial (``k_rigid_contact_collider_points``):
# weight sum, deepest penetration, then depth-weighted sums of the push
# direction (3), contact point (3), collider velocity (3), friction and
# restitution.
PARTIAL_WIDTH = 15


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


@wp.func
def _vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _vec3f(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(float(v[0]), float(v[1]), float(v[2]))


@wp.func
def _inverse_inertia(body_const: wp.array2d(dtype=wp.float64), b: int,
                     rot: wp.mat33d) -> wp.mat33d:
    zero = wp.float64(0.0)
    unit = wp.mat33d(
        body_const[b, 3], body_const[b, 4], body_const[b, 5],
        body_const[b, 6], body_const[b, 7], body_const[b, 8],
        body_const[b, 9], body_const[b, 10], body_const[b, 11])
    inertia = (rot * unit * wp.transpose(rot)) * body_const[b, 0]
    if wp.determinant(inertia) != zero:
        return wp.inverse(inertia)
    return wp.mat33d(zero, zero, zero, zero, zero, zero, zero, zero, zero)


@wp.func
def _effective_mass(mass: wp.float64, inverse: wp.mat33d, r: wp.vec3d,
                    d: wp.vec3d) -> wp.float64:
    """Inverse effective mass of a body along ``d`` at arm ``r``."""
    return wp.float64(1.0) / mass + wp.dot(wp.cross(inverse * wp.cross(r, d), r), d)


@wp.kernel
def k_rigid_contact_points(
    body_state: wp.array2d(dtype=wp.float64),
    body_const: wp.array2d(dtype=wp.float64),
    contact_points: wp.array(dtype=wp.vec3d),
    contact_owner: wp.array(dtype=wp.int32),
    body_meshes: wp.array(dtype=wp.uint64),
    body_count: wp.int32,
    meshes: wp.array(dtype=wp.uint64),
    mesh_lin: wp.array(dtype=wp.vec3),
    mesh_ang: wp.array(dtype=wp.vec3),
    mesh_center: wp.array(dtype=wp.vec3),
    mesh_material: wp.array(dtype=wp.vec2),
    mesh_count: wp.int32,
    max_dist: float,
    out_depth: wp.array(dtype=float),
    out_normal: wp.array(dtype=wp.vec3),
    out_point: wp.array(dtype=wp.vec3d),
    out_velocity: wp.array(dtype=wp.vec3),
    out_material: wp.array(dtype=wp.vec2),
    out_partner: wp.array(dtype=wp.int32),
):
    """Penetration of one body contact point into anything solid.

    One thread per contact point (a body's mesh points and its boundary
    samples, so an edge or face crossing a collider edge is seen too). The
    point is tested against the scene's Static and Animated colliders at their
    current substep pose and against every other Dynamic body's mesh at its
    current pose. A point inside one (negative signed distance, found within
    ``max_dist`` of the surface) records its depth, the direction that takes
    it out, its world position, the collider's velocity there (colliders
    only), the (friction, restitution) of the pair and the partner: -1 for a
    collider, else the other body's index. The deepest partner wins; depth 0
    means no contact. Every point is independent, so the result is bitwise
    repeatable; ``k_rigid_contact_resolve`` reads it in point order.
    """
    tid = wp.tid()
    b = contact_owner[tid]
    world = _body_vec(body_state, b, 1) + _body_rot(body_state, b) * contact_points[tid]
    p = _vec3f(world)
    best = float(0.0)  # noqa: UP018 - Warp needs a dynamic variable here
    partner = int(-1)  # noqa: UP018 - Warp needs a dynamic variable here
    normal = wp.vec3(0.0, 0.0, 0.0)
    velocity = wp.vec3(0.0, 0.0, 0.0)
    material = wp.vec2(0.0, 0.0)
    for m in range(mesh_count):
        q = wp.mesh_query_point_sign_normal(meshes[m], p, max_dist)
        if q.result and q.sign < 0.0:
            cp = wp.mesh_eval_position(meshes[m], q.face, q.u, q.v)
            out = cp - p
            depth = wp.length(out)
            if depth > best:
                best = depth
                if depth > 1.0e-7:
                    normal = out / depth
                else:
                    normal = wp.normalize(wp.mesh_eval_face_normal(meshes[m], q.face))
                velocity = mesh_lin[m] + wp.cross(mesh_ang[m], p - mesh_center[m])
                material = mesh_material[m]
                partner = -1
    own = wp.vec2(float(body_const[b, 22]), float(body_const[b, 23]))
    for c in range(body_count):
        if c == b:
            continue
        center = _body_vec(body_state, c, 1)
        if wp.length(world - center) > body_const[c, 2] + wp.float64(max_dist):
            continue
        rot = _body_rot(body_state, c)
        local = _vec3f(wp.transpose(rot) * (world - center))
        q = wp.mesh_query_point_sign_normal(body_meshes[c], local, max_dist)
        if q.result and q.sign < 0.0:
            cp = wp.mesh_eval_position(body_meshes[c], q.face, q.u, q.v)
            out = cp - local
            depth = wp.length(out)
            if depth > best:
                best = depth
                away = wp.vec3(0.0, 0.0, 0.0)
                if depth > 1.0e-7:
                    away = out / depth
                else:
                    away = wp.normalize(wp.mesh_eval_face_normal(body_meshes[c], q.face))
                normal = _vec3f(rot * _vec3d(away))
                velocity = wp.vec3(0.0, 0.0, 0.0)
                material = 0.5 * (own + wp.vec2(float(body_const[c, 22]),
                                                float(body_const[c, 23])))
                partner = c
    out_depth[tid] = best
    out_normal[tid] = normal
    out_point[tid] = world
    out_velocity[tid] = velocity
    out_material[tid] = material
    out_partner[tid] = partner


@wp.kernel
def k_rigid_contact_collider_points(
    body_state: wp.array2d(dtype=wp.float64),
    body_const: wp.array2d(dtype=wp.float64),
    body_meshes: wp.array(dtype=wp.uint64),
    meshes: wp.array(dtype=wp.uint64),
    mesh_lin: wp.array(dtype=wp.vec3),
    mesh_ang: wp.array(dtype=wp.vec3),
    mesh_center: wp.array(dtype=wp.vec3),
    mesh_material: wp.array(dtype=wp.vec2),
    chunk_mesh: wp.array(dtype=wp.int32),
    chunk_first: wp.array(dtype=wp.int32),
    chunk_count: wp.array(dtype=wp.int32),
    chunks: wp.int32,
    max_dist: float,
    partials: wp.array2d(dtype=wp.float64),
):
    """Collider vertices inside one body, as one partial per (body, chunk).

    The other half of collider contact: a sharp collider point (a spike, a
    corner, an edge vertex) pressed into a flat body face has no body point
    inside the collider, so it is found from the collider side. One thread
    walks one chunk of one collider mesh's points in order and sums, depth
    weighted, the direction that pushes the body off each point inside it,
    the point, the collider's velocity there and its material (layout:
    ``PARTIAL_WIDTH``). ``k_rigid_contact_resolve`` adds a body's partials in
    chunk order, so the sum is bitwise repeatable.
    """
    tid = wp.tid()
    b = tid // chunks
    k = tid - b * chunks
    zero = wp.float64(0.0)
    wsum = wp.float64(0.0)
    dmax = wp.float64(0.0)
    nsum = wp.vec3d(zero, zero, zero)
    psum = wp.vec3d(zero, zero, zero)
    vsum = wp.vec3d(zero, zero, zero)
    fsum = wp.float64(0.0)
    esum = wp.float64(0.0)
    m = chunk_mesh[k]
    points = wp.mesh_get(meshes[m]).points
    center = _body_vec(body_state, b, 1)
    rot = _body_rot(body_state, b)
    reach = body_const[b, 2] + wp.float64(max_dist)
    material = mesh_material[m]
    first = chunk_first[k]
    for v in range(first, first + chunk_count[k]):
        pw = points[v]
        pd = _vec3d(pw)
        if wp.length(pd - center) > reach:
            continue
        local = _vec3f(wp.transpose(rot) * (pd - center))
        q = wp.mesh_query_point_sign_normal(body_meshes[b], local, max_dist)
        if q.result and q.sign < 0.0:
            out = wp.mesh_eval_position(body_meshes[b], q.face, q.u, q.v) - local
            depth = wp.length(out)
            if depth > 1.0e-7:
                w = wp.float64(depth)
                # The point leaves the body through its nearest surface
                # point, so the body moves the other way.
                nsum = nsum - (rot * _vec3d(out / depth)) * w
                psum = psum + pd * w
                vel = mesh_lin[m] + wp.cross(mesh_ang[m], pw - mesh_center[m])
                vsum = vsum + _vec3d(vel) * w
                fsum = fsum + wp.float64(material[0]) * w
                esum = esum + wp.float64(material[1]) * w
                wsum = wsum + w
                dmax = wp.max(dmax, w)
    partials[tid, 0] = wsum
    partials[tid, 1] = dmax
    for a in range(3):
        partials[tid, 2 + a] = nsum[a]
        partials[tid, 5 + a] = psum[a]
        partials[tid, 8 + a] = vsum[a]
    partials[tid, 11] = fsum
    partials[tid, 12] = esum
    partials[tid, 13] = zero
    partials[tid, 14] = zero


@wp.kernel
def k_rigid_contact_pairs(
    depth: wp.array(dtype=float),
    normal: wp.array(dtype=wp.vec3),
    point: wp.array(dtype=wp.vec3d),
    velocity: wp.array(dtype=wp.vec3),
    material: wp.array(dtype=wp.vec2),
    partner: wp.array(dtype=wp.int32),
    chunk_first: wp.array(dtype=wp.int32),
    chunk_count: wp.array(dtype=wp.int32),
    chunks: wp.int32,
    pairs: wp.array2d(dtype=wp.float64),
):
    """Body contact points reduced per (partner, chunk), in point order.

    Row ``(g + 1) * chunks + k`` holds, depth weighted (layout:
    ``PARTIAL_WIDTH``), the contacts of the points in chunk ``k`` (one body's
    run of ``CONTACT_CHUNK`` points) whose partner is ``g`` (-1 for the
    colliders). Contact points do not change between the resolve passes, so
    this runs once per substep and every pass reads rows, not points; each
    row is one thread's fixed-order sum, so the result is bitwise repeatable.
    """
    tid = wp.tid()
    s = tid // chunks
    k = tid - s * chunks
    g = s - 1
    zero = wp.float64(0.0)
    wsum = wp.float64(0.0)
    dmax = wp.float64(0.0)
    nsum = wp.vec3d(zero, zero, zero)
    psum = wp.vec3d(zero, zero, zero)
    vsum = wp.vec3d(zero, zero, zero)
    fsum = wp.float64(0.0)
    esum = wp.float64(0.0)
    first = chunk_first[k]
    for i in range(first, first + chunk_count[k]):
        if depth[i] > 0.0 and partner[i] == g:
            w = wp.float64(depth[i])
            nsum = nsum + _vec3d(normal[i]) * w
            psum = psum + point[i] * w
            vsum = vsum + _vec3d(velocity[i]) * w
            fsum = fsum + wp.float64(material[i][0]) * w
            esum = esum + wp.float64(material[i][1]) * w
            wsum = wsum + w
            dmax = wp.max(dmax, w)
    pairs[tid, 0] = wsum
    pairs[tid, 1] = dmax
    for a in range(3):
        pairs[tid, 2 + a] = nsum[a]
        pairs[tid, 5 + a] = psum[a]
        pairs[tid, 8 + a] = vsum[a]
    pairs[tid, 11] = fsum
    pairs[tid, 12] = esum
    pairs[tid, 13] = zero
    pairs[tid, 14] = zero


@wp.kernel
def k_rigid_contact_resolve(
    body_const: wp.array2d(dtype=wp.float64),
    prev_state: wp.array2d(dtype=wp.float64),
    pairs: wp.array2d(dtype=wp.float64),
    pair_chunks: wp.int32,
    body_chunk_first: wp.array(dtype=wp.int32),
    body_chunk_count: wp.array(dtype=wp.int32),
    partials: wp.array2d(dtype=wp.float64),
    chunks: wp.int32,
    body_count: wp.int32,
    max_dist: float,
    gravity: wp.vec3d,
    dt: wp.float64,
    push: wp.int32,
    body_points: wp.array(dtype=wp.vec3d),
    domain_lo: wp.vec3d,
    domain_hi: wp.vec3d,
    prev_support: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxels_dim: wp.vec3i,
    box_margin: wp.int32,
    body_state: wp.array2d(dtype=wp.float64),
    support: wp.array(dtype=wp.int32),
):
    """Push one body out of what it touches and bounce it off.

    One thread per body; ``prev_state`` and ``prev_support`` are copies taken
    before the launch, so a thread reads its partners without racing their
    writes. Contacts are grouped by partner: first the Static and Animated
    colliders together (the body's points inside them plus the collider
    points inside the body), then each other Dynamic body (this body's points
    inside it and its points inside this body, so both bodies see the same
    contact with opposite normals). Each group is reduced, depth weighted, in
    chunk order from the ``k_rigid_contact_pairs`` and collider partial rows
    to one contact: a normal ``n``, a contact point and the partner's
    velocity there.

    One impulse along ``n`` at the contact arm cancels the approach speed
    (restitution below the resting threshold ``2 |g| dt`` is dropped so a
    resting body does not jitter) and a Coulomb friction impulse removes
    tangential slip; the impulse acts at the arm, so an off-centre contact
    tips the body. Against another body the impulse uses the pair's effective
    mass and each body applies its own half, so the pair's momentum is kept.
    When ``push`` is set the body is also translated out along ``n`` by the
    deepest penetration (its inverse-mass share of it against a body).

    A stack needs support to travel up it (shock propagation): a body is
    supported when the domain floor or a collider holds it up against
    gravity, or when it rests on a supported body. A supported body below
    acts as immovable ground for the body above it, and a supported body
    ignores the bodies resting on it, so the weight of a stack does not push
    its bottom through the floor. Repeating the launch (Jacobi iterations)
    carries support one body up per launch. Axis locks hold.
    """
    b = wp.tid()
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    pos0 = _body_vec(prev_state, b, 1)
    rot = _body_rot(prev_state, b)
    pos = _body_vec(prev_state, b, 1)
    lin = _body_vec(prev_state, b, 13)
    omega = _body_vec(prev_state, b, 16)
    mass = body_const[b, 0] * body_const[b, 1]
    inverse = _inverse_inertia(body_const, b, rot)
    radius = body_const[b, 2]
    rest = wp.float64(2.0) * wp.length(gravity) * dt
    down = wp.vec3d(zero, zero, zero)
    if wp.length(gravity) > zero:
        down = wp.normalize(gravity)
    supported = int(prev_support[b])
    # The domain floor (the wall gravity points at) holds the body up.
    pfirst = int(body_const[b, 18])
    pcount = int(body_const[b, 19])
    if pcount > 0 and supported == 0:
        low = rot * body_points[pfirst]
        high = low
        for i in range(pfirst + 1, pfirst + pcount):
            corner = rot * body_points[i]
            low = wp.min(low, corner)
            high = wp.max(high, corner)
        tol =wp.float64(0.05 * voxel_size)
        for axis in range(3):
            if down[axis] <= wp.float64(-0.5) and pos0[axis] + low[axis] <= domain_lo[axis] + tol:
                supported = 1
            if down[axis] >= wp.float64(0.5) and pos0[axis] + high[axis] >= domain_hi[axis] - tol:
                supported = 1
    for g in range(-1, body_count):
        if g == b:
            continue
        if g >= 0:
            gap = wp.length(_body_vec(prev_state, g, 1) - pos0)
            if gap > radius + body_const[g, 2] + wp.float64(2.0 * max_dist):
                continue
        wsum = wp.float64(0.0)
        dmax = wp.float64(0.0)
        nsum = wp.vec3d(zero, zero, zero)
        psum = wp.vec3d(zero, zero, zero)
        vsum = wp.vec3d(zero, zero, zero)
        fsum = wp.float64(0.0)
        esum = wp.float64(0.0)
        own = (g + 1) * pair_chunks
        cfirst = body_chunk_first[b]
        for k in range(cfirst, cfirst + body_chunk_count[b]):
            row = own + k
            if pairs[row, 0] > zero:
                wsum = wsum + pairs[row, 0]
                dmax = wp.max(dmax, pairs[row, 1])
                nsum = nsum + wp.vec3d(pairs[row, 2], pairs[row, 3], pairs[row, 4])
                psum = psum + wp.vec3d(pairs[row, 5], pairs[row, 6], pairs[row, 7])
                vsum = vsum + wp.vec3d(pairs[row, 8], pairs[row, 9], pairs[row, 10])
                fsum = fsum + pairs[row, 11]
                esum = esum + pairs[row, 12]
        if g < 0:
            for k in range(chunks):
                row = b * chunks + k
                if partials[row, 0] > zero:
                    wsum = wsum + partials[row, 0]
                    dmax = wp.max(dmax, partials[row, 1])
                    nsum = nsum + wp.vec3d(partials[row, 2], partials[row, 3], partials[row, 4])
                    psum = psum + wp.vec3d(partials[row, 5], partials[row, 6], partials[row, 7])
                    vsum = vsum + wp.vec3d(partials[row, 8], partials[row, 9], partials[row, 10])
                    fsum = fsum + partials[row, 11]
                    esum = esum + partials[row, 12]
        else:
            # ``g``'s points inside this body: the same contact, seen from g.
            mine = (b + 1) * pair_chunks
            gfirst = body_chunk_first[g]
            for k in range(gfirst, gfirst + body_chunk_count[g]):
                row = mine + k
                if pairs[row, 0] > zero:
                    wsum = wsum + pairs[row, 0]
                    dmax = wp.max(dmax, pairs[row, 1])
                    nsum = nsum - wp.vec3d(pairs[row, 2], pairs[row, 3], pairs[row, 4])
                    psum = psum + wp.vec3d(pairs[row, 5], pairs[row, 6], pairs[row, 7])
                    fsum = fsum + pairs[row, 11]
                    esum = esum + pairs[row, 12]
        if wsum <= zero:
            continue
        nlen = wp.length(nsum)
        if nlen <= wp.float64(1.0e-12):
            continue
        n = nsum / nlen
        lift = wp.dot(n, -down)
        if g < 0:
            if lift >= wp.float64(0.5):
                supported = 1
        elif lift <= wp.float64(-0.5) and supported != 0:
            # ``g`` rests on this supported body: it does not push back.
            continue
        contact = psum / wsum
        r = contact - pos0
        mu = wp.min(wp.max(fsum / wsum, zero), one)
        restitution = wp.min(wp.max(esum / wsum, zero), one)
        vc = vsum / wsum
        share = wp.float64(1.0)
        rg = wp.vec3d(zero, zero, zero)
        mass_g = wp.float64(1.0)
        inverse_g = wp.mat33d(zero, zero, zero, zero, zero, zero, zero, zero, zero)
        movable = int(0)  # noqa: UP018 - Warp needs a dynamic variable here
        if g >= 0:
            rg = contact - _body_vec(prev_state, g, 1)
            vc = _body_vec(prev_state, g, 13) + wp.cross(_body_vec(prev_state, g, 16), rg)
            if lift >= wp.float64(0.5) and prev_support[g] != 0:
                # A supported body underneath is ground.
                supported = 1
            else:
                movable = 1
                mass_g = body_const[g, 0] * body_const[g, 1]
                inverse_g = _inverse_inertia(body_const, g, _body_rot(prev_state, g))
                share = (one / mass) / (one / mass + one / mass_g)
        if push != 0:
            for axis in range(3):
                if body_const[b, 12 + axis] == zero:
                    pos[axis] = pos[axis] + n[axis] * dmax * share
        vrel = lin + wp.cross(omega, r) - vc
        vn = wp.dot(vrel, n)
        if vn < zero:
            kn = _effective_mass(mass, inverse, r, n)
            if movable != 0:
                kn = kn + _effective_mass(mass_g, inverse_g, rg, n)
            e = restitution
            if -vn < rest:
                e = zero
            j = -(one + e) * vn / kn
            lin = lin + n * (j / mass)
            omega = omega + inverse * (wp.cross(r, n) * j)
            vslip = lin + wp.cross(omega, r) - vc
            vt = vslip - n * wp.dot(vslip, n)
            vtl = wp.length(vt)
            if vtl > wp.float64(1.0e-12) and mu > zero:
                t = vt / vtl
                kt = _effective_mass(mass, inverse, r, t)
                if movable != 0:
                    kt = kt + _effective_mass(mass_g, inverse_g, rg, t)
                jt = wp.min(mu * j, vtl / kt)
                lin = lin - t * (jt / mass)
                omega = omega - inverse * (wp.cross(r, t) * jt)
        for axis in range(3):
            if body_const[b, 12 + axis] != zero:
                lin[axis] = zero
            if body_const[b, 15 + axis] != zero:
                omega[axis] = zero

    support[b] = supported
    h = wp.float64(voxel_size)
    for k in range(3):
        body_state[b, 1 + k] = pos[k]
        body_state[b, 13 + k] = lin[k]
        body_state[b, 16 + k] = omega[k]
        extent = int(body_state[b, 23 + k])
        lo = int(wp.floor((pos[k] - radius - wp.float64(origin[k])) / h)) - box_margin
        lo = wp.min(wp.max(lo, 0), voxels_dim[k] - extent)
        body_state[b, 20 + k] = wp.float64(lo)


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
