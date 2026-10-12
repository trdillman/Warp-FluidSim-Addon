"""GPU source helpers and kernels: analytic shapes, fields, and particles.
Includes volume (NanoVDB via wp.Volume) and particle-attribute sourcing.

Ports of the fluid_sourcing group:
  fluidSimSourceShapes, fluidSimSourceShapesParticlesAttributes,
  fluidSimSourceHeight, fluidSimSourceResolveType, fluidSimSourceParticles,
  fluidSimSourceParticleVectorFieldAttribute,
  fluidSimSourceParticleScalarFieldAttribute,
  fluidSimSourceParticleScalarIntegerAttribute,
  fluidSimSourceParticleScalarFloatAttribute,
  fluidSimSourceParticleScalarHalfAttribute,
  fluidSimResetSourceShapeTrackedVoxelType (registered in kernels_fluid)

and the core_sourcing_vdb_volume_particle group:
  coreSourceFloat, coreSourceVector, coreSourceStaggeredVector,
  coreSourceFloatParticleAttribute, coreSourceVectorParticleAttribute,
  coreSourceNanoVDBTypeFog, coreSourceNanoVDBTypeFogWithTemperature,
  coreSourceNanoVDBMaskFog, coreSourceNanoVDBFloat, coreSourceNanoVDBVector,
  coreSourceOpenVDBMask, coreSourceOpenVDBType,
  coreSourceOpenVDBTypeWithTemperature, coreSourceOpenVDBFloat,
  coreSourceOpenVDBVectorComponent

OpenVDB grids are converted to NanoVDB (wp.Volume) by the Blender bridge, so
the coreSourceOpenVDB* names map to the same wp.Volume sampling kernels.
"""

from __future__ import annotations

import warp as wp

from . import constants as C
from .fields import (
    leaf_linear_index,
    sample_scalar_trilinear,
    voxel_center_world,
    voxel_coord_of_flat,
    voxel_flat_index,
    world_to_voxel,
)
from .registry import register_fn

# Blend helper (BlendMode: 0 Add, 1 Replace, 2 Maximum, 3 Pull, 4 Blend)
from .source_contracts import (
    MeshSource,
    SourceShape,
)

wp.set_module_options({"enable_backward": False})


@wp.func
def blend_scalar(old: float, new: float, mode: int, t: float):
    if mode == 0:
        return old + new * t
    if mode == 1:
        return new
    if mode == 2:
        return wp.max(old, new)
    if mode == 3:
        return old + wp.clamp(new - old, -t, t)
    return old + (new - old) * wp.clamp(t, 0.0, 1.0)


@wp.func
def blend_vec(old: wp.vec3, new: wp.vec3, mode: int, t: float):
    if mode == 0:
        return old + new * t
    if mode == 1:
        return new
    if mode == 2:
        if wp.length_sq(new) > wp.length_sq(old):
            return new
        return old
    if mode == 3:
        d = new - old
        length = wp.length(d)
        if length > t:
            d = d * (t / length)
        return old + d
    return old + (new - old) * wp.clamp(t, 0.0, 1.0)


# Source shape description (GPU struct array)


@wp.func
def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Squared Euclidean distance from a point to a non-degenerate triangle."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.length_sq(ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.length_sq(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.length_sq(p - (a + v * ab))
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.length_sq(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.length_sq(p - (a + w * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.length_sq(p - (b + w * (c - b)))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.length_sq(p - (a + ab * v + ac * w))


@wp.func
def _segment_distance(p: wp.vec3, half_height: float):
    dz = wp.max(wp.abs(p[2]) - half_height, 0.0)
    return wp.sqrt(p[0] * p[0] + p[1] * p[1] + dz * dz)


@wp.func
def _cone_sdf(p: wp.vec3, radius: float, half_height: float):
    if half_height <= 1.0e-8:
        radial = wp.sqrt(p[0] * p[0] + p[1] * p[1])
        dr = wp.max(radial - wp.max(radius, 0.0), 0.0)
        return wp.sqrt(dr * dr + p[2] * p[2])
    if radius <= 1.0e-8:
        return _segment_distance(p, half_height)
    q = wp.vec2(wp.sqrt(p[0] * p[0] + p[1] * p[1]), p[2])
    k1 = wp.vec2(0.0, half_height)
    k2 = wp.vec2(-radius, 2.0 * half_height)
    cap_radius = wp.where(q[1] < 0.0, radius, 0.0)
    ca = wp.vec2(q[0] - wp.min(q[0], cap_radius), wp.abs(q[1]) - half_height)
    t = wp.clamp(wp.dot(k1 - q, k2) / wp.dot(k2, k2), 0.0, 1.0)
    cb = q - k1 + k2 * t
    sign = 1.0
    if cb[0] < 0.0 and ca[1] < 0.0:
        sign = -1.0
    return sign * wp.sqrt(wp.min(wp.dot(ca, ca), wp.dot(cb, cb)))


@wp.func
def _pyramid_sdf(p: wp.vec3, radius: float, half_height: float):
    if half_height <= 1.0e-8:
        dx = wp.max(wp.abs(p[0]) - wp.max(radius, 0.0), 0.0)
        dy = wp.max(wp.abs(p[1]) - wp.max(radius, 0.0), 0.0)
        return wp.sqrt(dx * dx + dy * dy + p[2] * p[2])
    if radius <= 1.0e-8:
        return _segment_distance(p, half_height)
    z0 = -half_height
    apex = wp.vec3(0.0, 0.0, half_height)
    v0 = wp.vec3(-radius, -radius, z0)
    v1 = wp.vec3(radius, -radius, z0)
    v2 = wp.vec3(radius, radius, z0)
    v3 = wp.vec3(-radius, radius, z0)
    distance_sq = _point_triangle_distance_sq(p, v0, v1, apex)
    distance_sq = wp.min(distance_sq, _point_triangle_distance_sq(p, v1, v2, apex))
    distance_sq = wp.min(distance_sq, _point_triangle_distance_sq(p, v2, v3, apex))
    distance_sq = wp.min(distance_sq, _point_triangle_distance_sq(p, v3, v0, apex))
    distance_sq = wp.min(distance_sq, _point_triangle_distance_sq(p, v0, v2, v1))
    distance_sq = wp.min(distance_sq, _point_triangle_distance_sq(p, v0, v3, v2))
    distance = wp.sqrt(distance_sq)
    cross_radius = radius * (half_height - p[2]) / (2.0 * half_height)
    if (
        p[2] >= -half_height
        and p[2] <= half_height
        and wp.abs(p[0]) <= cross_radius
        and wp.abs(p[1]) <= cross_radius
    ):
        return -distance
    return distance


@wp.func
def shape_local_sdf(shape: int, p: wp.vec3, half_extents: wp.vec3, radius: float, height: float):
    """Exact Euclidean signed distance in shape-local space."""
    if shape == 0:  # Sphere
        return wp.length(p) - radius
    if shape == 1:  # Cylinder (axis z)
        dr = wp.length(wp.vec3(p[0], p[1], 0.0)) - radius
        dz = wp.abs(p[2]) - height
        outside = wp.vec3(wp.max(dr, 0.0), wp.max(dz, 0.0), 0.0)
        return wp.min(wp.max(dr, dz), 0.0) + wp.length(outside)
    if shape == 2:  # Box
        q = wp.vec3(
            wp.abs(p[0]) - half_extents[0],
            wp.abs(p[1]) - half_extents[1],
            wp.abs(p[2]) - half_extents[2],
        )
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        return wp.length(outside) + inside
    if shape == 3:  # Pyramid (square base at z=-height, apex at z=+height)
        return _pyramid_sdf(p, radius, height)
    if shape == 4:  # Cone (base at z=-height radius, apex z=+height)
        return _cone_sdf(p, radius, height)
    # Plane: half space z < 0.
    return p[2]


@wp.func
def shape_sdf_world(s: SourceShape, pw: wp.vec3):
    pl = wp.transform_point(s.inv_xform, pw)
    return shape_local_sdf(s.shape, pl, s.half_extents, s.radius, s.height)


@wp.func
def shape_point_velocity(s: SourceShape, pw: wp.vec3):
    """Velocity of the shape (rigid motion) at world point pw."""
    center = wp.transform_get_translation(s.xform)
    return s.linear_velocity + wp.cross(s.angular_velocity, pw - center)


@wp.func
def shape_force_velocity(s: SourceShape, pw: wp.vec3):
    """Emission/force velocity contribution at world point pw."""
    if s.force_type == 0:
        return s.velocity
    center = wp.transform_get_translation(s.xform)
    rel = pw - center
    if s.force_type == 1:  # Directional
        return s.velocity + s.force_dir * s.force_strength
    if s.force_type == 2:  # Omnidirectional (outward)
        length = wp.length(rel)
        if length > 1.0e-8:
            return s.velocity + rel * (s.force_strength / length)
        return s.velocity
    if s.force_type == 3:  # Spin around force_dir axis
        axis = s.force_dir
        if wp.length(axis) < 1.0e-8:
            axis = wp.vec3(0.0, 0.0, 1.0)
        axis = wp.normalize(axis)
        return s.velocity + wp.cross(axis, rel) * s.force_strength
    # Radial (inward)
    length = wp.length(rel)
    if length > 1.0e-8:
        return s.velocity - rel * (s.force_strength / length)
    return s.velocity


@wp.func
def shape_force_acceleration(s: SourceShape, pw: wp.vec3):
    """Acceleration from an analytic FORCE region at world point ``pw``.

    Unlike ``shape_force_velocity`` (particle-emission initial velocity), this
    deliberately excludes ``s.velocity`` and is integrated as acceleration in
    the solver's forces phase, after P2G has established valid face weights.
    """
    center = wp.transform_get_translation(s.xform)
    rel = pw - center
    if s.force_type == 1:  # Directional
        return s.force_dir * s.force_strength
    if s.force_type == 2:  # Omnidirectional (outward)
        length = wp.length(rel)
        if length > 1.0e-8:
            return rel * (s.force_strength / length)
        return wp.vec3(0.0)
    if s.force_type == 3:  # Spin around force_dir axis
        axis = s.force_dir
        if wp.length(axis) < 1.0e-8:
            axis = wp.vec3(0.0, 0.0, 1.0)
        return wp.cross(wp.normalize(axis), rel) * s.force_strength
    if s.force_type == 4:  # Radial (inward)
        length = wp.length(rel)
        if length > 1.0e-8:
            return -rel * (s.force_strength / length)
    return wp.vec3(0.0)


@wp.func
def mesh_source_force_velocity(source: MeshSource, pw: wp.vec3):
    """Emission/force velocity contribution for an evaluated mesh source."""
    if source.force_type == 0:
        return source.velocity
    rel = pw - source.force_center
    if source.force_type == 1:  # Directional
        return source.velocity + source.force_dir * source.force_strength
    if source.force_type == 2:  # Omnidirectional (outward)
        length = wp.length(rel)
        if length > 1.0e-8:
            return source.velocity + rel * (source.force_strength / length)
        return source.velocity
    if source.force_type == 3:  # Spin around force_dir axis
        axis = source.force_dir
        if wp.length(axis) < 1.0e-8:
            axis = wp.vec3(0.0, 0.0, 1.0)
        return source.velocity + wp.cross(wp.normalize(axis), rel) * source.force_strength
    # Radial (inward)
    length = wp.length(rel)
    if length > 1.0e-8:
        return source.velocity - rel * (source.force_strength / length)
    return source.velocity


# Analytic FORCE regions: force-stage application to valid MAC faces


@wp.kernel
def k_source_shape_forces(
    shapes: wp.array(dtype=SourceShape),
    num_shapes: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    wu: wp.array(dtype=float),
    wv: wp.array(dtype=float),
    ww: wp.array(dtype=float),
    dt: float,
):
    """Apply analytic FORCE objects after P2G, only to valid MAC faces.

    Scene FORCE objects are represented as source shapes with ``mode=FIELD``
    and ``emit_rate=0`` for ABI compatibility, but their velocity influence
    belongs to the forces phase. Applying it here avoids the old ordering bug
    where sourcing wrote unweighted MAC values that P2G subsequently divided
    or ignored.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)

    for axis in range(3):
        valid = wp.where(axis == 0, wu[tid], wp.where(axis == 1, wv[tid], ww[tid]))
        if valid <= 0.0:
            continue
        point = origin + wp.vec3(
            (float(ijk[0]) + wp.where(axis == 0, 0.0, 0.5)) * voxel_size,
            (float(ijk[1]) + wp.where(axis == 1, 0.0, 0.5)) * voxel_size,
            (float(ijk[2]) + wp.where(axis == 2, 0.0, 0.5)) * voxel_size,
        )
        acceleration = wp.vec3(0.0)
        for si in range(num_shapes):
            shape = shapes[si]
            if shape.enabled == 0 or shape.stype != 0 or shape.mode != 0 or shape.force_type == 0:
                continue
            if shape_sdf_world(shape, point) > shape.feather:
                continue
            acceleration += shape_force_acceleration(shape, point)
        if axis == 0:
            u[tid] = u[tid] + acceleration[0] * dt
        elif axis == 1:
            v[tid] = v[tid] + acceleration[1] * dt
        else:
            w[tid] = w[tid] + acceleration[2] * dt


register_fn(
    k_source_shape_forces,
    "fluidSimSourceShapeForces",
    notes="Port-internal: apply analytic FORCE regions after P2G.",
)


# fluidSimSourceShapes: collisions, sinks, activation, and FIELD compatibility


@wp.kernel
def k_source_shapes(
    shapes: wp.array(dtype=SourceShape),
    num_shapes: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    dt: float,
    voxel_type: wp.array(dtype=wp.int32),
    collision_sdf: wp.array(dtype=float),
    collision_vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
):
    """Apply source-shape grid roles other than liquid emission.

    FIELD sources are skipped here: they are gathered as compiled field plans
    (pressure by default) and applied by ``source_execution``, so this kernel
    never creates liquid, sets tracking bits, or writes MAC
    velocity/temperature/viscosity for them. Analytic FORCE objects use
    ``k_source_shape_forces`` later in the forces phase.

    Collision: min into collision SDF + write collider velocity.
    Sink: set SINK bit.
    Activation/CameraFrustum: set ACTIVE bit.
    """
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)

    vt = voxel_type[tid]
    csdf = collision_sdf[tid]
    cvel = collision_vel[tid]
    vel_u = u[tid]
    vel_v = v[tid]
    vel_w = w[tid]
    temp = temperature[tid]
    visc = viscosity[tid]

    for si in range(num_shapes):
        s = shapes[si]
        if s.enabled == 0:
            continue
        d = shape_sdf_world(s, pw)
        if s.stype == 1:
            # Collision shape: contributes to collider SDF regardless of mode.
            if d < csdf:
                csdf = d
            if d < voxel_size:
                cvel = shape_point_velocity(s, pw)
            continue
        if d > s.feather:
            continue
        if s.stype == 0 and s.mode == 0:
            # FIELD mode creates no liquid; its pressure/field write is a
            # compiled plan applied by source_execution. FORCE-role velocity
            # influence is applied after P2G by k_source_shape_forces.
            continue
        elif s.stype == 2:
            vt |= C.VOXEL_SINK
        elif s.stype == 3 or s.stype == 4:
            vt |= C.VOXEL_ACTIVE

    voxel_type[tid] = vt
    collision_sdf[tid] = csdf
    collision_vel[tid] = cvel
    u[tid] = vel_u
    v[tid] = vel_v
    w[tid] = vel_w
    temperature[tid] = temp
    viscosity[tid] = visc


register_fn(
    k_source_shapes,
    "fluidSimSourceShapes",
    notes="Analytic collision/sink/activation; FIELD pressure deferred.",
)


# fluidSimSourceShapesParticlesAttributes: particle-mode emission


@wp.kernel
def k_source_shapes_particle_demand(
    shapes: wp.array(dtype=SourceShape),
    num_shapes: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
    demand: wp.array(dtype=wp.int32),
):
    """Per-voxel particle count :func:`k_source_shapes_particles` will emit.

    Predicate-identical to the commit pass. An exclusive scan of ``demand``
    gives every voxel its own slot range, so emitted particles land in voxel
    order instead of thread-arrival order (see k_reserve_owned_point_slots
    for why arrival order makes the whole bake irreproducible).
    """
    tid = wp.tid()
    demand[tid] = 0
    if emission_state[1] == 0 or tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    have = voxel_counts[tid]
    total = int(0)
    for si in range(num_shapes):
        s = shapes[si]
        if s.enabled == 0 or s.stype != 0 or s.mode != 1:
            continue
        if s.emission_mode == 0 and s.armed == 0:
            continue
        if shape_sdf_world(s, pw) > 0.0:
            continue
        want = int(s.emit_rate)
        if want > have:
            total += want - have
    demand[tid] = total


@wp.kernel
def k_reserve_scanned_slots(
    demand: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    n: int,
    capacity: int,
    emission_state: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
):
    """Reserve one contiguous block for a scanned per-voxel emission batch."""
    if emission_state[1] == 0 or n <= 0:
        slot_base[0] = 0
        slot_limit[0] = 0
        return
    base = count[0]
    admitted = wp.min(offsets[n - 1] + demand[n - 1], wp.max(0, capacity - base))
    slot_base[0] = base
    slot_limit[0] = admitted
    count[0] = base + admitted


@wp.kernel
def k_source_shapes_particles(
    shapes: wp.array(dtype=SourceShape),
    num_shapes: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
    seed: int,
    color_enabled: int,
    source_id_enabled: int,
    offsets: wp.array(dtype=wp.int32),
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    product_color: wp.array(dtype=wp.vec4),
    product_source_id: wp.array(dtype=wp.int32),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Emit particles inside particle-mode source shapes after batch admission.

    Voxel ``tid`` writes slots ``slot_base + offsets[tid] + k`` from the block
    :func:`k_reserve_scanned_slots` reserved, so the layout is a function of
    the grid alone. The admission gate already guarantees the batch fits, so
    the short-reservation branch is a defensive guard.
    """
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    local = offsets[tid]

    for si in range(num_shapes):
        s = shapes[si]
        if s.enabled == 0 or s.stype != 0 or s.mode != 1:
            continue
        # Initial-fill sources emit only while armed; the host disarms them
        # after the firing substep so the region fills exactly once per run.
        if s.emission_mode == 0 and s.armed == 0:
            continue
        d = shape_sdf_world(s, pw)
        if d > 0.0:
            continue
        have = voxel_counts[tid]
        want = int(s.emit_rate)
        if want <= have:
            continue
        emit_n = want - have
        state = wp.rand_init(seed, tid * 31 + si)
        for _emit_index in range(emit_n):
            if local >= slot_limit[0]:
                emission_state[3] = 1
                return
            slot = slot_base[0] + local
            local += 1
            jitter = wp.vec3(
                (wp.randf(state) - 0.5) * voxel_size,
                (wp.randf(state) - 0.5) * voxel_size,
                (wp.randf(state) - 0.5) * voxel_size,
            )
            p = pw + jitter
            positions[slot] = p
            velocities[slot] = shape_force_velocity(s, p)
            age[slot] = 0.0
            if s.use_lifespan != 0:
                life[slot] = s.life_min + wp.randf(state) * (s.life_max - s.life_min)
            else:
                life[slot] = 1.0e30
            temperature[slot] = s.temperature
            viscosity[slot] = s.viscosity
            density[slot] = s.density
            if color_enabled != 0:
                product_color[slot] = s.color
            if source_id_enabled != 0:
                product_source_id[slot] = s.source_numeric_id
            time_offset[slot] = 0.0
            alive[slot] = 1
            wp.atomic_add(emission_state, 2, 1)


register_fn(
    k_source_shapes_particles,
    "fluidSimSourceShapesParticlesAttributes",
    notes="Particle-mode emission with per-voxel top-up to emit_rate.",
)


# Evaluated-mesh surface sources and solid/surface sinks


@wp.func
def _watertight_mesh_contains(mesh_id: wp.uint64, point: wp.vec3, query_radius: float):
    """Orientation-independent inside test for validated watertight meshes."""
    query = wp.mesh_query_point_sign_parity(mesh_id, point, query_radius, 3, 0.1)
    return query.result and query.sign < 0.0


@wp.kernel
def k_mesh_source_field(
    mesh_id: wp.uint64,
    source: MeshSource,
    query_radius: float,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
):
    """Stamp a triangle shell source or a solid/surface mesh sink."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME or source.enabled == 0:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    p = voxel_center_world(ijk, origin, voxel_size)
    half_diagonal = 0.8660254037844386 * voxel_size
    shell = wp.max(source.surface_thickness, 0.0) + half_diagonal
    member = 0

    if source.stype == 2 and source.sink_region_mode == 0:
        if _watertight_mesh_contains(mesh_id, p, query_radius):
            member = 1
    else:
        query = wp.mesh_query_point_no_sign(mesh_id, p, shell)
        if query.result:
            closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            if wp.length(p - closest) <= shell:
                member = 1

    if member == 0:
        return
    vt = voxel_type[tid]
    if source.stype == 2:
        voxel_type[tid] = vt | C.VOXEL_SINK
    elif source.stype == 0 and source.mode == 0:
        # Compatibility no-op. Original FIELD mode is pressure-only and does
        # not create liquid or write MAC/attributes. Mesh FIELD sourcing has
        # no original counterpart and is not exposed in the normal UI.
        return


register_fn(
    k_mesh_source_field,
    "fluidSimMeshSourceField",
    notes="Evaluated-triangle shell sourcing and winding-number mesh sinks.",
)


@wp.kernel
def k_mesh_source_particles(
    mesh_id: wp.uint64,
    source: MeshSource,
    query_radius: float,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    voxel_counts: wp.array(dtype=wp.int32),
    seed: int,
    color_enabled: int,
    source_id_enabled: int,
    capacity: int,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    product_color: wp.array(dtype=wp.vec4),
    product_source_id: wp.array(dtype=wp.int32),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Top up intersecting voxels after whole-batch admission."""
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if (
        tid >= num_leaves[0] * C.LEAF_VOLUME
        or source.enabled == 0
        or source.stype != 0
        or source.mode != 1
        or (source.emission_mode == 0 and source.armed == 0)
    ):
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, voxel_size)
    half_diagonal = 0.8660254037844386 * voxel_size
    shell = wp.max(source.surface_thickness, 0.0) + half_diagonal
    projection_radius = wp.max(query_radius, shell + half_diagonal)
    center_query = wp.mesh_query_point_no_sign(mesh_id, center, shell)
    if not center_query.result:
        return
    center_closest = wp.mesh_eval_position(
        mesh_id, center_query.face, center_query.u, center_query.v
    )
    if wp.length(center - center_closest) > shell:
        return

    have = voxel_counts[tid]
    want = int(source.emit_rate)
    if want <= have:
        return
    state = wp.rand_init(seed, tid * 31)
    for _emit_index in range(want - have):
        jitter = wp.vec3(
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
        )
        query = wp.mesh_query_point_no_sign(mesh_id, center + jitter, projection_radius)
        if not query.result:
            continue
        slot = wp.atomic_add(count, 0, 1)
        if slot >= capacity:
            wp.atomic_sub(count, 0, 1)
            emission_state[3] = 1
            return
        position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        positions[slot] = position
        velocities[slot] = mesh_source_force_velocity(source, position)
        age[slot] = 0.0
        if source.use_lifespan != 0:
            life[slot] = source.life_min + wp.randf(state) * (source.life_max - source.life_min)
        else:
            life[slot] = 1.0e30
        temperature[slot] = source.temperature
        viscosity[slot] = source.viscosity
        density[slot] = source.density
        if color_enabled != 0:
            product_color[slot] = source.color
        if source_id_enabled != 0:
            product_source_id[slot] = source.source_numeric_id
        time_offset[slot] = 0.0
        alive[slot] = 1
        wp.atomic_add(emission_state, 2, 1)


register_fn(
    k_mesh_source_particles,
    "fluidSimMeshSourceParticles",
    notes="Triangle-projected mesh emission with per-voxel top-up.",
)


# fluidSimSourceHeight: height-field sourcing (flat level or profile grid)


@wp.kernel
def k_source_height(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    base_height: float,
    back_mode: int,  # 0 Infinite, 1 Flat, 2 Profile
    profile: wp.array2d(dtype=float),
    profile_res: wp.vec2i,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    velocity: wp.vec3,
    require_admission: int,
    emission_state: wp.array(dtype=wp.int32),
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Fill voxels below the height surface, guarded when particles emit."""
    if require_admission != 0 and emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)

    h = base_height
    if back_mode == 2 and profile_res[0] > 1 and profile_res[1] > 1:
        fx = (pw[0] - domain_min[0]) / wp.max(domain_max[0] - domain_min[0], 1.0e-8)
        fy = (pw[1] - domain_min[1]) / wp.max(domain_max[1] - domain_min[1], 1.0e-8)
        fx = wp.clamp(fx, 0.0, 1.0) * float(profile_res[0] - 1)
        fy = wp.clamp(fy, 0.0, 1.0) * float(profile_res[1] - 1)
        x0 = int(wp.floor(fx))
        y0 = int(wp.floor(fy))
        x1 = wp.min(x0 + 1, profile_res[0] - 1)
        y1 = wp.min(y0 + 1, profile_res[1] - 1)
        tx = fx - float(x0)
        ty = fy - float(y0)
        h00 = profile[x0, y0]
        h10 = profile[x1, y0]
        h01 = profile[x0, y1]
        h11 = profile[x1, y1]
        h = base_height + (
            h00 * (1.0 - tx) * (1.0 - ty)
            + h10 * tx * (1.0 - ty)
            + h01 * (1.0 - tx) * ty
            + h11 * tx * ty
        )
    # back_mode 0 (Infinite) and 1 (Flat) both use base_height for the level;
    # Infinite refills continuously (handled by re-running each frame).

    if pw[2] <= h:
        vt = voxel_type[tid]
        if (vt & C.VOXEL_COLLISION) == 0:
            voxel_type[tid] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            u[tid] = velocity[0]
            v[tid] = velocity[1]
            w[tid] = velocity[2]


register_fn(
    k_source_height, "fluidSimSourceHeight", notes="Flat / profile-grid height sourcing along +z."
)


# fluidSimSourceResolveType: precedence resolution after all sourcing


@wp.kernel
def k_source_resolve_type(
    voxel_type: wp.array(dtype=wp.int32),
    num_leaves: wp.array(dtype=wp.int32),
):
    """Collision dominates: a collision voxel can hold neither fluid nor sink.
    Sink voxels drop the tracked-source bit so sinks are never re-sourced."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    vt = voxel_type[tid]
    if (vt & C.VOXEL_COLLISION) != 0:
        vt &= ~(C.VOXEL_FLUID | C.VOXEL_SINK | C.VOXEL_SOURCE_REGION_TRACKING)
    if (vt & C.VOXEL_SINK) != 0:
        vt &= ~C.VOXEL_SOURCE_REGION_TRACKING
    voxel_type[tid] = vt


register_fn(
    k_source_resolve_type,
    "fluidSimSourceResolveType",
    notes="Voxel type precedence: collision > sink > fluid.",
)


# fluidSimSourceParticles: append externally provided particles


# Retained for the synthesized snippet-inventory compatibility contract. Live
# direct injection uses the admission-gated owned-packet kernels below.
@wp.kernel
def k_source_particles(
    src_positions: wp.array(dtype=wp.vec3),
    src_velocities: wp.array(dtype=wp.vec3),
    src_count: int,
    default_life: float,
    use_lifespan: int,
    temperature_value: float,
    viscosity_value: float,
    density_value: float,
    capacity: int,
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    first_slot: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= src_count:
        return
    slot = wp.atomic_add(count, 0, 1)
    if slot >= capacity:
        wp.atomic_sub(count, 0, 1)
        return
    if tid == 0:
        first_slot[0] = slot
    positions[slot] = src_positions[tid]
    velocities[slot] = src_velocities[tid]
    age[slot] = 0.0
    if use_lifespan != 0:
        life[slot] = default_life
    else:
        life[slot] = 1.0e30
    temperature[slot] = temperature_value
    viscosity[slot] = viscosity_value
    density[slot] = density_value
    time_offset[slot] = 0.0
    alive[slot] = 1


register_fn(
    k_source_particles,
    "fluidSimSourceParticles",
    notes="Append external particle list (point cloud / particle system).",
)


@wp.kernel
def k_reserve_owned_point_slots(
    src_count: int,
    capacity: int,
    emission_state: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
):
    """Reserve one contiguous slot block for the next owned-point packet.

    Single-threaded on purpose: it replaces the per-point atomic cursor in
    :func:`k_source_owned_points` so a packet lands at ``slot_base + tid``
    instead of in thread-arrival order.  Arrival order permutes the particle
    arrays differently on every run, and both the bin sort and the P2G
    accumulation read those arrays positionally, so the whole bake becomes
    irreproducible.  The block reservation is exact and order-free, so the
    stored layout is a function of the input point order alone.
    """
    if emission_state[1] == 0:
        slot_base[0] = 0
        slot_limit[0] = 0
        return
    base = count[0]
    room = capacity - base
    admitted = src_count
    if admitted > room:
        admitted = room
    if admitted < 0:
        admitted = 0
    slot_base[0] = base
    slot_limit[0] = admitted
    count[0] = base + admitted


@wp.kernel
def k_source_owned_points(
    src_positions: wp.array(dtype=wp.vec3),
    src_velocities: wp.array(dtype=wp.vec3),
    src_temperature: wp.array(dtype=float),
    src_viscosity: wp.array(dtype=float),
    src_density: wp.array(dtype=float),
    src_lifespan: wp.array(dtype=float),
    src_count: int,
    has_velocity: int,
    has_temperature: int,
    has_viscosity: int,
    has_density: int,
    has_lifespan: int,
    default_velocity: wp.vec3,
    default_temperature: float,
    default_viscosity: float,
    default_density: float,
    default_color: wp.vec4,
    default_source_id: int,
    color_enabled: int,
    source_id_enabled: int,
    default_lifespan: float,
    use_lifespan: int,
    capacity: int,
    slot_base: wp.array(dtype=wp.int32),
    slot_limit: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    product_color: wp.array(dtype=wp.vec4),
    product_source_id: wp.array(dtype=wp.int32),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Direct point append preserving every owned per-point attribute.

    Placement is ``slot_base + tid``, from the block
    :func:`k_reserve_owned_point_slots` reserved just before this launch, so a
    packet occupies a contiguous run in source-point order rather than in
    thread-arrival order.  The admission gate already guarantees the whole
    batch fits, so the short-reservation branch below is a defensive guard.
    """
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= src_count:
        return
    if tid >= slot_limit[0]:
        result[1] = 1
        emission_state[3] = 1
        return
    slot = slot_base[0] + tid
    positions[slot] = src_positions[tid]
    velocities[slot] = default_velocity
    if has_velocity != 0:
        velocities[slot] = src_velocities[tid]
    age[slot] = 0.0
    life[slot] = 1.0e30
    if use_lifespan != 0:
        life[slot] = default_lifespan
    if has_lifespan != 0:
        life[slot] = src_lifespan[tid]
    temperature[slot] = default_temperature
    if has_temperature != 0:
        temperature[slot] = src_temperature[tid]
    viscosity[slot] = default_viscosity
    if has_viscosity != 0:
        viscosity[slot] = src_viscosity[tid]
    density[slot] = default_density
    if has_density != 0:
        density[slot] = src_density[tid]
    if color_enabled != 0:
        product_color[slot] = default_color
    if source_id_enabled != 0:
        product_source_id[slot] = default_source_id
    time_offset[slot] = 0.0
    alive[slot] = 1
    wp.atomic_add(result, 0, 1)
    wp.atomic_add(emission_state, 2, 1)


# Particle attribute sourcing for a contiguous slot range [start, start+n)


@wp.kernel
def k_source_particle_scalar_field_attribute(
    field: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    positions: wp.array(dtype=wp.vec3),
    start: int,
    n: int,
    attr: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= n:
        return
    i = start + tid
    vp = world_to_voxel(positions[i], origin, voxel_size)
    attr[i] = sample_scalar_trilinear(field, leaf_table, voxels_dim, leaves_dim, vp, 0.0)


register_fn(
    k_source_particle_scalar_field_attribute,
    "fluidSimSourceParticleScalarFieldAttribute",
    notes="Sample scalar field into per-particle attribute at spawn.",
)


@wp.kernel
def k_source_particle_vector_field_attribute(
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    positions: wp.array(dtype=wp.vec3),
    start: int,
    n: int,
    out_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= n:
        return
    i = start + tid
    vp = world_to_voxel(positions[i], origin, voxel_size)
    x = sample_scalar_trilinear(u, leaf_table, voxels_dim, leaves_dim, vp, 0.0)
    y = sample_scalar_trilinear(v, leaf_table, voxels_dim, leaves_dim, vp, 0.0)
    z = sample_scalar_trilinear(w, leaf_table, voxels_dim, leaves_dim, vp, 0.0)
    out_vec[i] = wp.vec3(x, y, z)


register_fn(
    k_source_particle_vector_field_attribute,
    "fluidSimSourceParticleVectorFieldAttribute",
    notes="Sample vector field into per-particle vec attribute at spawn.",
)


@wp.kernel
def k_source_particle_scalar_constant(
    value: float,
    start: int,
    n: int,
    attr: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= n:
        return
    attr[start + tid] = value


register_fn(
    k_source_particle_scalar_constant,
    "fluidSimSourceParticleScalarFloatAttribute",
    "fluidSimSourceParticleScalarHalfAttribute",
    notes="Constant float/half attribute fill; half stored as f32 attr.",
)


@wp.kernel
def k_source_particle_integer_constant(
    value: int,
    start: int,
    n: int,
    attr: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= n:
        return
    attr[start + tid] = value


register_fn(
    k_source_particle_integer_constant,
    "fluidSimSourceParticleScalarIntegerAttribute",
    notes="Constant integer attribute fill.",
)


# core field-to-field sourcing


@wp.kernel
def k_core_source_float(
    src: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    blend_mode: int,
    t: float,
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    dst[tid] = blend_scalar(dst[tid], src[tid], blend_mode, t)


register_fn(k_core_source_float, "coreSourceFloat", notes="Blend scalar field into scalar field.")


@wp.kernel
def k_core_source_vector(
    src: wp.array(dtype=wp.vec3),
    num_leaves: wp.array(dtype=wp.int32),
    blend_mode: int,
    t: float,
    dst: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    dst[tid] = blend_vec(dst[tid], src[tid], blend_mode, t)


register_fn(k_core_source_vector, "coreSourceVector", notes="Blend vec3 field into vec3 field.")


@wp.kernel
def k_core_source_staggered_vector(
    src_u: wp.array(dtype=float),
    src_v: wp.array(dtype=float),
    src_w: wp.array(dtype=float),
    num_leaves: wp.array(dtype=wp.int32),
    blend_mode: int,
    t: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    u[tid] = blend_scalar(u[tid], src_u[tid], blend_mode, t)
    v[tid] = blend_scalar(v[tid], src_v[tid], blend_mode, t)
    w[tid] = blend_scalar(w[tid], src_w[tid], blend_mode, t)


register_fn(
    k_core_source_staggered_vector,
    "coreSourceStaggeredVector",
    notes="Blend staggered MAC field into MAC field.",
)


@wp.kernel
def k_core_source_float_particle_attribute(
    positions: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=float),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    accum: wp.array(dtype=float),
    weight: wp.array(dtype=float),
):
    """Splat a per-particle scalar attribute to the grid (accumulate)."""
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    flat = voxel_flat_index(
        leaf_table,
        voxels_dim,
        leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    wp.atomic_add(accum, flat, attr[tid])
    wp.atomic_add(weight, flat, 1.0)


register_fn(
    k_core_source_float_particle_attribute,
    "coreSourceFloatParticleAttribute",
    notes="Particle scalar attr -> grid accumulate; normalize separately.",
)


@wp.kernel
def k_core_source_vector_particle_attribute(
    positions: wp.array(dtype=wp.vec3),
    attr: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=wp.int32),
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    accum: wp.array(dtype=wp.vec3),
    weight: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= count[0]:
        return
    vp = world_to_voxel(positions[tid], origin, voxel_size)
    flat = voxel_flat_index(
        leaf_table,
        voxels_dim,
        leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    wp.atomic_add(accum, flat, attr[tid])
    wp.atomic_add(weight, flat, 1.0)


register_fn(
    k_core_source_vector_particle_attribute,
    "coreSourceVectorParticleAttribute",
    notes="Particle vec3 attr -> grid accumulate; normalize separately.",
)


# Volume (NanoVDB via wp.Volume) sourcing. OpenVDB names are aliases: the
# Blender bridge converts OpenVDB grids to NanoVDB volumes on load.


@wp.kernel
def k_source_volume_float(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    blend_mode: int,
    t: float,
    scale: float,
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    val = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR) * scale
    dst[tid] = blend_scalar(dst[tid], val, blend_mode, t)


register_fn(
    k_source_volume_float,
    "coreSourceNanoVDBFloat",
    "coreSourceOpenVDBFloat",
    notes="Sample wp.Volume float grid into scalar field.",
)


@wp.kernel
def k_source_volume_vector(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    blend_mode: int,
    t: float,
    scale: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    val = wp.volume_sample_v(volume, uvw, wp.Volume.LINEAR) * scale
    old = wp.vec3(u[tid], v[tid], w[tid])
    new = blend_vec(old, val, blend_mode, t)
    u[tid] = new[0]
    v[tid] = new[1]
    w[tid] = new[2]


register_fn(
    k_source_volume_vector,
    "coreSourceNanoVDBVector",
    notes="Sample wp.Volume vec3 grid into velocity components.",
)


@wp.kernel
def k_source_volume_vector_component(
    volume: wp.uint64,
    component: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    blend_mode: int,
    t: float,
    scale: float,
    dst: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    val = wp.volume_sample_v(volume, uvw, wp.Volume.LINEAR)
    dst[tid] = blend_scalar(dst[tid], val[component] * scale, blend_mode, t)


register_fn(
    k_source_volume_vector_component,
    "coreSourceOpenVDBVectorComponent",
    notes="Sample one component of a vec3 volume into a scalar field.",
)


@wp.kernel
def k_source_volume_type_fog(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    fog_threshold: float,
    emission_velocity: wp.vec3,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Mark voxels as fluid where fog density exceeds the threshold."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    fog = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if fog > fog_threshold:
        vt = voxel_type[tid]
        if (vt & C.VOXEL_COLLISION) == 0:
            voxel_type[tid] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            u[tid] = emission_velocity[0]
            v[tid] = emission_velocity[1]
            w[tid] = emission_velocity[2]


register_fn(
    k_source_volume_type_fog,
    "coreSourceNanoVDBTypeFog",
    "coreSourceOpenVDBType",
    notes="Fog density -> fluid voxel marking with emission velocity.",
)


@wp.kernel
def k_source_volume_type_fog_with_temperature(
    volume: wp.uint64,
    temperature_volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    fog_threshold: float,
    emission_velocity: wp.vec3,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    fog = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if fog > fog_threshold:
        vt = voxel_type[tid]
        if (vt & C.VOXEL_COLLISION) == 0:
            voxel_type[tid] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            u[tid] = emission_velocity[0]
            v[tid] = emission_velocity[1]
            w[tid] = emission_velocity[2]
            tuvw = wp.volume_world_to_index(temperature_volume, pw)
            temperature[tid] = wp.volume_sample_f(temperature_volume, tuvw, wp.Volume.LINEAR)


register_fn(
    k_source_volume_type_fog_with_temperature,
    "coreSourceNanoVDBTypeFogWithTemperature",
    "coreSourceOpenVDBTypeWithTemperature",
    notes="Fog sourcing that also samples a temperature volume.",
)


@wp.kernel
def k_source_volume_mask_fog(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    fog_threshold: float,
    mask_bit: int,
    voxel_type: wp.array(dtype=wp.int32),
):
    """Set a voxel-type mask bit (e.g. Active or Sink) where fog is present."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    fog = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if fog > fog_threshold:
        # tid indexes a unique voxel, so a plain RMW is race-free here.
        voxel_type[tid] = voxel_type[tid] | mask_bit


register_fn(
    k_source_volume_mask_fog,
    "coreSourceNanoVDBMaskFog",
    "coreSourceOpenVDBMask",
    notes="Fog density -> voxel-type mask bit.",
)


@wp.kernel
def k_source_volume_type_sdf(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    emission_velocity: wp.vec3,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Mark voxels as fluid where an SDF volume is negative (inside)."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    d = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if d < 0.0:
        vt = voxel_type[tid]
        if (vt & C.VOXEL_COLLISION) == 0:
            voxel_type[tid] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            u[tid] = emission_velocity[0]
            v[tid] = emission_velocity[1]
            w[tid] = emission_velocity[2]


register_fn(
    k_source_volume_type_sdf,
    "coreSourceNanoVDBTypeSDF",
    notes="SDF volume inside test -> fluid voxel marking.",
)


@wp.kernel
def k_source_volume_type_sdf_with_temperature(
    volume: wp.uint64,
    temperature_volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    emission_velocity: wp.vec3,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    d = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if d < 0.0:
        vt = voxel_type[tid]
        if (vt & C.VOXEL_COLLISION) == 0:
            voxel_type[tid] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            u[tid] = emission_velocity[0]
            v[tid] = emission_velocity[1]
            w[tid] = emission_velocity[2]
            tuvw = wp.volume_world_to_index(temperature_volume, pw)
            temperature[tid] = wp.volume_sample_f(temperature_volume, tuvw, wp.Volume.LINEAR)


register_fn(
    k_source_volume_type_sdf_with_temperature,
    "coreSourceNanoVDBTypeSDFWithTemperature",
    notes="SDF volume sourcing with temperature volume sampling.",
)


@wp.kernel
def k_source_volume_mask_sdf(
    volume: wp.uint64,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    mask_bit: int,
    voxel_type: wp.array(dtype=wp.int32),
):
    """Set a voxel-type mask bit where an SDF volume is negative."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    uvw = wp.volume_world_to_index(volume, pw)
    d = wp.volume_sample_f(volume, uvw, wp.Volume.LINEAR)
    if d < 0.0:
        voxel_type[tid] = voxel_type[tid] | mask_bit


register_fn(
    k_source_volume_mask_sdf,
    "coreSourceNanoVDBMaskSDF",
    notes="SDF volume inside test -> voxel-type mask bit.",
)


@wp.kernel
def k_apply_compiled_source(
    operation: int,
    coverage: wp.array(dtype=float),
    scalar_values: wp.array(dtype=float),
    vector_values: wp.array(dtype=wp.vec3),
    source_leaf_coords: wp.array(dtype=wp.vec3i),
    source_leaf_count: int,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    constant_scalar: float,
    constant_vector: wp.vec3,
    blend_mode: int,
    blend_amount: float,
    dt: float,
    voxel_type: wp.array(dtype=wp.int32),
    pressure: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Apply one compiled flat-leaf source without geometry branching."""
    tid = wp.tid()
    if tid >= source_leaf_count * C.LEAF_VOLUME:
        return
    cov = coverage[tid]
    if cov <= 0.0:
        return
    source_slot = tid // C.LEAF_VOLUME
    local = tid - source_slot * C.LEAF_VOLUME
    lc = source_leaf_coords[source_slot]
    ijk = wp.vec3i(
        (lc[0] << C.LEAF_SHIFT) + (local & C.LEAF_MASK),
        (lc[1] << C.LEAF_SHIFT) + ((local >> C.LEAF_SHIFT) & C.LEAF_MASK),
        (lc[2] << C.LEAF_SHIFT) + ((local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK),
    )
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return
    vt = voxel_type[flat]
    if operation == 3:  # COLLISION, highest precedence
        voxel_type[flat] = (vt | C.VOXEL_COLLISION) & ~(
            C.VOXEL_SINK | C.VOXEL_FLUID | C.VOXEL_SOURCE_REGION_TRACKING
        )
        return
    if (vt & C.VOXEL_COLLISION) != 0:
        return
    if operation == 1:  # SINK
        voxel_type[flat] = (vt | C.VOXEL_SINK) & ~(C.VOXEL_FLUID | C.VOXEL_SOURCE_REGION_TRACKING)
        return
    if operation == 4:  # ACTIVATION
        voxel_type[flat] = vt | C.VOXEL_ACTIVE
        return
    if (vt & C.VOXEL_SINK) != 0:
        return
    if operation == 0:  # LIQUID region field; particles are appended separately
        voxel_type[flat] = vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
        u[flat] = constant_vector[0]
        v[flat] = constant_vector[1]
        w[flat] = constant_vector[2]
        return
    source_scalar = constant_scalar
    if scalar_values.shape[0] > tid:
        source_scalar = scalar_values[tid]
    source_vector = constant_vector
    if vector_values.shape[0] > tid:
        source_vector = vector_values[tid]
    amount = blend_amount * cov
    if operation == 5:  # PRESSURE: FIELD's only original write
        pressure[flat] = blend_scalar(
            pressure[flat],
            source_scalar,
            blend_mode,
            wp.where(blend_mode == 0, amount * dt, amount),
        )
    elif operation == 6:  # SCALAR_FIELD defaults to temperature
        temperature[flat] = blend_scalar(
            temperature[flat],
            source_scalar,
            blend_mode,
            wp.where(blend_mode == 0, amount * dt, amount),
        )
    elif operation == 7 or operation == 8:  # vector/staggered velocity
        old = wp.vec3(u[flat], v[flat], w[flat])
        value = blend_vec(
            old, source_vector, blend_mode, wp.where(blend_mode == 0, amount * dt, amount)
        )
        u[flat] = value[0]
        v[flat] = value[1]
        w[flat] = value[2]


@wp.kernel
def k_apply_compiled_staggered_source(
    coverage: wp.array(dtype=float),
    source_u: wp.array(dtype=float),
    source_v: wp.array(dtype=float),
    source_w: wp.array(dtype=float),
    source_leaf_coords: wp.array(dtype=wp.vec3i),
    source_leaf_count: int,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    constant_vector: wp.vec3,
    use_face_values: int,
    blend_mode: int,
    blend_amount: float,
    dt: float,
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Apply independently sampled x/y/z face values to the MAC fields."""
    tid = wp.tid()
    if tid >= source_leaf_count * C.LEAF_VOLUME:
        return
    cov = coverage[tid]
    if cov <= 0.0:
        return
    source_slot = tid // C.LEAF_VOLUME
    local = tid - source_slot * C.LEAF_VOLUME
    lc = source_leaf_coords[source_slot]
    ijk = wp.vec3i(
        (lc[0] << C.LEAF_SHIFT) + (local & C.LEAF_MASK),
        (lc[1] << C.LEAF_SHIFT) + ((local >> C.LEAF_SHIFT) & C.LEAF_MASK),
        (lc[2] << C.LEAF_SHIFT) + ((local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK),
    )
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return
    value_u = constant_vector[0]
    value_v = constant_vector[1]
    value_w = constant_vector[2]
    if use_face_values != 0:
        value_u = source_u[tid]
        value_v = source_v[tid]
        value_w = source_w[tid]
    amount = blend_amount * cov
    if blend_mode == 0:
        amount = amount * dt
    u[flat] = blend_scalar(u[flat], value_u, blend_mode, amount)
    v[flat] = blend_scalar(v[flat], value_v, blend_mode, amount)
    w[flat] = blend_scalar(w[flat], value_w, blend_mode, amount)


@wp.kernel
def k_emit_compiled_source_particles(
    coverage: wp.array(dtype=float),
    source_leaf_coords: wp.array(dtype=wp.vec3i),
    source_leaf_count: int,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    origin: wp.vec3,
    voxel_size: float,
    target_per_voxel: int,
    emission_velocity: wp.vec3,
    temperature_value: float,
    viscosity_value: float,
    density_value: float,
    color_value: wp.vec4,
    source_id_value: int,
    color_enabled: int,
    source_id_enabled: int,
    life_value: float,
    use_lifespan: int,
    capacity: int,
    seed: int,
    voxel_counts: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    temperature: wp.array(dtype=float),
    viscosity: wp.array(dtype=float),
    density: wp.array(dtype=float),
    product_color: wp.array(dtype=wp.vec4),
    product_source_id: wp.array(dtype=wp.int32),
    time_offset: wp.array(dtype=float),
    alive: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Top covered voxels up to their target occupancy.

    Each particle is jittered inside its voxel like legacy shape emission;
    stacking them on the centre left them inseparable (relaxation skips
    zero-distance pairs). The random stream is keyed by the transaction
    ``seed``, the voxel's global index and its occupancy before this pass,
    never by slot or thread order, so the emitted positions are
    reproducible even though slots are taken with an atomic cursor.
    """
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= source_leaf_count * C.LEAF_VOLUME or coverage[tid] <= 0.0:
        return
    source_slot = tid // C.LEAF_VOLUME
    local = tid - source_slot * C.LEAF_VOLUME
    lc = source_leaf_coords[source_slot]
    ijk = wp.vec3i(
        (lc[0] << C.LEAF_SHIFT) + (local & C.LEAF_MASK),
        (lc[1] << C.LEAF_SHIFT) + ((local >> C.LEAF_SHIFT) & C.LEAF_MASK),
        (lc[2] << C.LEAF_SHIFT) + ((local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK),
    )
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return
    desired = int(wp.ceil(float(target_per_voxel) * coverage[tid]))
    have = voxel_counts[flat]
    missing = wp.max(0, desired - have)
    center = voxel_center_world(ijk, origin, voxel_size)
    voxel_key = ijk[0] + voxels_dim[0] * (ijk[1] + voxels_dim[1] * ijk[2])
    state = wp.rand_init(seed + have * 7919, voxel_key)
    for sample in range(C.MAX_PARTICLES_PER_VOXEL):
        if sample >= missing:
            break
        slot = wp.atomic_add(count, 0, 1)
        if slot >= capacity:
            wp.atomic_sub(count, 0, 1)
            result[1] = 1
            emission_state[3] = 1
            break
        positions[slot] = center + wp.vec3(
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
        )
        velocities[slot] = emission_velocity
        age[slot] = 0.0
        life[slot] = wp.where(use_lifespan != 0, life_value, 1.0e30)
        temperature[slot] = temperature_value
        viscosity[slot] = viscosity_value
        density[slot] = density_value
        if color_enabled != 0:
            product_color[slot] = color_value
        if source_id_enabled != 0:
            product_source_id[slot] = source_id_value
        time_offset[slot] = 0.0
        alive[slot] = 1
        wp.atomic_add(voxel_counts, flat, 1)
        wp.atomic_add(result, 0, 1)
        wp.atomic_add(emission_state, 2, 1)


@wp.kernel
def k_mark_source_leaves(
    source_leaf_coords: wp.array(dtype=wp.vec3i),
    source_leaf_count: int,
    leaves_dim: wp.vec3i,
    leaf_flags: wp.array(dtype=wp.int32),
):
    """Flag exactly the compiled source leaves for the topology rebuild.

    One thread per leaf coordinate. Writing the constant 1 is
    order-independent, so concurrent duplicates are harmless.
    """
    tid = wp.tid()
    if tid >= source_leaf_count:
        return
    lc = source_leaf_coords[tid]
    if lc[0] < 0 or lc[1] < 0 or lc[2] < 0:
        return
    if lc[0] >= leaves_dim[0] or lc[1] >= leaves_dim[1] or lc[2] >= leaves_dim[2]:
        return
    leaf_flags[leaf_linear_index(lc, leaves_dim)] = 1


@wp.kernel
def k_assign_emitted_uids(
    start: int,
    emitted: int,
    uid_base: int,
    uid: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid < emitted:
        uid[start + tid] = uid_base + tid


@wp.kernel
def k_add_emission_demand(amount: int, emission_state: wp.array(dtype=wp.int32)):
    if wp.tid() == 0 and amount > 0:
        wp.atomic_add(emission_state, 0, amount)


@wp.kernel
def k_admit_emission_batch(
    particle_count: wp.array(dtype=wp.int32),
    capacity: int,
    emission_state: wp.array(dtype=wp.int32),
):
    if wp.tid() == 0:
        available = wp.max(0, capacity - particle_count[0])
        emission_state[4] = available
        emission_state[1] = wp.where(emission_state[0] <= available, 1, 0)


@wp.kernel
def k_count_compiled_source_particles(
    coverage: wp.array(dtype=float),
    source_leaf_coords: wp.array(dtype=wp.vec3i),
    source_leaf_count: int,
    leaf_table: wp.array(dtype=wp.int32),
    voxels_dim: wp.vec3i,
    leaves_dim: wp.vec3i,
    target_per_voxel: int,
    scratch_counts: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Count compiled top-up demand and reserve occupancy only in scratch."""
    tid = wp.tid()
    if tid >= source_leaf_count * C.LEAF_VOLUME or coverage[tid] <= 0.0:
        return
    source_slot = tid // C.LEAF_VOLUME
    local = tid - source_slot * C.LEAF_VOLUME
    lc = source_leaf_coords[source_slot]
    ijk = wp.vec3i(
        (lc[0] << C.LEAF_SHIFT) + (local & C.LEAF_MASK),
        (lc[1] << C.LEAF_SHIFT) + ((local >> C.LEAF_SHIFT) & C.LEAF_MASK),
        (lc[2] << C.LEAF_SHIFT) + ((local >> (2 * C.LEAF_SHIFT)) & C.LEAF_MASK),
    )
    flat = voxel_flat_index(leaf_table, voxels_dim, leaves_dim, ijk)
    if flat < 0:
        return
    desired = int(wp.ceil(float(target_per_voxel) * coverage[tid]))
    missing = wp.max(0, desired - scratch_counts[flat])
    if missing > 0:
        scratch_counts[flat] = scratch_counts[flat] + missing
        wp.atomic_add(emission_state, 0, missing)


@wp.kernel
def k_count_source_shapes_particles(
    shapes: wp.array(dtype=SourceShape),
    num_shapes: int,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    scratch_counts: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Predicate-identical demand pass for analytic and derived height shapes."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)
    have = scratch_counts[tid]
    for si in range(num_shapes):
        s = shapes[si]
        if s.enabled == 0 or s.stype != 0 or s.mode != 1 or (s.emission_mode == 0 and s.armed == 0):
            continue
        if shape_sdf_world(s, pw) > 0.0:
            continue
        want = int(s.emit_rate)
        if want > have:
            # The accepted legacy commit kernel does not update voxel_counts
            # between overlapping shapes; mirror that behavior exactly.
            wp.atomic_add(emission_state, 0, want - have)


@wp.kernel
def k_count_mesh_source_particles(
    mesh_id: wp.uint64,
    source: MeshSource,
    query_radius: float,
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    scratch_counts: wp.array(dtype=wp.int32),
    seed: int,
    emission_state: wp.array(dtype=wp.int32),
):
    """Run the same projected-spawn predicates as the mesh commit pass."""
    tid = wp.tid()
    if (
        tid >= num_leaves[0] * C.LEAF_VOLUME
        or source.enabled == 0
        or source.stype != 0
        or source.mode != 1
        or (source.emission_mode == 0 and source.armed == 0)
    ):
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, voxel_size)
    half_diagonal = 0.8660254037844386 * voxel_size
    shell = wp.max(source.surface_thickness, 0.0) + half_diagonal
    projection_radius = wp.max(query_radius, shell + half_diagonal)
    center_query = wp.mesh_query_point_no_sign(mesh_id, center, shell)
    if not center_query.result:
        return
    center_closest = wp.mesh_eval_position(
        mesh_id, center_query.face, center_query.u, center_query.v
    )
    if wp.length(center - center_closest) > shell:
        return
    have = scratch_counts[tid]
    want = int(source.emit_rate)
    if want <= have:
        return
    state = wp.rand_init(seed, tid * 31)
    for _emit_index in range(want - have):
        jitter = wp.vec3(
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
        )
        query = wp.mesh_query_point_no_sign(mesh_id, center + jitter, projection_radius)
        if query.result:
            wp.atomic_add(emission_state, 0, 1)
            # Commit consumes one additional random value for lifespan before
            # the next projected spawn; preserve predicate parity.
            if source.use_lifespan != 0:
                wp.randf(state)
