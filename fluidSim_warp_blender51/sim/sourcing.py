"""Sourcing kernels: analytic shapes, height fields, particle sources,
volume (NanoVDB via wp.Volume) sources and particle-attribute sourcing.

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

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

from . import constants as C
from .fields import (
    voxel_coord_of_flat,
    voxel_flat_index,
    voxel_center_world,
    world_to_voxel,
    sample_scalar_trilinear,
)
from .registry import register_fn


# ---------------------------------------------------------------------------
# Blend helper (BlendMode: 0 Add, 1 Replace, 2 Maximum, 3 Pull, 4 Blend)
# ---------------------------------------------------------------------------


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
        l = wp.length(d)
        if l > t:
            d = d * (t / l)
        return old + d
    return old + (new - old) * wp.clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Source shape description (GPU struct array)
# ---------------------------------------------------------------------------


@wp.struct
class SourceShape:
    enabled: wp.int32
    shape: wp.int32        # SourceShapeShape: 0 Sphere 1 Cylinder 2 Box 3 Pyramid 4 Cone 5 Plane
    stype: wp.int32        # SourceShapeType: 0 Source 1 Collision 2 Sink 3 Activation 4 CameraFrustum
    mode: wp.int32         # SourceShapeSourceMode: 0 Field 1 Particles
    xform: wp.transform    # local -> world
    inv_xform: wp.transform
    half_extents: wp.vec3  # box half extents / general scale
    radius: wp.float32
    height: wp.float32     # cylinder/cone/pyramid half height
    feather: wp.float32    # soft boundary width (world units)
    velocity: wp.vec3      # emission velocity (world)
    linear_velocity: wp.vec3   # rigid motion of the shape itself (colliders)
    angular_velocity: wp.vec3
    blend_mode: wp.int32
    blend_amount: wp.float32
    force_type: wp.int32   # 0 Off 1 Directional 2 Omnidirectional 3 Spin 4 Radial
    force_strength: wp.float32
    force_dir: wp.vec3
    emit_rate: wp.float32          # target particles per voxel
    temperature: wp.float32
    viscosity: wp.float32
    density: wp.float32
    life_min: wp.float32
    life_max: wp.float32
    use_lifespan: wp.int32


@wp.struct
class MeshSource:
    """GPU attributes for one mesh source or sink.

    The mesh handle remains a separate kernel argument so this contract does
    not alter the analytic ``SourceShape`` ABI.
    """

    enabled: wp.int32
    stype: wp.int32
    mode: wp.int32
    surface_thickness: wp.float32
    sink_region_mode: wp.int32
    velocity: wp.vec3
    blend_mode: wp.int32
    blend_amount: wp.float32
    emit_rate: wp.float32
    temperature: wp.float32
    viscosity: wp.float32
    density: wp.float32
    life_min: wp.float32
    life_max: wp.float32
    use_lifespan: wp.int32


@dataclass
class SourceShapeDesc:
    """Host description for one analytic source shape."""

    enabled: bool = True
    shape: int = 0
    stype: int = 0
    mode: int = 1
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)
    half_extents: tuple = (0.5, 0.5, 0.5)
    radius: float = 0.5
    height: float = 0.5
    feather: float = 0.0
    velocity: tuple = (0.0, 0.0, 0.0)
    linear_velocity: tuple = (0.0, 0.0, 0.0)
    angular_velocity: tuple = (0.0, 0.0, 0.0)
    blend_mode: int = 1
    blend_amount: float = 1.0
    force_type: int = 0
    force_strength: float = 0.0
    force_dir: tuple = (0.0, 0.0, 1.0)
    emit_rate: float = 8.0
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False

    def aabb(self):
        """Return a conservative world AABB for topology marking."""
        radius = max(
            self.radius,
            self.height,
            self.half_extents[0],
            self.half_extents[1],
            self.half_extents[2],
        ) + max(self.feather, 0.0)
        x, y, z = self.position
        return ((x - radius, y - radius, z - radius),
                (x + radius, y + radius, z + radius))

    def to_struct(self):
        shape = SourceShape()
        shape.enabled = 1 if self.enabled else 0
        shape.shape = int(self.shape)
        shape.stype = int(self.stype)
        shape.mode = int(self.mode)
        rotation = wp.quat(*[float(value) for value in self.rotation])
        shape.xform = wp.transform(
            wp.vec3(*[float(value) for value in self.position]), rotation)
        shape.inv_xform = wp.transform_inverse(shape.xform)
        shape.half_extents = wp.vec3(
            *[float(value) for value in self.half_extents])
        shape.radius = float(self.radius)
        shape.height = float(self.height)
        shape.feather = float(self.feather)
        shape.velocity = wp.vec3(*[float(value) for value in self.velocity])
        shape.linear_velocity = wp.vec3(
            *[float(value) for value in self.linear_velocity])
        shape.angular_velocity = wp.vec3(
            *[float(value) for value in self.angular_velocity])
        shape.blend_mode = int(self.blend_mode)
        shape.blend_amount = float(self.blend_amount)
        shape.force_type = int(self.force_type)
        shape.force_strength = float(self.force_strength)
        shape.force_dir = wp.vec3(*[float(value) for value in self.force_dir])
        shape.emit_rate = float(self.emit_rate)
        shape.temperature = float(self.temperature)
        shape.viscosity = float(self.viscosity)
        shape.density = float(self.density)
        shape.life_min = float(self.life_min)
        shape.life_max = float(self.life_max)
        shape.use_lifespan = 1 if self.use_lifespan else 0
        return shape


@dataclass
class MeshSourceDesc:
    """Host description for evaluated-mesh surface sourcing and sinks."""

    enabled: bool = True
    stype: int = 0
    mode: int = 1
    mesh: object = None
    aabb_min: tuple = (0.0, 0.0, 0.0)
    aabb_max: tuple = (0.0, 0.0, 0.0)
    surface_thickness: float = 0.0
    sink_region_mode: int = 0  # 0 Solid, 1 Surface
    velocity: tuple = (0.0, 0.0, 0.0)
    blend_mode: int = 1
    blend_amount: float = 1.0
    emit_rate: float = 8.0
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False

    @property
    def role(self):
        """Semantic source role, exposed separately from the analytic ABI."""
        return self.stype

    @role.setter
    def role(self, value):
        self.stype = int(value)

    def aabb(self):
        band = max(float(self.surface_thickness), 0.0)
        return (
            tuple(float(value) - band for value in self.aabb_min),
            tuple(float(value) + band for value in self.aabb_max),
        )

    def query_radius(self):
        extent = tuple(
            float(self.aabb_max[i]) - float(self.aabb_min[i])
            for i in range(3)
        )
        return max(1.0e-6, float(np.linalg.norm(extent)) +
                   max(float(self.surface_thickness), 0.0))

    def to_struct(self):
        source = MeshSource()
        source.enabled = 1 if self.enabled else 0
        source.stype = int(self.stype)
        source.mode = int(self.mode)
        source.surface_thickness = float(self.surface_thickness)
        source.sink_region_mode = int(self.sink_region_mode)
        source.velocity = wp.vec3(*[float(value) for value in self.velocity])
        source.blend_mode = int(self.blend_mode)
        source.blend_amount = float(self.blend_amount)
        source.emit_rate = float(self.emit_rate)
        source.temperature = float(self.temperature)
        source.viscosity = float(self.viscosity)
        source.density = float(self.density)
        source.life_min = float(self.life_min)
        source.life_max = float(self.life_max)
        source.use_lifespan = 1 if self.use_lifespan else 0
        return source


@dataclass
class HeightSourceDesc:
    """Host description for the optional height-field source."""

    enabled: bool = False
    base_height: float = 0.0
    back_mode: int = 1
    velocity: tuple = (0.0, 0.0, 0.0)
    profile: object = None
    emit_particles: bool = True
    emit_rate: float = 8.0


@wp.func
def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3,
                                c: wp.vec3):
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
    ca = wp.vec2(q[0] - wp.min(q[0], cap_radius),
                 wp.abs(q[1]) - half_height)
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
    if (p[2] >= -half_height and p[2] <= half_height
            and wp.abs(p[0]) <= cross_radius
            and wp.abs(p[1]) <= cross_radius):
        return -distance
    return distance


@wp.func
def shape_local_sdf(shape: int, p: wp.vec3, half_extents: wp.vec3,
                    radius: float, height: float):
    """Exact Euclidean signed distance in shape-local space."""
    if shape == 0:  # Sphere
        return wp.length(p) - radius
    if shape == 1:  # Cylinder (axis z)
        dr = wp.length(wp.vec3(p[0], p[1], 0.0)) - radius
        dz = wp.abs(p[2]) - height
        outside = wp.vec3(wp.max(dr, 0.0), wp.max(dz, 0.0), 0.0)
        return wp.min(wp.max(dr, dz), 0.0) + wp.length(outside)
    if shape == 2:  # Box
        q = wp.vec3(wp.abs(p[0]) - half_extents[0],
                    wp.abs(p[1]) - half_extents[1],
                    wp.abs(p[2]) - half_extents[2])
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
        l = wp.length(rel)
        if l > 1.0e-8:
            return s.velocity + rel * (s.force_strength / l)
        return s.velocity
    if s.force_type == 3:  # Spin around force_dir axis
        axis = s.force_dir
        if wp.length(axis) < 1.0e-8:
            axis = wp.vec3(0.0, 0.0, 1.0)
        axis = wp.normalize(axis)
        return s.velocity + wp.cross(axis, rel) * s.force_strength
    # Radial (inward)
    l = wp.length(rel)
    if l > 1.0e-8:
        return s.velocity - rel * (s.force_strength / l)
    return s.velocity


# ---------------------------------------------------------------------------
# fluidSimSourceShapes: field-mode application of all shapes to the grid
# ---------------------------------------------------------------------------


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
    """Apply field-mode source shapes to voxels:
      Source: mark FLUID + SOURCE_REGION_TRACKING, blend velocity/attributes.
      Collision: min into collision SDF + write collider velocity.
      Sink: set SINK bit.
      Activation: set ACTIVE bit.
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
            # Field-mode fluid source.
            vt |= C.VOXEL_FLUID | C.VOXEL_ACTIVE | C.VOXEL_SOURCE_REGION_TRACKING
            ev = shape_force_velocity(s, pw)
            new_v = blend_vec(wp.vec3(vel_u, vel_v, vel_w), ev,
                              s.blend_mode, s.blend_amount)
            vel_u = new_v[0]
            vel_v = new_v[1]
            vel_w = new_v[2]
            temp = blend_scalar(temp, s.temperature, s.blend_mode, s.blend_amount)
            visc = blend_scalar(visc, s.viscosity, s.blend_mode, s.blend_amount)
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


register_fn(k_source_shapes, "fluidSimSourceShapes",
            notes="Field-mode analytic shape sourcing/collision/sink/activation.")


# ---------------------------------------------------------------------------
# fluidSimSourceShapesParticlesAttributes: particle-mode emission
# ---------------------------------------------------------------------------


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
):
    """Emit particles inside particle-mode source shapes, topping each voxel
    up to the shape's emit_rate target using the current particle counts."""
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)

    for si in range(num_shapes):
        s = shapes[si]
        if s.enabled == 0 or s.stype != 0 or s.mode != 1:
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
        for e in range(emit_n):
            slot = wp.atomic_add(count, 0, 1)
            if slot >= capacity:
                wp.atomic_sub(count, 0, 1)
                return
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
            time_offset[slot] = 0.0
            alive[slot] = 1


register_fn(k_source_shapes_particles, "fluidSimSourceShapesParticlesAttributes",
            notes="Particle-mode emission with per-voxel top-up to emit_rate.")


# ---------------------------------------------------------------------------
# Evaluated-mesh surface sources and solid/surface sinks
# ---------------------------------------------------------------------------


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
        query = wp.mesh_query_point_sign_winding_number(
            mesh_id, p, query_radius, 2.0, 0.5)
        if query.result and query.sign < 0.0:
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
        voxel_type[tid] = (
            vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE |
            C.VOXEL_SOURCE_REGION_TRACKING
        )
        old_velocity = wp.vec3(u[tid], v[tid], w[tid])
        new_velocity = blend_vec(
            old_velocity, source.velocity,
            source.blend_mode, source.blend_amount)
        u[tid] = new_velocity[0]
        v[tid] = new_velocity[1]
        w[tid] = new_velocity[2]
        temperature[tid] = blend_scalar(
            temperature[tid], source.temperature,
            source.blend_mode, source.blend_amount)
        viscosity[tid] = blend_scalar(
            viscosity[tid], source.viscosity,
            source.blend_mode, source.blend_amount)


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
):
    """Top up intersecting voxels and project every spawn onto a triangle."""
    tid = wp.tid()
    if (tid >= num_leaves[0] * C.LEAF_VOLUME or source.enabled == 0
            or source.stype != 0 or source.mode != 1):
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    center = voxel_center_world(ijk, origin, voxel_size)
    half_diagonal = 0.8660254037844386 * voxel_size
    shell = wp.max(source.surface_thickness, 0.0) + half_diagonal
    projection_radius = wp.max(
        query_radius, shell + half_diagonal)
    center_query = wp.mesh_query_point_no_sign(mesh_id, center, shell)
    if not center_query.result:
        return
    center_closest = wp.mesh_eval_position(
        mesh_id, center_query.face, center_query.u, center_query.v)
    if wp.length(center - center_closest) > shell:
        return

    have = voxel_counts[tid]
    want = int(source.emit_rate)
    if want <= have:
        return
    state = wp.rand_init(seed, tid * 31)
    for emit_index in range(want - have):
        jitter = wp.vec3(
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
            (wp.randf(state) - 0.5) * voxel_size,
        )
        query = wp.mesh_query_point_no_sign(
            mesh_id, center + jitter, projection_radius)
        if not query.result:
            continue
        slot = wp.atomic_add(count, 0, 1)
        if slot >= capacity:
            wp.atomic_sub(count, 0, 1)
            return
        positions[slot] = wp.mesh_eval_position(
            mesh_id, query.face, query.u, query.v)
        velocities[slot] = source.velocity
        age[slot] = 0.0
        if source.use_lifespan != 0:
            life[slot] = source.life_min + wp.randf(state) * (
                source.life_max - source.life_min)
        else:
            life[slot] = 1.0e30
        temperature[slot] = source.temperature
        viscosity[slot] = source.viscosity
        density[slot] = source.density
        time_offset[slot] = 0.0
        alive[slot] = 1


register_fn(
    k_mesh_source_particles,
    "fluidSimMeshSourceParticles",
    notes="Triangle-projected mesh emission with per-voxel top-up.",
)


# ---------------------------------------------------------------------------
# fluidSimSourceHeight: height-field sourcing (flat level or profile grid)
# ---------------------------------------------------------------------------


@wp.kernel
def k_source_height(
    leaf_coords: wp.array(dtype=wp.vec3i),
    num_leaves: wp.array(dtype=wp.int32),
    origin: wp.vec3,
    voxel_size: float,
    base_height: float,
    back_mode: int,           # 0 Infinite, 1 Flat, 2 Profile
    profile: wp.array2d(dtype=float),
    profile_res: wp.vec2i,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
    velocity: wp.vec3,
    voxel_type: wp.array(dtype=wp.int32),
    u: wp.array(dtype=float),
    v: wp.array(dtype=float),
    w: wp.array(dtype=float),
):
    """Fill voxels whose center lies below the height surface with fluid."""
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


register_fn(k_source_height, "fluidSimSourceHeight",
            notes="Flat / profile-grid height sourcing along +z.")


# ---------------------------------------------------------------------------
# fluidSimSourceResolveType: precedence resolution after all sourcing
# ---------------------------------------------------------------------------


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


register_fn(k_source_resolve_type, "fluidSimSourceResolveType",
            notes="Voxel type precedence: collision > sink > fluid.")


# ---------------------------------------------------------------------------
# fluidSimSourceParticles: append externally provided particles
# ---------------------------------------------------------------------------


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


register_fn(k_source_particles, "fluidSimSourceParticles",
            notes="Append external particle list (point cloud / particle system).")


# ---------------------------------------------------------------------------
# Particle attribute sourcing for a contiguous slot range [start, start+n)
# ---------------------------------------------------------------------------


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


register_fn(k_source_particle_scalar_field_attribute,
            "fluidSimSourceParticleScalarFieldAttribute",
            notes="Sample scalar field into per-particle attribute at spawn.")


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


register_fn(k_source_particle_vector_field_attribute,
            "fluidSimSourceParticleVectorFieldAttribute",
            notes="Sample vector field into per-particle vec attribute at spawn.")


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


register_fn(k_source_particle_scalar_constant,
            "fluidSimSourceParticleScalarFloatAttribute",
            "fluidSimSourceParticleScalarHalfAttribute",
            notes="Constant float/half attribute fill; half stored as f32 attr.")


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


register_fn(k_source_particle_integer_constant,
            "fluidSimSourceParticleScalarIntegerAttribute",
            notes="Constant integer attribute fill.")


# ---------------------------------------------------------------------------
# core field-to-field sourcing
# ---------------------------------------------------------------------------


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


register_fn(k_core_source_float, "coreSourceFloat",
            notes="Blend scalar field into scalar field.")


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


register_fn(k_core_source_vector, "coreSourceVector",
            notes="Blend vec3 field into vec3 field.")


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


register_fn(k_core_source_staggered_vector, "coreSourceStaggeredVector",
            notes="Blend staggered MAC field into MAC field.")


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
        leaf_table, voxels_dim, leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    wp.atomic_add(accum, flat, attr[tid])
    wp.atomic_add(weight, flat, 1.0)


register_fn(k_core_source_float_particle_attribute,
            "coreSourceFloatParticleAttribute",
            notes="Particle scalar attr -> grid accumulate; normalize separately.")


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
        leaf_table, voxels_dim, leaves_dim,
        wp.vec3i(int(wp.floor(vp[0])), int(wp.floor(vp[1])), int(wp.floor(vp[2]))),
    )
    if flat < 0:
        return
    wp.atomic_add(accum, flat, attr[tid])
    wp.atomic_add(weight, flat, 1.0)


register_fn(k_core_source_vector_particle_attribute,
            "coreSourceVectorParticleAttribute",
            notes="Particle vec3 attr -> grid accumulate; normalize separately.")


# ---------------------------------------------------------------------------
# Volume (NanoVDB via wp.Volume) sourcing. OpenVDB names are aliases: the
# Blender bridge converts OpenVDB grids to NanoVDB volumes on load.
# ---------------------------------------------------------------------------


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


register_fn(k_source_volume_float, "coreSourceNanoVDBFloat", "coreSourceOpenVDBFloat",
            notes="Sample wp.Volume float grid into scalar field.")


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


register_fn(k_source_volume_vector, "coreSourceNanoVDBVector",
            notes="Sample wp.Volume vec3 grid into velocity components.")


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


register_fn(k_source_volume_vector_component, "coreSourceOpenVDBVectorComponent",
            notes="Sample one component of a vec3 volume into a scalar field.")


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


register_fn(k_source_volume_type_fog, "coreSourceNanoVDBTypeFog", "coreSourceOpenVDBType",
            notes="Fog density -> fluid voxel marking with emission velocity.")


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


register_fn(k_source_volume_type_fog_with_temperature,
            "coreSourceNanoVDBTypeFogWithTemperature",
            "coreSourceOpenVDBTypeWithTemperature",
            notes="Fog sourcing that also samples a temperature volume.")


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


register_fn(k_source_volume_mask_fog, "coreSourceNanoVDBMaskFog", "coreSourceOpenVDBMask",
            notes="Fog density -> voxel-type mask bit.")


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


register_fn(k_source_volume_type_sdf, "coreSourceNanoVDBTypeSDF",
            notes="SDF volume inside test -> fluid voxel marking.")


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


register_fn(k_source_volume_type_sdf_with_temperature,
            "coreSourceNanoVDBTypeSDFWithTemperature",
            notes="SDF volume sourcing with temperature volume sampling.")


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


register_fn(k_source_volume_mask_sdf, "coreSourceNanoVDBMaskSDF",
            notes="SDF volume inside test -> voxel-type mask bit.")


class SourcePipeline:
    """Own source state and the complete source-launch transaction."""

    def __init__(self, owner):
        self.owner = owner
        self.device = owner.device
        self.sources: list[SourceShapeDesc | MeshSourceDesc] = []
        self.analytic_sources: list[SourceShapeDesc] = []
        self.mesh_sources: list[MeshSourceDesc] = []
        self.height_source = HeightSourceDesc()
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None
        self._height_profile = wp.zeros(
            (2, 2), dtype=float, device=self.device)
        self._height_profile_res = (2, 2)
        self._emit_seed = 12345
        self._first_slot_dev = wp.zeros(
            1, dtype=wp.int32, device=self.device)

    def set_sources(self, sources):
        self.sources = list(sources)
        self.analytic_sources = [
            source for source in self.sources
            if isinstance(source, SourceShapeDesc)
        ]
        self.mesh_sources = [
            source for source in self.sources
            if isinstance(source, MeshSourceDesc)
        ]
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None

    def set_height_source(self, desc):
        desc = HeightSourceDesc() if desc is None else desc
        self.height_source = desc
        self._emit_shape_array = None
        self._emit_shape_key = None
        if desc.profile is not None:
            profile = np.ascontiguousarray(desc.profile, dtype=np.float32)
            self._height_profile = wp.array(
                profile, dtype=float, device=self.device)
            self._height_profile_res = profile.shape
        else:
            self._height_profile_res = (2, 2)

    def _shapes(self):
        if self._shape_array is None:
            structs = [source.to_struct() for source in self.analytic_sources]
            if not structs:
                structs = [SourceShapeDesc(enabled=False).to_struct()]
            self._shape_array = wp.array(
                structs, dtype=SourceShape, device=self.device)
        return self._shape_array

    def apply(self, dt):
        owner = self.owner
        grid = owner.grid
        settings = owner.settings
        shapes = self._shapes()
        analytic_sources = getattr(self, "analytic_sources", self.sources)
        mesh_sources = getattr(self, "mesh_sources", ())
        shape_count = len(analytic_sources)

        if shape_count:
            wp.launch(
                k_source_shapes, dim=owner._nvox,
                inputs=[shapes, shape_count, grid.leaf_coords, grid.num_leaves,
                        grid.origin, settings.voxel_size, dt],
                outputs=[owner.voxel_type.data, owner.collision_sdf.data,
                         owner.collision_vel, owner.vel.u, owner.vel.v,
                         owner.vel.w, owner.temperature.data,
                         owner.viscosity_field.data],
                device=self.device,
            )

        for source in mesh_sources:
            if not source.enabled or source.mesh is None:
                continue
            wp.launch(
                k_mesh_source_field, dim=owner._nvox,
                inputs=[source.mesh.id, source.to_struct(),
                        source.query_radius(), grid.leaf_coords,
                        grid.num_leaves, grid.origin, settings.voxel_size],
                outputs=[owner.voxel_type.data, owner.vel.u, owner.vel.v,
                         owner.vel.w, owner.temperature.data,
                         owner.viscosity_field.data],
                device=self.device,
            )

        height = self.height_source
        if height.enabled:
            profile_res = wp.vec2i(
                int(self._height_profile_res[0]),
                int(self._height_profile_res[1]),
            )
            wp.launch(
                k_source_height, dim=owner._nvox,
                inputs=[grid.leaf_coords, grid.num_leaves, grid.origin,
                        settings.voxel_size, height.base_height,
                        int(height.back_mode), self._height_profile, profile_res,
                        wp.vec3(*settings.domain_min),
                        wp.vec3(*settings.domain_max),
                        wp.vec3(*height.velocity)],
                outputs=[owner.voxel_type.data, owner.vel.u, owner.vel.v,
                         owner.vel.w],
                device=self.device,
            )

        wp.launch(
            k_source_resolve_type, dim=owner._nvox,
            inputs=[owner.voxel_type.data, grid.num_leaves],
            device=self.device,
        )

        self._emit_seed = (
            self._emit_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        emit_shapes = shape_count > 0 and any(
            source.enabled and source.stype == 0 and source.mode == 1
            for source in analytic_sources
        )
        emit_mesh = any(
            source.enabled and source.mesh is not None
            and source.stype == 0 and source.mode == 1
            for source in mesh_sources
        )
        emit_height = height.enabled and height.emit_particles
        particles = owner.particles
        if ((emit_shapes or emit_mesh or emit_height)
                and particles.host_count < particles.capacity):
            emit_count = shape_count
            if emit_height:
                domain_min = settings.domain_min
                domain_max = settings.domain_max
                key = (
                    shape_count,
                    height.base_height,
                    tuple(height.velocity),
                    height.emit_rate,
                    tuple(domain_min),
                    tuple(domain_max),
                )
                if (self._emit_shape_array is None
                        or self._emit_shape_key != key):
                    center_z = (domain_min[2] + height.base_height) * 0.5
                    half_z = max(
                        (height.base_height - domain_min[2]) * 0.5, 0.0)
                    emit_sources = list(analytic_sources)
                    emit_sources.append(SourceShapeDesc(
                        enabled=True,
                        shape=2,
                        stype=0,
                        mode=1,
                        position=(
                            (domain_min[0] + domain_max[0]) * 0.5,
                            (domain_min[1] + domain_max[1]) * 0.5,
                            center_z,
                        ),
                        half_extents=(
                            (domain_max[0] - domain_min[0]) * 0.5,
                            (domain_max[1] - domain_min[1]) * 0.5,
                            half_z,
                        ),
                        velocity=tuple(height.velocity),
                        emit_rate=height.emit_rate,
                    ))
                    self._emit_shape_array = wp.array(
                        [source.to_struct() for source in emit_sources],
                        dtype=SourceShape,
                        device=self.device,
                    )
                    self._emit_shape_key = key
                shapes = self._emit_shape_array
                emit_count = shape_count + 1

            wp.launch(
                k_source_shapes_particles, dim=owner._nvox,
                inputs=[shapes, emit_count, grid.leaf_coords, grid.num_leaves,
                        grid.origin, settings.voxel_size,
                        owner.voxel_counts.data, self._emit_seed,
                        particles.capacity],
                outputs=[particles.position, particles.velocity,
                         particles.attr("age"), particles.attr("life"),
                         particles.attr("temperature"),
                         particles.attr("viscosity"),
                         particles.attr("density"), particles.time_offset,
                         particles.alive, particles.count],
                device=self.device,
            )
            for mesh_index, source in enumerate(mesh_sources):
                if (not source.enabled or source.mesh is None
                        or source.stype != 0 or source.mode != 1):
                    continue
                wp.launch(
                    k_mesh_source_particles, dim=owner._nvox,
                    inputs=[source.mesh.id, source.to_struct(),
                            source.query_radius(), grid.leaf_coords,
                            grid.num_leaves, grid.origin, settings.voxel_size,
                            owner.voxel_counts.data,
                            self._emit_seed + mesh_index + 1,
                            particles.capacity],
                    outputs=[particles.position, particles.velocity,
                             particles.attr("age"), particles.attr("life"),
                             particles.attr("temperature"),
                             particles.attr("viscosity"),
                             particles.attr("density"), particles.time_offset,
                             particles.alive, particles.count],
                    device=self.device,
                )
            particles.refresh_host_count()
            return True
        return False

    def inject_particles(self, positions, velocity=(0.0, 0.0, 0.0),
                         velocities=None, temperature=0.0, viscosity=0.0,
                         density=1.0, life=0.0, use_lifespan=False):
        """Append external world-space points as liquid particles."""
        points = np.ascontiguousarray(positions, dtype=np.float32)
        count = len(points)
        if count == 0:
            return 0

        source_positions = wp.array(
            points, dtype=wp.vec3, device=self.device)
        if velocities is None:
            velocities = np.tile(
                np.asarray(velocity, dtype=np.float32), (count, 1))
        else:
            velocities = np.ascontiguousarray(
                velocities, dtype=np.float32)
        source_velocities = wp.array(
            velocities, dtype=wp.vec3, device=self.device)

        particles = self.owner.particles
        wp.launch(
            k_source_particles, dim=count,
            inputs=[source_positions, source_velocities, count, float(life),
                    1 if use_lifespan else 0, float(temperature),
                    float(viscosity), float(density), particles.capacity,
                    particles.position, particles.velocity,
                    particles.attr("age"), particles.attr("life"),
                    particles.attr("temperature"),
                    particles.attr("viscosity"), particles.attr("density"),
                    particles.time_offset, particles.alive, particles.count,
                    self._first_slot_dev],
            device=self.device,
        )
        self.owner._host_frame_cache = None
        return particles.refresh_host_count()
