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
from .enums import (BlendMode, MeshSinkRegionMode, MeshSourceRegion,
                    SourceEmissionMode, SourceEvaluationCadence,
                    SourceLifecycle, SourceOperation, SourcePayloadKind,
                    SourceShapeForceType, SourceShapeShape,
                    SourceShapeSourceMode, SourceShapeType)
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
    emission_mode: wp.int32   # SourceEmissionMode: 0 Initial Fill, 1 Continuous Inflow
    armed: wp.int32           # 1 = may emit initial fill; host disarms after first fire


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
    force_type: wp.int32   # 0 Off 1 Directional 2 Omnidirectional 3 Spin 4 Radial
    force_strength: wp.float32
    force_dir: wp.vec3
    force_center: wp.vec3
    blend_mode: wp.int32
    blend_amount: wp.float32
    emit_rate: wp.float32
    temperature: wp.float32
    viscosity: wp.float32
    density: wp.float32
    life_min: wp.float32
    life_max: wp.float32
    use_lifespan: wp.int32
    emission_mode: wp.int32   # SourceEmissionMode: 0 Initial Fill, 1 Continuous Inflow
    armed: wp.int32           # 1 = may emit initial fill; host disarms after first fire


@dataclass
class SourceShapeDesc:
    """Host description for one analytic source shape."""

    enabled: bool = True
    shape: int = SourceShapeShape.SPHERE
    stype: int = SourceShapeType.SOURCE
    mode: int = SourceShapeSourceMode.PARTICLES
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0, 1.0)
    half_extents: tuple = (0.5, 0.5, 0.5)
    radius: float = 0.5
    height: float = 0.5
    feather: float = 0.0
    velocity: tuple = (0.0, 0.0, 0.0)
    linear_velocity: tuple = (0.0, 0.0, 0.0)
    angular_velocity: tuple = (0.0, 0.0, 0.0)
    blend_mode: int = BlendMode.REPLACE
    blend_amount: float = 1.0
    force_type: int = SourceShapeForceType.OFF
    force_strength: float = 0.0
    force_dir: tuple = (0.0, 0.0, 1.0)
    emit_rate: float = 8.0
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False
    emission_mode: int = SourceEmissionMode.CONTINUOUS_INFLOW
    source_id: str = ""
    runtime_source_key: str = ""

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
        shape.emission_mode = int(self.emission_mode)
        shape.armed = 1
        return shape


@dataclass
class MeshSourceDesc:
    """Host description for evaluated-mesh surface sourcing and sinks."""

    enabled: bool = True
    stype: int = SourceShapeType.SOURCE
    mode: int = SourceShapeSourceMode.PARTICLES
    mesh: object = None
    aabb_min: tuple = (0.0, 0.0, 0.0)
    aabb_max: tuple = (0.0, 0.0, 0.0)
    surface_thickness: float = 0.0
    sink_region_mode: int = MeshSinkRegionMode.SOLID
    velocity: tuple = (0.0, 0.0, 0.0)
    force_type: int = SourceShapeForceType.OFF
    force_strength: float = 0.0
    force_dir: tuple = (0.0, 0.0, 1.0)
    force_center: tuple = (0.0, 0.0, 0.0)
    blend_mode: int = BlendMode.REPLACE
    blend_amount: float = 1.0
    emit_rate: float = 8.0
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False
    emission_mode: int = SourceEmissionMode.CONTINUOUS_INFLOW
    source_id: str = ""
    runtime_source_key: str = ""
    source_region: int = MeshSourceRegion.SURFACE_SHELL
    evaluation_cadence: int = SourceEvaluationCadence.FIRST_SIM_FRAME
    vertices: object = None
    triangles: object = None

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
        source.force_type = int(self.force_type)
        source.force_strength = float(self.force_strength)
        source.force_dir = wp.vec3(*[float(value) for value in self.force_dir])
        source.force_center = wp.vec3(
            *[float(value) for value in self.force_center])
        source.blend_mode = int(self.blend_mode)
        source.blend_amount = float(self.blend_amount)
        source.emit_rate = float(self.emit_rate)
        source.temperature = float(self.temperature)
        source.viscosity = float(self.viscosity)
        source.density = float(self.density)
        source.life_min = float(self.life_min)
        source.life_max = float(self.life_max)
        source.use_lifespan = 1 if self.use_lifespan else 0
        source.emission_mode = int(self.emission_mode)
        source.armed = 1
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
    # A fill fills. Standing behaviour for a height fill is to establish the
    # liquid volume once; continuous inflow is the opt-in pump.
    emission_mode: int = SourceEmissionMode.INITIAL_FILL
    source_id: str = "builtin:height"


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
        return (source.velocity + wp.cross(wp.normalize(axis), rel)
                * source.force_strength)
    # Radial (inward)
    length = wp.length(rel)
    if length > 1.0e-8:
        return source.velocity - rel * (source.force_strength / length)
    return source.velocity


# ---------------------------------------------------------------------------
# Analytic FORCE regions: force-stage application to valid MAC faces
# ---------------------------------------------------------------------------


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
            if (shape.enabled == 0 or shape.stype != 0 or shape.mode != 0
                    or shape.force_type == 0):
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


# ---------------------------------------------------------------------------
# fluidSimSourceShapes: collisions, sinks, activation, and FIELD compatibility
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
    """Apply source-shape grid roles other than liquid emission.

    The original FIELD branch blends source pressure only; this port does not
    yet expose/source that pressure field. Therefore FIELD sources are a
    compatibility no-op here: they do not create liquid, set tracking bits, or
    write MAC velocity/temperature/viscosity. Analytic FORCE objects use
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
            # Original semantics: FIELD mode does not create liquid. The only
            # original grid write is source pressure, which is not exposed by
            # this port yet (docs/source_shape_spec.md). FORCE-role velocity
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


register_fn(k_source_shapes, "fluidSimSourceShapes",
            notes="Analytic collision/sink/activation; FIELD pressure deferred.")


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
    emission_state: wp.array(dtype=wp.int32),
):
    """Emit particles inside particle-mode source shapes after batch admission."""
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= num_leaves[0] * C.LEAF_VOLUME:
        return
    ijk = voxel_coord_of_flat(leaf_coords, tid)
    pw = voxel_center_world(ijk, origin, voxel_size)

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
        for e in range(emit_n):
            slot = wp.atomic_add(count, 0, 1)
            if slot >= capacity:
                wp.atomic_sub(count, 0, 1)
                emission_state[3] = 1
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
            wp.atomic_add(emission_state, 2, 1)


register_fn(k_source_shapes_particles, "fluidSimSourceShapesParticlesAttributes",
            notes="Particle-mode emission with per-voxel top-up to emit_rate.")


# ---------------------------------------------------------------------------
# Evaluated-mesh surface sources and solid/surface sinks
# ---------------------------------------------------------------------------


@wp.func
def _watertight_mesh_contains(mesh_id: wp.uint64, point: wp.vec3,
                              query_radius: float):
    """Orientation-independent inside test for validated watertight meshes."""
    query = wp.mesh_query_point_sign_parity(
        mesh_id, point, query_radius, 3, 0.1)
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
    emission_state: wp.array(dtype=wp.int32),
):
    """Top up intersecting voxels after whole-batch admission."""
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if (tid >= num_leaves[0] * C.LEAF_VOLUME or source.enabled == 0
            or source.stype != 0 or source.mode != 1
            or (source.emission_mode == 0 and source.armed == 0)):
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
            emission_state[3] = 1
            return
        position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        positions[slot] = position
        velocities[slot] = mesh_source_force_velocity(source, position)
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
        wp.atomic_add(emission_state, 2, 1)


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


register_fn(k_source_particles, "fluidSimSourceParticles",
            notes="Append external particle list (point cloud / particle system).")


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
    default_lifespan: float,
    use_lifespan: int,
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
    result: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
    """Direct point append preserving every owned per-point attribute."""
    if emission_state[1] == 0:
        return
    tid = wp.tid()
    if tid >= src_count:
        return
    slot = wp.atomic_add(count, 0, 1)
    if slot >= capacity:
        wp.atomic_sub(count, 0, 1)
        result[1] = 1
        emission_state[3] = 1
        return
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
    time_offset[slot] = 0.0
    alive[slot] = 1
    wp.atomic_add(result, 0, 1)
    wp.atomic_add(emission_state, 2, 1)


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
        voxel_type[flat] = ((vt | C.VOXEL_COLLISION)
                            & ~(C.VOXEL_SINK | C.VOXEL_FLUID
                                | C.VOXEL_SOURCE_REGION_TRACKING))
        return
    if (vt & C.VOXEL_COLLISION) != 0:
        return
    if operation == 1:  # SINK
        voxel_type[flat] = ((vt | C.VOXEL_SINK)
                            & ~(C.VOXEL_FLUID | C.VOXEL_SOURCE_REGION_TRACKING))
        return
    if operation == 4:  # ACTIVATION
        voxel_type[flat] = vt | C.VOXEL_ACTIVE
        return
    if (vt & C.VOXEL_SINK) != 0:
        return
    if operation == 0:  # LIQUID region field; particles are appended separately
        voxel_type[flat] = (vt | C.VOXEL_FLUID | C.VOXEL_ACTIVE
                            | C.VOXEL_SOURCE_REGION_TRACKING)
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
            pressure[flat], source_scalar, blend_mode,
            wp.where(blend_mode == 0, amount * dt, amount))
    elif operation == 6:  # SCALAR_FIELD defaults to temperature
        temperature[flat] = blend_scalar(
            temperature[flat], source_scalar, blend_mode,
            wp.where(blend_mode == 0, amount * dt, amount))
    elif operation == 7 or operation == 8:  # vector/staggered velocity
        old = wp.vec3(u[flat], v[flat], w[flat])
        value = blend_vec(old, source_vector, blend_mode,
                          wp.where(blend_mode == 0, amount * dt, amount))
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
    life_value: float,
    use_lifespan: int,
    capacity: int,
    voxel_counts: wp.array(dtype=wp.int32),
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
    result: wp.array(dtype=wp.int32),
    emission_state: wp.array(dtype=wp.int32),
):
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
    missing = wp.max(0, desired - voxel_counts[flat])
    center = voxel_center_world(ijk, origin, voxel_size)
    for sample in range(64):
        if sample >= missing:
            break
        slot = wp.atomic_add(count, 0, 1)
        if slot >= capacity:
            wp.atomic_sub(count, 0, 1)
            result[1] = 1
            emission_state[3] = 1
            break
        positions[slot] = center
        velocities[slot] = emission_velocity
        age[slot] = 0.0
        life[slot] = wp.where(use_lifespan != 0, life_value, 1.0e30)
        temperature[slot] = temperature_value
        viscosity[slot] = viscosity_value
        density[slot] = density_value
        time_offset[slot] = 0.0
        alive[slot] = 1
        wp.atomic_add(voxel_counts, flat, 1)
        wp.atomic_add(result, 0, 1)
        wp.atomic_add(emission_state, 2, 1)


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
        if (s.enabled == 0 or s.stype != 0 or s.mode != 1
                or (s.emission_mode == 0 and s.armed == 0)):
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
    if (tid >= num_leaves[0] * C.LEAF_VOLUME or source.enabled == 0
            or source.stype != 0 or source.mode != 1
            or (source.emission_mode == 0 and source.armed == 0)):
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
        mesh_id, center_query.face, center_query.u, center_query.v)
    if wp.length(center - center_closest) > shell:
        return
    have = scratch_counts[tid]
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
        if query.result:
            wp.atomic_add(emission_state, 0, 1)
            # Commit consumes one additional random value for lifespan before
            # the next projected spawn; preserve predicate parity.
            if source.use_lifespan != 0:
                wp.randf(state)


@dataclass(frozen=True)
class SourceBatch:
    """One runtime source snapshot with unambiguous representation ownership."""

    legacy_descriptors: tuple[SourceShapeDesc | MeshSourceDesc, ...] = ()
    native_plans: tuple = ()


@dataclass(frozen=True)
class DirectInjectionPacket:
    positions: np.ndarray
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocities: np.ndarray | None = None
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    life: float = 0.0
    use_lifespan: bool = False

    def __post_init__(self):
        positions = np.array(
            self.positions, dtype=np.float32, copy=True, order="C"
        ).reshape((-1, 3))
        object.__setattr__(self, "positions", positions)
        if self.velocities is not None:
            velocities = np.array(
                self.velocities, dtype=np.float32, copy=True, order="C"
            ).reshape((-1, 3))
            if len(velocities) != len(positions):
                raise ValueError("per-particle velocities must match positions")
            object.__setattr__(self, "velocities", velocities)


@dataclass(frozen=True)
class InjectionResult:
    admitted: bool
    emitted: int
    required: int
    available: int
    capacity_limited: bool
    total: int


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
        self._compiled_result = wp.zeros(
            2, dtype=wp.int32, device=self.device)
        # [required, admitted, emitted, invariant-overflow, available]. Count
        # passes and guarded commit kernels remain on-device until one
        # aggregate result is read after the whole source batch.
        self._emission_state = wp.zeros(
            5, dtype=wp.int32, device=self.device)
        self._emission_scratch_counts = wp.zeros(
            max(1, int(getattr(owner.grid, "voxel_capacity",
                               getattr(owner, "_nvox", 1)))),
            dtype=wp.int32, device=self.device)
        self._emission_transaction_active = False
        self._emission_pending_fired: set[str] = set()
        self._emission_initial_count = 0
        self._emission_available = 0
        self._emission_legacy_attempted = False
        self._transaction_shapes = None
        self._transaction_shape_count = 0
        self._transaction_mesh_sources = ()
        self._transaction_height_armed = False
        self._compiled_empty_scalar = None
        self._compiled_empty_vector = None
        self.last_apply_result = {
            "emitted": 0, "capacity_limited": False,
            "required": 0, "available": int(owner.particles.capacity),
            "admitted": True,
        }
        self.last_injection_result = None
        self.last_pressure_apply_count = 0
        self.last_post_p2g_apply_count = 0
        # Per-source initial-fill armed state, keyed by the stable descriptor
        # ``source_id`` (Blender object name). Persists across per-frame
        # ``set_sources`` calls; cleared by ``reset`` (Initialize/Reset).
        self._initial_fired: set[str] = set()
        from .source_regions import SourceCompiler
        self.compiler = SourceCompiler()
        self.batch = SourceBatch()
        self.plans = ()
        self._compiled_batch = None
        self._session_generation = 0
        self.last_diagnostics = ()

    def reset(self):
        """Re-arm sources and release compiler-owned device/cache state."""
        self._initial_fired = set()
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None
        self._session_generation = getattr(self, "_session_generation", 0) + 1
        compiler = getattr(self, "compiler", None)
        if compiler is not None:
            compiler.release()
        self._compiled_batch = None
        self.last_diagnostics = ()
        self.last_injection_result = None
        self.last_pressure_apply_count = 0
        self.last_post_p2g_apply_count = 0
        self._emission_transaction_active = False
        self._emission_pending_fired = set()
        owner = getattr(self, "owner", None)
        particles = getattr(owner, "particles", None)
        self.last_apply_result = {
            "emitted": 0, "capacity_limited": False,
            "required": 0,
            "available": int(getattr(particles, "capacity", 0)),
            "admitted": True,
        }

    def release(self):
        """Release all compiler-owned GPU values on solver teardown/failure."""
        self.reset()
        self.owner = None

    def set_source_batch(self, batch):
        """Install both runtime representations without compatibility copies."""
        from .source_regions import SourcePlan
        if not isinstance(batch, SourceBatch):
            raise TypeError("set_source_batch accepts a SourceBatch")
        descriptors = tuple(batch.legacy_descriptors)
        plans = tuple(batch.native_plans)
        if not all(isinstance(plan, SourcePlan) for plan in plans):
            raise TypeError("native_plans accepts SourcePlan values only")
        if not all(isinstance(value, (SourceShapeDesc, MeshSourceDesc))
                   for value in descriptors):
            raise TypeError("legacy_descriptors accepts source descriptors only")
        self.batch = SourceBatch(descriptors, plans)
        self.sources = list(descriptors)
        self.analytic_sources = [
            source for source in descriptors if isinstance(source, SourceShapeDesc)
        ]
        self.mesh_sources = [
            source for source in descriptors if isinstance(source, MeshSourceDesc)
        ]
        self.plans = plans
        self._compiled_batch = None
        self._shape_array = None
        self._emit_shape_array = None
        self._emit_shape_key = None

    def set_plans(self, plans):
        """Compatibility seam: native plans replace all legacy descriptors."""
        self.set_source_batch(SourceBatch(native_plans=tuple(plans)))

    def set_sources(self, sources):
        """Compatibility seam: descriptors replace all native plans."""
        from .source_regions import SourcePlan
        values = tuple(sources)
        if values and all(isinstance(value, SourcePlan) for value in values):
            self.set_plans(values)
        else:
            self.set_source_batch(SourceBatch(legacy_descriptors=values))

    def required_leaf_boxes(self):
        """World-space plan bounds used by solver topology marking."""
        for plan in self.plans:
            bounds = getattr(plan.region, "bounds", None)
            if bounds is not None and plan.activation.enabled:
                yield bounds()

    def compile(self, frame=0, substep=0, *, upload=True):
        """Compile current plans for the owner's sparse-grid context."""
        from .source_regions import CompileContext
        owner, settings = self.owner, self.owner.settings
        context = CompileContext(
            frame=int(frame), substep=int(substep),
            session_generation=self._session_generation,
            device=self.device, origin=tuple(float(v) for v in owner.grid.origin),
            voxel_size=float(settings.voxel_size),
            domain_min=tuple(float(v) for v in settings.domain_min),
            domain_max=tuple(float(v) for v in settings.domain_max),
            leaf_capacity=int(getattr(owner.grid, "capacity", C.DEFAULT_LEAF_CAPACITY)),
            particle_capacity=int(owner.particles.capacity), upload=upload,
        )
        self._compiled_batch = self.compiler.compile(self.plans, context)
        self.last_diagnostics = self._compiled_batch.diagnostics
        return self._compiled_batch

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

    def _bake_struct(self, source):
        """Build the GPU struct with initial-fill armed state applied.

        Continuous sources are always armed (emit every substep). An initial-
        fill source is armed only until it has fired once this run.
        """
        struct = source.to_struct()
        if (int(getattr(source, "emission_mode",
                        SourceEmissionMode.CONTINUOUS_INFLOW))
                == SourceEmissionMode.INITIAL_FILL
                and getattr(source, "source_id", "") in self._initial_fired):
            struct.armed = 0
        return struct

    def _shapes(self):
        if self._shape_array is None:
            structs = [self._bake_struct(source)
                       for source in self.analytic_sources]
            if not structs:
                structs = [SourceShapeDesc(enabled=False).to_struct()]
            self._shape_array = wp.array(
                structs, dtype=SourceShape, device=self.device)
        return self._shape_array

    def apply_forces(self, dt):
        """Apply analytic FORCE-role shapes in the solver forces phase."""
        analytic_sources = getattr(self, "analytic_sources", self.sources)
        if not any(
            source.enabled
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.FIELD
            and source.force_type != SourceShapeForceType.OFF
            for source in analytic_sources
        ):
            return
        owner = self.owner
        grid = owner.grid
        wp.launch(
            k_source_shape_forces, dim=owner._nvox,
            inputs=[self._shapes(), len(analytic_sources),
                    grid.leaf_coords, grid.num_leaves,
                    grid.origin, owner.settings.voxel_size,
                    owner.vel.u, owner.vel.v, owner.vel.w,
                    owner.vel.wu, owner.vel.wv, owner.vel.ww, dt],
            device=self.device,
        )

    @staticmethod
    def _legacy_source_armed(source, fired):
        return not (
            int(getattr(source, "emission_mode",
                        SourceEmissionMode.CONTINUOUS_INFLOW))
            == SourceEmissionMode.INITIAL_FILL
            and getattr(source, "source_id", "") in fired
        )

    @staticmethod
    def _legacy_liquid_target(source):
        # Legacy kernels historically truncate emit_rate; preserve that
        # behavior while making the zero predicate explicit and shared.
        return max(0, min(64, int(float(getattr(source, "emit_rate", 0.0)))))

    @classmethod
    def legacy_descriptor_requires_topology(cls, source):
        if not getattr(source, "enabled", False):
            return False
        if (getattr(source, "stype", SourceShapeType.SOURCE)
                != SourceShapeType.SOURCE
                or getattr(source, "mode", SourceShapeSourceMode.PARTICLES)
                != SourceShapeSourceMode.PARTICLES):
            return True
        return cls._legacy_liquid_target(source) > 0

    def _prepare_legacy_emission(self):
        """Freeze legacy emitters and their deterministic seed for one batch."""
        owner, settings = self.owner, self.owner.settings
        analytic = tuple(getattr(self, "analytic_sources", self.sources))
        meshes = tuple(getattr(self, "mesh_sources", ()))
        height = self.height_source
        height_armed = self._legacy_source_armed(height, self._initial_fired)
        emit_height = bool(
            height.enabled and height.emit_particles and height_armed
            and self._legacy_liquid_target(height) > 0
        )
        emit_shapes = any(
            source.enabled
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.PARTICLES
            and self._legacy_liquid_target(source) > 0
            and self._legacy_source_armed(source, self._initial_fired)
            for source in analytic
        )
        emit_meshes = tuple(
            source for source in meshes
            if source.enabled and source.mesh is not None
            and source.stype == SourceShapeType.SOURCE
            and source.mode == SourceShapeSourceMode.PARTICLES
            and self._legacy_liquid_target(source) > 0
            and self._legacy_source_armed(source, self._initial_fired)
        )
        self._transaction_height_armed = height_armed
        self._transaction_emit_seed = (
            self._emit_seed * 1664525 + 1013904223) & 0x7FFFFFFF
        self._transaction_mesh_sources = emit_meshes
        self._transaction_shapes = None
        self._transaction_shape_count = 0

        if emit_shapes or emit_height:
            emit_sources = list(analytic)
            if emit_height:
                domain_min, domain_max = settings.domain_min, settings.domain_max
                center_z = (domain_min[2] + height.base_height) * 0.5
                half_z = max((height.base_height - domain_min[2]) * 0.5, 0.0)
                emit_sources.append(SourceShapeDesc(
                    enabled=True, shape=2, stype=0, mode=1,
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
                    emission_mode=int(getattr(
                        height, "emission_mode",
                        SourceEmissionMode.CONTINUOUS_INFLOW)),
                    source_id=getattr(height, "source_id", ""),
                ))
            key = tuple(repr(source) for source in emit_sources)
            if self._emit_shape_array is None or self._emit_shape_key != key:
                self._emit_shape_array = wp.array(
                    [self._bake_struct(source) for source in emit_sources],
                    dtype=SourceShape, device=self.device)
                self._emit_shape_key = key
            self._transaction_shapes = self._emit_shape_array
            self._transaction_shape_count = len(emit_sources)

        pending = set()
        if (emit_height
                and int(getattr(height, "emission_mode",
                                SourceEmissionMode.CONTINUOUS_INFLOW))
                == SourceEmissionMode.INITIAL_FILL
                and getattr(height, "source_id", "")):
            pending.add(height.source_id)
        for source in tuple(analytic) + emit_meshes:
            if (source.enabled
                    and source.stype == SourceShapeType.SOURCE
                    and source.mode == SourceShapeSourceMode.PARTICLES
                    and self._legacy_liquid_target(source) > 0
                    and int(getattr(source, "emission_mode",
                                    SourceEmissionMode.CONTINUOUS_INFLOW))
                    == SourceEmissionMode.INITIAL_FILL
                    and getattr(source, "source_id", "")
                    and self._legacy_source_armed(source, self._initial_fired)):
                pending.add(source.source_id)
        return bool(emit_shapes or emit_height or emit_meshes), pending

    def _begin_emission_transaction(self, batch=None, *, include_legacy=False):
        """Count a complete source batch and enqueue one device admission gate."""
        from .source_regions import (
            liquid_coverage_emission_enabled, normalized_liquid_target,
        )
        owner, grid, particles = self.owner, self.owner.grid, self.owner.particles
        compiled_candidates = []
        self._emission_pending_fired = set()
        if batch is not None:
            for compiled in batch.sources:
                plan, region = compiled.plan, compiled.region
                if (plan.operation != SourceOperation.LIQUID
                        or any(item.severity == "ERROR"
                               for item in region.diagnostics)):
                    continue
                stable_id = str(plan.stable_id)
                if (plan.lifecycle in {SourceLifecycle.ONCE,
                                       SourceLifecycle.DIRECT_APPEND}
                        and stable_id in self._initial_fired):
                    continue
                if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
                    if region.points is None or not len(region.points.positions):
                        continue
                elif not liquid_coverage_emission_enabled(plan):
                    continue
                compiled_candidates.append(compiled)
                if plan.lifecycle in {SourceLifecycle.ONCE,
                                      SourceLifecycle.DIRECT_APPEND}:
                    self._emission_pending_fired.add(stable_id)

        self._emission_legacy_attempted = False
        if include_legacy:
            legacy_attempted, pending = self._prepare_legacy_emission()
            self._emission_legacy_attempted = legacy_attempted
            self._emission_pending_fired |= pending

        attempted = bool(compiled_candidates or self._emission_legacy_attempted)
        self._emission_initial_count = int(particles.host_count)
        self._emission_available = max(
            0, int(particles.capacity) - self._emission_initial_count)
        self._emission_transaction_active = attempted
        if not attempted:
            self.last_apply_result = {
                "emitted": 0, "capacity_limited": False,
                "required": 0, "available": self._emission_available,
                "admitted": True,
            }
            return False

        self._emission_state.zero_()
        self._compiled_result.zero_()
        wp.copy(self._emission_scratch_counts, owner.voxel_counts.data,
                count=int(owner._nvox))

        for compiled in compiled_candidates:
            plan, region = compiled.plan, compiled.region
            if (plan.lifecycle == SourceLifecycle.DIRECT_APPEND
                    and region.points is not None):
                count = int(len(region.points.positions))
                if count:
                    wp.launch(
                        k_add_emission_demand, dim=1,
                        inputs=[count, self._emission_state],
                        device=self.device)
            elif region.leaf_count:
                wp.launch(
                    k_count_compiled_source_particles,
                    dim=region.leaf_count * C.LEAF_VOLUME,
                    inputs=[
                        region.device_coverage, region.device_leaf_coords,
                        region.leaf_count, grid.leaf_table, grid.voxels_dim,
                        grid.leaves_dim,
                        normalized_liquid_target(plan),
                        self._emission_scratch_counts,
                        self._emission_state,
                    ], device=self.device)

        if self._transaction_shape_count:
            wp.launch(
                k_count_source_shapes_particles, dim=owner._nvox,
                inputs=[
                    self._transaction_shapes,
                    self._transaction_shape_count,
                    grid.leaf_coords, grid.num_leaves, grid.origin,
                    owner.settings.voxel_size,
                    self._emission_scratch_counts,
                    self._emission_state,
                ], device=self.device)
        for mesh_index, source in enumerate(self._transaction_mesh_sources):
            wp.launch(
                k_count_mesh_source_particles, dim=owner._nvox,
                inputs=[
                    source.mesh.id, self._bake_struct(source),
                    source.query_radius(), grid.leaf_coords,
                    grid.num_leaves, grid.origin,
                    owner.settings.voxel_size,
                    self._emission_scratch_counts,
                    self._transaction_emit_seed + mesh_index + 1,
                    self._emission_state,
                ], device=self.device)

        wp.launch(
            k_admit_emission_batch, dim=1,
            inputs=[particles.count, particles.capacity,
                    self._emission_state],
            device=self.device)
        return True

    def _request_particle_growth(self, required_total):
        grow = getattr(self.owner, "ensure_particle_capacity", None)
        if callable(grow):
            return grow(required_total)
        from .particle_capacity import ParticleGrowthResult

        current = int(getattr(self.owner.particles, "capacity", 0))
        return ParticleGrowthResult(
            "DISABLED", current, current, int(required_total), current,
            reason="owner does not expose transactional particle growth")

    def _finish_emission_transaction(self):
        if not self._emission_transaction_active:
            return False
        values = np.asarray(self._emission_state.numpy(), dtype=np.int64)
        required, admitted, emitted, invariant = map(int, values[:4])
        self._emission_available = int(values[4])
        self._emission_initial_count = max(
            0, int(self.owner.particles.capacity) - self._emission_available)
        self._emission_transaction_active = False
        if invariant:
            self.last_apply_result = {
                "emitted": emitted, "capacity_limited": True,
                "required": required, "available": self._emission_available,
                "admitted": False, "invariant_violation": True,
            }
            raise RuntimeError(
                "source emission exceeded capacity after admission; "
                "the solver session must be discarded")
        if admitted:
            if self._emission_legacy_attempted:
                self._emit_seed = self._transaction_emit_seed
            if self._emission_pending_fired:
                self._initial_fired |= self._emission_pending_fired
                self._shape_array = None
                self._emit_shape_array = None
                self._emit_shape_key = None
            if emitted:
                self.owner.particles.refresh_host_count()
                self.owner._host_frame_cache = None
            self.last_apply_result = {
                "emitted": emitted,
                "capacity_limited": False,
                "required": required,
                "available": self._emission_available,
                "admitted": True,
            }
            return bool(emitted)

        # Guarded commit kernels have made no particle/lifecycle/RNG changes.
        # Grow only after that fact is known; the prepared source remains armed
        # and retries on the next source phase without stopping the solver.
        total_required = self._emission_initial_count + required
        growth = self._request_particle_growth(total_required)
        policy = str(getattr(
            self.owner.settings, "particle_growth_failure_policy", "DEFER"
        )).upper()
        pause_requested = not growth.grew and policy == "PAUSE"
        self.last_apply_result = {
            "emitted": 0,
            "capacity_limited": True,
            "required": required,
            "available": self._emission_available,
            "admitted": False,
            "deferred": True,
            "pause_requested": pause_requested,
            "growth_status": growth.status,
            "growth_previous": growth.previous,
            "growth_current": growth.current,
            "growth_target": growth.target,
            "growth_reason": growth.reason,
        }
        return False

    def apply_compiled(self, batch, dt, *, transaction_ready=False,
                       finalize=True):
        """Apply compiled operations in fixed precedence/semantic order."""
        from .source_regions import (
            liquid_coverage_emission_enabled, normalized_liquid_target,
        )
        if not transaction_ready:
            self._begin_emission_transaction(batch, include_legacy=False)
        owner, grid = self.owner, self.owner.grid
        if self._compiled_empty_scalar is None:
            self._compiled_empty_scalar = wp.empty(
                0, dtype=float, device=self.device)
            self._compiled_empty_vector = wp.empty(
                0, dtype=wp.vec3, device=self.device)
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        pressure = owner.pressure_solver.pressure
        particles = owner.particles
        order = {
            SourceOperation.COLLISION: 0,
            SourceOperation.SINK: 1,
            SourceOperation.ACTIVATION: 2,
            SourceOperation.LIQUID: 3,
            SourceOperation.PRESSURE: 4,
            SourceOperation.SCALAR_FIELD: 5,
            SourceOperation.VECTOR_FIELD: 6,
            SourceOperation.STAGGERED_VELOCITY: 7,
            SourceOperation.FORCE: 8,
        }
        for compiled in sorted(batch.sources,
                               key=lambda value: order[value.plan.operation]):
            plan, region = compiled.plan, compiled.region
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            stable_id = str(plan.stable_id)
            if (plan.lifecycle in {SourceLifecycle.ONCE,
                                   SourceLifecycle.DIRECT_APPEND}
                    and stable_id in self._initial_fired):
                continue
            payload = plan.payload
            if plan.operation == SourceOperation.LIQUID:
                if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
                    if region.points is None or not len(region.points.positions):
                        continue
                elif not liquid_coverage_emission_enabled(plan):
                    continue
            # Liquid coverage is materialized by the particle kernel and the
            # immediate re-index/P2G path. Avoid a redundant field-stamp launch;
            # particle indexing truthfully owns Fluid/Active classification.
            if plan.operation in {
                    SourceOperation.COLLISION, SourceOperation.SINK,
                    SourceOperation.ACTIVATION}:
                wp.launch(
                    k_apply_compiled_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        int(plan.operation), region.device_coverage,
                        (region.device_scalar if region.device_scalar is not None
                         else empty_scalar),
                        (region.device_vector if region.device_vector is not None
                         else empty_vector),
                        region.device_leaf_coords, region.leaf_count,
                        grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                        float(payload.pressure if plan.operation == SourceOperation.PRESSURE
                              else payload.scalar),
                        wp.vec3(*(payload.vector if plan.operation in {
                            SourceOperation.VECTOR_FIELD,
                            SourceOperation.STAGGERED_VELOCITY} else payload.velocity)),
                        int(plan.blend.mode), float(plan.blend.amount), float(dt),
                    ],
                    outputs=[owner.voxel_type.data, pressure,
                             owner.temperature.data, owner.viscosity_field.data,
                             owner.density.data, owner.vel.u, owner.vel.v, owner.vel.w],
                    device=self.device,
                )
            if (plan.operation == SourceOperation.LIQUID
                    and region.points is not None
                    and plan.lifecycle == SourceLifecycle.DIRECT_APPEND):
                point_values = region.points
                count = len(point_values.positions)
                wp.launch(
                    k_source_owned_points, dim=max(1, count),
                    inputs=[
                        point_values.device_positions,
                        (point_values.device_velocities
                         if point_values.device_velocities is not None else empty_vector),
                        (point_values.device_temperature
                         if point_values.device_temperature is not None else empty_scalar),
                        (point_values.device_viscosity
                         if point_values.device_viscosity is not None else empty_scalar),
                        (point_values.device_density
                         if point_values.device_density is not None else empty_scalar),
                        (point_values.device_lifespan
                         if point_values.device_lifespan is not None else empty_scalar),
                        count,
                        1 if point_values.device_velocities is not None else 0,
                        1 if point_values.device_temperature is not None else 0,
                        1 if point_values.device_viscosity is not None else 0,
                        1 if point_values.device_density is not None else 0,
                        1 if point_values.device_lifespan is not None else 0,
                        wp.vec3(*payload.velocity), payload.temperature,
                        payload.viscosity, payload.density, payload.lifespan,
                        1 if payload.use_lifespan else 0, particles.capacity,
                    ],
                    outputs=[particles.position, particles.velocity,
                             particles.attr("age"), particles.attr("life"),
                             particles.attr("temperature"),
                             particles.attr("viscosity"), particles.attr("density"),
                             particles.time_offset, particles.alive,
                             particles.count, self._compiled_result,
                             self._emission_state],
                    device=self.device,
                )
            elif plan.operation == SourceOperation.LIQUID:
                if not region.leaf_count:
                    continue
                wp.launch(
                    k_emit_compiled_source_particles,
                    dim=region.leaf_count * C.LEAF_VOLUME,
                    inputs=[
                        region.device_coverage, region.device_leaf_coords,
                        region.leaf_count, grid.leaf_table, grid.voxels_dim,
                        grid.leaves_dim, grid.origin,
                        self.owner.settings.voxel_size,
                        normalized_liquid_target(plan),
                        wp.vec3(*payload.velocity), payload.temperature,
                        payload.viscosity, payload.density, payload.lifespan,
                        1 if payload.use_lifespan else 0, particles.capacity,
                        owner.voxel_counts.data,
                    ],
                    outputs=[particles.position, particles.velocity,
                             particles.attr("age"), particles.attr("life"),
                             particles.attr("temperature"),
                             particles.attr("viscosity"), particles.attr("density"),
                             particles.time_offset, particles.alive,
                             particles.count, self._compiled_result,
                             self._emission_state],
                    device=self.device,
                )
            if (plan.lifecycle == SourceLifecycle.ONCE
                    and plan.operation in {
                        SourceOperation.COLLISION, SourceOperation.SINK,
                        SourceOperation.ACTIVATION}):
                self._initial_fired.add(stable_id)
        if finalize:
            return self._finish_emission_transaction()
        return False

    def apply_post_p2g_fields(self, dt):
        """Apply scalar/vector/staggered fields after P2G so they persist."""
        batch = self._compiled_batch
        self.last_post_p2g_apply_count = 0
        if batch is None:
            return 0
        owner, grid = self.owner, self.owner.grid
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        operations = {
            SourceOperation.SCALAR_FIELD, SourceOperation.VECTOR_FIELD,
            SourceOperation.STAGGERED_VELOCITY,
        }
        for compiled in batch.sources:
            plan, region = compiled.plan, compiled.region
            if plan.operation not in operations:
                continue
            stable_id = str(plan.stable_id)
            if (plan.lifecycle == SourceLifecycle.ONCE
                    and stable_id in self._initial_fired):
                continue
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            payload = plan.payload
            if plan.operation == SourceOperation.STAGGERED_VELOCITY:
                has_faces = all(value is not None for value in (
                    region.device_staggered_u, region.device_staggered_v,
                    region.device_staggered_w))
                wp.launch(
                    k_apply_compiled_staggered_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        region.device_coverage,
                        (region.device_staggered_u if has_faces else empty_scalar),
                        (region.device_staggered_v if has_faces else empty_scalar),
                        (region.device_staggered_w if has_faces else empty_scalar),
                        region.device_leaf_coords, region.leaf_count,
                        grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                        wp.vec3(*payload.vector), 1 if has_faces else 0,
                        int(plan.blend.mode), float(plan.blend.amount), float(dt),
                    ],
                    outputs=[owner.vel.u, owner.vel.v, owner.vel.w],
                    device=self.device,
                )
            else:
                wp.launch(
                    k_apply_compiled_source,
                    dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                    inputs=[
                        int(plan.operation), region.device_coverage,
                        (region.device_scalar if region.device_scalar is not None
                         else empty_scalar),
                        (region.device_vector if region.device_vector is not None
                         else empty_vector),
                        region.device_leaf_coords, region.leaf_count,
                        grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                        float(payload.scalar), wp.vec3(*payload.vector),
                        int(plan.blend.mode), float(plan.blend.amount), float(dt),
                    ],
                    outputs=[owner.voxel_type.data, owner.pressure_solver.pressure,
                             owner.temperature.data, owner.viscosity_field.data,
                             owner.density.data, owner.vel.u, owner.vel.v, owner.vel.w],
                    device=self.device,
                )
            self.last_post_p2g_apply_count += int(region.covered_voxel_count)
            if plan.lifecycle == SourceLifecycle.ONCE:
                self._initial_fired.add(stable_id)
        return self.last_post_p2g_apply_count

    def apply_pressure_sources(self, dt):
        """Blend FIELD pressure after the solve and immediately before project."""
        batch = self._compiled_batch
        self.last_pressure_apply_count = 0
        if batch is None:
            return 0
        owner, grid = self.owner, self.owner.grid
        empty_scalar = self._compiled_empty_scalar
        empty_vector = self._compiled_empty_vector
        for compiled in batch.sources:
            plan, region = compiled.plan, compiled.region
            if plan.operation != SourceOperation.PRESSURE:
                continue
            stable_id = str(plan.stable_id)
            if (plan.lifecycle == SourceLifecycle.ONCE
                    and stable_id in self._initial_fired):
                continue
            if any(item.severity == "ERROR" for item in region.diagnostics):
                continue
            payload = plan.payload
            wp.launch(
                k_apply_compiled_source,
                dim=max(1, region.leaf_count * C.LEAF_VOLUME),
                inputs=[
                    int(SourceOperation.PRESSURE), region.device_coverage,
                    (region.device_scalar if region.device_scalar is not None
                     else empty_scalar), empty_vector,
                    region.device_leaf_coords, region.leaf_count,
                    grid.leaf_table, grid.voxels_dim, grid.leaves_dim,
                    float(payload.pressure), wp.vec3(0.0, 0.0, 0.0),
                    int(plan.blend.mode), float(plan.blend.amount), float(dt),
                ],
                outputs=[owner.voxel_type.data, owner.pressure_solver.pressure,
                         owner.temperature.data, owner.viscosity_field.data,
                         owner.density.data, owner.vel.u, owner.vel.v, owner.vel.w],
                device=self.device,
            )
            self.last_pressure_apply_count += int(region.covered_voxel_count)
            if plan.lifecycle == SourceLifecycle.ONCE:
                self._initial_fired.add(stable_id)
        return self.last_pressure_apply_count

    def apply(self, dt):
        """Apply one all-or-nothing particle-emission transaction."""
        batch = None
        if getattr(self, "plans", ()):
            dynamic = any(
                plan.cadence == SourceEvaluationCadence.EVERY_SIM_FRAME
                for plan in self.plans)
            batch = getattr(self, "_compiled_batch", None)
            if batch is None or dynamic:
                batch = self.compile(
                    frame=int(getattr(self.owner, "frame", 0)), substep=0)

        include_legacy = bool(self.sources or self.height_source.enabled)
        self._begin_emission_transaction(
            batch, include_legacy=include_legacy)
        if batch is not None:
            self.apply_compiled(
                batch, dt, transaction_ready=True, finalize=False)
        if not include_legacy:
            return self._finish_emission_transaction()

        owner = self.owner
        grid = owner.grid
        settings = owner.settings
        analytic_sources = tuple(getattr(
            self, "analytic_sources", self.sources))
        mesh_sources = tuple(getattr(self, "mesh_sources", ()))
        shape_count = len(analytic_sources)
        shapes = self._shapes()

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
        if height.enabled and self._transaction_height_armed:
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
                        wp.vec3(*height.velocity),
                        1 if (height.emit_particles
                              and self._legacy_liquid_target(height) > 0) else 0,
                        self._emission_state],
                outputs=[owner.voxel_type.data, owner.vel.u, owner.vel.v,
                         owner.vel.w],
                device=self.device,
            )

        wp.launch(
            k_source_resolve_type, dim=owner._nvox,
            inputs=[owner.voxel_type.data, grid.num_leaves],
            device=self.device,
        )

        particles = owner.particles
        if self._transaction_shape_count:
            wp.launch(
                k_source_shapes_particles, dim=owner._nvox,
                inputs=[
                    self._transaction_shapes, self._transaction_shape_count,
                    grid.leaf_coords, grid.num_leaves, grid.origin,
                    settings.voxel_size, owner.voxel_counts.data,
                    self._transaction_emit_seed, particles.capacity,
                ],
                outputs=[particles.position, particles.velocity,
                         particles.attr("age"), particles.attr("life"),
                         particles.attr("temperature"),
                         particles.attr("viscosity"),
                         particles.attr("density"), particles.time_offset,
                         particles.alive, particles.count,
                         self._emission_state],
                device=self.device,
            )
        for mesh_index, source in enumerate(self._transaction_mesh_sources):
            wp.launch(
                k_mesh_source_particles, dim=owner._nvox,
                inputs=[
                    source.mesh.id, self._bake_struct(source),
                    source.query_radius(), grid.leaf_coords,
                    grid.num_leaves, grid.origin, settings.voxel_size,
                    owner.voxel_counts.data,
                    self._transaction_emit_seed + mesh_index + 1,
                    particles.capacity,
                ],
                outputs=[particles.position, particles.velocity,
                         particles.attr("age"), particles.attr("life"),
                         particles.attr("temperature"),
                         particles.attr("viscosity"),
                         particles.attr("density"), particles.time_offset,
                         particles.alive, particles.count,
                         self._emission_state],
                device=self.device,
            )
        return self._finish_emission_transaction()

    def _upload_direct_packets(self, packets):
        if self._compiled_empty_scalar is None:
            self._compiled_empty_scalar = wp.empty(
                0, dtype=float, device=self.device)
            self._compiled_empty_vector = wp.empty(
                0, dtype=wp.vec3, device=self.device)
        uploaded = []
        for packet in packets:
            positions = wp.array(
                packet.positions, dtype=wp.vec3, device=self.device)
            velocities = (
                wp.array(packet.velocities, dtype=wp.vec3, device=self.device)
                if packet.velocities is not None else self._compiled_empty_vector
            )
            uploaded.append((packet, positions, velocities))
        return tuple(uploaded)

    def _direct_packet_attempt(self, uploaded, required):
        """Run one device-authoritative admit/commit attempt for all packets."""
        particles = self.owner.particles
        self._emission_state.zero_()
        self._compiled_result.zero_()
        wp.launch(
            k_add_emission_demand, dim=1,
            inputs=[required, self._emission_state], device=self.device)
        wp.launch(
            k_admit_emission_batch, dim=1,
            inputs=[particles.count, particles.capacity, self._emission_state],
            device=self.device)
        empty_scalar = self._compiled_empty_scalar
        for packet, positions, velocities in uploaded:
            count = len(packet.positions)
            wp.launch(
                k_source_owned_points, dim=max(1, count),
                inputs=[
                    positions, velocities, empty_scalar, empty_scalar,
                    empty_scalar, empty_scalar, count,
                    1 if packet.velocities is not None else 0,
                    0, 0, 0, 0, wp.vec3(*packet.velocity),
                    float(packet.temperature), float(packet.viscosity),
                    float(packet.density), float(packet.life),
                    1 if packet.use_lifespan else 0, particles.capacity,
                ],
                outputs=[
                    particles.position, particles.velocity,
                    particles.attr("age"), particles.attr("life"),
                    particles.attr("temperature"),
                    particles.attr("viscosity"), particles.attr("density"),
                    particles.time_offset, particles.alive, particles.count,
                    self._compiled_result, self._emission_state,
                ], device=self.device)

        values = np.asarray(self._emission_state.numpy(), dtype=np.int64)
        demand, admitted, emitted, invariant, available = map(
            int, values[:5])
        initial = max(0, int(particles.capacity) - available)
        if invariant or (admitted and emitted != demand):
            raise RuntimeError(
                "direct particle transaction violated its admitted count")
        if admitted:
            total = particles.refresh_host_count() if emitted else initial
            if emitted:
                self.owner._host_frame_cache = None
            return InjectionResult(
                True, emitted, demand, available, False, int(total))
        return InjectionResult(False, 0, demand, available, True, initial)

    def inject_particle_packets_exact(self, packets):
        """Append every packet or none and return exact device-derived truth."""
        normalized = tuple(
            packet if isinstance(packet, DirectInjectionPacket)
            else DirectInjectionPacket(
                positions=packet.positions,
                velocity=tuple(packet.velocity),
                temperature=float(packet.temperature),
                viscosity=float(packet.viscosity),
                density=float(packet.density),
                life=float(packet.life),
                use_lifespan=bool(packet.use_lifespan),
            )
            for packet in packets
            if len(packet.positions)
        )
        required = sum(len(packet.positions) for packet in normalized)
        if not required:
            particles = self.owner.particles
            result = InjectionResult(
                True, 0, 0,
                max(0, int(particles.capacity) - int(particles.host_count)),
                False, int(particles.host_count))
            self.last_injection_result = result
            self.last_apply_result = {
                "emitted": 0, "capacity_limited": False,
                "required": 0, "available": result.available,
                "admitted": True, "total": result.total,
            }
            return result

        uploaded = self._upload_direct_packets(normalized)
        result = self._direct_packet_attempt(uploaded, required)
        growth = None
        if not result.admitted:
            growth = self._request_particle_growth(result.total + required)
            if growth.grew:
                result = self._direct_packet_attempt(uploaded, required)

        self.last_injection_result = result
        self.last_apply_result = {
            "emitted": result.emitted,
            "capacity_limited": result.capacity_limited,
            "required": result.required,
            "available": result.available,
            "admitted": result.admitted,
            "total": result.total,
        }
        if not result.admitted:
            policy = str(getattr(
                self.owner.settings,
                "particle_growth_failure_policy", "DEFER",
            )).upper()
            self.last_apply_result.update({
                "deferred": True,
                "pause_requested": policy == "PAUSE",
                "growth_status": growth.status,
                "growth_previous": growth.previous,
                "growth_current": growth.current,
                "growth_target": growth.target,
                "growth_reason": growth.reason,
            })
        return result

    def inject_particles_exact(self, positions, velocity=(0.0, 0.0, 0.0),
                               velocities=None, temperature=0.0, viscosity=0.0,
                               density=1.0, life=0.0, use_lifespan=False):
        packet = DirectInjectionPacket(
            positions=positions, velocity=tuple(velocity),
            velocities=velocities, temperature=float(temperature),
            viscosity=float(viscosity), density=float(density),
            life=float(life), use_lifespan=bool(use_lifespan))
        return self.inject_particle_packets_exact((packet,))

    def inject_particles(self, positions, velocity=(0.0, 0.0, 0.0),
                         velocities=None, temperature=0.0, viscosity=0.0,
                         density=1.0, life=0.0, use_lifespan=False):
        """Compatibility API: final total on success, zero on refusal/empty."""
        result = self.inject_particles_exact(
            positions, velocity=velocity, velocities=velocities,
            temperature=temperature, viscosity=viscosity, density=density,
            life=life, use_lifespan=use_lifespan)
        return result.total if result.admitted and result.emitted else 0
