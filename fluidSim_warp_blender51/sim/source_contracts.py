"""Stable source descriptors and transaction result contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import warp as wp

from .attributes import normalize_color, stable_source_id
from .enums import (
    BlendMode,
    MeshSinkRegionMode,
    MeshSourceRegion,
    SourceEmissionMode,
    SourceEvaluationCadence,
    SourceShapeForceType,
    SourceShapeShape,
    SourceShapeSourceMode,
    SourceShapeType,
)


def analytic_world_half_extents(shape, rotation, half_extents, radius, height, feather):
    """Per-axis world half extents that bound one rotated analytic shape.

    The historical bound is a cube of the largest authored size. It stays the
    floor, so every unrotated shape keeps its exact previous bound, and each
    axis widens to the rotated local box (sum of |R| times the local half
    extents) so a rotated shape's corners are no longer clipped. Spheres are
    rotation-invariant and planes are half-spaces, so both keep the cube.
    """
    feather = max(0.0, float(feather))
    cube = max(abs(float(radius)), abs(float(height)),
               *(abs(float(value)) for value in half_extents)) + feather
    shape = int(shape)
    if shape in (SourceShapeShape.SPHERE, SourceShapeShape.PLANE):
        return (cube, cube, cube)
    if shape == SourceShapeShape.BOX:
        local = np.abs(np.asarray(half_extents, np.float64))
    else:
        # Cylinder, cone and pyramid: radius across x/y, half height on z.
        local = np.abs(np.array((radius, radius, height), np.float64))
    q = np.asarray(rotation, np.float64)
    norm = float(np.linalg.norm(q))
    x, y, z, w = (q / norm) if norm > 0.0 else (0.0, 0.0, 0.0, 1.0)
    matrix = np.array([
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
        (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y),
    ])
    world = np.abs(matrix) @ local + feather
    return tuple(max(cube, float(value)) for value in world)


@wp.struct
class SourceShape:
    enabled: wp.int32
    shape: wp.int32  # SourceShapeShape: 0 Sphere 1 Cylinder 2 Box 3 Pyramid 4 Cone 5 Plane
    stype: wp.int32  # SourceShapeType: 0 Source 1 Collision 2 Sink 3 Activation 4 CameraFrustum
    mode: wp.int32  # SourceShapeSourceMode: 0 Field 1 Particles
    xform: wp.transform  # local -> world
    inv_xform: wp.transform
    half_extents: wp.vec3  # box half extents / general scale
    radius: wp.float32
    height: wp.float32  # cylinder/cone/pyramid half height
    feather: wp.float32  # soft boundary width (world units)
    velocity: wp.vec3  # emission velocity (world)
    linear_velocity: wp.vec3  # rigid motion of the shape itself (colliders)
    angular_velocity: wp.vec3
    blend_mode: wp.int32
    blend_amount: wp.float32
    force_type: wp.int32  # 0 Off 1 Directional 2 Omnidirectional 3 Spin 4 Radial
    force_strength: wp.float32
    force_dir: wp.vec3
    emit_rate: wp.float32  # target particles per voxel
    temperature: wp.float32
    viscosity: wp.float32
    density: wp.float32
    color: wp.vec4
    source_numeric_id: wp.int32
    life_min: wp.float32
    life_max: wp.float32
    use_lifespan: wp.int32
    emission_mode: wp.int32  # SourceEmissionMode: 0 Initial Fill, 1 Continuous Inflow
    armed: wp.int32  # 1 = may emit initial fill; host disarms after first fire


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
    force_type: wp.int32  # 0 Off 1 Directional 2 Omnidirectional 3 Spin 4 Radial
    force_strength: wp.float32
    force_dir: wp.vec3
    force_center: wp.vec3
    blend_mode: wp.int32
    blend_amount: wp.float32
    emit_rate: wp.float32
    temperature: wp.float32
    viscosity: wp.float32
    density: wp.float32
    color: wp.vec4
    source_numeric_id: wp.int32
    life_min: wp.float32
    life_max: wp.float32
    use_lifespan: wp.int32
    emission_mode: wp.int32  # SourceEmissionMode: 0 Initial Fill, 1 Continuous Inflow
    armed: wp.int32  # 1 = may emit initial fill; host disarms after first fire


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
    color: tuple = (0.1, 0.45, 1.0, 1.0)
    source_numeric_id: int = 0
    life_min: float = 1.0e30
    life_max: float = 1.0e30
    use_lifespan: bool = False
    emission_mode: int = SourceEmissionMode.CONTINUOUS_INFLOW
    source_id: str = ""
    runtime_source_key: str = ""

    def aabb(self):
        """Return a conservative world AABB for topology marking."""
        ex, ey, ez = analytic_world_half_extents(
            self.shape, self.rotation, self.half_extents, self.radius,
            self.height, self.feather)
        x, y, z = self.position
        return ((x - ex, y - ey, z - ez), (x + ex, y + ey, z + ez))

    def to_struct(self):
        shape = SourceShape()
        shape.enabled = 1 if self.enabled else 0
        shape.shape = int(self.shape)
        shape.stype = int(self.stype)
        shape.mode = int(self.mode)
        rotation = wp.quat(*[float(value) for value in self.rotation])
        shape.xform = wp.transform(wp.vec3(*[float(value) for value in self.position]), rotation)
        shape.inv_xform = wp.transform_inverse(shape.xform)
        shape.half_extents = wp.vec3(*[float(value) for value in self.half_extents])
        shape.radius = float(self.radius)
        shape.height = float(self.height)
        shape.feather = float(self.feather)
        shape.velocity = wp.vec3(*[float(value) for value in self.velocity])
        shape.linear_velocity = wp.vec3(*[float(value) for value in self.linear_velocity])
        shape.angular_velocity = wp.vec3(*[float(value) for value in self.angular_velocity])
        shape.blend_mode = int(self.blend_mode)
        shape.blend_amount = float(self.blend_amount)
        shape.force_type = int(self.force_type)
        shape.force_strength = float(self.force_strength)
        shape.force_dir = wp.vec3(*[float(value) for value in self.force_dir])
        shape.emit_rate = float(self.emit_rate)
        shape.temperature = float(self.temperature)
        shape.viscosity = float(self.viscosity)
        shape.density = float(self.density)
        shape.color = wp.vec4(*normalize_color(self.color))
        shape.source_numeric_id = int(self.source_numeric_id or stable_source_id(self.source_id))
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
    color: tuple = (0.1, 0.45, 1.0, 1.0)
    source_numeric_id: int = 0
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
        extent = tuple(float(self.aabb_max[i]) - float(self.aabb_min[i]) for i in range(3))
        return max(1.0e-6, float(np.linalg.norm(extent)) + max(float(self.surface_thickness), 0.0))

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
        source.force_center = wp.vec3(*[float(value) for value in self.force_center])
        source.blend_mode = int(self.blend_mode)
        source.blend_amount = float(self.blend_amount)
        source.emit_rate = float(self.emit_rate)
        source.temperature = float(self.temperature)
        source.viscosity = float(self.viscosity)
        source.density = float(self.density)
        source.color = wp.vec4(*normalize_color(self.color))
        source.source_numeric_id = int(self.source_numeric_id or stable_source_id(self.source_id))
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
    color: tuple = (0.1, 0.45, 1.0, 1.0)
    source_numeric_id: int = 0


@dataclass(frozen=True)
class NormalizedEmissionPlan:
    """One immutable admission input spanning native and legacy emitters."""

    compiled_sources: tuple = ()
    legacy_attempted: bool = False
    pending_fired: frozenset[str] = frozenset()

    @property
    def attempted(self) -> bool:
        return bool(self.compiled_sources or self.legacy_attempted)


@dataclass(frozen=True)
class SourceApplyResult(Mapping[str, object]):
    """Complete immutable receipt for one source application attempt."""

    emitted: int
    capacity_limited: bool
    required: int
    available: int
    admitted: bool
    deferred: bool = False
    pause_requested: bool = False
    growth_status: str = "UNCHANGED"
    growth_previous: int = 0
    growth_current: int = 0
    growth_target: int = 0
    growth_reason: str = ""
    invariant_violation: bool = False
    total: int = 0

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "emitted",
        "capacity_limited",
        "required",
        "available",
        "admitted",
        "deferred",
        "pause_requested",
        "growth_status",
        "growth_previous",
        "growth_current",
        "growth_target",
        "growth_reason",
        "invariant_violation",
        "total",
    )

    def __getitem__(self, key: str) -> object:
        if key not in self._FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELDS)

    def __len__(self) -> int:
        return len(self._FIELDS)


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
    color: tuple[float, float, float, float] = (0.1, 0.45, 1.0, 1.0)
    source_id: int = 0
    life: float = 0.0
    use_lifespan: bool = False

    def __post_init__(self):
        positions = np.array(self.positions, dtype=np.float32, copy=True, order="C").reshape(
            (-1, 3)
        )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "color", normalize_color(self.color))
        source_id = int(self.source_id)
        if source_id < 0 or source_id > np.iinfo(np.int32).max:
            raise ValueError("source_id must fit a non-negative int32")
        object.__setattr__(self, "source_id", source_id)
        if self.velocities is not None:
            velocities = np.array(self.velocities, dtype=np.float32, copy=True, order="C").reshape(
                (-1, 3)
            )
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
