"""Phase 2 normalized source-region compiler.

This is the Blender-independent deep module for source geometry, validation,
cache ownership, deterministic sampling and flat 8x8x8 source leaves. OpenVDB
objects are staging values only and never survive :meth:`SourceCompiler.compile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import math
import os
from pathlib import Path
from typing import NewType, Optional, Sequence, Union

import numpy as np
import warp as wp

from . import constants as C
from .capacity_planner import DemandQuality
from .source_contracts import analytic_world_half_extents
from .enums import (
    BlendMode,
    SourceEvaluationCadence,
    SourceHeightBackMode,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceRegionKind,
    SourceShapeShape,
)

SourceKey = NewType("SourceKey", str)
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


@wp.kernel
def _k_query_compiled_mesh(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        solid: int,
        shell_width: float,
        coverage: wp.array(dtype=float),
        sdf: wp.array(dtype=float)):
    """BVH-accelerated mesh containment/distance staging kernel."""
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point(mesh_id, point, 1.0e20)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance = wp.length(closest - point)
        if solid != 0:
            signed = distance * query.sign
            sdf[tid] = signed
            if signed <= 0.0:
                coverage[tid] = 1.0
        else:
            sdf[tid] = distance
            if distance <= shell_width:
                coverage[tid] = 1.0


@wp.kernel
def _k_query_compiled_mesh_parity(
        mesh_id: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        solid: int,
        shell_width: float,
        coverage: wp.array(dtype=float),
        sdf: wp.array(dtype=float)):
    """CPU-BVH staging matching the NumPy fallback's semantics.

    Unsigned closest distance plus orientation-independent ray parity for
    solids, like ``SourceCompiler._mesh_inside``; the CUDA kernel above keeps
    its winding-number sign unchanged.
    """
    tid = wp.tid()
    point = points[tid]
    query = wp.mesh_query_point_no_sign(mesh_id, point, 1.0e20)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance = wp.length(closest - point)
        sdf[tid] = distance
        if solid != 0:
            inside = wp.mesh_query_point_sign_parity(mesh_id, point, 1.0e20, 3, 0.1)
            if inside.result and inside.sign < 0.0:
                sdf[tid] = -distance
                coverage[tid] = 1.0
        elif distance <= shell_width:
            coverage[tid] = 1.0


def _owned_array(value, dtype, shape_tail=()):
    if value is None:
        return None
    array = np.array(value, dtype=dtype, copy=True, order="C")
    shape_tail = tuple(shape_tail)
    expected_rank = 1 + len(shape_tail)
    if array.ndim != expected_rank or array.shape[1:] != shape_tail:
        expected = (
            "(N,)" if not shape_tail
            else f"(N, {', '.join(map(str, shape_tail))})"
        )
        raise ValueError(
            f"array must have shape {expected}; received {array.shape}")
    array.setflags(write=False)
    return array


# Every array-valued field of any region plan; all are part of the cache key.
_REGION_ARRAY_FIELDS = (
    "vertices", "triangles", "positions", "velocities", "temperature",
    "viscosity", "density", "lifespan", "profile",
)


def _array_digest(value):
    if value is None:
        return "-"
    value = np.ascontiguousarray(value)
    return sha256(value.view(np.uint8)).hexdigest()


def _vdb_index_to_world_matrix(grid):
    """Extract a linear OpenVDB grid transform as a row-major 4x4 matrix."""
    transform = getattr(grid, "transform", None)
    if transform is None or not hasattr(transform, "indexToWorld"):
        return np.eye(4, dtype=np.float64)
    is_linear = getattr(transform, "isLinear", True)
    if callable(is_linear):
        is_linear = is_linear()
    if not is_linear:
        raise ValueError("non-linear OpenVDB grid transforms are unsupported")
    origin = np.asarray(transform.indexToWorld((0.0, 0.0, 0.0)), np.float64)
    matrix = np.eye(4, dtype=np.float64)
    for axis in range(3):
        basis = [0.0, 0.0, 0.0]
        basis[axis] = 1.0
        matrix[:3, axis] = (
            np.asarray(transform.indexToWorld(tuple(basis)), np.float64) - origin)
    matrix[:3, 3] = origin
    return matrix


@dataclass(frozen=True)
class SourceDiagnostic:
    severity: str
    code: str
    message: str
    stable_id: SourceKey


@dataclass(frozen=True)
class AnalyticRegionPlan:
    shape: SourceShapeShape = SourceShapeShape.SPHERE
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    half_extents: Vec3 = (0.5, 0.5, 0.5)
    radius: float = 0.5
    height: float = 0.5
    feather: float = 0.0
    kind: SourceRegionKind = field(
        default=SourceRegionKind.ANALYTIC_SDF, init=False)

    def bounds(self):
        extent = np.asarray(analytic_world_half_extents(
            self.shape, self.rotation, self.half_extents, self.radius,
            self.height, self.feather), dtype=np.float64)
        p = np.asarray(self.position, dtype=np.float64)
        return tuple(p - extent), tuple(p + extent)


@dataclass(frozen=True)
class MeshRegionPlan:
    vertices: np.ndarray
    triangles: np.ndarray
    solid: bool = False
    shell_half_width: float = 0.0
    kind: SourceRegionKind = field(init=False)

    def __post_init__(self):
        vertices = _owned_array(self.vertices, np.float32, (3,))
        triangles = _owned_array(self.triangles, np.int32, (3,))
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(
            self, "kind",
            SourceRegionKind.MESH_SOLID if self.solid
            else SourceRegionKind.MESH_SHELL,
        )

    def bounds(self):
        if not len(self.vertices):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        band = 0.0 if self.solid else max(0.0, float(self.shell_half_width))
        return (tuple(self.vertices.min(axis=0) - band),
                tuple(self.vertices.max(axis=0) + band))


@dataclass(frozen=True)
class PointRegionPlan:
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    viscosity: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None
    lifespan: Optional[np.ndarray] = None
    rasterize_radius: float = 0.0
    kind: SourceRegionKind = field(default=SourceRegionKind.POINTS, init=False)

    def __post_init__(self):
        positions = _owned_array(self.positions, np.float32, (3,))
        count = len(positions)
        object.__setattr__(self, "positions", positions)
        for name, dtype, tail in (
            ("velocities", np.float32, (3,)),
            ("temperature", np.float32, ()),
            ("viscosity", np.float32, ()),
            ("density", np.float32, ()),
            ("lifespan", np.float32, ()),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = _owned_array(value, dtype, tail)
            if len(array) != count:
                raise ValueError(f"point attribute {name} length must match positions")
            object.__setattr__(self, name, array)

    def bounds(self):
        if not len(self.positions):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        radius = max(0.0, float(self.rasterize_radius))
        return (tuple(self.positions.min(axis=0) - radius),
                tuple(self.positions.max(axis=0) + radius))


@dataclass(frozen=True)
class VolumeRegionPlan:
    filepath: str
    mask_grid: str = ""
    sdf_grid: str = ""
    scalar_grid: str = ""
    vector_grid: str = ""
    vector_x_grid: str = ""
    vector_y_grid: str = ""
    vector_z_grid: str = ""
    transform: tuple[float, ...] = (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    fog_threshold: float = 0.0
    kind: SourceRegionKind = field(default=SourceRegionKind.VOLUME_GRID, init=False)

    def __post_init__(self):
        object.__setattr__(self, "filepath", os.path.abspath(os.fspath(self.filepath)))
        if len(self.transform) != 16:
            raise ValueError("volume transform must contain 16 values")
        if not any((self.mask_grid, self.sdf_grid, self.scalar_grid,
                    self.vector_grid, self.vector_x_grid,
                    self.vector_y_grid, self.vector_z_grid)):
            raise ValueError("volume region requires at least one named grid")
        components = (self.vector_x_grid, self.vector_y_grid, self.vector_z_grid)
        if any(components) and not all(components):
            raise ValueError("vector-component volume input requires x, y, and z grids")

    def _named_grids(self):
        return tuple(name for name in (
            self.mask_grid, self.sdf_grid, self.scalar_grid, self.vector_grid,
            self.vector_x_grid, self.vector_y_grid, self.vector_z_grid) if name)

    def bounds(self):
        """Return transformed active topology bounds from staging VDB metadata.

        Blender's RNA exposes no voxel values, so the bundled OpenVDB binding is
        the truthful metadata/value seam. Grid objects remain local and are
        released before this method returns.
        """
        if not Path(self.filepath).is_file():
            raise ValueError(f"volume file does not exist: {self.filepath}")
        try:
            import openvdb
        except ImportError as exc:
            raise RuntimeError("Blender-bundled openvdb is unavailable") from exc
        grids = []
        try:
            for name in self._named_grids():
                try:
                    grids.append(openvdb.read(self.filepath, name))
                except Exception as exc:
                    raise ValueError(
                        f"missing or unreadable volume grid {name!r}: {exc}") from exc
            plan_matrix = np.asarray(self.transform, np.float64).reshape(4, 4)
            world_corners = []
            for grid in grids:
                lo, hi = grid.evalActiveVoxelBoundingBox()
                corners = np.array([
                    (x, y, z, 1.0) for x in (lo[0], hi[0] + 1)
                    for y in (lo[1], hi[1] + 1)
                    for z in (lo[2], hi[2] + 1)])
                combined = plan_matrix @ _vdb_index_to_world_matrix(grid)
                world_corners.append(corners @ combined.T)
            world = np.concatenate(world_corners, axis=0)
            return (tuple(world[:, :3].min(axis=0)),
                    tuple(world[:, :3].max(axis=0)))
        finally:
            grids.clear()


@dataclass(frozen=True)
class HeightRegionPlan:
    base_height: float
    back_mode: SourceHeightBackMode = SourceHeightBackMode.FLAT
    profile: Optional[np.ndarray] = None
    kind: SourceRegionKind = field(default=SourceRegionKind.HEIGHT, init=False)

    def __post_init__(self):
        profile = self.profile
        if profile is not None:
            profile = np.array(profile, dtype=np.float32, copy=True, order="C")
            if profile.ndim != 2 or min(profile.shape) < 2:
                raise ValueError("height profile must be a 2D array of at least 2x2")
            profile.setflags(write=False)
            object.__setattr__(self, "profile", profile)
        if self.back_mode == SourceHeightBackMode.PROFILE and profile is None:
            raise ValueError("PROFILE height mode requires an explicit profile")
        if self.back_mode == SourceHeightBackMode.INFINITE:
            raise ValueError(
                "INFINITE height back mode is not defined by source_shape_spec; "
                "it remains hidden and unsupported")


RegionPlan = Union[
    AnalyticRegionPlan,
    MeshRegionPlan,
    PointRegionPlan,
    VolumeRegionPlan,
    HeightRegionPlan,
]


@dataclass(frozen=True)
class PayloadPlan:
    kind: SourcePayloadKind = SourcePayloadKind.NONE
    velocity: Vec3 = (0.0, 0.0, 0.0)
    pressure: float = 0.0
    scalar: float = 0.0
    vector: Vec3 = (0.0, 0.0, 0.0)
    temperature: float = 0.0
    viscosity: float = 0.0
    density: float = 1.0
    color: Vec4 = (0.1, 0.45, 1.0, 1.0)
    source_id: int = 0
    lifespan: float = 0.0
    use_lifespan: bool = False
    target_particles_per_voxel: float = 8.0


@dataclass(frozen=True)
class ActivationPlan:
    enabled: bool = True
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None

    def active(self, frame):
        return (self.enabled
                and (self.frame_start is None or frame >= self.frame_start)
                and (self.frame_end is None or frame <= self.frame_end))


@dataclass(frozen=True)
class BlendPlan:
    mode: BlendMode = BlendMode.REPLACE
    amount: float = 1.0


@dataclass(frozen=True)
class SourcePlan:
    stable_id: SourceKey
    operation: SourceOperation
    lifecycle: SourceLifecycle
    cadence: SourceEvaluationCadence
    region: RegionPlan
    payload: PayloadPlan = PayloadPlan()
    activation: ActivationPlan = ActivationPlan()
    blend: BlendPlan = BlendPlan()

    def __post_init__(self):
        if not str(self.stable_id):
            raise ValueError("stable source identity must be non-empty")
        if self.lifecycle == SourceLifecycle.DIRECT_APPEND:
            if self.region.kind != SourceRegionKind.POINTS:
                raise ValueError("DIRECT_APPEND is valid only for point inputs")
            if self.operation != SourceOperation.LIQUID:
                raise ValueError("DIRECT_APPEND is valid only for liquid point input")
        if (self.operation == SourceOperation.LIQUID
                and self.payload.kind in {
                    SourcePayloadKind.PRESSURE, SourcePayloadKind.SCALAR,
                    SourcePayloadKind.VECTOR, SourcePayloadKind.STAGGERED_VECTOR,
                }):
            raise ValueError("field payloads cannot be represented as liquid emission")
        expected = {
            SourceOperation.PRESSURE: SourcePayloadKind.PRESSURE,
            SourceOperation.SCALAR_FIELD: SourcePayloadKind.SCALAR,
            SourceOperation.VECTOR_FIELD: SourcePayloadKind.VECTOR,
            SourceOperation.STAGGERED_VELOCITY: SourcePayloadKind.STAGGERED_VECTOR,
        }
        if self.operation in expected and self.payload.kind != expected[self.operation]:
            raise ValueError(
                f"{self.operation.name} requires {expected[self.operation].name} payload")


@dataclass(frozen=True)
class CompileContext:
    frame: int
    substep: int
    session_generation: int
    device: str
    origin: Vec3
    voxel_size: float
    domain_min: Vec3
    domain_max: Vec3
    leaf_capacity: int = C.DEFAULT_LEAF_CAPACITY
    particle_capacity: int = C.DEFAULT_MAX_PARTICLES
    max_source_bytes: int = 512 * 1024 * 1024
    upload: bool = True


@dataclass
class CompiledPoints:
    positions: np.ndarray
    velocities: Optional[np.ndarray]
    temperature: Optional[np.ndarray]
    viscosity: Optional[np.ndarray]
    density: Optional[np.ndarray]
    lifespan: Optional[np.ndarray]
    device_positions: object = None
    device_velocities: object = None
    device_temperature: object = None
    device_viscosity: object = None
    device_density: object = None
    device_lifespan: object = None


@dataclass
class CompiledRegion:
    kind: SourceRegionKind
    cache_key: str
    leaf_coords: np.ndarray
    coverage: np.ndarray
    sdf: Optional[np.ndarray] = None
    scalar: Optional[np.ndarray] = None
    vector: Optional[np.ndarray] = None
    staggered_u: Optional[np.ndarray] = None
    staggered_v: Optional[np.ndarray] = None
    staggered_w: Optional[np.ndarray] = None
    points: Optional[CompiledPoints] = None
    diagnostics: tuple[SourceDiagnostic, ...] = ()
    covered_voxel_count: int = 0
    topology_leaf_coords: Optional[np.ndarray] = None
    device_leaf_coords: object = None
    device_coverage: object = None
    device_sdf: object = None
    device_scalar: object = None
    device_vector: object = None
    device_staggered_u: object = None
    device_staggered_v: object = None
    device_staggered_w: object = None

    @property
    def leaf_count(self):
        return int(len(self.leaf_coords))

    @property
    def device_bytes(self):
        total = sum(value.nbytes for value in (
            self.leaf_coords, self.coverage, self.sdf, self.scalar, self.vector,
            self.staggered_u, self.staggered_v, self.staggered_w)
                    if value is not None)
        if self.points is not None:
            total += sum(value.nbytes for value in (
                self.points.positions, self.points.velocities,
                self.points.temperature, self.points.viscosity,
                self.points.density, self.points.lifespan)
                         if value is not None)
        return int(total)

    def release(self):
        self.device_leaf_coords = None
        self.device_coverage = None
        self.device_sdf = None
        self.device_scalar = None
        self.device_vector = None
        self.device_staggered_u = None
        self.device_staggered_v = None
        self.device_staggered_w = None
        if self.points is not None:
            self.points.device_positions = None
            self.points.device_velocities = None
            self.points.device_temperature = None
            self.points.device_viscosity = None
            self.points.device_density = None
            self.points.device_lifespan = None


def normalized_liquid_target(plan: SourcePlan) -> int:
    """Return the canonical bounded liquid occupancy target."""
    return max(0, min(C.MAX_PARTICLES_PER_VOXEL, int(round(
        float(plan.payload.target_particles_per_voxel)))))


def liquid_coverage_emission_enabled(plan: SourcePlan) -> bool:
    """Whether a coverage-backed liquid plan can create particles.

    Authored point cardinality is independent of occupancy.  Non-liquid plans
    are likewise governed by their own operation semantics.
    """
    return (
        plan.operation == SourceOperation.LIQUID
        and plan.lifecycle != SourceLifecycle.DIRECT_APPEND
        and normalized_liquid_target(plan) > 0
    )


def plan_requires_source_topology(plan: SourcePlan) -> bool:
    if plan.operation != SourceOperation.LIQUID:
        return True
    if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
        return True
    return liquid_coverage_emission_enabled(plan)


@dataclass(frozen=True)
class CompiledSource:
    plan: SourcePlan
    region: CompiledRegion


@dataclass(frozen=True)
class CompiledSourceFootprint:
    """Host-only capacity facts produced by the accepted source rasterizer."""

    stable_id: str
    particle_demand: Optional[int]
    particle_quality: DemandQuality
    leaf_coords: tuple[tuple[int, int, int], ...]
    staging_bytes: int
    reason: str = ""


def _compiled_footprint(source: CompiledSource) -> CompiledSourceFootprint:
    plan, region = source.plan, source.region
    errors = tuple(item for item in region.diagnostics
                   if item.severity == "ERROR")
    topology_coords = (region.topology_leaf_coords
                       if region.topology_leaf_coords is not None
                       else region.leaf_coords)
    coords = tuple(tuple(int(axis) for axis in coord)
                   for coord in topology_coords)
    if errors:
        return CompiledSourceFootprint(
            str(plan.stable_id), None, DemandQuality.UNKNOWN, coords,
            region.device_bytes, "; ".join(item.message for item in errors),
        )
    if plan.operation != SourceOperation.LIQUID:
        return CompiledSourceFootprint(
            str(plan.stable_id), 0, DemandQuality.EXACT, coords,
            region.device_bytes,
        )
    if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
        if region.points is None:
            return CompiledSourceFootprint(
                str(plan.stable_id), None, DemandQuality.UNKNOWN, coords,
                region.device_bytes, "direct point payload is unavailable",
            )
        return CompiledSourceFootprint(
            str(plan.stable_id), len(region.points.positions),
            DemandQuality.EXACT, coords, region.device_bytes,
        )
    target = normalized_liquid_target(plan)
    if target == 0:
        return CompiledSourceFootprint(
            str(plan.stable_id), 0, DemandQuality.EXACT, (),
            region.device_bytes,
        )
    demand = int(np.ceil(
        np.asarray(region.coverage, np.float64) * target).sum(dtype=np.int64))
    return CompiledSourceFootprint(
        str(plan.stable_id), demand, DemandQuality.UPPER_BOUND, coords,
        region.device_bytes,
    )


@dataclass
class CompiledSourceBatch:
    context_key: tuple
    sources: tuple[CompiledSource, ...]
    required_leaf_coords: np.ndarray
    diagnostics: tuple[SourceDiagnostic, ...]
    device_bytes: int
    # (included source indices, device leaf coords, count) uploaded for
    # topology marking; owned by SourcePipeline.mark_source_topology.
    topology_upload: object = field(default=None, repr=False)

    @property
    def footprints(self) -> tuple[CompiledSourceFootprint, ...]:
        return tuple(_compiled_footprint(source) for source in self.sources)

    def release(self):
        self.topology_upload = None
        for source in self.sources:
            source.region.release()


class SourceCompiler:
    """Compile immutable plans into bounded flat source leaves.

    The cache owns all device uploads. Static/first-frame cache keys exclude the
    current frame; every-frame keys include it. No whole-domain dense array or
    hierarchical live grid is constructed.
    """

    def __init__(self):
        self._cache: dict[str, CompiledRegion] = {}
        # Keep at most one compiled generation per stable source for every
        # cadence. Edits, domain changes, and reset/session changes replace and
        # release the prior upload instead of leaking an unreachable cache row.
        self._source_keys: dict[SourceKey, str] = {}
        self.compile_count = 0
        self.cache_hits = 0

    @property
    def live_cache_bytes(self):
        return int(sum(region.device_bytes for region in self._cache.values()))

    def release(self):
        for region in self._cache.values():
            region.release()
        self._cache.clear()
        self._source_keys.clear()

    def _diagnostic(self, plan, severity, code, message):
        return SourceDiagnostic(severity, code, message, plan.stable_id)

    def _cache_key(self, plan, context):
        region = plan.region
        # repr() summarizes large NumPy arrays, so every region array is
        # digested in full; an interior edit must never reuse a stale upload.
        parts = [
            str(plan.stable_id), str(int(plan.operation)), str(int(plan.lifecycle)),
            str(int(plan.cadence)), str(int(region.kind)), repr(region),
            *(_array_digest(getattr(region, name, None))
              for name in _REGION_ARRAY_FIELDS),
            repr((context.origin, context.voxel_size, context.domain_min,
                  context.domain_max, context.session_generation,
                  context.device, context.upload)),
        ]
        if plan.cadence == SourceEvaluationCadence.EVERY_SIM_FRAME:
            parts.append(repr((context.frame, context.substep)))
        return sha256("|".join(parts).encode("utf8")).hexdigest()

    def compile(self, plans: Sequence[SourcePlan], context: CompileContext):
        compiled, diagnostics = [], []
        if context.voxel_size <= 0.0:
            raise ValueError("voxel size must be positive")
        occurrences: dict[SourceKey, int] = {}
        in_use: set[str] = set()
        for plan in plans:
            if not plan.activation.active(context.frame):
                continue
            # A duplicated object (Shift+D copies its stored identity) can
            # submit two plans with one stable id. Give each occurrence its
            # own cache slot: sharing one let the second compile release the
            # first region's uploads while it was still in this batch.
            occurrence = occurrences.get(plan.stable_id, 0)
            occurrences[plan.stable_id] = occurrence + 1
            slot = plan.stable_id
            if occurrence:
                slot = SourceKey(f"{plan.stable_id}#duplicate{occurrence}")
                diagnostics.append(self._diagnostic(
                    plan, "WARNING", "DUPLICATE_SOURCE_ID",
                    "Another source shares this source identity; duplicated objects should be given their own identity"))
            key = self._cache_key(plan, context)
            region = self._cache.get(key)
            if region is not None:
                self.cache_hits += 1
            else:
                previous = self._source_keys.get(slot)
                if (previous is not None and previous != key
                        and previous not in in_use):
                    stale = self._cache.pop(previous, None)
                    if stale is not None:
                        stale.release()
                region = self._compile_region(plan, context, key)
                self.compile_count += 1
                if not any(d.severity == "ERROR" for d in region.diagnostics):
                    self._cache[key] = region
                    self._source_keys[slot] = key
            in_use.add(key)
            diagnostics.extend(region.diagnostics)
            compiled.append(CompiledSource(plan, region))
        coords = [source.region.leaf_coords for source in compiled
                  if len(source.region.leaf_coords)]
        required = (np.unique(np.concatenate(coords), axis=0).astype(np.int32)
                    if coords else np.empty((0, 3), dtype=np.int32))
        device_bytes = sum(source.region.device_bytes for source in compiled)
        if len(required) > context.leaf_capacity:
            diagnostics.append(SourceDiagnostic(
                "ERROR", "SOURCE_LEAF_CAPACITY",
                f"source requires {len(required)} leaves; capacity is {context.leaf_capacity}",
                SourceKey("batch")))
        if device_bytes > context.max_source_bytes:
            diagnostics.append(SourceDiagnostic(
                "ERROR", "SOURCE_MEMORY_CAPACITY",
                f"source staging uses {device_bytes} bytes; limit is {context.max_source_bytes}",
                SourceKey("batch")))
        return CompiledSourceBatch(
            (context.frame, context.substep, context.session_generation,
             context.device, context.voxel_size),
            tuple(compiled), required, tuple(diagnostics), int(device_bytes))

    def _compile_region(self, plan, context, key):
        try:
            if plan.operation == SourceOperation.FORCE:
                return CompiledRegion(
                    plan.region.kind, key, np.empty((0, 3), np.int32),
                    np.empty((0, C.LEAF_VOLUME), np.float32),
                    diagnostics=(self._diagnostic(
                        plan, "ERROR", "COMPILED_FORCE_UNSUPPORTED",
                        "Compiled FORCE plans are unsupported; use the accepted legacy analytic FORCE descriptor path"),))
            if isinstance(plan.region, AnalyticRegionPlan):
                region = self._compile_analytic(plan, context, key)
            elif isinstance(plan.region, MeshRegionPlan):
                region = self._compile_mesh(plan, context, key)
            elif isinstance(plan.region, PointRegionPlan):
                region = self._compile_points(plan, context, key)
            elif isinstance(plan.region, HeightRegionPlan):
                region = self._compile_height(plan, context, key)
            elif isinstance(plan.region, VolumeRegionPlan):
                region = self._compile_volume(plan, context, key)
            else:
                raise TypeError(f"unsupported source region {type(plan.region).__name__}")
            region = self._compact_region(region)
            if plan.payload.target_particles_per_voxel > 64.0:
                region.diagnostics += (self._diagnostic(
                    plan, "WARNING", "TARGET_OCCUPANCY_CLAMPED",
                    "Target Particles / Voxel exceeds the GPU emission limit of 64 and will be clamped"),)
            if region.leaf_count > context.leaf_capacity:
                region.diagnostics += (self._diagnostic(
                    plan, "ERROR", "SOURCE_LEAF_CAPACITY",
                    f"source requires {region.leaf_count} leaves; capacity is {context.leaf_capacity}"),)
            if region.device_bytes > context.max_source_bytes:
                region.diagnostics += (self._diagnostic(
                    plan, "ERROR", "SOURCE_MEMORY_CAPACITY",
                    f"source uses {region.device_bytes} bytes; limit is {context.max_source_bytes}"),)
            if context.upload and not any(d.severity == "ERROR" for d in region.diagnostics):
                self._upload(region, context.device)
            return region
        except Exception as exc:
            return CompiledRegion(
                plan.region.kind, key, np.empty((0, 3), np.int32),
                np.empty((0, C.LEAF_VOLUME), np.float32),
                diagnostics=(self._diagnostic(
                    plan, "ERROR", "SOURCE_COMPILE_FAILED", str(exc)),))

    @staticmethod
    def _upload(region, device):
        import warp as wp
        region.device_leaf_coords = wp.array(
            region.leaf_coords, dtype=wp.vec3i, device=device)
        region.device_coverage = wp.array(
            region.coverage.reshape(-1), dtype=float, device=device)
        if region.sdf is not None:
            region.device_sdf = wp.array(region.sdf.reshape(-1), dtype=float, device=device)
        if region.scalar is not None:
            region.device_scalar = wp.array(region.scalar.reshape(-1), dtype=float, device=device)
        if region.vector is not None:
            region.device_vector = wp.array(
                region.vector.reshape(-1, 3), dtype=wp.vec3, device=device)
        for name in ("staggered_u", "staggered_v", "staggered_w"):
            values = getattr(region, name)
            if values is not None:
                setattr(region, f"device_{name}", wp.array(
                    values.reshape(-1), dtype=float, device=device))
        if region.points is not None:
            region.points.device_positions = wp.array(
                region.points.positions, dtype=wp.vec3, device=device)
            if region.points.velocities is not None:
                region.points.device_velocities = wp.array(
                    region.points.velocities, dtype=wp.vec3, device=device)
            for name in ("temperature", "viscosity", "density", "lifespan"):
                values = getattr(region.points, name)
                if values is not None:
                    setattr(region.points, f"device_{name}", wp.array(
                        values, dtype=float, device=device))

    @staticmethod
    def deterministic_seed(stable_id, voxel, frame, sample_index):
        raw = f"{stable_id}|{voxel[0]},{voxel[1]},{voxel[2]}|{frame}|{sample_index}"
        return int.from_bytes(sha256(raw.encode("utf8")).digest()[:8], "little") & 0x7FFFFFFF

    @staticmethod
    def _candidate_leaves(bounds, context):
        minimum = np.maximum(np.asarray(bounds[0], np.float64), context.domain_min)
        maximum = np.minimum(np.asarray(bounds[1], np.float64), context.domain_max)
        if np.any(maximum < minimum):
            return np.empty((0, 3), np.int32)
        origin = np.asarray(context.origin, np.float64)
        lo = np.floor((minimum - origin) / context.voxel_size).astype(np.int64)
        hi = np.floor((maximum - origin) / context.voxel_size).astype(np.int64)
        domain_lo = np.floor((np.asarray(context.domain_min) - origin)
                             / context.voxel_size).astype(np.int64)
        domain_hi = np.ceil((np.asarray(context.domain_max) - origin)
                            / context.voxel_size).astype(np.int64) - 1
        lo, hi = np.maximum(lo, domain_lo), np.minimum(hi, domain_hi)
        leaf_lo = np.floor_divide(lo, C.LEAF_DIM)
        leaf_hi = np.floor_divide(hi, C.LEAF_DIM)
        if np.any(leaf_hi < leaf_lo):
            return np.empty((0, 3), np.int32)
        grid = np.stack(np.meshgrid(
            np.arange(leaf_lo[0], leaf_hi[0] + 1),
            np.arange(leaf_lo[1], leaf_hi[1] + 1),
            np.arange(leaf_lo[2], leaf_hi[2] + 1), indexing="ij"), axis=-1)
        return np.ascontiguousarray(grid.reshape(-1, 3), dtype=np.int32)

    def _preflight_leaves(self, plan, leaves, context, *, output_arrays=2,
                          vector_arrays=0):
        """Reject impossible staging before allocating leaf voxel positions.

        The estimate includes the temporary float64 voxel-centre array, flat
        leaf coordinates, and the requested float32 output payloads. It is
        deliberately conservative so the configured source-memory limit is a
        real allocation guard rather than a post-allocation warning.
        """
        count = int(len(leaves))
        if count > int(context.leaf_capacity):
            return CompiledRegion(
                plan.region.kind, "", np.empty((0, 3), np.int32),
                np.empty((0, C.LEAF_VOLUME), np.float32),
                diagnostics=(self._diagnostic(
                    plan, "ERROR", "SOURCE_LEAF_CAPACITY",
                    f"source requires {count} candidate leaves; capacity is {context.leaf_capacity}"),))
        temporary = count * C.LEAF_VOLUME * 3 * np.dtype(np.float64).itemsize
        outputs = count * C.LEAF_VOLUME * (
            output_arrays + 3 * vector_arrays) * np.dtype(np.float32).itemsize
        estimate = count * 3 * np.dtype(np.int32).itemsize + temporary + outputs
        if estimate > int(context.max_source_bytes):
            return CompiledRegion(
                plan.region.kind, "", np.empty((0, 3), np.int32),
                np.empty((0, C.LEAF_VOLUME), np.float32),
                diagnostics=(self._diagnostic(
                    plan, "ERROR", "SOURCE_MEMORY_CAPACITY",
                    f"source staging requires at least {estimate} bytes before allocation; limit is {context.max_source_bytes}"),))
        return None

    @staticmethod
    def _compact_region(region):
        """Drop empty flat leaves before upload and cache ownership."""
        if region.leaf_count == 0:
            region.covered_voxel_count = 0
            return region
        keep = np.any(region.coverage > 0.0, axis=1)
        if not np.all(keep):
            region.leaf_coords = np.ascontiguousarray(region.leaf_coords[keep])
            for name in ("coverage", "sdf", "scalar", "vector",
                         "staggered_u", "staggered_v", "staggered_w"):
                value = getattr(region, name)
                if value is not None:
                    setattr(region, name, np.ascontiguousarray(value[keep]))
        region.covered_voxel_count = int(np.count_nonzero(region.coverage))
        return region

    @staticmethod
    def _voxel_points(leaves, context):
        local = np.stack(np.meshgrid(
            np.arange(C.LEAF_DIM), np.arange(C.LEAF_DIM),
            np.arange(C.LEAF_DIM), indexing="ij"), axis=-1).reshape(-1, 3)
        # Solver flat order is x fastest. meshgrid above has z fastest; only
        # compilation arrays consume this order, so normalize explicitly.
        local = local[:, (2, 1, 0)]
        voxels = leaves[:, None, :] * C.LEAF_DIM + local[None, :, :]
        points = np.asarray(context.origin) + (voxels + 0.5) * context.voxel_size
        return voxels, points

    @staticmethod
    def _quat_inverse_rotate(points, position, quaternion):
        q = np.asarray(quaternion, np.float64)
        norm = np.linalg.norm(q)
        if norm == 0.0:
            q = np.array((0.0, 0.0, 0.0, 1.0))
        else:
            q /= norm
        x, y, z, w = q
        matrix = np.array([
            (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
            (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w),
            (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y),
        ])
        return (points - np.asarray(position)) @ matrix

    @staticmethod
    def _analytic_sdf(region, points):
        local = SourceCompiler._quat_inverse_rotate(
            points, region.position, region.rotation)
        shape = int(region.shape)
        if shape == SourceShapeShape.SPHERE:
            return np.linalg.norm(local, axis=-1) - region.radius
        if shape == SourceShapeShape.BOX:
            q = np.abs(local) - np.asarray(region.half_extents)
            return (np.linalg.norm(np.maximum(q, 0.0), axis=-1)
                    + np.minimum(np.max(q, axis=-1), 0.0))
        if shape == SourceShapeShape.CYLINDER:
            radial = np.linalg.norm(local[..., :2], axis=-1) - region.radius
            vertical = np.abs(local[..., 2]) - region.height
            return (np.linalg.norm(np.stack((np.maximum(radial, 0.0),
                                             np.maximum(vertical, 0.0)), axis=-1), axis=-1)
                    + np.minimum(np.maximum(radial, vertical), 0.0))
        if shape == SourceShapeShape.PLANE:
            return local[..., 2]
        # Cone/pyramid membership is exact for the linear cross-section. The
        # signed magnitude is conservative; emission truth relies on its sign.
        z = local[..., 2]
        t = np.clip((region.height - z) / max(2.0 * region.height, 1e-12), 0.0, 1.0)
        cross = region.radius * t
        radial = (np.max(np.abs(local[..., :2]), axis=-1)
                  if shape == SourceShapeShape.PYRAMID
                  else np.linalg.norm(local[..., :2], axis=-1))
        outside = np.maximum.reduce((radial - cross,
                                     -region.height - z,
                                     z - region.height))
        return outside

    def _compile_analytic(self, plan, context, key):
        leaves = self._candidate_leaves(plan.region.bounds(), context)
        rejected = self._preflight_leaves(plan, leaves, context, output_arrays=2)
        if rejected is not None:
            rejected.cache_key = key
            return rejected
        _, points = self._voxel_points(leaves, context)
        sdf = self._analytic_sdf(plan.region, points).astype(np.float32)
        feather = max(0.0, float(plan.region.feather))
        if feather:
            coverage = np.clip(0.5 - sdf / max(feather, 1e-12), 0.0, 1.0)
        else:
            coverage = (sdf <= 0.0).astype(np.float32)
        return CompiledRegion(plan.region.kind, key, leaves, coverage, sdf=sdf)

    @staticmethod
    def _watertight(triangles):
        if not len(triangles):
            return False
        edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)],
                                triangles[:, (2, 0)]))
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(len(counts) and np.all(counts == 2))

    @staticmethod
    def _point_triangle_distance(point, a, b, c):
        ab, ac, ap = b-a, c-a, point-a
        d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
        if d1 <= 0 and d2 <= 0: return np.linalg.norm(ap)
        bp = point-b; d3, d4 = np.dot(ab,bp), np.dot(ac,bp)
        if d3 >= 0 and d4 <= d3: return np.linalg.norm(bp)
        vc = d1*d4-d3*d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            return np.linalg.norm(point-(a+ab*(d1/(d1-d3))))
        cp = point-c; d5, d6 = np.dot(ab,cp), np.dot(ac,cp)
        if d6 >= 0 and d5 <= d6: return np.linalg.norm(cp)
        vb = d5*d2-d1*d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            return np.linalg.norm(point-(a+ac*(d2/(d2-d6))))
        va = d3*d6-d5*d4
        if va <= 0 and d4-d3 >= 0 and d5-d6 >= 0:
            return np.linalg.norm(point-(b+(c-b)*((d4-d3)/((d4-d3)+(d5-d6)))))
        normal = np.cross(ab, ac)
        return abs(np.dot(point-a, normal)) / max(np.linalg.norm(normal), 1e-20)

    @staticmethod
    def _mesh_distances(points, vertices, triangles, chunk=256):
        """Chunked vectorized unsigned triangle distance.

        Work is O(points*triangles), but Python work is O(chunks) and temporary
        memory is bounded by ``chunk**2`` instead of the full Cartesian product.
        """
        points = np.asarray(points, np.float64)
        vertices = np.asarray(vertices, np.float64)
        triangles = np.asarray(triangles, np.int32)
        result = np.full(len(points), np.inf, np.float64)
        for ps in range(0, len(points), chunk):
            p = points[ps:ps + chunk]
            best = np.full(len(p), np.inf, np.float64)
            for ts in range(0, len(triangles), chunk):
                tri = triangles[ts:ts + chunk]
                a, b, c = vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]]
                ab, ac = b - a, c - a
                normal = np.cross(ab, ac)
                normal_len = np.linalg.norm(normal, axis=1)
                valid_tri = normal_len > 1.0e-20
                unit = normal / np.maximum(normal_len[:, None], 1.0e-20)
                ap = p[:, None, :] - a[None, :, :]
                signed = np.einsum("pti,ti->pt", ap, unit)
                projected = p[:, None, :] - signed[:, :, None] * unit[None, :, :]
                vp = projected - a[None, :, :]
                d00 = np.einsum("ti,ti->t", ab, ab)
                d01 = np.einsum("ti,ti->t", ab, ac)
                d11 = np.einsum("ti,ti->t", ac, ac)
                denom = d00 * d11 - d01 * d01
                d20 = np.einsum("pti,ti->pt", vp, ab)
                d21 = np.einsum("pti,ti->pt", vp, ac)
                safe = np.where(np.abs(denom) > 1.0e-20, denom, 1.0)
                bary_v = (d11[None, :] * d20 - d01[None, :] * d21) / safe
                bary_w = (d00[None, :] * d21 - d01[None, :] * d20) / safe
                inside = (valid_tri[None, :] & (bary_v >= 0.0)
                          & (bary_w >= 0.0) & (bary_v + bary_w <= 1.0))
                tri_distance = np.where(inside, np.abs(signed), np.inf)
                for start, end in ((a, b), (b, c), (c, a)):
                    edge = end - start
                    edge_len2 = np.einsum("ti,ti->t", edge, edge)
                    rel = p[:, None, :] - start[None, :, :]
                    t = np.clip(np.einsum("pti,ti->pt", rel, edge)
                                / np.maximum(edge_len2[None, :], 1.0e-20), 0.0, 1.0)
                    closest = start[None, :, :] + t[:, :, None] * edge[None, :, :]
                    segment = np.linalg.norm(p[:, None, :] - closest, axis=2)
                    tri_distance = np.minimum(tri_distance, segment)
                best = np.minimum(best, np.min(tri_distance, axis=1))
            result[ps:ps + len(p)] = best
        return result.astype(np.float32)

    @staticmethod
    def _mesh_inside(points, vertices, triangles, point_chunk=512, tri_chunk=512):
        """Chunked vectorized orientation-independent ray parity."""
        points = np.asarray(points, np.float64)
        vertices = np.asarray(vertices, np.float64)
        triangles = np.asarray(triangles, np.int32)
        direction = np.array((1.0, 0.00000173, 0.00000231), np.float64)
        result = np.zeros(len(points), dtype=bool)
        for ps in range(0, len(points), point_chunk):
            p = points[ps:ps + point_chunk]
            hits = np.zeros(len(p), np.int64)
            for ts in range(0, len(triangles), tri_chunk):
                tri = triangles[ts:ts + tri_chunk]
                a, b, c = vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]]
                e1, e2 = b - a, c - a
                h = np.cross(np.broadcast_to(direction, e2.shape), e2)
                det = np.einsum("ti,ti->t", e1, h)
                valid = np.abs(det) >= 1.0e-12
                inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
                s = p[:, None, :] - a[None, :, :]
                u = np.einsum("pti,ti->pt", s, h) * inv[None, :]
                q = np.cross(s, e1[None, :, :])
                v = np.einsum("i,pti->pt", direction, q) * inv[None, :]
                distance = np.einsum("ti,pti->pt", e2, q) * inv[None, :]
                intersections = (valid[None, :] & (u >= 0.0) & (u <= 1.0)
                                 & (v >= 0.0) & (u + v <= 1.0)
                                 & (distance > 1.0e-9))
                hits += np.count_nonzero(intersections, axis=1)
            result[ps:ps + len(p)] = (hits & 1).astype(bool)
        return result

    @classmethod
    def _compile_mesh_cpu_bvh(cls, points, vertices, triangles, solid, shell_width):
        """Warp CPU BVH staging with the NumPy fallback's exact decisions.

        The BVH runs in float32. Voxels whose BVH distance sits within a
        float32-scale tolerance of the coverage decision (the shell width, or
        the surface for solids) are recomputed with the float64 NumPy path,
        so coverage matches the full NumPy result instead of tie-breaking
        differently on grid-aligned meshes. SDF magnitudes elsewhere differ
        from the NumPy values only at float32 rounding (~1e-7); no runtime
        kernel reads the compiled mesh SDF.
        """
        coverage, sdf = cls._compile_mesh_gpu(
            points, vertices, triangles, solid, shell_width, "cpu",
            kernel=_k_query_compiled_mesh_parity)
        scale = max(1.0, float(np.max(np.abs(vertices))),
                    float(np.max(np.abs(points))))
        threshold = 0.0 if solid else float(shell_width)
        near = np.nonzero(np.abs(np.abs(sdf) - threshold) <= 1.0e-5 * scale)[0]
        if len(near):
            exact = np.asarray(points, np.float64)[near]
            distances = cls._mesh_distances(exact, vertices, triangles)
            if solid:
                inside = cls._mesh_inside(exact, vertices, triangles)
                sdf[near] = np.where(inside, -distances, distances)
                coverage[near] = inside
            else:
                sdf[near] = distances
                coverage[near] = distances <= shell_width
        return coverage, sdf

    @staticmethod
    def _compile_mesh_gpu(points, vertices, triangles, solid, shell_width, device,
                          kernel=None):
        """Use Warp's mesh BVH for production-scale static mesh compilation."""
        points = np.ascontiguousarray(points, dtype=np.float32)
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        indices = np.ascontiguousarray(triangles, dtype=np.int32).reshape(-1)
        device_points = wp.array(points, dtype=wp.vec3, device=device)
        device_vertices = wp.array(vertices, dtype=wp.vec3, device=device)
        device_indices = wp.array(indices, dtype=int, device=device)
        mesh = wp.Mesh(points=device_vertices, indices=device_indices)
        device_coverage = wp.zeros(len(points), dtype=float, device=device)
        device_sdf = wp.zeros(len(points), dtype=float, device=device)
        wp.launch(
            kernel or _k_query_compiled_mesh, dim=len(points),
            inputs=[mesh.id, device_points, int(bool(solid)), float(shell_width),
                    device_coverage, device_sdf],
            device=device,
        )
        wp.synchronize_device(device)
        coverage = device_coverage.numpy().astype(np.float32, copy=False)
        sdf = device_sdf.numpy().astype(np.float32, copy=False)
        # Mesh and temporary arrays are staging-only; no BVH enters live state.
        del mesh, device_points, device_vertices, device_indices
        del device_coverage, device_sdf
        return coverage, sdf

    def _compile_mesh(self, plan, context, key):
        region = plan.region
        diagnostics = []
        if not len(region.vertices) or not len(region.triangles):
            diagnostics.append(self._diagnostic(
                plan, "ERROR", "MESH_EMPTY", "mesh source has no triangles"))
            return CompiledRegion(region.kind, key, np.empty((0,3), np.int32),
                                  np.empty((0,C.LEAF_VOLUME), np.float32),
                                  diagnostics=tuple(diagnostics))
        if region.triangles.min() < 0 or region.triangles.max() >= len(region.vertices):
            raise ValueError("mesh triangle index is out of range")
        if region.solid and not self._watertight(region.triangles):
            diagnostics.append(self._diagnostic(
                plan, "ERROR", "MESH_NOT_WATERTIGHT",
                "Solid Volume requires a non-degenerate watertight mesh; use Surface Shell explicitly for open input"))
            return CompiledRegion(region.kind, key, np.empty((0,3), np.int32),
                                  np.empty((0,C.LEAF_VOLUME), np.float32),
                                  diagnostics=tuple(diagnostics))
        leaves = self._candidate_leaves(region.bounds(), context)
        rejected = self._preflight_leaves(plan, leaves, context, output_arrays=2)
        if rejected is not None:
            rejected.cache_key = key
            return rejected
        comparisons = int(len(leaves)) * C.LEAF_VOLUME * int(len(region.triangles))
        # The approved interactive p95 budget is 33.3 ms. The measured NumPy
        # staging rate is ~0.25 us/comparison, so dynamic CPU compilation is
        # conservatively capped at 40k comparisons; measured 73,728-work took
        # 27.8 ms median, so this bound preserves headroom below both approved
        # 16.7 ms median and 33.3 ms p95 limits. Larger
        # meshes remain available as STATIC/FIRST_SIM_FRAME and use Warp's BVH
        # on CUDA instead of the Cartesian O(points*triangles) host path.
        if (plan.cadence == SourceEvaluationCadence.EVERY_SIM_FRAME
                and comparisons > 40_000):
            diagnostics.append(self._diagnostic(
                plan, "ERROR", "SOURCE_DYNAMIC_MESH_BUDGET",
                f"animated mesh compile requires {comparisons} comparisons; "
                "the pre-approved 16.7/33.3 ms budget permits 40000 on the "
                "measured CPU staging path; use First Sim Frame or lower source resolution"))
            return CompiledRegion(region.kind, key, np.empty((0, 3), np.int32),
                                  np.empty((0, C.LEAF_VOLUME), np.float32),
                                  diagnostics=tuple(diagnostics))
        if (not str(context.device).startswith("cuda")
                and comparisons > 50_000_000):
            diagnostics.append(self._diagnostic(
                plan, "ERROR", "SOURCE_MESH_CUDA_REQUIRED",
                f"mesh compile requires {comparisons} comparisons; production-scale "
                "static mesh compilation requires a CUDA device"))
            return CompiledRegion(region.kind, key, np.empty((0, 3), np.int32),
                                  np.empty((0, C.LEAF_VOLUME), np.float32),
                                  diagnostics=tuple(diagnostics))
        _, points = self._voxel_points(leaves, context)
        flat = points.reshape(-1,3)
        requested_width = max(0.0, float(region.shell_half_width))
        width = max(requested_width, context.voxel_size * 0.5)
        on_cuda = str(context.device).startswith("cuda")
        if on_cuda and comparisons > 500_000:
            flat_coverage, flat_sdf = self._compile_mesh_gpu(
                flat, region.vertices, region.triangles, region.solid, width,
                context.device)
            coverage = flat_coverage.reshape(len(leaves), C.LEAF_VOLUME)
            sdf = flat_sdf.reshape(len(leaves), C.LEAF_VOLUME)
        elif not on_cuda and comparisons > 500_000:
            # CPU-staged admission: the Cartesian NumPy path took seconds per
            # mesh (13-17 s for a 960-triangle sphere) on Initialize/Reset.
            flat_coverage, flat_sdf = self._compile_mesh_cpu_bvh(
                flat, region.vertices, region.triangles, region.solid, width)
            coverage = flat_coverage.reshape(len(leaves), C.LEAF_VOLUME)
            sdf = flat_sdf.reshape(len(leaves), C.LEAF_VOLUME)
        else:
            distances = self._mesh_distances(
                flat.astype(np.float64), region.vertices, region.triangles)
            distances = distances.reshape(len(leaves), C.LEAF_VOLUME)
            if region.solid:
                inside = self._mesh_inside(flat, region.vertices, region.triangles).reshape(
                    len(leaves), C.LEAF_VOLUME)
                sdf = np.where(inside, -distances, distances).astype(np.float32)
                coverage = inside.astype(np.float32)
            else:
                sdf = distances.astype(np.float32)
                coverage = (distances <= width).astype(np.float32)
        if not region.solid:
            if width > requested_width:
                diagnostics.append(self._diagnostic(
                    plan, "INFO", "SHELL_WIDTH_VOXEL_FLOOR",
                    f"Surface Shell half-width {requested_width:g} is below half a voxel; effective half-width is {width:g}"))
        return CompiledRegion(region.kind, key, leaves, coverage, sdf=sdf,
                              diagnostics=tuple(diagnostics))

    def _compile_points(self, plan, context, key):
        r = plan.region
        points = CompiledPoints(r.positions, r.velocities, r.temperature,
                                r.viscosity, r.density, r.lifespan)
        if plan.lifecycle == SourceLifecycle.DIRECT_APPEND:
            region = CompiledRegion(r.kind, key, np.empty((0,3),np.int32),
                                    np.empty((0,C.LEAF_VOLUME),np.float32), points=points)
            # No coverage, but the leaves holding the points must be active on
            # the frame they are appended: topology is rebuilt before
            # emission, so otherwise they get no grid support until the next
            # rebuild sees them.
            region.topology_leaf_coords = self._point_leaves(r.positions, context)
            return region
        leaves = self._candidate_leaves(r.bounds(), context)
        rejected = self._preflight_leaves(plan, leaves, context, output_arrays=1)
        if rejected is not None:
            rejected.cache_key = key
            return rejected
        radius = max(float(r.rasterize_radius), context.voxel_size * 0.5)
        coverage = self._rasterize_points(r.positions, leaves, context, radius)
        return CompiledRegion(r.kind, key, leaves, coverage, points=points)

    @staticmethod
    def _point_leaves(positions, context):
        """Unique flat leaves containing in-domain points (domain voxel range)."""
        if not len(positions):
            return np.empty((0, 3), np.int32)
        origin = np.asarray(context.origin, np.float64)
        voxels = np.floor((np.asarray(positions, np.float64) - origin)
                          / context.voxel_size).astype(np.int64)
        domain_lo = np.floor((np.asarray(context.domain_min) - origin)
                             / context.voxel_size).astype(np.int64)
        domain_hi = np.ceil((np.asarray(context.domain_max) - origin)
                            / context.voxel_size).astype(np.int64) - 1
        inside = np.all((voxels >= domain_lo) & (voxels <= domain_hi), axis=1)
        leaves = np.floor_divide(voxels[inside], C.LEAF_DIM)
        return np.ascontiguousarray(np.unique(leaves, axis=0), dtype=np.int32)

    @staticmethod
    def _rasterize_points(positions, leaves, context, radius, max_rows=1_000_000):
        """Cover candidate voxels whose centre lies within ``radius`` of a point.

        Each point visits only the voxel stencil around it rather than every
        candidate voxel, so the work is points x stencil instead of points x
        region. Centres and distances use the same float64 arithmetic as
        :meth:`_voxel_points`, so coverage equals the former full scan.
        """
        coverage = np.zeros((len(leaves), C.LEAF_VOLUME), np.float32)
        if not len(positions) or not len(leaves):
            return coverage
        leaves = np.asarray(leaves, np.int64)
        leaf_lo = leaves.min(axis=0)
        leaf_dims = leaves.max(axis=0) - leaf_lo + 1
        rows = np.full(int(np.prod(leaf_dims)), -1, np.int64)
        relative = leaves - leaf_lo
        rows[(relative[:, 0] * leaf_dims[1] + relative[:, 1]) * leaf_dims[2]
             + relative[:, 2]] = np.arange(len(leaves))
        origin = np.asarray(context.origin)
        voxel_size = context.voxel_size
        # A centre within radius lies at most ceil(radius/h) + 2 voxels from the
        # point's own voxel on each axis (half-voxel centre offset + flooring).
        reach = int(math.ceil(radius / voxel_size)) + 2
        span = np.arange(-reach, reach + 1, dtype=np.int64)
        stencil = np.stack(np.meshgrid(span, span, span, indexing="ij"),
                           axis=-1).reshape(-1, 3)
        # Drop stencil voxels no point of the base voxel can reach. The
        # per-axis gap is under-estimated by a whole voxel (|d| - 1), so float
        # rounding can never exclude a voxel the exact test would accept.
        gap = np.maximum(np.abs(stencil) - 1, 0) * voxel_size
        stencil = stencil[np.sum(gap * gap, axis=1) <= radius * radius]
        chunk = max(1, int(max_rows) // len(stencil))
        for start in range(0, len(positions), chunk):
            points = positions[start:start + chunk]
            base = np.floor((points.astype(np.float64) - origin)
                            / voxel_size).astype(np.int64)
            voxels = (base[:, None, :] + stencil[None, :, :]).reshape(-1, 3)
            owner = np.repeat(np.arange(len(points)), len(stencil))
            leaf = np.floor_divide(voxels, C.LEAF_DIM)
            relative = leaf - leaf_lo
            inside = np.all((relative >= 0) & (relative < leaf_dims), axis=1)
            voxels, owner, leaf, relative = (
                voxels[inside], owner[inside], leaf[inside], relative[inside])
            row = rows[(relative[:, 0] * leaf_dims[1] + relative[:, 1])
                       * leaf_dims[2] + relative[:, 2]]
            candidate = row >= 0
            voxels, owner, leaf, row = (
                voxels[candidate], owner[candidate], leaf[candidate], row[candidate])
            centers = origin + (voxels + 0.5) * voxel_size
            hit = np.linalg.norm(centers - points[owner], axis=-1) <= radius
            local = voxels[hit] - leaf[hit] * C.LEAF_DIM
            # Flat in-leaf order is x fastest (see _voxel_points).
            coverage[row[hit], local[:, 0] + C.LEAF_DIM * (
                local[:, 1] + C.LEAF_DIM * local[:, 2])] = 1.0
        return coverage

    def _compile_height(self, plan, context, key):
        r = plan.region
        # Match FluidSolver._rebuild_topology's four-voxel height safety band;
        # planner/compiler topology must not under-count the runtime contributor.
        bounds = (context.domain_min,
                  (context.domain_max[0], context.domain_max[1],
                   min(context.domain_max[2],
                       r.base_height + 4.0 * context.voxel_size)))
        leaves = self._candidate_leaves(bounds, context)
        rejected = self._preflight_leaves(plan, leaves, context, output_arrays=2)
        if rejected is not None:
            rejected.cache_key = key
            return rejected
        _, points = self._voxel_points(leaves, context)
        if r.back_mode == SourceHeightBackMode.PROFILE:
            x = np.clip((points[...,0]-context.domain_min[0]) /
                        max(context.domain_max[0]-context.domain_min[0],1e-12),0,1)
            y = np.clip((points[...,1]-context.domain_min[1]) /
                        max(context.domain_max[1]-context.domain_min[1],1e-12),0,1)
            ix = np.rint(x*(r.profile.shape[0]-1)).astype(int)
            iy = np.rint(y*(r.profile.shape[1]-1)).astype(int)
            top = r.base_height + r.profile[ix,iy]
        else:
            top = r.base_height
        coverage = (points[...,2] <= top).astype(np.float32)
        sdf = (points[...,2]-top).astype(np.float32)
        region = CompiledRegion(r.kind, key, leaves, coverage, sdf=sdf)
        region.topology_leaf_coords = np.ascontiguousarray(leaves)
        return region

    @staticmethod
    def _accessor_value(accessor, coord):
        coord = tuple(int(v) for v in coord)
        if hasattr(accessor, "getValue"):
            return accessor.getValue(coord)
        return accessor[coord]

    @classmethod
    def _sample_accessor(cls, accessor, coord):
        """Trilinear OpenVDB staging sample for scalar or vector grids."""
        base = np.floor(np.asarray(coord, dtype=np.float64)).astype(np.int64)
        fraction = np.asarray(coord, dtype=np.float64) - base
        result = None
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    weight = ((fraction[0] if dx else 1.0 - fraction[0])
                              * (fraction[1] if dy else 1.0 - fraction[1])
                              * (fraction[2] if dz else 1.0 - fraction[2]))
                    value = np.asarray(cls._accessor_value(
                        accessor, base + (dx, dy, dz)), dtype=np.float64)
                    result = value * weight if result is None else result + value * weight
        return result.item() if np.ndim(result) == 0 else result

    @staticmethod
    def _sample_dense_volume(values, coords, index_min, background=0.0):
        """Vectorized trilinear staging sample from an OpenVDB dense window.

        Samples outside the active window read the grid background, as
        OpenVDB's accessor does: clamping into the one-voxel pad lands on a
        background cell. Weighting them to zero instead turned exterior
        level-set padding into false coverage (zero passes ``sdf <= 0``).
        """
        values = np.asarray(values)
        vector = values.ndim == 4
        pad_width = ((1, 1), (1, 1), (1, 1)) + (((0, 0),) if vector else ())
        padded = np.pad(values, pad_width, constant_values=background)
        local = np.asarray(coords, np.float64) - np.asarray(index_min) + 1.0
        base = np.floor(local).astype(np.int64)
        fraction = local - base
        result_shape = (len(local), values.shape[-1]) if vector else (len(local),)
        result = np.zeros(result_shape, dtype=np.float64)
        limits = np.asarray(padded.shape[:3])
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    sample = base + (dx, dy, dz)
                    clipped = np.minimum(np.maximum(sample, 0), limits - 1)
                    weight = ((fraction[:, 0] if dx else 1.0 - fraction[:, 0])
                              * (fraction[:, 1] if dy else 1.0 - fraction[:, 1])
                              * (fraction[:, 2] if dz else 1.0 - fraction[:, 2]))
                    gathered = padded[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
                    if vector:
                        result += gathered * weight[:, None]
                    else:
                        result += gathered * weight
        return result

    def _compile_volume(self, plan, context, key):
        region = plan.region
        if not Path(region.filepath).is_file():
            raise ValueError(f"volume file does not exist: {region.filepath}")
        try:
            import openvdb
        except ImportError as exc:
            raise RuntimeError("Blender-bundled openvdb is unavailable") from exc
        grids = {}
        try:
            for role, name in (("mask", region.mask_grid), ("sdf", region.sdf_grid),
                               ("scalar", region.scalar_grid),
                               ("vector", region.vector_grid),
                               ("vector_x", region.vector_x_grid),
                               ("vector_y", region.vector_y_grid),
                               ("vector_z", region.vector_z_grid)):
                if name:
                    try:
                        grids[role] = openvdb.read(region.filepath, name)
                    except Exception as exc:
                        raise ValueError(f"missing or unreadable volume grid {name!r}: {exc}") from exc
            plan_matrix = np.asarray(region.transform, np.float64).reshape(4,4)
            combined_matrices = {
                role: plan_matrix @ _vdb_index_to_world_matrix(grid)
                for role, grid in grids.items()
            }
            world_corners = []
            for role, grid in grids.items():
                lo, hi = grid.evalActiveVoxelBoundingBox()
                corners = np.array([
                    (x,y,z,1.0) for x in (lo[0],hi[0]+1)
                    for y in (lo[1],hi[1]+1) for z in (lo[2],hi[2]+1)])
                world_corners.append(corners @ combined_matrices[role].T)
            world = np.concatenate(world_corners, axis=0)
            bounds = (tuple(world[:,:3].min(axis=0)), tuple(world[:,:3].max(axis=0)))
            leaves = self._candidate_leaves(bounds, context)
            has_vector = ("vector" in grids or all(
                role in grids for role in ("vector_x", "vector_y", "vector_z")))
            output_arrays = 1 + int("sdf" in grids) + int("scalar" in grids)
            rejected = self._preflight_leaves(
                plan, leaves, context, output_arrays=output_arrays,
                vector_arrays=int(has_vector))
            if rejected is not None:
                rejected.cache_key = key
                return rejected
            _, points = self._voxel_points(leaves, context)
            homogeneous = np.concatenate((points, np.ones(points.shape[:-1]+(1,))),axis=-1)
            inverse_matrices = {
                role: np.linalg.inv(matrix)
                for role, matrix in combined_matrices.items()
            }
            index_points = {
                role: (homogeneous @ inverse.T)[..., :3]
                for role, inverse in inverse_matrices.items()
            }
            coverage = np.zeros((len(leaves),C.LEAF_VOLUME),np.float32)
            sdf = (np.full_like(coverage, np.inf) if "sdf" in grids else None)
            scalar = (np.zeros_like(coverage) if "scalar" in grids else None)
            vector = (np.zeros((len(leaves),C.LEAF_VOLUME,3),np.float32)
                      if has_vector else None)
            staggered = (plan.operation == SourceOperation.STAGGERED_VELOCITY
                         and has_vector)
            staggered_u = (np.zeros_like(coverage) if staggered else None)
            staggered_v = (np.zeros_like(coverage) if staggered else None)
            staggered_w = (np.zeros_like(coverage) if staggered else None)
            world_face_offsets = (
                np.asarray((context.voxel_size * 0.5, 0.0, 0.0)),
                np.asarray((0.0, context.voxel_size * 0.5, 0.0)),
                np.asarray((0.0, 0.0, context.voxel_size * 0.5)),
            )
            face_offsets = {
                role: tuple(offset @ inverse[:3, :3].T
                            for offset in world_face_offsets)
                for role, inverse in inverse_matrices.items()
            }
            accessors = {name:grid.getConstAccessor() if hasattr(grid,"getConstAccessor")
                         else grid.getAccessor() for name,grid in grids.items()}
            dense = {}
            for role, grid in grids.items():
                bbox = grid.evalActiveVoxelBoundingBox()
                grid_min = np.asarray(bbox[0], dtype=np.int64)
                grid_max = np.asarray(bbox[1], dtype=np.int64)
                shape = tuple((grid_max - grid_min + 1).tolist())
                if role == "vector":
                    shape += (3,)
                values = np.empty(shape, dtype=np.float32)
                try:
                    grid.copyToArray(values, tuple(int(v) for v in grid_min))
                    dense[role] = (values, grid_min,
                                   float(getattr(grid, "background", 0.0))
                                   if role != "vector" else 0.0)
                except (AttributeError, TypeError):
                    # Test doubles and unusual bindings retain the bounded
                    # accessor fallback; Blender's bundled binding uses this
                    # fast dense staging path.
                    pass
            flat_index = {
                role: coords.reshape(-1, 3)
                for role, coords in index_points.items()
            }
            if len(dense) == len(grids):
                if "mask" in dense:
                    values, grid_min, background = dense["mask"]
                    sampled = self._sample_dense_volume(
                        values, flat_index["mask"], grid_min, background)
                    coverage[:] = (sampled.reshape(coverage.shape)
                                   > region.fog_threshold)
                if "sdf" in dense:
                    values, grid_min, background = dense["sdf"]
                    sampled = self._sample_dense_volume(
                        values, flat_index["sdf"], grid_min, background)
                    sdf[:] = sampled.reshape(sdf.shape)
                    coverage[:] = np.maximum(coverage, sdf <= 0.0)
                if "scalar" in dense:
                    values, grid_min, background = dense["scalar"]
                    scalar[:] = self._sample_dense_volume(
                        values, flat_index["scalar"], grid_min, background).reshape(scalar.shape)
                    if not ("mask" in dense or "sdf" in dense):
                        coverage[:] = 1.0
                if "vector" in dense:
                    values, grid_min, background = dense["vector"]
                    vector[:] = self._sample_dense_volume(
                        values, flat_index["vector"], grid_min, background).reshape(vector.shape)
                    if not ("mask" in dense or "sdf" in dense):
                        coverage[:] = 1.0
                    if staggered:
                        staggered_u[:] = self._sample_dense_volume(
                            values, flat_index["vector"] - face_offsets["vector"][0], grid_min,
                            background)[:, 0].reshape(staggered_u.shape)
                        staggered_v[:] = self._sample_dense_volume(
                            values, flat_index["vector"] - face_offsets["vector"][1], grid_min,
                            background)[:, 1].reshape(staggered_v.shape)
                        staggered_w[:] = self._sample_dense_volume(
                            values, flat_index["vector"] - face_offsets["vector"][2], grid_min,
                            background)[:, 2].reshape(staggered_w.shape)
                elif all(role in dense for role in ("vector_x", "vector_y", "vector_z")):
                    for component, role in enumerate(("vector_x", "vector_y", "vector_z")):
                        values, grid_min, background = dense[role]
                        vector[..., component] = self._sample_dense_volume(
                            values, flat_index[role], grid_min, background).reshape(coverage.shape)
                        if staggered:
                            target = (staggered_u, staggered_v, staggered_w)[component]
                            target[:] = self._sample_dense_volume(
                                values, flat_index[role] - face_offsets[role][component],
                                grid_min, background).reshape(target.shape)
                    if not ("mask" in dense or "sdf" in dense):
                        coverage[:] = 1.0
                return CompiledRegion(
                    region.kind, key, leaves, coverage, sdf=sdf, scalar=scalar,
                    vector=vector, staggered_u=staggered_u,
                    staggered_v=staggered_v, staggered_w=staggered_w)
            for leaf in range(len(leaves)):
                for voxel in range(C.LEAF_VOLUME):
                    if "mask" in accessors:
                        coord = index_points["mask"][leaf, voxel]
                        value = self._sample_accessor(accessors["mask"], coord)
                        coverage[leaf,voxel] = float(value > region.fog_threshold)
                    if "sdf" in accessors:
                        coord = index_points["sdf"][leaf, voxel]
                        value = float(self._sample_accessor(accessors["sdf"], coord))
                        sdf[leaf,voxel] = value
                        coverage[leaf,voxel] = max(coverage[leaf,voxel], float(value <= 0.0))
                    if "scalar" in accessors:
                        coord = index_points["scalar"][leaf, voxel]
                        scalar[leaf,voxel] = float(self._sample_accessor(
                            accessors["scalar"], coord))
                        if "mask" not in accessors and "sdf" not in accessors:
                            coverage[leaf, voxel] = 1.0
                    if "vector" in accessors:
                        coord = index_points["vector"][leaf, voxel]
                        vector[leaf,voxel] = self._sample_accessor(
                            accessors["vector"], coord)
                        if "mask" not in accessors and "sdf" not in accessors:
                            coverage[leaf, voxel] = 1.0
                        if staggered:
                            staggered_u[leaf, voxel] = self._sample_accessor(
                                accessors["vector"], coord - face_offsets["vector"][0])[0]
                            staggered_v[leaf, voxel] = self._sample_accessor(
                                accessors["vector"], coord - face_offsets["vector"][1])[1]
                            staggered_w[leaf, voxel] = self._sample_accessor(
                                accessors["vector"], coord - face_offsets["vector"][2])[2]
                    elif all(role in accessors for role in
                             ("vector_x", "vector_y", "vector_z")):
                        for component, role in enumerate(
                                ("vector_x", "vector_y", "vector_z")):
                            coord = index_points[role][leaf, voxel]
                            vector[leaf, voxel, component] = float(
                                self._sample_accessor(accessors[role], coord))
                            if staggered:
                                (staggered_u, staggered_v, staggered_w)[component][leaf, voxel] = float(
                                    self._sample_accessor(
                                        accessors[role],
                                        coord - face_offsets[role][component]))
                        if "mask" not in accessors and "sdf" not in accessors:
                            coverage[leaf, voxel] = 1.0
            return CompiledRegion(
                region.kind, key, leaves, coverage, sdf=sdf, scalar=scalar,
                vector=vector, staggered_u=staggered_u,
                staggered_v=staggered_v, staggered_w=staggered_w)
        finally:
            # Drop every OpenVDB tree/accessor before the compiled region enters
            # persistent solver state.
            grids.clear()
