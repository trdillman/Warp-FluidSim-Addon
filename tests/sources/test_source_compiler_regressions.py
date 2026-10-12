"""Regression tests for the source-region compiler (review 2026-10-11, lane S).

Each test was written red against the unmodified compiler before its fix.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np

from fluidSim_warp_blender51.sim.enums import (
    SourceEvaluationCadence,
    SourceLifecycle,
    SourceOperation,
    SourcePayloadKind,
    SourceShapeShape,
)
from fluidSim_warp_blender51.sim.source_regions import (
    CompileContext,
    PayloadPlan,
    SourceCompiler,
    SourceKey,
    SourcePlan,
    VolumeRegionPlan,
)


# B44: dense VDB staging must sample the grid background, not zero, outside
# the active window.

def test_dense_volume_sampling_uses_background_outside_the_window():
    values = np.full((4, 4, 4), 0.3, np.float32)
    coords = np.array([[1.5, 1.5, 1.5], [-0.5, 1.0, 1.0], [-1.5, 1.0, 1.0],
                       [-3.0, 1.0, 1.0], [10.0, 1.0, 1.0]], np.float64)
    sampled = SourceCompiler._sample_dense_volume(
        values, coords, np.array([0, 0, 0]), 0.3)
    np.testing.assert_allclose(sampled, 0.3, rtol=0.0, atol=1.0e-7)


_VDB_VOXEL = 0.01
_RADIUS = 0.2
_BACKGROUND = 3 * _VDB_VOXEL
_HALF = int(round(_RADIUS / _VDB_VOXEL)) + 3


def _sphere_sdf(i, j, k):
    distance = np.sqrt(np.asarray(i, float) ** 2 + np.asarray(j, float) ** 2
                       + np.asarray(k, float) ** 2) * _VDB_VOXEL - _RADIUS
    return np.clip(distance, -_BACKGROUND, _BACKGROUND)


class _Transform:
    def isLinear(self):
        return True

    def indexToWorld(self, coord):
        return tuple(np.asarray(coord, float) * _VDB_VOXEL)


class _Accessor:
    def getValue(self, coord):
        if min(coord) < -_HALF or max(coord) > _HALF:
            return _BACKGROUND
        return float(_sphere_sdf(*coord))


class _NarrowBandGrid:
    """A level set whose sim leaves extend past its active window."""

    transform = _Transform()
    background = _BACKGROUND

    def __init__(self, dense):
        self.dense = dense

    def evalActiveVoxelBoundingBox(self):
        return ((-_HALF,) * 3, (_HALF,) * 3)

    def getConstAccessor(self):
        return _Accessor()

    def copyToArray(self, values, ijk):
        if not self.dense:
            raise AttributeError("accessor fallback")
        index = np.arange(values.shape[0]) + ijk[0]
        i, j, k = np.meshgrid(index, index, index, indexing="ij")
        values[...] = _sphere_sdf(i, j, k)


def _compile_level_set(tmp_path, monkeypatch, dense):
    path = tmp_path / "sphere.vdb"
    path.write_bytes(b"fixture")
    monkeypatch.setitem(sys.modules, "openvdb", types.SimpleNamespace(
        read=lambda _path, _name: _NarrowBandGrid(dense)))
    plan = SourcePlan(
        SourceKey("level-set"), SourceOperation.LIQUID, SourceLifecycle.TOP_UP,
        SourceEvaluationCadence.STATIC,
        VolumeRegionPlan(str(path), sdf_grid="surface"),
        PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
    context = CompileContext(0, 0, 0, "cpu", (-1.0, -1.0, -1.0), 0.05,
                             (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), upload=False)
    return SourceCompiler().compile((plan,), context).sources[0].region


def _z_rotation(angle):
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def _rotate(quaternion, vector):
    x, y, z, w = quaternion
    matrix = np.array([
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
        (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y),
    ])
    return matrix @ np.asarray(vector, np.float64)


def _box_corners(position, rotation, half_extents):
    return np.array([
        np.asarray(position) + _rotate(rotation, (sx * half_extents[0],
                                                  sy * half_extents[1],
                                                  sz * half_extents[2]))
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


# B43: analytic bounds must contain the rotated shape, not only its
# unrotated largest extent.

def test_rotated_analytic_box_compiles_its_full_coverage():
    from fluidSim_warp_blender51.sim.source_regions import AnalyticRegionPlan

    context = CompileContext(0, 0, 0, "cpu", (-4.0, -4.0, -4.0), 0.05,
                             (-4.0, -4.0, -4.0), (4.0, 4.0, 4.0),
                             leaf_capacity=10 ** 6, upload=False)
    rotation = _z_rotation(math.pi / 4.0)
    region = AnalyticRegionPlan(
        shape=SourceShapeShape.BOX, rotation=rotation,
        half_extents=(1.5, 1.5, 0.1), radius=1.5, height=0.1)
    plan = SourcePlan(SourceKey("box"), SourceOperation.LIQUID,
                      SourceLifecycle.TOP_UP, SourceEvaluationCadence.STATIC,
                      region, PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))
    compiled = SourceCompiler().compile((plan,), context).sources[0].region
    whole = SourceCompiler._candidate_leaves(
        (context.domain_min, context.domain_max), context)
    _, points = SourceCompiler._voxel_points(whole, context)
    expected = int(np.count_nonzero(SourceCompiler._analytic_sdf(region, points) <= 0.0))
    assert compiled.covered_voxel_count == expected
    lo, hi = region.bounds()
    corners = _box_corners(region.position, rotation, region.half_extents)
    assert np.all(corners >= np.asarray(lo) - 1e-9)
    assert np.all(corners <= np.asarray(hi) + 1e-9)


def test_rotated_legacy_descriptor_aabb_contains_every_corner():
    from fluidSim_warp_blender51.sim.source_contracts import SourceShapeDesc

    rotation = _z_rotation(math.pi / 4.0)
    for shape, half_extents, radius, height in (
            (SourceShapeShape.BOX, (1.5, 1.5, 0.1), 1.5, 0.1),
            (SourceShapeShape.CYLINDER, (1.0, 1.0, 0.1), 1.0, 0.1)):
        tilt = (math.sin(math.pi / 8.0), 0.0, 0.0, math.cos(math.pi / 8.0))
        for quaternion in (rotation, tilt):
            desc = SourceShapeDesc(shape=int(shape), position=(0.2, -0.1, 0.3),
                                   rotation=quaternion, half_extents=half_extents,
                                   radius=radius, height=height)
            extents = (half_extents if shape == SourceShapeShape.BOX
                       else (radius, radius, height))
            corners = _box_corners(desc.position, quaternion, extents)
            lo, hi = desc.aabb()
            assert np.all(corners >= np.asarray(lo) - 1e-9), (shape, quaternion)
            assert np.all(corners <= np.asarray(hi) + 1e-9), (shape, quaternion)


def test_unrotated_analytic_bounds_are_unchanged():
    from fluidSim_warp_blender51.sim.source_contracts import SourceShapeDesc
    from fluidSim_warp_blender51.sim.source_regions import AnalyticRegionPlan

    for shape in SourceShapeShape:
        region = AnalyticRegionPlan(shape=shape, position=(0.5, 0.25, -0.5),
                                    half_extents=(0.3, 0.7, 0.2), radius=0.4,
                                    height=0.6, feather=0.05)
        extent = 0.7 + 0.05
        assert region.bounds() == ((0.5 - extent, 0.25 - extent, -0.5 - extent),
                                   (0.5 + extent, 0.25 + extent, -0.5 + extent))
        desc = SourceShapeDesc(shape=int(shape), position=(0.5, 0.25, -0.5),
                               half_extents=(0.3, 0.7, 0.2), radius=0.4,
                               height=0.6, feather=0.05)
        assert desc.aabb() == ((0.5 - extent, 0.25 - extent, -0.5 - extent),
                               (0.5 + extent, 0.25 + extent, -0.5 + extent))


def test_dense_level_set_coverage_matches_the_accessor_path(tmp_path, monkeypatch):
    accessor = _compile_level_set(tmp_path, monkeypatch, dense=False)
    dense = _compile_level_set(tmp_path, monkeypatch, dense=True)
    expected = 4.0 / 3.0 * math.pi * _RADIUS ** 3 / 0.05 ** 3
    assert abs(accessor.covered_voxel_count - expected) < 0.2 * expected
    assert dense.covered_voxel_count == accessor.covered_voxel_count
    np.testing.assert_array_equal(dense.leaf_coords, accessor.leaf_coords)
    np.testing.assert_array_equal(dense.coverage, accessor.coverage)


# B50: every point attribute array is part of the compiled-region key.

def _point_plan(velocities, temperature, *, cadence=SourceEvaluationCadence.STATIC,
                stable_id="points", positions=None):
    from fluidSim_warp_blender51.sim.source_regions import PointRegionPlan

    count = len(velocities)
    if positions is None:
        rng = np.random.default_rng(3)
        positions = rng.uniform(0.2, 0.8, (count, 3)).astype(np.float32)
    region = PointRegionPlan(positions, velocities=velocities,
                             temperature=temperature)
    return SourcePlan(SourceKey(stable_id), SourceOperation.LIQUID,
                      SourceLifecycle.DIRECT_APPEND, cadence, region,
                      PayloadPlan(kind=SourcePayloadKind.LIQUID_ATTRIBUTES))


def _small_context(frame=1):
    return CompileContext(frame, 0, 1, "cpu", (0.0, 0.0, 0.0), 0.1,
                          (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), upload=False)


def test_interior_point_attribute_edits_miss_the_compiler_cache():
    count = 2000
    velocities = np.zeros((count, 3), np.float32)
    temperature = np.zeros(count, np.float32)
    compiler = SourceCompiler()
    first = compiler.compile((_point_plan(velocities, temperature),), _small_context())
    assert first.sources[0].region.points.velocities[count // 2, 0] == 0.0

    edited_velocities = velocities.copy()
    edited_velocities[count // 2] = (4.0, 0.0, 0.0)
    second = compiler.compile(
        (_point_plan(edited_velocities, temperature),), _small_context())
    assert second.sources[0].region.points.velocities[count // 2, 0] == 4.0

    edited_temperature = temperature.copy()
    edited_temperature[count // 2] = 300.0
    third = compiler.compile(
        (_point_plan(edited_velocities, edited_temperature),), _small_context())
    assert third.sources[0].region.points.temperature[count // 2] == 300.0
    assert compiler.compile_count == 3


# B49: two plans sharing one stable id (a Shift+D duplicate) must not
# evict each other's compiled region.

def _pressure_box(x, stable_id="object:shared"):
    from fluidSim_warp_blender51.sim.source_regions import AnalyticRegionPlan

    return SourcePlan(
        SourceKey(stable_id), SourceOperation.PRESSURE, SourceLifecycle.EVERY_FRAME,
        SourceEvaluationCadence.FIRST_SIM_FRAME,
        AnalyticRegionPlan(shape=SourceShapeShape.BOX, position=(x, 0.5, 0.5),
                           half_extents=(0.2, 0.2, 0.2)),
        PayloadPlan(kind=SourcePayloadKind.PRESSURE, pressure=1.0))


def test_duplicate_stable_ids_keep_both_compiled_regions():
    context = CompileContext(1, 0, 1, "cpu", (0.0, 0.0, 0.0), 0.05,
                             (0.0, 0.0, 0.0), (2.0, 1.0, 1.0), upload=True)
    compiler = SourceCompiler()
    try:
        for _frame in range(3):
            batch = compiler.compile((_pressure_box(0.4), _pressure_box(1.4)), context)
            first, second = (source.region for source in batch.sources)
            assert first.device_coverage is not None
            assert second.device_coverage is not None
            assert first.covered_voxel_count and second.covered_voxel_count
            assert not np.array_equal(first.leaf_coords, second.leaf_coords)
        assert compiler.compile_count == 2
        assert any(item.code == "DUPLICATE_SOURCE_ID" for item in batch.diagnostics)
    finally:
        compiler.release()
