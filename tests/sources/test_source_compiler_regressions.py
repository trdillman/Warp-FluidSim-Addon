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


def test_dense_level_set_coverage_matches_the_accessor_path(tmp_path, monkeypatch):
    accessor = _compile_level_set(tmp_path, monkeypatch, dense=False)
    dense = _compile_level_set(tmp_path, monkeypatch, dense=True)
    expected = 4.0 / 3.0 * math.pi * _RADIUS ** 3 / 0.05 ** 3
    assert abs(accessor.covered_voxel_count - expected) < 0.2 * expected
    assert dense.covered_voxel_count == accessor.covered_voxel_count
    np.testing.assert_array_equal(dense.leaf_coords, accessor.leaf_coords)
    np.testing.assert_array_equal(dense.coverage, accessor.coverage)
