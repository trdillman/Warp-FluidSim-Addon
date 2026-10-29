"""object_points_world always releases the evaluated mesh (REVIEW-041)."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def sources(monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.context = SimpleNamespace()
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    monkeypatch.delitem(
        sys.modules, "fluidSim_warp_blender51.blender.sources", raising=False)
    return importlib.import_module("fluidSim_warp_blender51.blender.sources")


class _Vertices:
    def __init__(self, coords, fail=False):
        self._coords = np.asarray(coords, dtype=np.float32)
        self._fail = fail

    def __len__(self):
        return len(self._coords)

    def foreach_get(self, _name, out):
        if self._fail:
            raise RuntimeError("vertex read failed")
        out[:] = self._coords.ravel()


class _EvalObject:
    particle_systems = ()

    def __init__(self, vertices):
        self._mesh = SimpleNamespace(vertices=vertices)
        self.cleared = 0

    def to_mesh(self):
        return self._mesh

    def to_mesh_clear(self):
        self.cleared += 1


def _mesh_object(eval_obj, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = translation
    return SimpleNamespace(
        type="MESH", matrix_world=matrix,
        evaluated_get=lambda _depsgraph: eval_obj)


def test_mesh_points_are_world_space_and_mesh_is_released(sources):
    eval_obj = _EvalObject(_Vertices([(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)]))
    points = sources.object_points_world(
        _mesh_object(eval_obj, (10.0, 0.0, 0.0)), depsgraph=object())

    np.testing.assert_allclose(points, [(10.0, 0.0, 0.0), (11.0, 2.0, 3.0)])
    assert eval_obj.cleared == 1


def test_mesh_is_released_when_reading_vertices_raises(sources):
    eval_obj = _EvalObject(_Vertices([(0.0, 0.0, 0.0)], fail=True))

    with pytest.raises(RuntimeError, match="vertex read failed"):
        sources.object_points_world(_mesh_object(eval_obj), depsgraph=object())
    assert eval_obj.cleared == 1


def test_points_preserve_original_transform_when_evaluated_pose_differs(sources):
    eval_obj = _EvalObject(_Vertices([(1.0, 0.0, 0.0)]))
    eval_obj.matrix_world = np.eye(4)
    eval_obj.matrix_world[0, 3] = 99.0
    points = sources.object_points_world(
        _mesh_object(eval_obj, (10.0, 0.0, 0.0)), depsgraph=object())
    np.testing.assert_array_equal(points, [(11.0, 0.0, 0.0)])
    assert eval_obj.cleared == 1


def test_particle_system_points_are_already_world_space(sources):
    eval_obj = _EvalObject(_Vertices([]))
    eval_obj.particle_systems = [SimpleNamespace(
        particles=_Vertices([(2.0, 3.0, 4.0)]))]
    points = sources.object_points_world(
        _mesh_object(eval_obj, (10.0, 0.0, 0.0)), depsgraph=object())
    np.testing.assert_array_equal(points, [(2.0, 3.0, 4.0)])
    assert eval_obj.cleared == 0


def test_pointcloud_points_use_original_world_transform(sources):
    eval_obj = _EvalObject(_Vertices([]))
    points = _Vertices([(2.0, 3.0, 4.0)])
    eval_obj.data = SimpleNamespace(
        points=points, attributes={"position": SimpleNamespace(data=points)})
    obj = _mesh_object(eval_obj, (10.0, 0.0, 0.0))
    obj.type = "POINTCLOUD"
    np.testing.assert_array_equal(
        sources.object_points_world(obj, depsgraph=object()),
        [(12.0, 3.0, 4.0)])
    assert eval_obj.cleared == 0
