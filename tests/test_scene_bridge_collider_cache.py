"""Regression tests for collider Warp mesh cache invalidation."""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeMeshSeq:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def foreach_get(self, attr, out):
        if attr == "co":
            out[:] = np.asarray(self._items, dtype=np.float32).reshape(-1)
            return
        if attr == "vertices":
            out[:] = np.asarray(self._items, dtype=np.int32).reshape(-1)
            return
        raise AssertionError(attr)


class _FakeMesh:
    def __init__(self, verts, tris):
        self.vertices = _FakeMeshSeq(verts)
        self.loop_triangles = _FakeMeshSeq(tris)
        self.polygons = [object()] * len(tris)
        self.calc_loop_triangles_count = 0

    def calc_loop_triangles(self):
        self.calc_loop_triangles_count += 1


class _FakeMatrix:
    def __init__(self, tx=0.0):
        self._array = np.eye(4, dtype=np.float32)
        self._array[0, 3] = tx
        self.translation = SimpleNamespace(x=tx, y=0.0, z=0.0)

    def __array__(self, dtype=None, copy=None):
        return self._array.astype(dtype) if dtype is not None else self._array


class _FakeEvalObject:
    def __init__(self, owner):
        self.owner = owner
        self.clear_count = 0

    def to_mesh(self):
        self.owner.to_mesh_count += 1
        return self.owner.mesh

    def to_mesh_clear(self):
        self.clear_count += 1


class _FakeObject:
    def __init__(self, mesh):
        self.name = "collider"
        self.type = "MESH"
        self.hide_viewport = False
        self.fluidsim = SimpleNamespace(
            role="COLLIDER",
            enabled=True,
            transfer_velocity=False,
        )
        self.mesh = mesh
        self.matrix_world = _FakeMatrix()
        self.to_mesh_count = 0
        self._eval = _FakeEvalObject(self)

    def evaluated_get(self, depsgraph):
        return self._eval

    def as_pointer(self):
        return 42


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.context = SimpleNamespace(evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[])
    )

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    fake_warp = types.ModuleType("warp")
    fake_warp.meshes = []
    fake_warp.vec3 = object()
    fake_warp.int32 = np.int32
    fake_warp.array = lambda values, dtype=None, device=None: (device, dtype, values.copy())

    class _Mesh:
        def __init__(self, points, indices):
            self.points = points
            self.indices = indices
            fake_warp.meshes.append(self)

    fake_warp.Mesh = _Mesh

    fake_solver = types.ModuleType("fluidSim_warp_blender51.sim.solver")

    class _ColliderDesc:
        pass

    fake_solver.ColliderDesc = _ColliderDesc

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    scene_bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    return scene_bridge, fake_warp


def _triangle_mesh(extra_vert=False):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    tris = [(0, 1, 2)]
    if extra_vert:
        verts.append((0.0, 0.0, 1.0))
        tris.append((0, 2, 3))
    return _FakeMesh(verts, tris)


def test_repeated_collider_gather_reuses_cached_warp_mesh(monkeypatch):
    scene_bridge, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    scene = SimpleNamespace(objects=[obj])

    first = scene_bridge.gather_colliders(scene, sess, depsgraph=object())
    second = scene_bridge.gather_colliders(scene, sess, depsgraph=object())

    assert len(fake_warp.meshes) == 1
    assert first[0].mesh is second[0].mesh
    assert sess.collider_mesh_cache_stats == {"rebuilds": 1, "reuses": 1}


def test_collider_cache_rebuilds_when_topology_or_transform_changes(monkeypatch):
    scene_bridge, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    scene = SimpleNamespace(objects=[obj])

    first = scene_bridge.gather_colliders(scene, sess, depsgraph=object())[0]
    obj.mesh = _triangle_mesh(extra_vert=True)
    after_topology_change = scene_bridge.gather_colliders(scene, sess, depsgraph=object())[0]
    obj.matrix_world = _FakeMatrix(tx=2.0)
    after_transform_change = scene_bridge.gather_colliders(scene, sess, depsgraph=object())[0]

    assert len(fake_warp.meshes) == 3
    assert first.mesh is not after_topology_change.mesh
    assert after_topology_change.mesh is not after_transform_change.mesh
    assert sess.collider_mesh_cache_stats == {"rebuilds": 3, "reuses": 0}
