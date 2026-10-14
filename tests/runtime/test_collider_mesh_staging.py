"""Evaluated-mesh host work and admission staging stay output-identical (P12).

- Moving colliders rebuild their cache entry every frame. The two
  ``np.unique`` passes that found the referenced vertices for the bounds are
  replaced by a boolean mask; the bounds, the arrays handed to ``wp.Mesh`` and
  every cached array must be byte-identical to the old computation.
- Admission staged every mesh on the CPU and built a CPU BVH that nothing
  queried before the replacement built its own. Staging now binds a
  ``StagedMesh`` stand-in; the materialized mesh must be built from the same
  arrays and reach every descriptor, on CUDA and on a CPU solver device.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.blender import scene_inputs as si


class _Seq:
    def __init__(self, data):
        self.data = np.asarray(data)

    def __len__(self):
        return len(self.data)

    def foreach_get(self, _attr, out):
        out[:] = self.data.reshape(-1)


class _Object:
    name = "collider"

    def __init__(self, verts, tris, matrix):
        self.verts = np.asarray(verts, dtype=np.float32)
        self.tris = np.asarray(tris, dtype=np.int32)
        self.matrix_world = np.asarray(matrix, dtype=np.float32)
        self.data = object()

    def evaluated_get(self, _depsgraph):
        return self

    def to_mesh(self):
        return SimpleNamespace(vertices=_Seq(self.verts), loop_triangles=_Seq(self.tris),
                               calc_loop_triangles=lambda: None)

    def to_mesh_clear(self):
        pass

    def as_pointer(self):
        return id(self)


def _fake_warp(monkeypatch):
    fake = types.ModuleType("warp")
    fake.vec3, fake.int32 = "vec3", np.int32
    fake.built = []
    fake.array = lambda values, dtype=None, device=None: np.array(values, copy=True)

    class Mesh:
        def __init__(self, points, indices):
            self.points, self.indices = points, indices
            fake.built.append(self)

    fake.Mesh = Mesh
    monkeypatch.setitem(sys.modules, "warp", fake)
    return fake


def _reference(verts, tris, matrix, strict):
    """The pre-P12 computation, verbatim."""
    world = np.ascontiguousarray(
        verts @ matrix[:3, :3].T + matrix[:3, 3], dtype=np.float32)
    triangle_points = world[np.unique(tris)]
    diagonal = float(np.linalg.norm(
        triangle_points.max(axis=0) - triangle_points.min(axis=0)))
    edges_a = world[tris[:, 1]] - world[tris[:, 0]]
    edges_b = world[tris[:, 2]] - world[tris[:, 0]]
    area2 = np.linalg.norm(np.cross(edges_a, edges_b), axis=1)
    valid = area2 > (1.0e-12 * max(1.0, diagonal * diagonal))
    had_degenerate = not np.all(valid)
    if strict and had_degenerate:
        return None
    if had_degenerate:
        tris = tris[valid]
    tris = np.ascontiguousarray(tris, dtype=np.int32)
    triangle_points = world[np.unique(tris)]
    return world, tris, triangle_points.min(axis=0), triangle_points.max(axis=0), had_degenerate


def _random_mesh(rng, count, *, unused=0, degenerate=0):
    verts = rng.normal(size=(count + unused, 3)).astype(np.float32)
    tris = rng.integers(0, count, size=(4 * count, 3)).astype(np.int32)
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
                & (tris[:, 0] != tris[:, 2])]
    if degenerate:
        # Zero-area triangles: a vertex and an exact copy of it.
        extra = []
        for index in range(degenerate):
            verts = np.vstack([verts, verts[index]])
            extra.append((index, index + 1, len(verts) - 1))
        tris = np.vstack([tris, np.asarray(extra, dtype=np.int32)])
    return verts, tris


def _pose(angle, offset):
    m = np.eye(4, dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = offset
    return m


@pytest.mark.parametrize("unused,degenerate", [(0, 0), (17, 0), (5, 6), (0, 9)])
def test_moving_mesh_entries_match_the_unique_based_computation(
        monkeypatch, unused, degenerate):
    fake = _fake_warp(monkeypatch)
    rng = np.random.default_rng(unused * 31 + degenerate)
    verts, tris = _random_mesh(rng, 300, unused=unused, degenerate=degenerate)
    obj = _Object(verts, tris, np.eye(4))
    inputs = si.SceneInputs()
    inputs.device = "cuda:0"
    depsgraph = SimpleNamespace(updates=[SimpleNamespace(id=obj)])
    for frame in range(4):
        obj.matrix_world = _pose(0.3 * frame, (0.1 * frame, -0.2, 0.05 * frame))
        entry, error = inputs._evaluated_mesh(
            obj, depsgraph, ("collider", 1), sync_generation=inputs.begin_sync(),
            strict_degenerate=False)
        assert not error
        world, ref_tris, lo, hi, had_degenerate = _reference(
            verts, tris, obj.matrix_world, strict=False)
        assert entry["had_degenerate"] is had_degenerate is bool(degenerate)
        for actual, expected in ((entry["vertices"], world), (entry["triangles"], ref_tris),
                                 (entry["aabb_min"], lo), (entry["aabb_max"], hi),
                                 (fake.built[-1].points, world),
                                 (fake.built[-1].indices, ref_tris.reshape(-1))):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
    assert len(fake.built) == 4  # a moved collider still builds a fresh mesh


def test_strict_degenerate_refusal_is_unchanged(monkeypatch):
    _fake_warp(monkeypatch)
    rng = np.random.default_rng(5)
    verts, tris = _random_mesh(rng, 50, degenerate=2)
    inputs = si.SceneInputs()
    entry, error = inputs._evaluated_mesh(
        _Object(verts, tris, np.eye(4)), SimpleNamespace(updates=[]), ("c", 1))
    assert entry is None and "degenerate" in error


def _staged(device_for_meshes):
    rng = np.random.default_rng(9)
    verts, tris = _random_mesh(rng, 200, unused=3)
    inputs = si.SceneInputs()
    inputs.device = "cpu"
    inputs.stage_meshes = True  # what admission capture sets
    entry, error = inputs._evaluated_mesh(
        _Object(verts, tris, _pose(0.4, (1.0, 2.0, 3.0))),
        SimpleNamespace(updates=[]), ("collider", 7), sync_generation=inputs.begin_sync())
    assert not error and isinstance(entry["mesh"], si.StagedMesh)
    collider = SimpleNamespace(mesh=entry["mesh"])
    source = SimpleNamespace(mesh=entry["mesh"])
    # Admission's own registration call while staging builds nothing.
    assert inputs.materialize_device_meshes((source,), (collider,), "cpu") == 1
    assert isinstance(collider.mesh, si.StagedMesh)
    # Replacement configuration ends staging, then materializes.
    inputs.stage_meshes = False
    inputs.materialize_device_meshes((source,), (collider,), device_for_meshes)
    return inputs, entry, source, collider


@pytest.mark.parametrize("device_kind", ["cpu", "cuda"])
def test_materialized_mesh_comes_from_the_staged_arrays(device_kind, request):
    wp = pytest.importorskip("warp")
    device = (request.getfixturevalue("cuda_device")
              if device_kind == "cuda" else "cpu")
    inputs, entry, source, collider = _staged(device)
    mesh = entry["mesh"]
    assert isinstance(mesh, wp.Mesh)
    assert source.mesh is mesh and collider.mesh is mesh
    assert str(mesh.points.device) == str(wp.get_device(device))
    assert mesh.points.numpy().tobytes() == entry["vertices"].tobytes()
    assert mesh.indices.numpy().tobytes() == entry["triangles"].reshape(-1).tobytes()
    assert inputs.device == device
    inputs.clear()
    assert source.mesh is None and collider.mesh is None


def test_a_staged_mesh_cannot_be_queried():
    with pytest.raises(AttributeError):
        _ = si.StagedMesh().id


def test_admission_stages_and_configuration_unstages(monkeypatch):
    from fluidSim_warp_blender51.blender import simulation_admission as admission
    from fluidSim_warp_blender51.sim.effects import compiler

    fake_effects = types.ModuleType("fluidSim_warp_blender51.blender.effects")
    fake_effects.trust_store_for_scene = lambda _scene: None
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.blender.effects", fake_effects)
    from fluidSim_warp_blender51 import blender as blender_package
    monkeypatch.setattr(blender_package, "effects", fake_effects, raising=False)
    monkeypatch.setattr(compiler, "build_candidate_effects",
                        lambda *_a, **_k: SimpleNamespace(success=True, issues=()))
    seen = {}

    class Inputs:
        stage_meshes = True

        def materialize_device_meshes(self, _descriptors, _colliders, device):
            seen["stage_meshes"] = self.stage_meshes
            seen["device"] = device

    candidate = SimpleNamespace(
        set_source_batch=lambda *_a: None, set_colliders=lambda *_a: None,
        set_force_fields=lambda *_a: None, set_height_source=lambda *_a: None,
        set_custom_effect_candidate=lambda *_a: True)
    snapshot = SimpleNamespace(
        scene_inputs=Inputs(), descriptors=(), colliders=(), extra_plans=(),
        force_fields=(), height_source=None, effects=(), configured_device="cpu")
    admission.SimulationAdmission._configure_candidate(
        None, candidate, snapshot, SimpleNamespace())
    assert seen == {"stage_meshes": False, "device": "cpu"}
    source = open(admission.__file__, encoding="utf-8").read()
    assert "inputs.stage_meshes = True" in source
