"""Regression tests for collider Warp mesh cache invalidation."""

from __future__ import annotations

import importlib
import importlib.machinery
import math
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

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
    def __init__(self, tx=0.0, angle=0.0):
        self._array = np.eye(4, dtype=np.float32)
        self._array[0, 3] = tx
        self.translation = _FakeVector(tx, 0.0, 0.0)
        self.angle = angle

    def to_translation(self):
        return self.translation

    def to_quaternion(self):
        return _FakeQuaternion(self.angle)

    def __array__(self, dtype=None, copy=None):
        return self._array.astype(dtype) if dtype is not None else self._array


class _FakeVector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __array__(self, dtype=None, copy=None):
        values = np.asarray((self.x, self.y, self.z))
        return values.astype(dtype) if dtype is not None else values


class _FakeQuaternion:
    def __init__(self, angle):
        self.angle = angle
        self.x = 0.0
        self.y = 0.0
        self.z = math.sin(angle * 0.5)
        self.w = math.cos(angle * 0.5)

    def rotation_difference(self, other):
        delta = other.angle - self.angle
        return SimpleNamespace(
            angle=abs(delta),
            axis=_FakeVector(0.0, 0.0, 1.0 if delta >= 0.0 else -1.0),
        )

    def copy(self):
        return _FakeQuaternion(self.angle)


class _FakeEvalObject:
    def __init__(self, owner):
        self.owner = owner
        self.clear_count = 0

    def to_mesh(self):
        self.owner.to_mesh_count += 1
        return self.owner.mesh

    def to_mesh_clear(self):
        self.clear_count += 1

    @property
    def name(self):
        return self.owner.name

    @property
    def matrix_world(self):
        return self.owner.evaluated_matrix_world or self.owner.matrix_world


class _FakeObject:
    def __init__(self, mesh):
        self.name = "collider"
        self.type = "MESH"
        self.hide_viewport = False
        self.fluidsim = SimpleNamespace(
            role="COLLIDER",
            enabled=True,
            transfer_velocity=False,
            collider_friction=0.25,
            collider_restitution=0.75,
        )
        self.mesh = mesh
        self.data = object()
        self.matrix_world = _FakeMatrix()
        self.evaluated_matrix_world = None
        self.to_mesh_count = 0
        self._eval = _FakeEvalObject(self)

    def evaluated_get(self, depsgraph):
        return self._eval

    def as_pointer(self):
        return id(self)


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.context = SimpleNamespace(evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[])
    )

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    fake_warp = types.ModuleType("warp")
    fake_warp.__spec__ = importlib.machinery.ModuleSpec("warp", loader=None)
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

    class _SourceShapeDesc:
        pass

    class _MeshSourceDesc:
        pass

    class _HeightSourceDesc:
        pass

    class _ForceFieldDesc:
        def __init__(self, **values):
            self.__dict__.update(values)

    fake_solver.ColliderDesc = _ColliderDesc
    fake_solver.SourceShapeDesc = _SourceShapeDesc
    fake_solver.MeshSourceDesc = _MeshSourceDesc
    fake_solver.HeightSourceDesc = _HeightSourceDesc
    fake_solver.ForceFieldDesc = _ForceFieldDesc
    fake_sourcing = types.ModuleType("fluidSim_warp_blender51.sim.sourcing")
    fake_sourcing.MeshSourceDesc = _MeshSourceDesc

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    monkeypatch.setitem(sys.modules, "warp", fake_warp)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.sim.sourcing", fake_sourcing)
    sys.modules.pop("fluidSim_warp_blender51.blender.colliders", None)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    scene_bridge = importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")
    colliders = importlib.import_module("fluidSim_warp_blender51.blender.colliders")
    return scene_bridge, colliders, fake_warp


def _triangle_mesh(extra_vert=False):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    tris = [(0, 1, 2)]
    if extra_vert:
        verts.append((0.0, 0.0, 1.0))
        tris.append((0, 2, 3))
    return _FakeMesh(verts, tris)


def _cube_mesh():
    verts = [
        (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
    ]
    tris = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return _FakeMesh(verts, tris)


def _mesh_source_settings(role="SOURCE", **overrides):
    values = dict(
        role=role,
        enabled=True,
        geometry_mode="MESH",
        mesh_object=None,
        mesh_evaluation_mode="FIRST_SIM_FRAME",
        mesh_surface_thickness=0.05,
        sink_region_mode="SOLID",
        mesh_error="",
        shape="BOX",
        source_mode="PARTICLES",
        feather=0.0,
        initial_velocity=(1.0, 2.0, 3.0),
        use_object_velocity=False,
        blend_mode="REPLACE",
        emit_rate=3.0,
        emission_mode="CONTINUOUS_INFLOW",
        temperature=4.0,
        viscosity=5.0,
        density=6.0,
        use_lifespan=True,
        life_min=2.0,
        life_max=7.0,
        force_type="OFF",
        force_strength=0.0,
        force_direction=(0.0, 0.0, 1.0),
        transfer_velocity=False,
        collider_friction=0.0,
        collider_restitution=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_blender_force_fields_translate_to_gpu_proxies(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()

    def proxy(field_type, **overrides):
        obj = _FakeObject(_triangle_mesh())
        values = dict(
            type=field_type, strength=1.0, falloff_power=0.0,
            use_min_distance=False, distance_min=0.0,
            use_max_distance=False, distance_max=0.0,
            apply_to_location=True, shape="POINT", flow=0.0, noise=0.0,
            seed=17, size=1.0, rest_length=0.0,
            harmonic_damping=0.0, linear_drag=0.0, quadratic_drag=0.0,
            guide_minimum=0.0, guide_free=0.0, guide_clump_amount=0.0,
            texture=None, texture_mode="RGB", use_global_coords=False,
            use_smoke_density=False, source_object=None,
        )
        values.update(overrides)
        obj.field = SimpleNamespace(**values)
        return obj

    force = proxy(
        "FORCE", strength=-3.0, falloff_power=2.0,
        use_min_distance=True, distance_min=0.5,
        use_max_distance=True, distance_max=4.0)
    force.matrix_world = _FakeMatrix(tx=2.0)
    wind = proxy("WIND", strength=5.0)
    wind.matrix_world._array[:3, 2] = (0.0, 2.0, 0.0)
    remaining = [
        proxy("VORTEX"), proxy("MAGNET"),
        proxy("HARMONIC", rest_length=0.75, harmonic_damping=0.4),
        proxy("CHARGE"),
        proxy("LENNARDJ", use_min_distance=True, distance_min=0.3),
        proxy("TEXTURE", texture=SimpleNamespace(name="Cloud Proxy"),
              texture_mode="CURL", use_global_coords=True),
        proxy("GUIDE", guide_minimum=0.6, guide_clump_amount=0.2),
        proxy("BOID"), proxy("TURBULENCE", size=0.8, seed=23),
        proxy("DRAG", linear_drag=0.4, quadratic_drag=0.2),
        proxy("FLUID_FLOW", flow=0.7),
    ]

    fields = sess.scene_inputs.gather_force_fields(
        SimpleNamespace(objects=[force, wind, *remaining]))

    assert [field.force_type.name for field in fields] == [
        "FORCE", "WIND", "VORTEX", "MAGNETIC", "HARMONIC", "CHARGE",
        "LENNARD_JONES", "TEXTURE", "CURVE_GUIDE", "BOID",
        "TURBULENCE", "DRAG", "FLUID_FLOW"]
    assert fields[0].center == (2.0, 0.0, 0.0)
    assert fields[0].strength == -3.0
    assert fields[0].min_distance == 0.5
    assert fields[0].max_distance == 4.0
    assert fields[0].falloff_power == 2.0
    assert fields[1].direction == (0.0, 1.0, 0.0)
    assert fields[4].rest_length == 0.75
    assert fields[4].damping == 0.4
    assert fields[6].size == 0.3
    assert fields[7].texture_mode.name == "CURL"
    assert fields[7].flags & 1
    assert len(fields[8].guide_points) == 2
    assert fields[10].seed == 23
    assert fields[11].linear_drag == 0.4
    assert fields[11].quadratic_drag == 0.2
    assert fields[12].flow == 0.7


def test_scene_input_state_is_owned_outside_session(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)

    sess = scene_bridge.Session()

    assert sess.scene_inputs is not None
    assert not hasattr(sess, "_collider_prev")
    assert not hasattr(sess, "_collider_rotation_prev")
    assert not hasattr(sess, "_collider_mesh_cache")


def test_repeated_collider_gather_reuses_cached_warp_mesh(monkeypatch):
    scene_bridge, colliders, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    scene = SimpleNamespace(objects=[obj])

    first = sess.scene_inputs.gather_colliders(scene, depsgraph=object())
    second = colliders.gather_colliders(scene, sess, depsgraph=object())

    assert len(fake_warp.meshes) == 1
    assert obj.to_mesh_count == 1
    assert first[0].mesh is second[0].mesh
    assert sess.collider_mesh_cache_stats == {"rebuilds": 1, "reuses": 1}


def test_scene_input_owner_evicts_removed_collider_mesh(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    scene = SimpleNamespace(objects=[_FakeObject(_triangle_mesh())])

    sess.scene_inputs.gather_colliders(scene, depsgraph=object())
    scene.objects.clear()
    sess.scene_inputs.gather_colliders(scene, depsgraph=object())

    assert sess.scene_inputs._collider_mesh_cache == {}


def test_collider_gather_preserves_individual_material_response(monkeypatch):
    scene_bridge, colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())

    desc = colliders.gather_colliders(
        SimpleNamespace(objects=[obj]), sess, depsgraph=object())[0]

    assert desc.friction == 0.25
    assert desc.restitution == 0.75


def test_collider_gather_uses_evaluated_transform(monkeypatch):
    scene_bridge, colliders, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.evaluated_matrix_world = _FakeMatrix(tx=3.0)

    desc = colliders.gather_colliders(
        SimpleNamespace(objects=[obj]), sess, depsgraph=object())[0]

    assert desc.center == (3.0, 0.0, 0.0)
    assert fake_warp.meshes[0].points[2][0, 0] == 3.0


def test_animated_collider_reports_angular_velocity(monkeypatch):
    scene_bridge, colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.fluidsim.transfer_velocity = True
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1),
    )

    assert colliders.gather_colliders(scene, sess, depsgraph=object())[0].angular_velocity == (0.0, 0.0, 0.0)
    scene.frame_current = 2
    obj.evaluated_matrix_world = _FakeMatrix(angle=math.pi / 2.0)
    angular = colliders.gather_colliders(scene, sess, depsgraph=object())[0].angular_velocity

    assert angular == (0.0, 0.0, pytest.approx(12.0 * math.pi))


def test_animated_collider_reports_linear_velocity(monkeypatch):
    scene_bridge, colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.fluidsim.transfer_velocity = True
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1),
    )

    assert colliders.gather_colliders(scene, sess, depsgraph=object())[0].linear_velocity == (0.0, 0.0, 0.0)
    scene.frame_current = 2
    obj.evaluated_matrix_world = _FakeMatrix(tx=2.0)

    assert colliders.gather_colliders(scene, sess, depsgraph=object())[0].linear_velocity == (48.0, 0.0, 0.0)


def test_collider_cache_rebuilds_when_topology_or_transform_changes(monkeypatch):
    scene_bridge, colliders, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    scene = SimpleNamespace(objects=[obj])

    first = colliders.gather_colliders(scene, sess, depsgraph=object())[0]
    obj.mesh = _triangle_mesh(extra_vert=True)
    obj.data = object()
    after_topology_change = colliders.gather_colliders(
        scene, sess, depsgraph=object())[0]
    obj.matrix_world = _FakeMatrix(tx=2.0)
    after_transform_change = colliders.gather_colliders(
        scene, sess, depsgraph=object())[0]

    assert len(fake_warp.meshes) == 3
    assert first.mesh is not after_topology_change.mesh
    assert after_topology_change.mesh is not after_transform_change.mesh
    assert sess.collider_mesh_cache_stats == {"rebuilds": 3, "reuses": 0}


def test_collider_cache_rebuilds_for_depsgraph_object_update(monkeypatch):
    scene_bridge, colliders, fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    scene = SimpleNamespace(objects=[obj])

    colliders.gather_colliders(scene, sess, depsgraph=SimpleNamespace(updates=[]))
    obj.mesh = _triangle_mesh(extra_vert=True)
    update = SimpleNamespace(id=SimpleNamespace(original=obj))
    colliders.gather_colliders(
        scene, sess, depsgraph=SimpleNamespace(updates=[update]))

    assert obj.to_mesh_count == 2
    assert len(fake_warp.meshes) == 2
    assert sess.collider_mesh_cache_stats == {"rebuilds": 2, "reuses": 0}


def test_source_translation_uses_session_velocity_and_skips_disabled(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sources = importlib.import_module("fluidSim_warp_blender51.blender.sources")
    sess = scene_bridge.Session()
    settings = dict(
        role="SOURCE",
        enabled=True,
        shape="BOX",
        source_mode="PARTICLES",
        feather=0.1,
        initial_velocity=(1.0, 0.0, 0.0),
        use_object_velocity=True,
        blend_mode="ADD",
        emit_rate=2.0,
        emission_mode="CONTINUOUS_INFLOW",
        temperature=0.0,
        viscosity=0.0,
        density=1.0,
        use_lifespan=False,
        life_min=0.0,
        life_max=0.0,
        force_type="DIRECTIONAL",
        force_strength=0.0,
        force_direction=(0.0, 0.0, 1.0),
    )
    obj = SimpleNamespace(
        name="source",
        fluidsim=SimpleNamespace(**settings),
        hide_viewport=False,
        matrix_world=_FakeMatrix(),
        dimensions=SimpleNamespace(x=2.0, y=4.0, z=6.0),
    )
    disabled = SimpleNamespace(
        fluidsim=SimpleNamespace(**{**settings, "enabled": False}),
        hide_viewport=False,
    )
    scene = SimpleNamespace(
        objects=[obj, disabled],
        frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1),
    )

    descriptor = sources.gather_sources(scene, sess)[0]
    assert descriptor.velocity == (1.0, 0.0, 0.0)
    assert descriptor.emission_mode.name == "CONTINUOUS_INFLOW"
    source_id = descriptor.source_id
    scene.frame_current = 2
    obj.matrix_world = _FakeMatrix(tx=1.0)
    descriptors = sources.gather_sources(scene, sess)

    assert len(descriptors) == 1
    assert descriptors[0].velocity == (25.0, 0.0, 0.0)
    assert descriptors[0].half_extents == (1.0, 2.0, 3.0)

    obj.name = "renamed source"
    scene.frame_current = 3
    renamed = sources.gather_sources(scene, sess)
    assert renamed[0].source_id == source_id


def test_mesh_source_uses_own_or_referenced_evaluated_mesh(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    tagged = _FakeObject(_triangle_mesh())
    tagged.name = "source"
    tagged.fluidsim = _mesh_source_settings(
        force_type="SPIN", force_strength=2.5,
        force_direction=(0.0, 1.0, 0.0))
    referenced = _FakeObject(_triangle_mesh(extra_vert=True))
    referenced.name = "referenced"
    referenced.fluidsim.role = "NONE"
    scene = SimpleNamespace(
        objects=[tagged, referenced], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    own = sess.scene_inputs.gather_sources(scene, depsgraph=object())
    assert len(own) == 1
    assert own[0].mesh is not None
    assert tagged.to_mesh_count == 1
    assert own[0].velocity == (1.0, 2.0, 3.0)
    assert own[0].emit_rate == 3.0
    assert own[0].force_type.name == "SPIN"
    assert own[0].force_strength == 2.5
    assert own[0].force_dir == (0.0, 1.0, 0.0)
    assert own[0].force_center == (0.0, 0.0, 0.0)

    sess.scene_inputs.clear()
    tagged.fluidsim.mesh_object = referenced
    referenced.evaluated_matrix_world = _FakeMatrix(tx=4.0)
    selected = sess.scene_inputs.gather_sources(scene, depsgraph=object())
    assert len(selected) == 1
    assert referenced.to_mesh_count == 1
    assert selected[0].aabb_max == (5.0, 1.0, 1.0)
    assert selected[0].force_center == (4.0, 0.0, 0.0)


def test_mesh_sources_share_one_evaluated_reference_upload(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    referenced = _FakeObject(_triangle_mesh(extra_vert=True))
    referenced.name = "referenced"
    referenced.fluidsim.role = "NONE"
    owners = []
    for name in ("source-a", "source-b"):
        owner = _FakeObject(_triangle_mesh())
        owner.name = name
        owner.fluidsim = _mesh_source_settings(mesh_object=referenced)
        owners.append(owner)
    scene = SimpleNamespace(
        objects=[*owners, referenced], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    descriptors = sess.scene_inputs.gather_sources(scene, depsgraph=object())

    assert len(descriptors) == 2
    assert referenced.to_mesh_count == 1
    assert descriptors[0].mesh is descriptors[1].mesh


def test_dynamic_mesh_source_and_collider_share_one_upload(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    collider = _FakeObject(_cube_mesh())
    collider.name = "collider"
    owner = _FakeObject(_triangle_mesh())
    owner.name = "source"
    owner.fluidsim = _mesh_source_settings(
        mesh_object=collider, mesh_evaluation_mode="EVERY_SIM_FRAME")
    scene = SimpleNamespace(
        objects=[owner, collider], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    depsgraph = SimpleNamespace(updates=[SimpleNamespace(
        id=SimpleNamespace(original=collider))])
    sync_generation = sess.scene_inputs.begin_sync()
    sources = sess.scene_inputs.gather_sources(
        scene, depsgraph=depsgraph, sync_generation=sync_generation)
    colliders = sess.scene_inputs.gather_colliders(
        scene, depsgraph=depsgraph, sync_generation=sync_generation)

    assert len(sources) == len(colliders) == 1
    assert collider.to_mesh_count == 1
    assert sources[0].mesh is colliders[0].mesh


def test_first_frame_mesh_snapshot_is_immutable_until_clear(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.name = "source"
    obj.fluidsim = _mesh_source_settings()
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    first = sess.scene_inputs.gather_sources(scene, depsgraph=object())[0]
    obj.evaluated_matrix_world = _FakeMatrix(tx=4.0)
    scene.frame_current = 2
    second = sess.scene_inputs.gather_sources(scene, depsgraph=object())[0]

    assert obj.to_mesh_count == 1
    assert second.mesh is first.mesh
    assert second.aabb_min == first.aabb_min
    sess.scene_inputs.clear()
    third = sess.scene_inputs.gather_sources(scene, depsgraph=object())[0]
    assert obj.to_mesh_count == 2
    assert third.aabb_min[0] == pytest.approx(4.0)


def test_first_frame_snapshot_defers_until_simulation_capture(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.name = "source"
    obj.fluidsim = _mesh_source_settings()
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    pending = sess.scene_inputs.gather_sources(
        scene, depsgraph=object(), capture_first_mesh=False)
    assert pending == []
    assert obj.to_mesh_count == 0

    obj.evaluated_matrix_world = _FakeMatrix(tx=3.0)
    scene.frame_current = 8
    captured = sess.scene_inputs.gather_sources(
        scene, depsgraph=object(), capture_first_mesh=True)
    assert len(captured) == 1
    assert captured[0].aabb_min[0] == pytest.approx(3.0)


def test_every_frame_mesh_source_refreshes_transform_and_topology(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.name = "source"
    obj.fluidsim = _mesh_source_settings(
        mesh_evaluation_mode="EVERY_SIM_FRAME")
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    first = sess.scene_inputs.gather_sources(scene, depsgraph=object())[0]
    obj.mesh = _triangle_mesh(extra_vert=True)
    obj.evaluated_matrix_world = _FakeMatrix(tx=2.0)
    scene.frame_current = 2
    second = sess.scene_inputs.gather_sources(scene, depsgraph=object())[0]

    assert obj.to_mesh_count == 2
    assert second.mesh is not first.mesh
    assert second.aabb_min[0] == pytest.approx(2.0)
    assert second.aabb_max == (3.0, 1.0, 1.0)


def test_mesh_sink_validation_is_visible_and_never_falls_back(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.name = "sink"
    obj.fluidsim = _mesh_source_settings(role="SINK")
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))

    assert sess.scene_inputs.gather_sources(scene, depsgraph=object()) == []
    assert obj.fluidsim.mesh_error == "Solid mesh sink requires a watertight mesh"

    obj.fluidsim.sink_region_mode = "SURFACE"
    surface = sess.scene_inputs.gather_sources(scene, depsgraph=object())
    assert len(surface) == 1
    assert surface[0].sink_region_mode == 1
    assert obj.fluidsim.mesh_error == ""

    sess.scene_inputs.clear()
    obj.mesh = _cube_mesh()
    obj.fluidsim.sink_region_mode = "SOLID"
    solid = sess.scene_inputs.gather_sources(scene, depsgraph=object())
    assert len(solid) == 1
    assert solid[0].sink_region_mode == 0


def test_mesh_source_cache_drops_deleted_objects_and_degenerate_meshes(monkeypatch):
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    sess = scene_bridge.Session()
    obj = _FakeObject(_triangle_mesh())
    obj.name = "source"
    obj.fluidsim = _mesh_source_settings()
    scene = SimpleNamespace(
        objects=[obj], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))
    assert len(sess.scene_inputs.gather_sources(scene, depsgraph=object())) == 1
    assert sess.scene_inputs._collider_mesh_cache

    scene.objects = []
    assert sess.scene_inputs.gather_sources(scene, depsgraph=object()) == []
    assert sess.scene_inputs._collider_mesh_cache == {}

    obj.mesh = _FakeMesh(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        [(0, 1, 2)])
    scene.objects = [obj]
    assert sess.scene_inputs.gather_sources(scene, depsgraph=object()) == []
    assert "degenerate" in obj.fluidsim.mesh_error

    sess.scene_inputs.clear()
    obj.mesh = _FakeMesh(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2), (0, 0, 1)])
    assert sess.scene_inputs.gather_sources(scene, depsgraph=object()) == []
    assert "degenerate" in obj.fluidsim.mesh_error

    sess.scene_inputs.clear()
    obj.mesh = _FakeMesh([], [])
    assert sess.scene_inputs.gather_sources(scene, depsgraph=object()) == []
    assert "no triangles" in obj.fluidsim.mesh_error


def test_height_source_translation_preserves_scene_settings(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    sources = importlib.import_module("fluidSim_warp_blender51.blender.sources")
    scene = SimpleNamespace(fluidsim=SimpleNamespace(
        height_enabled=True,
        height_base=1.25,
        height_back_mode="FLAT",
        height_velocity=(1.0, 2.0, 3.0),
        height_emit_particles=True,
        particles_per_voxel=8.0,
    ))

    desc = sources.gather_height_source(scene)

    assert desc.enabled is True
    assert desc.base_height == 1.25
    assert desc.back_mode.name == "FLAT"
    assert desc.velocity == (1.0, 2.0, 3.0)
    assert desc.emit_particles is True
    assert desc.emit_rate == 8.0


def test_watertight_check_is_deferred_to_solid_sink_consumers(monkeypatch):
    """Watertight edge analysis must not run on the collider/source hot path.

    Measured before this contract: a plain COLLIDER with 122,880 evaluated
    triangles paid ~450 ms of pure-Python edge counting twice per playback
    frame, although the result is only ever read for a SOLID mesh sink.
    """
    scene_bridge, _colliders, _fake_warp = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import scene_inputs as si

    calls = []
    original = si._watertight_triangles

    def counting(tris):
        calls.append(len(tris))
        return original(tris)

    monkeypatch.setattr(si, "_watertight_triangles", counting)

    sess = scene_bridge.Session()
    collider = _FakeObject(_cube_mesh())
    collider.name = "collider"
    scene = SimpleNamespace(objects=[collider])
    assert sess.scene_inputs.gather_colliders(scene, depsgraph=object())
    assert calls == [], (
        f"watertight ran {len(calls)} time(s) for a non-sink collider")

    # A SOLID mesh sink is the only consumer, so it must still be evaluated.
    sess2 = scene_bridge.Session()
    sink = _FakeObject(_triangle_mesh())
    sink.name = "sink"
    sink.fluidsim = _mesh_source_settings(role="SINK")
    sink_scene = SimpleNamespace(
        objects=[sink], frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1))
    assert sess2.scene_inputs.gather_sources(sink_scene, depsgraph=object()) == []
    assert sink.fluidsim.mesh_error == "Solid mesh sink requires a watertight mesh"
    assert calls, "watertight must still be evaluated for a SOLID mesh sink"


@pytest.mark.parametrize("mesh_tris,expected", [
    ([], False),
    ([(0, 1, 2)], False),
    ([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
      (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
      (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)], True),
    # Two coincident triangles give every edge a count of 2, so the existing
    # edge-manifold criterion reports True. Recorded as-is: the vectorized
    # rewrite must preserve current semantics, not "improve" them.
    ([(0, 1, 2), (0, 1, 2)], True),
])
def test_watertight_triangles_semantics(mesh_tris, expected):
    """Guards the vectorized rewrite against the original scalar semantics."""
    from fluidSim_warp_blender51.blender import scene_inputs as si

    tris = np.asarray(mesh_tris, dtype=np.int32).reshape(-1, 3)
    assert si._watertight_triangles(tris) is expected
