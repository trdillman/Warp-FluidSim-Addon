"""Regression tests for cache playback frame-change routing."""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[])
    )

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    return importlib.import_module("fluidSim_warp_blender51.blender.scene_bridge")


def _scene(frame=7, *, playback=True):
    return SimpleNamespace(
        frame_current=frame,
        frame_start=1,
        frame_end=10,
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=playback,
            cache_use_custom_range=False,
            cache_frame_start=1,
            cache_frame_end=10,
            cache_path="//cache",
            status="",
            stat_particles=0,
            stat_voxels=0,
            stat_whitewater=0,
            stat_step_ms=0.0,
            stat_substeps=0,
        ),
    )


def _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport):
    import fluidSim_warp_blender51.blender as blender_pkg
    import fluidSim_warp_blender51.io as io_pkg

    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.viewport", fake_viewport)
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.io.cache", fake_cache)
    monkeypatch.setattr(blender_pkg, "viewport", fake_viewport, raising=False)
    monkeypatch.setattr(io_pkg, "cache", fake_cache, raising=False)


def test_nearest_values_native_and_fallback_agree(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    viewport = importlib.import_module("fluidSim_warp_blender51.blender.viewport")
    points = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=np.float32)
    samples = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    values = np.array([2.0, 7.0], dtype=np.float32)

    class FakeKDTree:
        def __init__(self, _size):
            self.items = []

        def insert(self, point, index):
            self.items.append((np.asarray(point), index))

        def balance(self):
            pass

        def find(self, point):
            point = np.asarray(point)
            sample, index = min(
                self.items, key=lambda item: np.sum((item[0] - point) ** 2))
            return sample, index, float(np.linalg.norm(sample - point))

    monkeypatch.setattr(viewport, "_KDTree", FakeKDTree)
    native = viewport._nearest_values(points, samples, values, 0.5)
    monkeypatch.setattr(viewport, "_KDTree", None)
    fallback = viewport._nearest_values(points, samples, values, 0.5)

    np.testing.assert_array_equal(native, [2.0, 7.0])
    np.testing.assert_array_equal(fallback, native)


def test_frame_change_cache_playback_uses_cached_snapshot_without_solver_step(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    calls = []

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.viewport")
    fake_viewport.update_outputs_from_cache_frame = (
        lambda scene_arg, data: calls.append((scene_arg.frame_current, int(data["frame"])))
    )
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.read_frame = lambda base, frame: {
        "frame": np.array(frame, dtype=np.int32),
        "positions": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.zeros((1, 3), dtype=np.float32),
    }
    fake_cache.resolve_frame_range = lambda *args, **kwargs: (1, 10)
    _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport)

    solver = SimpleNamespace(step_frame=lambda: (_ for _ in ()).throw(
        AssertionError("live solver should not step during cache playback")))
    scene_bridge.session().solver = solver
    scene_bridge.session().last_frame = -1

    scene_bridge._on_frame_change(scene, depsgraph=object())

    assert calls == [(7, 7)]
    assert scene.fluidsim.stat_particles == 2
    assert scene.fluidsim.stat_whitewater == 1
    assert scene.fluidsim.status == "Cache frame 7: 2 particles"


def test_live_property_cache_playback_does_not_initialize_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=True)

    monkeypatch.setattr(scene_bridge, "initialize_session", lambda _scene: (
        (_ for _ in ()).throw(AssertionError("cache playback must not initialize solver"))
    ))

    scene_bridge.release_session()
    scene_bridge.set_live_enabled(scene, True)

    assert not scene_bridge.session().active
    assert scene.fluidsim.status == "Cache playback enabled - press play"


def test_enabling_cache_playback_releases_live_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(playback=False)
    scene.fluidsim.live = True
    scene_bridge.session().solver = object()

    scene.fluidsim.cache_playback_enabled = True
    scene_bridge.set_cache_playback_enabled(scene, True)

    assert not scene_bridge.session().active
    assert scene.fluidsim.status == "Cache playback enabled - press play"


def test_frame_change_cache_playback_reports_missing_frame_without_fallback(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.viewport")
    fake_viewport.update_outputs_from_cache_frame = (
        lambda *_args: (_ for _ in ()).throw(AssertionError("no snapshot to draw"))
    )
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.read_frame = lambda base, frame: None
    fake_cache.resolve_frame_range = lambda *args, **kwargs: (1, 10)
    _install_cache_playback_modules(monkeypatch, fake_cache, fake_viewport)

    solver = SimpleNamespace(step_frame=lambda: (_ for _ in ()).throw(
        AssertionError("live solver fallback is not allowed")))
    scene_bridge.session().solver = solver
    scene_bridge.session().last_frame = -1

    scene_bridge._on_frame_change(scene, depsgraph=object())

    assert scene.fluidsim.status.startswith("Cache playback error:")
    assert "missing or corrupt" in scene_bridge.current_error(scene)


def test_cache_frame_output_matches_live_display_toggles_and_scales(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    import bpy

    bpy.data = SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: None),
        meshes=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        pointclouds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
    )
    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    fake_materials.ensure_surface_material = lambda *_args: None
    fake_materials.ensure_particle_material = lambda *_args: None
    fake_materials.ensure_whitewater_material = lambda *_args: None
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.blender.materials", fake_materials)
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    viewport = importlib.import_module("fluidSim_warp_blender51.blender.viewport")

    calls = []
    monkeypatch.setattr(viewport, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(
        viewport,
        "_ensure_object",
        lambda _scene, name, _factory: SimpleNamespace(name=name, data=SimpleNamespace()),
    )
    monkeypatch.setattr(
        viewport,
        "_swap_mesh",
        lambda obj, *_args, **_kwargs: calls.append(("mesh", obj.name)),
    )
    monkeypatch.setattr(
        viewport,
        "_swap_points",
        lambda obj, _pos, radius, _colors=None: calls.append(
            ("points", obj.name, radius)
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_clear_points",
        lambda name: calls.append(("clear_points", name)),
    )

    scene = SimpleNamespace(
        fluidsim=SimpleNamespace(
            show_surface=True,
            surface_enabled=False,
            show_particles=True,
            particle_display_fraction=1.0,
            particle_display_scale=2.0,
            particle_attribute_mode="NONE",
            surface_attribute_mode="NONE",
            show_whitewater=True,
            whitewater_enabled=False,
            whitewater_display_scale=3.0,
            whitewater_attribute_mode="NONE",
            voxel_size=0.2,
        )
    )
    frame_data = {
        "voxel_size": np.array(0.2, dtype=np.float32),
        "positions": np.ones((2, 3), dtype=np.float32),
        "velocities": np.zeros((2, 3), dtype=np.float32),
        "whitewater": np.ones((1, 3), dtype=np.float32),
        "surface_verts": np.ones((3, 3), dtype=np.float32),
        "surface_tris": np.array([[0, 1, 2]], dtype=np.int32),
    }

    viewport.update_outputs_from_cache_frame(scene, frame_data)

    assert ("mesh", viewport.SURFACE_NAME) not in calls
    assert ("clear_points", viewport.WHITEWATER_NAME) in calls
    assert any(
        call[0] == "points"
        and call[1] == viewport.PARTICLES_NAME
        and abs(call[2] - 0.1) < 1e-6
        for call in calls
    )
    assert all(call[:2] != ("points", viewport.WHITEWATER_NAME) for call in calls)


def test_live_and_cached_outputs_share_rendering_policy(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    import bpy

    bpy.data = SimpleNamespace(
        objects=SimpleNamespace(get=lambda _name: None),
        meshes=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
        pointclouds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
    )
    fake_materials = types.ModuleType("fluidSim_warp_blender51.blender.materials")
    calls = []
    fake_materials.ensure_surface_material = (
        lambda _scene, obj: calls.append(("material", "surface", obj.name)))
    fake_materials.ensure_particle_material = (
        lambda _scene, obj: calls.append(("material", "particles", obj.name)))
    fake_materials.ensure_whitewater_material = (
        lambda _scene, obj: calls.append(("material", "whitewater", obj.name)))
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.materials", fake_materials
    )
    sys.modules.pop("fluidSim_warp_blender51.blender.viewport", None)
    viewport = importlib.import_module("fluidSim_warp_blender51.blender.viewport")

    def values(array):
        if array is None:
            return None
        return np.asarray(array).round(6).tolist()

    monkeypatch.setattr(viewport, "_supports_pointcloud", lambda: True)
    monkeypatch.setattr(
        viewport,
        "_ensure_object",
        lambda _scene, name, _factory: SimpleNamespace(name=name, data=SimpleNamespace()),
    )
    monkeypatch.setattr(
        viewport,
        "_swap_mesh",
        lambda obj, verts, tris, colors=None: calls.append(
            ("mesh", obj.name, values(verts), values(tris), values(colors))
        ),
    )
    monkeypatch.setattr(
        viewport,
        "_swap_points",
        lambda obj, positions, radius, colors=None: calls.append(
            ("points", obj.name, values(positions), round(radius, 6), values(colors))
        ),
    )
    monkeypatch.setattr(
        viewport, "_clear_points", lambda name: calls.append(("clear", name))
    )

    fs = SimpleNamespace(
        show_surface=True,
        surface_enabled=False,
        surface_attribute_mode="NONE",
        show_particles=True,
        particle_display_fraction=1.0,
        particle_display_scale=1.0,
        particle_attribute_mode="NONE",
        show_whitewater=False,
        whitewater_enabled=False,
        whitewater_display_scale=1.0,
        whitewater_attribute_mode="NONE",
        voxel_size=0.2,
    )
    scene = SimpleNamespace(fluidsim=fs)
    positions = np.ones((2, 3), dtype=np.float32)
    velocities = np.zeros((2, 3), dtype=np.float32)
    verts = np.ones((3, 3), dtype=np.float32)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    solver = SimpleNamespace(
        settings=SimpleNamespace(
            surface_enabled=True,
            whitewater_enabled=False,
            voxel_size=0.2,
        ),
        surface_mesh=lambda: (verts, tris),
        particle_positions=lambda: positions,
        particle_velocities=lambda: velocities,
        whitewater_positions=lambda: np.zeros((0, 3), dtype=np.float32),
    )
    session = SimpleNamespace(active=True, solver=solver)
    frame_data = {
        "voxel_size": np.array(0.2, dtype=np.float32),
        "positions": positions,
        "velocities": velocities,
        "surface_verts": verts,
        "surface_tris": tris,
    }

    viewport.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    viewport.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls

    calls.clear()
    fs.surface_enabled = True
    fs.surface_attribute_mode = "HEIGHT"
    fs.particle_display_fraction = 0.5
    fs.particle_attribute_mode = "HEIGHT"
    fs.show_whitewater = True
    fs.whitewater_enabled = True
    fs.whitewater_attribute_mode = "HEIGHT"
    solver.settings.surface_enabled = True
    solver.settings.whitewater_enabled = True
    solver.whitewater_positions = lambda: np.array(
        ((0.0, 0.0, 0.5), (0.0, 0.0, 1.5)), dtype=np.float32)
    frame_data.pop("velocities")
    frame_data["whitewater"] = solver.whitewater_positions()

    viewport.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    viewport.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls

    calls.clear()
    solver.whitewater_positions = lambda: np.zeros((0, 3), dtype=np.float32)
    frame_data["whitewater"] = solver.whitewater_positions()

    viewport.update_outputs(scene, session)
    live_calls = list(calls)
    calls.clear()
    viewport.update_outputs_from_cache_frame(scene, frame_data)

    assert live_calls == calls
    assert any(
        call[0] == "points" and call[1] == viewport.WHITEWATER_NAME
        for call in calls
    )


def test_release_session_clears_all_runtime_state_and_error(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    sess = scene_bridge.session()
    sess.solver = object()
    sess.profiler = object()
    sess.start_frame = 9
    sess.last_frame = 12
    sess.frame_stats = [{"frame": 12}]
    sess.scene_inputs._collider_prev = {"Collider": (object(), 12)}
    sess.scene_inputs._collider_rotation_prev = {"Collider": (object(), 12)}
    sess.scene_inputs._collider_mesh_cache = {"Collider": object()}
    sess.scene_inputs.collider_mesh_cache_stats = {"rebuilds": 3, "reuses": 4}
    scene_bridge.set_error(scene, "boom")

    scene_bridge.release_session(scene)
    scene_bridge.release_session(scene)

    assert not sess.active
    assert sess.profiler is None
    assert sess.start_frame == 1
    assert sess.last_frame == -1
    assert sess.frame_stats == []
    assert sess.scene_inputs._collider_prev == {}
    assert sess.scene_inputs._collider_rotation_prev == {}
    assert sess.scene_inputs._collider_mesh_cache == {}
    assert sess.collider_mesh_cache_stats == {"rebuilds": 0, "reuses": 0}
    assert scene_bridge.current_error(scene) == ""


def test_initialize_failure_releases_partial_session(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.frame_start = 1
    scene.fluidsim.profiling_enabled = False
    scene.fluidsim.snippets = []
    fake_solver_module = types.ModuleType("fluidSim_warp_blender51.sim.solver")
    fake_solver_module.FluidSolver = lambda *_args, **_kwargs: object()
    fake_profiler_module = types.ModuleType(
        "fluidSim_warp_blender51.diagnostics.profiler")
    fake_profiler_module.Profiler = object
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.sim.solver", fake_solver_module)
    monkeypatch.setitem(
        sys.modules,
        "fluidSim_warp_blender51.diagnostics.profiler",
        fake_profiler_module,
    )
    monkeypatch.setattr(
        scene_bridge, "build_settings", lambda _scene: SimpleNamespace(device="cuda:0"))
    sess = scene_bridge.session()
    monkeypatch.setattr(
        sess,
        "sync_scene",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        sess.initialize(scene)

    assert not sess.active
    assert sess.profiler is None
    assert sess.frame_stats == []
    assert sess.scene_inputs._collider_prev == {}
    assert sess.scene_inputs._collider_mesh_cache == {}


def test_session_step_retains_only_latest_thousand_frame_stats(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene()
    scene.fluidsim.cache_enabled = False
    sess = scene_bridge.session()
    stats = SimpleNamespace(
        active_particles=3,
        active_voxels=4,
        whitewater_particles=5,
        step_time=0.01,
        substeps=2,
        as_dict=lambda: {"frame": 1001},
    )
    sess.solver = SimpleNamespace(frame=1001, step_frame=lambda: stats)
    sess.frame_stats = [{"frame": index} for index in range(1, 1001)]
    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.viewport")
    fake_viewport.update_outputs = lambda *_args: None
    import fluidSim_warp_blender51.blender as blender_pkg

    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.viewport", fake_viewport)
    monkeypatch.setattr(blender_pkg, "viewport", fake_viewport, raising=False)

    scene_bridge.step_frame(scene, sess)

    assert len(sess.frame_stats) == 1000
    assert sess.frame_stats[0] == {"frame": 2}
    assert sess.frame_stats[-1] == {"frame": 1001}


def test_frame_change_rewind_delegates_to_session_reset(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=3, playback=False)
    sess = scene_bridge.session()
    sess.solver = object()
    sess.start_frame = 1
    sess.last_frame = 8
    calls = []
    monkeypatch.setattr(sess, "reset", lambda scene_arg: calls.append(scene_arg))

    scene_bridge._on_frame_change(scene, depsgraph=object())

    assert calls == [scene]
    assert sess.last_frame == 3


def test_frame_change_reinitializes_saved_live_scene_without_step_button(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=2, playback=False)
    sess = scene_bridge.session()
    calls = []

    def initialize(scene_arg):
        calls.append(("initialize", scene_arg.frame_current))
        sess.solver = SimpleNamespace(frame=0)
        sess.start_frame = scene_arg.frame_start
        return sess

    monkeypatch.setattr(sess, "initialize", initialize)
    monkeypatch.setattr(
        sess, "sync_scene", lambda *_args: calls.append(("sync", scene.frame_current)))
    monkeypatch.setattr(
        sess, "step", lambda *_args: calls.append(("step", scene.frame_current)))

    scene_bridge._on_frame_change(scene, depsgraph=object())

    assert calls == [("initialize", 2), ("sync", 2), ("step", 2)]


def test_cache_export_restores_frame_and_live_after_failure(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    scene = _scene(frame=7, playback=False)
    scene.fluidsim.live = True
    frames = []

    def frame_set(frame):
        frames.append(frame)
        scene.frame_current = frame

    scene.frame_set = frame_set
    sess = scene_bridge.session()
    sess.solver = SimpleNamespace(frame=0)
    monkeypatch.setattr(sess, "initialize", lambda _scene: sess)
    monkeypatch.setattr(sess, "sync_scene", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sess,
        "step",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("step failed")),
    )
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.write_frame = lambda *_args: None
    fake_cache.write_sequence_metadata = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "fluidSim_warp_blender51.io.cache", fake_cache)

    with pytest.raises(RuntimeError, match="step failed"):
        sess.export_cache_sequence(scene, 2, 4)

    assert frames == [2, 7]
    assert scene.frame_current == 7
    assert scene.fluidsim.live is True
