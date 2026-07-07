"""Regression tests for cache playback frame-change routing."""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

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
