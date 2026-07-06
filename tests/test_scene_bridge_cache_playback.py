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
