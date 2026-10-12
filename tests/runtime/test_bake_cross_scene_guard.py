"""A Production Bake in one scene owns the process-wide Solver Session.

Review finding (a), 2026-10-12: ticking Live, scrubbing an already-live scene
and render-init in *another* scene asked only "is this scene baking?", so
they re-initialized the singleton the bake was stepping. The bake's next tick
then failed with a misleading "solver advanced to frame X, expected Y".
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _load(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        scene=None, evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(handlers=SimpleNamespace(
        frame_change_post=[], render_init=[], persistent=lambda fn: fn))
    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    for name in ("playback", "scene_bridge", "render"):
        sys.modules.pop(f"fluidSim_warp_blender51.blender.{name}", None)
    scene_bridge = importlib.import_module(
        "fluidSim_warp_blender51.blender.scene_bridge")
    playback = importlib.import_module("fluidSim_warp_blender51.blender.playback")
    render = importlib.import_module("fluidSim_warp_blender51.blender.render")
    from fluidSim_warp_blender51.blender import production_bake

    monkeypatch.setattr(production_bake, "_ACTIVE_BAKES", {})
    monkeypatch.setattr(render, "ensure_render_visibility", lambda _scene: None)
    return scene_bridge, playback, render, production_bake


class _Scene:
    def __init__(self, token, *, live=True):
        self._token = token
        self.frame_current = 5
        self.frame_start = 1
        self.frame_end = 250
        self.objects = ()
        self.fluidsim = SimpleNamespace(
            live=live, cache_playback_enabled=False, cache_enabled=False,
            cache_path="//cache", max_frame_catchup=8, status="")

    def as_pointer(self):
        return self._token


class _BakingSession:
    """The singleton while scene A's bake steps it."""

    def __init__(self, owner):
        self.owner = owner
        self.active = True
        self.start_frame = 1
        self.last_frame = 4
        self.calls = []

    def owns_scene(self, scene):
        return scene is self.owner

    def initialize(self, scene, **_kwargs):
        self.calls.append(("initialize", scene))
        self.owner = scene

    def reset(self, scene, **_kwargs):
        self.calls.append(("reset", scene))

    def sync_scene(self, scene, *_args):
        self.calls.append(("sync", scene))

    def set_error(self, scene, message):
        self.calls.append(("error", scene, message))


def _bake_in(production_bake, scene):
    controller = SimpleNamespace(
        status=production_bake.BakeStatus.RUNNING, _reserved_cache_path=None,
        backend=SimpleNamespace(base_dir=None))
    production_bake.register_active_bake(scene, controller)
    return controller


@pytest.fixture
def world(monkeypatch):
    scene_bridge, playback, render, production_bake = _load(monkeypatch)
    baking, other = _Scene(101), _Scene(202)
    _bake_in(production_bake, baking)
    sess = _BakingSession(baking)
    monkeypatch.setattr(scene_bridge, "session", lambda: sess)
    monkeypatch.setattr(
        scene_bridge, "step_frame",
        lambda scene, *_a, **_k: sess.calls.append(("step", scene)))
    import fluidSim_warp_blender51.runtime.warp_runtime as warp_runtime

    monkeypatch.setattr(warp_runtime, "prewarm_simulation_kernels", lambda: None)
    monkeypatch.setattr(warp_runtime, "is_kernels_prewarmed", lambda: True)
    return SimpleNamespace(
        playback=playback, render=render, sess=sess, baking=baking, other=other)


def test_ticking_live_in_another_scene_leaves_the_baking_session(world):
    world.other.fluidsim.live = False

    assert world.playback.set_live_enabled(world.other, True) is False

    assert world.sess.calls == []
    assert world.sess.owns_scene(world.baking)
    assert world.other.fluidsim.live is False
    assert "Production Bake" in world.other.fluidsim.status


def test_scrubbing_another_live_scene_leaves_the_baking_session(world):
    world.other.frame_current = 9

    world.playback.handle_frame_change(world.sess, world.other, object())

    assert world.sess.calls == []
    assert world.sess.owns_scene(world.baking)
    assert "Production Bake" in world.other.fluidsim.status


def test_rendering_another_live_scene_leaves_the_baking_session(world):
    world.render._on_render_init(world.other)

    assert world.sess.calls == []
    assert world.sess.owns_scene(world.baking)
    assert "Production Bake" in world.other.fluidsim.status


def test_without_a_bake_another_scene_still_takes_the_free_session(monkeypatch):
    """Control: the guard is about the bake, not about other scenes."""
    scene_bridge, playback, _render, _production_bake = _load(monkeypatch)
    owner, other = _Scene(101), _Scene(202, live=False)
    sess = _BakingSession(owner)
    monkeypatch.setattr(scene_bridge, "session", lambda: sess)
    import fluidSim_warp_blender51.runtime.warp_runtime as warp_runtime

    monkeypatch.setattr(warp_runtime, "prewarm_simulation_kernels", lambda: None)

    assert playback.set_live_enabled(other, True) is True
    assert sess.calls == [("initialize", other)]


def test_a_second_bake_in_another_scene_is_refused(monkeypatch):
    """Finding (b): the registry was per scene, the session is per process."""
    _scene_bridge, _playback, _render, production_bake = _load(monkeypatch)
    baking, other = _Scene(101), _Scene(202)
    first = _bake_in(production_bake, baking)

    with pytest.raises(RuntimeError, match="already running"):
        _bake_in(production_bake, other)
    assert production_bake.active_bake(other) is None

    first.status = production_bake.BakeStatus.COMPLETE  # a finished bake frees it
    _bake_in(production_bake, other)
    assert production_bake.bake_is_active(other)
