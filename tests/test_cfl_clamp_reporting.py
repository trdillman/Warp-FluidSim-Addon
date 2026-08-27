"""Regression tests for surfacing the CFL substep clamp.

``_compute_substeps`` ends in ``min(max_substeps, n)``. When ``n`` exceeds the
cap the frame integrates at a timestep known to violate the CFL condition, and
the old code threw that fact away -- so the resulting artefact had no
explanation anywhere in the UI. The flag only helps if it survives all four
hops to the panel, and only if it is honest on the frames that did not clamp:
a latched True is as useless as no flag at all.

The two ``_compute_substeps`` return values are pinned next to the existing
substep tests in ``test_pressure_projection_regressions.py``; this file covers
the reporting path, which needs no GPU.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_scene_bridge_stubs(monkeypatch):
    fake_bpy = types.ModuleType("bpy")
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.context = SimpleNamespace(
        scene=None, evaluated_depsgraph_get=lambda: object())
    fake_bpy.app = SimpleNamespace(
        handlers=SimpleNamespace(frame_change_post=[]))

    fake_mathutils = types.ModuleType("mathutils")
    fake_mathutils.Vector = lambda value: value

    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", fake_mathutils)
    sys.modules.pop("fluidSim_warp_blender51.blender.playback", None)
    sys.modules.pop("fluidSim_warp_blender51.blender.scene_bridge", None)
    scene_bridge = importlib.import_module(
        "fluidSim_warp_blender51.blender.scene_bridge")
    scene_bridge.playback = importlib.import_module(
        "fluidSim_warp_blender51.blender.playback")
    return scene_bridge


def _stub_viewport(monkeypatch):
    import fluidSim_warp_blender51.blender as blender_pkg

    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    fake_viewport.update_outputs = lambda scene, sess: None
    fake_viewport.update_outputs_from_cache_frame = lambda scene, data: None
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", fake_viewport)
    monkeypatch.setattr(blender_pkg, "output_publication", fake_viewport, raising=False)


def _scene():
    return SimpleNamespace(
        frame_current=5,
        frame_start=1,
        frame_end=250,
        objects=(),
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=False,
            cache_enabled=False,
            cache_use_custom_range=False,
            cache_path="//cache",
            status="",
            stat_particles=0,
            stat_voxels=0,
            stat_whitewater=0,
            stat_step_ms=0.0,
            stat_substeps=0,
            stat_cfl_clamped=False,
        ),
    )


def _stats(clamped):
    return SimpleNamespace(
        active_particles=7, active_voxels=8, whitewater_particles=3,
        whitewater_required=5, whitewater_admitted=3,
        whitewater_dropped=2, whitewater_policy="CLAMP",
        whitewater_status="DROPPED",
        whitewater_message="Whitewater admitted 3/5; dropped 2",
        step_time=0.002, substeps=4, cfl_substeps_clamped=clamped,
        as_dict=lambda: {})


def test_frame_stats_default_false_and_carry_the_flag_into_the_report():
    """The report dict is the only record a finished run leaves behind."""
    pytest.importorskip("warp")
    from fluidSim_warp_blender51.sim.solver import FrameStats

    stats = FrameStats()
    assert stats.cfl_substeps_clamped is False
    assert stats.as_dict()["cflSubstepsClamped"] is False

    stats.cfl_substeps_clamped = True
    assert stats.as_dict()["cflSubstepsClamped"] is True


def test_the_clamp_flag_reaches_the_panel_and_does_not_latch(monkeypatch):
    """``solver.stats`` outlives the frame, so a stale True is the hazard.

    A flag that latches on the first violent frame stays lit for the rest of
    the session, which is indistinguishable from a broken sim to the user.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    _stub_viewport(monkeypatch)

    reported = [True]
    scene = _scene()
    sess = scene_bridge.Session()
    sess.solver = SimpleNamespace(
        frame=5, sourcing=None,
        step_frame=lambda: _stats(reported[0]))
    sess.scene_identity = scene_bridge._scene_identity(scene)

    sess.step(scene)
    assert scene.fluidsim.stat_cfl_clamped is True

    reported[0] = False
    sess.step(scene)
    assert scene.fluidsim.stat_cfl_clamped is False


def test_whitewater_truth_reaches_live_scene_status(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    _stub_viewport(monkeypatch)

    scene = _scene()
    sess = scene_bridge.Session()
    sess.solver = SimpleNamespace(
        frame=5, sourcing=None, step_frame=lambda: _stats(False))
    sess.scene_identity = scene_bridge._scene_identity(scene)

    sess.step(scene)

    assert scene.fluidsim.stat_whitewater == 3
    assert scene.fluidsim.stat_whitewater_required == 5
    assert scene.fluidsim.stat_whitewater_admitted == 3
    assert scene.fluidsim.stat_whitewater_dropped == 2
    assert scene.fluidsim.stat_whitewater_policy == "CLAMP"
    assert scene.fluidsim.stat_whitewater_status == "DROPPED"
    assert "Whitewater admitted 3/5; dropped 2" in scene.fluidsim.status


def test_cache_playback_clears_the_clamp_flag(monkeypatch):
    """A cached frame was integrated by some earlier run, maybe not this one.

    ``play_cache_frame`` zeroes every other ``stat_*`` for exactly this
    reason; leaving this one set reports a violation against a frame the
    current session never stepped.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    _stub_viewport(monkeypatch)

    import fluidSim_warp_blender51.io as io_pkg

    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.read_frame = lambda base, frame: {"positions": [(0.0, 0.0, 0.0)]}
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.io.cache", fake_cache)
    # `from ..io import cache` binds the package attribute, which an earlier
    # test in the same process may already have populated with the real module.
    monkeypatch.setattr(io_pkg, "cache", fake_cache, raising=False)
    monkeypatch.setattr(
        scene_bridge.playback, "_check_cache_scene_key",
        lambda scene, base, data, playback_read=None: None)

    sess = scene_bridge.Session()
    scene = _scene()
    scene.fluidsim.stat_cfl_clamped = True

    scene_bridge.playback.play_cache_frame(sess, scene, frame=5)

    assert scene.fluidsim.stat_cfl_clamped is False
