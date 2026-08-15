"""Regression tests for the live frame-change catch-up gate and handler teardown.

Blender drops frames during playback, so a forward jump larger than one is
normal. Taking exactly one solver step per frame change ran the simulation in
slow motion against the timeline and punched holes in the cache, with no
symptom the user could see. These tests pin the three facts that fix depends
on: how many steps a frame change is worth, which timeline index each one
lands at, and that a jump too large to serve refuses instead of pretending.
"""

from __future__ import annotations

import importlib
import inspect
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


def _scene(frame, *, catchup=8):
    return SimpleNamespace(
        frame_current=frame,
        frame_start=1,
        frame_end=250,
        objects=(),
        fluidsim=SimpleNamespace(
            live=True,
            cache_playback_enabled=False,
            cache_enabled=False,
            cache_path="//cache",
            max_frame_catchup=catchup,
            status="",
        ),
    )


def _live_session(scene_bridge, *, last_frame):
    """A Session that reports active, with the solver and sync stubbed out.

    ``handle_frame_change`` decides *how many* steps a frame change is worth
    and *which* timeline index each one belongs at. Recording those two facts
    needs no solver, so this stays a host-only test.
    """
    sess = scene_bridge.Session()
    sess.solver = object()  # Session.active is `self.solver is not None`
    sess.start_frame = 1
    sess.last_frame = last_frame
    steps = []
    sess.sync_scene = lambda *args, **kwargs: None
    sess.step = lambda scene, **kwargs: steps.append(kwargs.get("timeline_frame"))
    return sess, steps


def test_dropped_playback_frames_step_the_whole_gap_at_the_right_indices(
        monkeypatch):
    """A four-frame jump is four solver steps, at frames 4 through 7.

    One step per frame change is the silent-wrong-answer failure: the solver
    falls behind the timeline at a fixed ratio and the cache gains holes,
    because ``write_frame`` keys on the timeline index, not the solver's.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    sess, steps = _live_session(scene_bridge, last_frame=3)

    scene_bridge.playback.handle_frame_change(sess, _scene(7), None)

    assert steps == [4, 5, 6, 7]
    assert sess.last_frame == 7


def test_topology_capacity_refusal_pauses_without_releasing_live_solver(monkeypatch):
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.sim.fields import TopologyCapacityError

    scene = _scene(4)
    sess, _steps = _live_session(scene_bridge, last_frame=3)
    old_solver = sess.solver
    releases = []
    sess.release = lambda *args, **kwargs: releases.append((args, kwargs))

    def refuse(_scene, **_kwargs):
        raise TopologyCapacityError(required=9, capacity=8)

    sess.step = refuse
    scene_bridge.playback.handle_frame_change(sess, scene, None)

    assert sess.solver is old_solver
    assert releases == []
    assert sess.last_frame == 3
    assert scene.fluidsim.live is False
    assert "solver preserved" in scene.fluidsim.status.lower()
    assert "need 9" in scene.fluidsim.status.lower()


def test_first_frame_of_a_fresh_session_is_a_single_step(monkeypatch):
    """``last_frame`` starts at -1, which is not a 6-frame backlog.

    Without this guard the gap arithmetic reads the sentinel as a real
    timeline position and the first frame change bursts a full catch-up.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    sess, steps = _live_session(scene_bridge, last_frame=-1)

    scene_bridge.playback.handle_frame_change(sess, _scene(5), None)

    assert steps == [5]


def test_a_jump_past_the_catchup_bound_reports_desync_and_takes_no_step(
        monkeypatch):
    """A long scrub must refuse, not stall Blender inside the handler.

    Stepping 47 frames synchronously from ``frame_change_post`` freezes the
    UI; taking one step instead silently desyncs. The bound picks neither.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    sess, steps = _live_session(scene_bridge, last_frame=3)
    scene = _scene(50)

    scene_bridge.playback.handle_frame_change(sess, scene, None)

    assert steps == []
    assert sess.last_frame == 50
    assert "desync" in scene.fluidsim.status.lower()
    error = sess.current_error(scene)
    assert "47" in error
    assert "Maximum Playback Recovery Frames" in error
    assert "Advanced Playback" in error


def test_the_catchup_bound_is_the_scene_property_not_a_constant(monkeypatch):
    """The bound is user-facing, so the handler must read it per scene."""
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)

    sess, steps = _live_session(scene_bridge, last_frame=3)
    scene_bridge.playback.handle_frame_change(
        sess, _scene(20, catchup=2), None)
    assert steps == []

    sess, steps = _live_session(scene_bridge, last_frame=3)
    scene_bridge.playback.handle_frame_change(
        sess, _scene(20, catchup=32), None)
    assert steps == list(range(4, 21))


def test_step_never_implicitly_writes_legacy_cache(monkeypatch):
    """Live and catch-up stepping never capture authoritative cache data."""
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    import fluidSim_warp_blender51.blender as blender_pkg
    import fluidSim_warp_blender51.io as io_pkg

    written = []
    fake_cache = types.ModuleType("fluidSim_warp_blender51.io.cache")
    fake_cache.write_frame = (
        lambda scene, sess, frame: written.append(int(frame)))
    fake_cache.write_sequence_metadata = lambda *args, **kwargs: None
    fake_viewport = types.ModuleType("fluidSim_warp_blender51.blender.output_publication")
    fake_viewport.update_outputs = lambda scene, sess: None
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.io.cache", fake_cache)
    monkeypatch.setitem(
        sys.modules, "fluidSim_warp_blender51.blender.output_publication", fake_viewport)
    monkeypatch.setattr(io_pkg, "cache", fake_cache, raising=False)
    monkeypatch.setattr(blender_pkg, "output_publication", fake_viewport, raising=False)

    stats = SimpleNamespace(
        active_particles=3, active_voxels=4, whitewater_particles=0,
        step_time=0.001, substeps=1, cfl_substeps_clamped=False,
        as_dict=lambda: {})
    sess = scene_bridge.Session()
    sess.solver = SimpleNamespace(
        frame=9, sourcing=None, step_frame=lambda: stats)
    scene = _scene(9)
    scene.fluidsim.cache_enabled = True

    sess.step(scene, timeline_frame=4)
    sess.step(scene)

    assert written == []


def test_unregister_handlers_releases_the_session(monkeypatch):
    """Reload Scripts reaches teardown here, never through ``unregister()``.

    Without the release the whole GPU allocation is orphaned with no owner
    left to free it short of restarting Blender. Registering, by contrast,
    deduplicates only -- it must never tear down a live session.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    released = []
    monkeypatch.setattr(
        scene_bridge.Session, "release",
        lambda self, *args, **kwargs: released.append(True))
    handlers = sys.modules["bpy"].app.handlers.frame_change_post

    scene_bridge.playback.register_handlers()
    scene_bridge.playback.register_handlers()
    assert len(handlers) == 1
    assert released == []

    scene_bridge.playback.unregister_handlers()
    assert handlers == []
    assert released == [True]


def test_a_nested_frame_change_takes_no_second_pass(monkeypatch):
    """``scene.frame_set`` from inside the handler re-enters it synchronously.

    The bake loop already does exactly that. A nested pass would step the
    solver again against a depsgraph the outer call still owns, so the
    handler refuses re-entry rather than relying on every caller to
    suppress it.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    seen = []

    def _handle(_sess, scn, depsgraph):
        seen.append(scn.frame_current)
        if len(seen) == 1:
            scene_bridge.playback._on_frame_change(_scene(6), depsgraph)

    monkeypatch.setattr(scene_bridge.playback, "handle_frame_change", _handle)

    scene_bridge.playback._on_frame_change(_scene(5), None)

    assert seen == [5]
    # The guard resets, so the next real frame change is served normally.
    scene_bridge.playback._on_frame_change(_scene(7), None)
    assert seen == [5, 7]


def test_the_guard_resets_when_the_handler_raises(monkeypatch):
    """A one-shot failure must not wedge playback for the rest of the session."""
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    calls = []

    def _handle(_sess, scn, depsgraph):
        calls.append(scn.frame_current)
        if len(calls) == 1:
            raise RuntimeError("handler exploded")

    monkeypatch.setattr(scene_bridge.playback, "handle_frame_change", _handle)

    with pytest.raises(RuntimeError):
        scene_bridge.playback._on_frame_change(_scene(5), None)
    scene_bridge.playback._on_frame_change(_scene(6), None)

    assert calls == [5, 6]


def test_a_live_write_from_inside_the_handler_cannot_re_enter_the_session(
        monkeypatch):
    """``fs.live = False`` on the error path runs ``_on_live_changed`` inline.

    Blender fires the RNA update callback synchronously on assignment, so
    the handler's teardown lands back in ``set_live_enabled`` while the
    depsgraph it was handed is still mid-evaluation. Re-entering there can
    allocate a whole GPU solver from inside ``frame_change_post``.
    """
    scene_bridge = _install_scene_bridge_stubs(monkeypatch)
    initialized = []
    monkeypatch.setattr(
        scene_bridge.Session, "initialize",
        lambda self, scn, **kwargs: initialized.append(True))
    scene = _scene(5)

    def _handle(_sess, scn, depsgraph):
        # Stands in for properties._on_live_changed, which Blender runs
        # inline on the fs.live assignment at the end of the error path.
        scene_bridge.playback.set_live_enabled(scn, True)
        scene_bridge.playback.set_live_enabled(scn, False)

    monkeypatch.setattr(scene_bridge.playback, "handle_frame_change", _handle)

    scene_bridge.playback._on_frame_change(scene, None)

    assert initialized == []
    assert scene.fluidsim.status == ""

    # Outside the handler the same call does its normal work.
    scene_bridge.playback.set_live_enabled(scene, False)
    assert scene.fluidsim.status == "Live simulation disabled"


def test_vram_snapshot_costs_no_sleep_and_no_subprocess(monkeypatch):
    """The capacity check sits on the frame-change path; telemetry must not.

    ``resource_snapshot`` blocks 50 ms and launches nvidia-smi with a 2 s
    timeout. ``initialize`` is reachable from ``frame_change_post``, and
    ``assess_configuration`` reads only the two VRAM fields, so the handler
    path pays for neither.
    """
    fake_warp = types.ModuleType("warp")
    fake_warp.get_device = lambda name: SimpleNamespace(
        total_memory=16 * 1024**3, free_memory=8 * 1024**3, is_cuda=True,
        ordinal=0)
    monkeypatch.setitem(sys.modules, "warp", fake_warp)

    from fluidSim_warp_blender51.sim import memory_safety

    def _forbidden(*args, **kwargs):
        raise AssertionError("frame-change path must not block")

    monkeypatch.setattr(memory_safety.subprocess, "run", _forbidden)
    monkeypatch.setattr(memory_safety.time, "sleep", _forbidden)

    snapshot = memory_safety.vram_snapshot("cuda:0")

    assert snapshot["gpu_total_bytes"] == 16 * 1024**3
    assert snapshot["gpu_free_bytes"] == 8 * 1024**3
    with pytest.raises(AssertionError):
        memory_safety.resource_snapshot("cuda:0")


def test_initialize_reads_the_cheap_snapshot(monkeypatch):
    """Pin the caller, not just the callee.

    A cheap sampler nothing calls is decoration. Running the real
    ``initialize`` needs a GPU, so this reads the source instead -- the
    smallest check that fails if the blocking call comes back.
    """
    _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import simulation_admission

    source = inspect.getsource(simulation_admission.SimulationAdmission._preflight)
    # Comments stripped: the comment there explains why the blocking sampler
    # is not used, and naming it is not calling it.
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))

    assert "vram_snapshot" in code
    assert "resource_snapshot" not in code
