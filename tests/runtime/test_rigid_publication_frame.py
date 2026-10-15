"""Live Dynamic collider poses key the solver's frame (4.0 lane R).

A run starts at ``sess.start_frame`` and each step advances one scene frame, so
a loop that steps without moving the timeline (bake loops, demo builders) must
still key each pose on its own frame, never overwrite ``frame_current``.
"""

from __future__ import annotations

from types import SimpleNamespace

from fluidSim_warp_blender51.blender import rigid_publication


def _sess(start, steps):
    return SimpleNamespace(start_frame=start, solver=SimpleNamespace(frame=steps))


def test_live_frame_is_start_plus_solver_steps_not_the_timeline():
    scene = SimpleNamespace(frame_current=1)
    assert rigid_publication.live_frame(scene, _sess(1, 0)) == 1
    assert rigid_publication.live_frame(scene, _sess(1, 7)) == 8
    assert rigid_publication.live_frame(scene, _sess(10, 3)) == 13


def test_live_frame_falls_back_to_the_timeline_without_session_bookkeeping():
    scene = SimpleNamespace(frame_current=42)
    assert rigid_publication.live_frame(scene, SimpleNamespace()) == 42
    assert rigid_publication.live_frame(scene, _sess(None, 3)) == 42
