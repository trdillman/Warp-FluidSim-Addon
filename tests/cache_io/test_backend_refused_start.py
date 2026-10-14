"""A start refused inside ``ProductionBakeBackend.initialize`` must not leave live off.

``initialize`` switches live playback off before it looks at the cache, so a
refusal (a 2.x cache, existing data, a changed animation) raised from there
used to leave the artist's live toggle off: ``fail`` only restored state once
the backend was fully initialized.
"""

from __future__ import annotations

from types import SimpleNamespace

from fluidSim_warp_blender51 import blender as blender_package
from fluidSim_warp_blender51.blender.production_bake import ProductionBakeBackend


def _scene(live=True):
    return SimpleNamespace(
        frame_current=7,
        fluidsim=SimpleNamespace(live=live),
    )


def _snapshot(monkeypatch, backend, scene):
    # The snapshot imports scene_bridge and cache_identity, which need bpy.
    # Only the session lookup matters here, so stand in for both.
    session = SimpleNamespace(owns_scene=lambda _scene: False)
    monkeypatch.setattr(
        blender_package, "scene_bridge",
        SimpleNamespace(session=lambda: session), raising=False)
    monkeypatch.setattr(
        blender_package, "cache_identity", SimpleNamespace(), raising=False)
    backend._snapshot_live_state(scene)


def test_fail_before_initialization_completes_restores_the_live_flag(monkeypatch):
    backend = ProductionBakeBackend()
    scene = _scene(live=True)
    _snapshot(monkeypatch, backend, scene)
    scene.fluidsim.live = False  # what initialize() does right after the snapshot

    backend.fail(scene, ValueError("refused"))

    assert scene.fluidsim.live is True


def test_fail_restores_a_live_flag_that_was_already_off(monkeypatch):
    backend = ProductionBakeBackend()
    scene = _scene(live=False)
    _snapshot(monkeypatch, backend, scene)

    backend.fail(scene, ValueError("refused"))

    assert scene.fluidsim.live is False


def test_fail_without_a_snapshot_leaves_the_scene_alone():
    backend = ProductionBakeBackend()
    scene = _scene(live=True)

    backend.fail(scene, ValueError("never started"))

    assert scene.fluidsim.live is True


def test_a_second_fail_does_not_overwrite_a_later_live_change(monkeypatch):
    backend = ProductionBakeBackend()
    scene = _scene(live=True)
    _snapshot(monkeypatch, backend, scene)
    scene.fluidsim.live = False
    backend.fail(scene, ValueError("refused"))

    scene.fluidsim.live = False  # the artist turned it off afterwards
    backend.fail(scene, ValueError("again"))

    assert scene.fluidsim.live is False
