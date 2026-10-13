"""Resume refusal and "Re-bake from frame N" through the bake lifecycle.

The backend decides whether a cache can continue (its own tests need Blender);
these pin what the controller and ``BakeRun`` do with that decision: a refusal
raises with the decision attached, a changed animation leaves a pointer the
Bake stage turns into a "Re-bake from frame N" button, and the re-bake frame
reaches the backend's ``initialize``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import cache_bake, production_bake
from fluidSim_warp_blender51.blender.cache_bake import BakeRun
from fluidSim_warp_blender51.blender.production_bake import (
    BakeController,
    BakeStatus,
    ResumeRefused,
)
from fluidSim_warp_blender51.io.cache_schema import (
    CacheManifestV3,
    ResumeDecision,
)


@pytest.fixture(autouse=True)
def _clean_registries():
    production_bake._ACTIVE_BAKES.clear()
    cache_bake._REBAKE_OFFERS.clear()
    yield
    production_bake._ACTIVE_BAKES.clear()
    cache_bake._REBAKE_OFFERS.clear()


def _manifest():
    return CacheManifestV3.new(
        solver_version="3.0.0",
        scene_key="scene",
        settings_signature="settings",
        source_signature="sources",
        frame_start=1,
        frame_end=10,
        source_schema_version=1,
        transfer_mode="FLIP",
        grid_spec={
            "resolution_divisions": 64, "voxel_size": 0.0625,
            "domain_size": [4.0, 4.0, 4.0], "leaf_dim": 8},
        narrow_band={"enabled": False, "band_voxels": 0.0},
        platform={
            "os": "Windows-11", "gpu_name": "NVIDIA RTX",
            "driver": "560.94", "warp": "1.17.0"},
    )


class _Backend:
    def __init__(self, cache_dir: Path, decision: ResumeDecision):
        self.cache_dir = cache_dir
        self.base_dir = cache_dir
        self.decision = decision
        self.manifest = _manifest()
        self.decision_calls: list[dict] = []
        self.initialize_calls: list[dict] = []

    def cache_path(self, _scene):
        return self.cache_dir

    def load_manifest(self, _scene):
        return self.manifest

    def resume_decision(self, _scene, *, rebake_from=None):
        self.decision_calls.append({"rebake_from": rebake_from})
        return self.decision

    def initialize(self, _scene, start, end, **kwargs):
        self.initialize_calls.append({"start": start, "end": end, **kwargs})

    def step_and_commit(self, _scene, frame):
        pass

    def finish(self, _scene):
        pass

    def cancel(self, _scene):
        pass

    def fail(self, _scene, _error):
        pass


class _PlainBackend(_Backend):
    """A backend that predates re-bake: no ``rebake_from`` anywhere."""

    def resume_decision(self, _scene):
        self.decision_calls.append({})
        return self.decision

    def initialize(self, _scene, start, end, *, resume=False, restart=False):
        self.initialize_calls.append(
            {"start": start, "end": end, "resume": resume, "restart": restart})


def _scene(token=11):
    return SimpleNamespace(
        as_pointer=lambda: token,
        fluidsim=SimpleNamespace(
            bake_start=1, bake_end=10, checkpoint_interval=1, cache_compress=False),
    )


_CHANGED = ResumeDecision(
    False, None, "SOURCE_ANIMATION_CHANGED",
    "The animation of a source or collider changed.",
    first_changed_frame=17,
)
_READY = ResumeDecision(True, 4, "RESUME_READY", "Resume from verified checkpoint 3")
_REBAKE_READY = ResumeDecision(True, 4, "REBAKE_READY", "Keep frames 1 to 3")


def test_a_refused_resume_raises_with_the_decision_attached(tmp_path):
    controller = BakeController(backend=_Backend(tmp_path, _CHANGED))

    with pytest.raises(ResumeRefused) as caught:
        controller.resume(_scene(), end=10)

    assert caught.value.decision is _CHANGED
    assert caught.value.decision.first_changed_frame == 17
    assert str(caught.value).startswith("SOURCE_ANIMATION_CHANGED: ")
    assert isinstance(caught.value, ValueError)
    assert controller.status is BakeStatus.IDLE


def test_a_rebake_asks_the_backend_and_hands_the_frame_to_initialize(tmp_path):
    backend = _Backend(tmp_path, _REBAKE_READY)
    controller = BakeController(backend=backend)

    controller.resume(_scene(), end=10, rebake_from=4)
    assert backend.decision_calls == [{"rebake_from": 4}]
    assert controller.next_frame == 4

    controller.tick(_scene())
    assert backend.initialize_calls == [
        {"start": 1, "end": 10, "resume": True, "restart": False, "rebake_from": 4}]


def test_a_plain_resume_never_passes_rebake_from(tmp_path):
    backend = _PlainBackend(tmp_path, _READY)
    controller = BakeController(backend=backend)

    controller.resume(_scene(), end=10)
    controller.tick(_scene())

    assert backend.decision_calls == [{}]
    assert backend.initialize_calls == [
        {"start": 1, "end": 10, "resume": True, "restart": False}]


def test_a_rebake_does_not_leak_into_the_next_plain_start(tmp_path):
    backend = _Backend(tmp_path, _REBAKE_READY)
    controller = BakeController(backend=backend)
    controller.resume(_scene(), end=10, rebake_from=4)
    controller.status = BakeStatus.CANCELLED

    controller.start(_scene(), 1, 10)
    controller.tick(_scene())

    assert "rebake_from" not in backend.initialize_calls[-1]


def test_a_refused_resume_leaves_a_rebake_offer_for_the_bake_stage(tmp_path):
    scene = _scene()
    with pytest.raises(ResumeRefused):
        BakeRun(scene, resume=True, backend=_Backend(tmp_path, _CHANGED))

    assert cache_bake.rebake_offer(scene) == 17
    assert production_bake.active_bake(scene) is None


def test_other_refusals_clear_the_offer(tmp_path):
    scene = _scene()
    with pytest.raises(ResumeRefused):
        BakeRun(scene, resume=True, backend=_Backend(tmp_path, _CHANGED))
    other = ResumeDecision(False, None, "CACHE_INTEGRITY_FAILURE", "missing file")

    with pytest.raises(ResumeRefused):
        BakeRun(scene, resume=True, backend=_Backend(tmp_path, other))

    assert cache_bake.rebake_offer(scene) is None


def test_a_bake_that_starts_clears_the_offer(tmp_path):
    scene = _scene()
    with pytest.raises(ResumeRefused):
        BakeRun(scene, resume=True, backend=_Backend(tmp_path, _CHANGED))

    run = BakeRun(scene, resume=True, rebake_from=17,
                  backend=_Backend(tmp_path, _REBAKE_READY))
    try:
        assert cache_bake.rebake_offer(scene) is None
    finally:
        run.close()


def test_offers_are_per_scene(tmp_path):
    with pytest.raises(ResumeRefused):
        BakeRun(_scene(1), resume=True, backend=_Backend(tmp_path, _CHANGED))

    assert cache_bake.rebake_offer(_scene(1)) == 17
    assert cache_bake.rebake_offer(_scene(2)) is None
    cache_bake.clear_rebake_offer(_scene(1))
    assert cache_bake.rebake_offer(_scene(1)) is None


def test_the_production_bake_delegate_forwards_rebake_from(monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return "done"

    monkeypatch.setattr(cache_bake, "run_synchronous", fake)

    result = production_bake.run_synchronous(
        "scene", end=9, resume=True, rebake_from=4)

    assert result == "done"
    assert seen["rebake_from"] == 4 and seen["resume"] is True


def test_a_rebake_needs_a_cache_to_continue(tmp_path):
    with pytest.raises(ValueError, match="resume=True"):
        BakeRun(_scene(), rebake_from=4, backend=_Backend(tmp_path, _REBAKE_READY))
