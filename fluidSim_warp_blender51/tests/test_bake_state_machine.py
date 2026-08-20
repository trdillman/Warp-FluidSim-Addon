from __future__ import annotations

import pytest
from fluidsim_testpkg.blender.production_bake import BakeController, BakeStatus
from fluidsim_testpkg.io.cache_schema import CacheManifestV2, ResumeDecision


class FakeBackend:
    def __init__(self, *, fail_frame: int | None = None):
        self.frames: list[int] = []
        self.events: list[str] = []
        self.fail_frame = fail_frame
        self.base_dir = "/cache"
        self.manifest = CacheManifestV2.new(
            solver_version="1.0.248",
            scene_key="scene",
            settings_signature="settings",
            source_signature="sources",
            frame_start=1,
            frame_end=5,
        )

    def initialize(self, scene, start, end, *, resume=False, restart=False):
        self.events.append(f"initialize:{start}:{end}:{resume}:{restart}")
        return 3 if resume else start

    def step_and_commit(self, scene, frame):
        if frame == self.fail_frame:
            raise RuntimeError("synthetic commit failure")
        self.frames.append(frame)

    def finish(self, scene):
        self.events.append("finish")

    def cancel(self, scene):
        self.events.append("cancel")

    def fail(self, scene, error):
        self.events.append(f"fail:{type(error).__name__}")

    def resume_decision(self, scene):
        return ResumeDecision(True, 3, "RESUME_READY", "resume frame 3")

    def load_manifest(self, scene):
        return self.manifest


def test_controller_bakes_one_frame_per_tick():
    backend = FakeBackend()
    controller = BakeController(backend=backend)
    controller.start(scene=object(), start=1, end=3)
    assert controller.status is BakeStatus.INITIALIZING

    controller.tick(object())
    assert controller.status is BakeStatus.RUNNING
    assert backend.frames == []

    controller.tick(object())
    assert backend.frames == [1]
    assert controller.progress == pytest.approx(1 / 3)

    controller.tick(object())
    assert backend.frames == [1, 2]
    controller.tick(object())
    assert backend.frames == [1, 2, 3]
    controller.tick(object())
    assert controller.status is BakeStatus.COMPLETE
    assert backend.events[-1] == "finish"


def test_cancel_stops_at_last_committed_boundary():
    backend = FakeBackend()
    controller = BakeController(backend=backend)
    controller.start(object(), 1, 10)
    controller.tick(object())
    controller.tick(object())
    assert backend.frames == [1]

    controller.request_cancel()
    assert controller.status is BakeStatus.CANCELLING
    controller.tick(object())

    assert controller.status is BakeStatus.CANCELLED
    assert controller.last_committed_frame == 1
    assert backend.frames == [1]
    assert backend.events[-1] == "cancel"


def test_resume_uses_backend_verified_resume_frame():
    backend = FakeBackend()
    controller = BakeController(backend=backend)
    controller.resume(object(), end=5)
    assert controller.next_frame == 3
    assert controller.last_committed_frame == 2

    controller.tick(object())
    controller.tick(object())
    assert backend.frames == [3]


def test_failure_is_terminal_and_cleanup_runs():
    backend = FakeBackend(fail_frame=2)
    events: list[dict] = []
    controller = BakeController(backend=backend, progress_callback=events.append)
    controller.start(object(), 1, 3)
    controller.tick(object())
    controller.tick(object())
    controller.tick(object())

    assert controller.status is BakeStatus.FAILED
    assert controller.last_committed_frame == 1
    assert "synthetic commit failure" in controller.error
    assert backend.events[-1] == "fail:RuntimeError"
    assert events[-1]["event"] == "bake_failed"


def test_invalid_range_and_overlapping_start_are_rejected():
    controller = BakeController(backend=FakeBackend())
    with pytest.raises(ValueError, match="before start"):
        controller.start(object(), 5, 4)
    controller.start(object(), 1, 2)
    with pytest.raises(RuntimeError, match="already active"):
        controller.start(object(), 1, 2)
