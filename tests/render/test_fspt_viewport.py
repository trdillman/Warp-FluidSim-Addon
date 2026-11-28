"""Bounded Blender-independent viewport capture, submission and retirement."""
from dataclasses import replace
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt.contracts import (
    CameraSnapshot,
    Material,
    RenderFrame,
    SceneSnapshot,
    frame_key,
)
from fluidSim_warp_blender51.fspt.scheduler import DeviceArbiter


class PendingBackend:
    def __init__(self, device, arbiter):
        self.device, self.arbiter, self.owner = device, arbiter, object()
        self.pending = self.closed = self.ready = False
        self.samples = 0
        self.key = None
        self.lease = self.submission = None
        self.calls = []

    def submit(self, scene, camera, width, height, **options):
        self.calls.append(("submit", scene.frame))
        lease = self.arbiter.try_acquire(self.owner, final=options.get("final", False))
        if lease is None:
            return False
        self.lease, self.pending = lease, True
        key = frame_key(scene, camera, width, height, options["max_bounces"], options["seed"])
        if key != self.key:
            self.key, self.samples = key, 0
        self.submission = scene, key, options["samples"]
        return True

    def collect(self, *, wait=False):
        self.calls.append(("collect", wait))
        if not self.pending or not (wait or self.ready):
            return None
        scene, key, count = self.submission
        self.samples += count
        self.pending = self.ready = False
        self.arbiter.release(self.lease)
        self.lease = None
        return RenderFrame(np.ones((key[3], key[2], 4)), key, self.samples,
                           str(self.device), "test", "test", scene.frame, scene.publications)

    def close(self):
        self.calls.append(("close",))
        self.arbiter.cancel(self.owner)
        if self.pending:
            self.collect(wait=True)
        self.closed = True


@pytest.fixture
def viewport():
    from fluidSim_warp_blender51.fspt.viewport import (
        ViewportCoordinator,
        ViewportState,
        ViewRequest,
    )

    arbiter, backends = DeviceArbiter(), []
    def factory(device):
        backend = PendingBackend(device, arbiter)
        backends.append(backend)
        return backend
    coordinator = ViewportCoordinator()
    view = ViewportState(coordinator, factory)
    request = ViewRequest(CameraSnapshot(np.eye(4)), 4, 3, 2, 6, 31, "cuda:0")
    scene = SceneSnapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0],
                          (Material(),), (0.1, 0.2, 0.3), frame=1)
    view.request_update(request)
    yield SimpleNamespace(view=view, scene=scene, request=request, arbiter=arbiter,
                          backends=backends, factory=factory, coordinator=coordinator,
                          ViewportState=ViewportState)
    for state in tuple(coordinator.views):
        state.close()


def test_busy_device_refuses_capture_before_extraction(viewport):
    v = viewport
    held = v.arbiter.try_acquire(object())
    def forbidden():
        raise AssertionError("geometry extraction ran before admission")
    assert not v.view.capture(forbidden)
    assert v.view.snapshot is None
    v.arbiter.release(held)
    assert v.view.capture(lambda: v.scene)
    assert v.view.snapshot is v.scene


def test_pending_job_retains_owned_snapshot_and_only_newest_request(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    assert v.view.tick(lambda _: True) == "submitted"
    backend = v.backends[0]
    v.view.invalidate()
    latest = replace(v.request, width=6)
    v.view.request_update(latest)
    def forbidden():
        raise AssertionError("pending view captured another geometry allocation")
    assert not v.view.capture(forbidden)
    assert v.view.request is latest and v.view.snapshot is v.scene
    backend.ready = True
    assert v.view.tick(lambda _: True) == "capture"
    assert v.view.presentable is None
    assert not backend.pending


def test_collect_never_waits_and_refines_to_goal_without_extra_work(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    assert v.view.tick(lambda _: True) == "submitted"
    backend = v.backends[0]
    assert v.view.tick(lambda _: True) == "pending"
    backend.ready = True
    assert v.view.tick(lambda _: True) == "submitted"
    assert v.view.presentable.samples == 1
    backend.ready = True
    assert v.view.tick(lambda _: True) == "idle"
    assert v.view.presentable.samples == 2 and not v.view.needs_tick
    before = list(backend.calls)
    assert v.view.tick(lambda _: True) == "idle"
    assert backend.calls == before
    assert ("collect", True) not in backend.calls


def test_old_source_completion_is_discarded_before_presentation(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    v.backends[0].ready = True
    assert v.view.tick(lambda _: False) == "capture"
    assert v.view.presentable is None and v.view.dirty


def test_camera_and_sample_goal_changes_reuse_captured_geometry(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    v.backends[0].ready = True
    v.view.tick(lambda _: True)
    v.backends[0].ready = True
    v.view.tick(lambda _: True)
    v.view.request_update(replace(v.request, target_samples=3))
    assert not v.view.dirty and v.view.presentable.samples == 2
    assert v.view.tick(lambda _: True) == "submitted"
    v.backends[0].ready = True
    v.view.tick(lambda _: True)
    matrix = np.eye(4); matrix[0, 3] = 0.2
    v.view.request_update(replace(v.request, camera=CameraSnapshot(matrix)))
    assert not v.view.dirty and v.view.presentable is None
    assert v.view.snapshot is v.scene
    assert v.view.tick(lambda _: True) == "submitted"


def test_capture_invalidation_during_extraction_drops_candidate(viewport):
    v = viewport
    def capture():
        v.view.invalidate()
        return v.scene
    assert not v.view.capture(capture)
    assert v.view.snapshot is None and v.view.dirty


def test_final_pauses_same_device_and_retires_only_owned_pending_jobs(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    with v.coordinator.final("cuda:0"):
        assert not v.backends[0].pending
        assert v.view.tick(lambda _: True) == "suspended"
        assert not v.view.capture(lambda: (_ for _ in ()).throw(AssertionError("captured during final")))
    assert v.view.needs_tick
    assert v.view.tick(lambda _: True) == "submitted"


def test_scene_rebind_removes_old_image_even_when_frame_number_repeats(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    v.backends[0].ready = True
    v.view.tick(lambda _: True)
    assert v.view.presentable is not None
    v.view.rebind()
    assert v.view.presentable is None and v.view.dirty


def test_capture_failure_stops_polling_until_a_real_input_change(viewport):
    v = viewport
    def fail():
        raise ValueError("unsupported material")
    with pytest.raises(ValueError, match="unsupported"):
        v.view.capture(fail)
    assert not v.view.needs_tick
    assert "unsupported material" in v.view.error
    v.view.invalidate()
    assert v.view.capture(lambda: v.scene)
    assert not v.view.error


def test_close_cancels_queued_demand_and_never_resubmits(viewport):
    v = viewport
    holder = v.arbiter.try_acquire(object())
    assert not v.view.capture(lambda: v.scene)
    v.view.close()
    v.arbiter.release(holder)
    next_owner = object()
    grant = v.arbiter.try_acquire(next_owner)
    assert grant is not None
    v.arbiter.release(grant)
    assert not v.view.needs_tick and v.view.tick(lambda _: True) == "closed"


def test_device_edit_defers_teardown_until_safe_capture_boundary(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    old = v.backends[0]
    v.view.request_update(replace(v.request, device="cuda:1"))
    assert old.pending and not old.closed
    assert v.view.tick(lambda _: True) == "capture"
    assert old.closed
    assert v.view.capture(lambda: v.scene)
    assert str(v.backends[-1].device) == "cuda:1"


def test_failed_close_keeps_backend_for_a_checked_retirement_retry(viewport, monkeypatch):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    backend = v.backends[0]
    close = backend.close
    def failed():
        raise RuntimeError("retirement failed")
    monkeypatch.setattr(backend, "close", failed)
    with pytest.raises(RuntimeError, match="retirement"):
        v.view.close()
    assert v.view.closed and v.view.backend is backend and backend.pending
    monkeypatch.setattr(backend, "close", close)
    v.view.close()
    assert backend.closed and v.view.backend is None


def test_lowering_sample_goal_withdraws_dormant_submission_intent(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    assert v.view.tick(lambda _: True) == "submitted"
    other_owner = object()
    assert v.arbiter.try_acquire(other_owner) is None
    v.backends[0].ready = True
    assert v.view.tick(lambda _: True) == "admission"
    other_lease = v.arbiter.try_acquire(other_owner)
    assert other_lease is not None
    v.view.request_update(replace(v.request, target_samples=1))
    assert v.view.at_goal and not v.view.needs_tick
    v.arbiter.release(other_lease)
    third_lease = v.arbiter.try_acquire(object())
    assert third_lease is not None, "idle viewport left an unserviceable FIFO head"
    v.arbiter.release(third_lease)


def test_final_waits_for_in_progress_tick_before_claiming_admission(viewport, monkeypatch):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    backend = v.backends[0]
    backend.ready = True
    collected, resume_tick, final_started, final_entered, resume_final = (Event() for _ in range(5))
    errors, final_pending = [], []
    original_collect = backend.collect
    def collect(*, wait=False):
        result = original_collect(wait=wait)
        if result is not None and not wait:
            collected.set()
            assert resume_tick.wait(2), "test did not release the collected frame"
        return result
    monkeypatch.setattr(backend, "collect", collect)
    def tick():
        try:
            v.view.tick(lambda _: True)
        except BaseException as exc:
            errors.append(exc)
    def final():
        try:
            final_started.set()
            with v.coordinator.final("cuda:0"):
                final_pending.append(backend.pending)
                final_entered.set()
                assert resume_final.wait(2)
        except BaseException as exc:
            errors.append(exc)
    tick_thread, final_thread = Thread(target=tick), Thread(target=final)
    try:
        tick_thread.start()
        assert collected.wait(2)
        final_thread.start()
        assert final_started.wait(2)
        assert not final_entered.wait(0.05), "final raced past a still-running viewport quantum"
        resume_tick.set()
        assert final_entered.wait(2)
        assert final_pending == [False]
        assert v.view.tick(lambda _: True) == "suspended"
        assert not backend.pending
    finally:
        resume_tick.set()
        resume_final.set()
        tick_thread.join(2)
        if final_thread.ident is not None:
            final_thread.join(2)
    assert not errors


def test_final_waits_for_admitted_capture_before_reading_its_scene(viewport):
    v = viewport
    capturing, resume_capture, final_started, final_entered, resume_final = (Event() for _ in range(5))
    errors = []
    def extract():
        capturing.set()
        assert resume_capture.wait(2)
        return v.scene
    def capture():
        try:
            v.view.capture(extract)
        except BaseException as exc:
            errors.append(exc)
    def final():
        try:
            final_started.set()
            with v.coordinator.final("cuda:0"):
                final_entered.set()
                assert v.view.snapshot is v.scene
                assert resume_final.wait(2)
        except BaseException as exc:
            errors.append(exc)
    capture_thread, final_thread = Thread(target=capture), Thread(target=final)
    try:
        capture_thread.start()
        assert capturing.wait(2)
        final_thread.start()
        assert final_started.wait(2)
        assert not final_entered.wait(0.05), "final extraction overlapped an admitted viewport capture"
        resume_capture.set()
        assert final_entered.wait(2)
        assert not v.view.capture(lambda: pytest.fail("captured while final owned the scene"))
    finally:
        resume_capture.set()
        resume_final.set()
        capture_thread.join(2)
        if final_thread.ident is not None:
            final_thread.join(2)
    assert not errors


def test_final_admission_does_not_suspend_a_different_device(viewport):
    v = viewport
    with v.coordinator.final("cuda:1"):
        assert v.view.capture(lambda: v.scene)
        assert v.view.tick(lambda _: True) == "submitted"
        assert v.backends[0].pending


def test_final_retirement_keeps_exact_backend_when_view_migrates_devices(viewport, monkeypatch):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    old = v.backends[0]
    selected, resume, entered, finish = (Event() for _ in range(4))
    errors = []
    cancel = old.arbiter.cancel
    def paused_cancel(owner):
        if current_thread() is final_thread and not selected.is_set():
            selected.set()
            assert resume.wait(2)
        return cancel(owner)
    monkeypatch.setattr(old.arbiter, "cancel", paused_cancel)
    def replacement(device):
        backend = PendingBackend(device, DeviceArbiter())
        v.backends.append(backend)
        return backend
    monkeypatch.setattr(v.view, "backend_factory", replacement)
    def final():
        try:
            with v.coordinator.final("cuda:0"):
                entered.set()
                assert finish.wait(2)
        except BaseException as exc:
            errors.append(exc)
    final_thread = Thread(target=final)
    try:
        final_thread.start()
        assert selected.wait(2)
        v.view.request_update(replace(v.request, device="cuda:1"))
        assert v.view.tick(lambda _: True) == "capture"
        assert v.view.capture(lambda: v.scene)
        assert v.view.tick(lambda _: True) == "submitted"
        new = v.backends[-1]
        assert new.pending and old.closed
        resume.set()
        assert entered.wait(2)
        assert new.pending, "final drained a different device after backend migration"
        assert ("collect", True) not in new.calls
        assert v.view._completed is None
    finally:
        resume.set()
        finish.set()
        final_thread.join(2)
    assert not errors


def test_forward_publication_updates_do_not_starve_owned_completed_images(viewport):
    v = viewport
    seen = []
    for number in range(10):
        candidate = replace(v.scene, frame=number + 1, vertices=v.scene.vertices + number * 0.1)
        assert v.view.capture(lambda: candidate)
        assert v.view.tick(lambda _: True) == "submitted"
        v.view.invalidate(surface_only=True)
        v.backends[0].ready = True
        assert v.view.tick(lambda _: False, presentation_eligible=lambda _: True) == "capture"
        assert v.view.presentable is not None, "every completed live image was discarded"
        assert v.view.presented_snapshot is candidate
        assert v.view.presentable.scene_frame == number + 1
        seen.append(v.view.presentable.key)
    assert len(set(seen)) == 10
    assert v.view.statistics["discarded"] == 0


def test_static_edit_still_rejects_a_pending_frame_after_soft_publication_change(viewport):
    v = viewport
    v.view.capture(lambda: v.scene)
    v.view.tick(lambda _: True)
    v.view.invalidate(surface_only=True)
    v.view.invalidate()
    v.backends[0].ready = True
    assert v.view.tick(lambda _: True, presentation_eligible=lambda _: True) == "capture"
    assert v.view.presentable is None
