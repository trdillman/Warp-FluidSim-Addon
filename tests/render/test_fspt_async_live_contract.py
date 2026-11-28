"""Owned asynchronous images require independent evaluated-geometry receipts."""
import importlib.util
from collections import Counter, deque
from pathlib import Path
from threading import local
from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.fspt.contracts import Material, SceneSnapshot


def harness():
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_async_live_acceptance.py"
    spec = importlib.util.spec_from_file_location("async_live_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene():
    return SceneSnapshot(np.eye(3), [[0, 1, 2]], [0], (Material(kind="glass"),), (0, 0, 0), 1)


def test_missing_independent_capture_receipt_rejects_an_owned_image():
    with pytest.raises(AssertionError, match="no independent"):
        harness().join_geometry_audit(scene(), {})


def test_geometry_receipt_must_match_actual_submitted_positions():
    module, snapshot = harness(), scene()
    with pytest.raises(AssertionError, match="differs from independently audited"):
        module.join_geometry_audit(snapshot, {snapshot.digest: {
            "geometry_sha256": "different-actual-geometry", "evaluated_triangle_count": 1}})
    digest, count = module.fluid_geometry(snapshot)
    audit = {"geometry_sha256": digest, "evaluated_triangle_count": count}
    assert module.join_geometry_audit(snapshot, {snapshot.digest: audit}) is audit


def test_draw_observer_joins_presented_snapshot_when_latest_capture_already_changed():
    module = harness()
    run = object.__new__(module.AsyncLiveAcceptance)
    run.callback = local()
    run.window, run.area, run.region = object(), object(), object()
    context = SimpleNamespace(window=run.window, area=run.area, region=run.region)
    old, newest = object(), object()
    frame = SimpleNamespace(key=("captured-old",), samples=1)
    renderer = SimpleNamespace(_view=SimpleNamespace(snapshot=newest, presented_snapshot=old,
                                                     presentable=frame), _texture_frame=frame,
                               _view_info={"status": "COMPLETE"})
    def draw(*args):
        run.engine._DRAW_COUNT += 1
    run.engine = SimpleNamespace(FSPTRenderEngine=SimpleNamespace(view_draw=draw), _DRAW_COUNT=0)
    run.stage = "live"
    run.observer_error = None
    run.last_draw_identity = None
    run.report = {"draw_observer": {"entered": 0, "matched": 0}}
    run.pending = deque(maxlen=4)
    run.drop_counts = Counter()
    seen = []
    def joined(context, graph, observed, snapshot, info, started):
        seen.append(snapshot)
        return {}, np.ones((1, 1, 4))
    run.capture_owned_draw = joined
    run.observe_draw()(renderer, context, object())
    assert run.observer_error is None
    assert seen == [old] and len(run.pending) == 1


def test_soak_rejects_frozen_pixels_when_independently_audited_geometry_changes():
    module = harness()
    prior = {"epoch": 1, "timeline_frame": 2, "solver_frame": 1, "state_revision": 1,
             "generation": 1, "geometry_sha256": "actual-old", "telemetry": {"key": ["old", "camera"]}}
    current = dict(prior, timeline_frame=3, solver_frame=2, state_revision=2, generation=2,
                   geometry_sha256="actual-new", telemetry={"key": ["new", "camera"]})
    pixels = np.ones((2, 3, 4))
    with pytest.raises(AssertionError, match="completed pixels remained frozen"):
        module.verify_soak_advance(prior, current, pixels, pixels)
    assert module.verify_soak_advance(prior, current, pixels, pixels + 0.1) == pytest.approx(0.1)


def test_soak_rejects_new_pixels_with_reused_source_identity():
    prior = {"epoch": 1, "timeline_frame": 2, "solver_frame": 1, "state_revision": 1,
             "generation": 1, "geometry_sha256": "actual-old", "telemetry": {"key": ["old", "camera"]}}
    pixels = np.ones((2, 3, 4))
    with pytest.raises(AssertionError, match="did not advance"):
        harness().verify_soak_advance(prior, prior, pixels, pixels + 0.1)
