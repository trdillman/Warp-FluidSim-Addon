"""Controls for the independent live-acceptance content oracle."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _harness():
    path = Path(__file__).resolve().parents[1] / "acceptance/blender_fspt_live_acceptance.py"
    spec = importlib.util.spec_from_file_location("fspt_live_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_content_oracle_rejects_frozen_geometry_even_with_new_solver_labels():
    accept = _harness()
    prior = {"timeline_frame": 4, "solver_frame": 3, "generation": 8, "state_revision": 3,
             "epoch": 2, "geometry_sha256": "retained-fluid"}
    fresh_labels = dict(prior, timeline_frame=5, solver_frame=4, generation=9, state_revision=4)
    pixels = np.ones((3, 4, 4), dtype=np.float32)
    with pytest.raises(accept.FrozenSurfaceError, match="solved state advanced but fluid geometry did not"):
        accept.verify_advance(prior, fresh_labels, pixels, pixels)


def test_live_content_oracle_rejects_image_unchanged_after_geometry_changes():
    accept = _harness()
    prior = {"timeline_frame": 4, "solver_frame": 3, "generation": 8, "state_revision": 3,
             "epoch": 2, "geometry_sha256": "old-positions"}
    result = dict(prior, timeline_frame=5, solver_frame=4, generation=9, state_revision=4,
                  geometry_sha256="new-positions")
    pixels = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(AssertionError, match="completed image remained unchanged"):
        accept.verify_advance(prior, result, pixels, pixels)


def test_live_content_oracle_rejects_moving_image_with_stale_publication():
    accept = _harness()
    prior = {"timeline_frame": 4, "solver_frame": 3, "generation": 8, "state_revision": 3,
             "epoch": 2, "geometry_sha256": "old-positions"}
    result = dict(prior, timeline_frame=5, solver_frame=4, state_revision=4,
                  geometry_sha256="new-positions")
    pixels = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(AssertionError, match="generation"):
        accept.verify_advance(prior, result, pixels, pixels + 0.1)


def test_surface_digest_depends_on_actual_positions_and_connectivity():
    accept = _harness()
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    tris = np.asarray([[0, 1, 2]], dtype=np.int32)
    digest = accept.surface_digest(vertices, tris)
    assert digest != accept.surface_digest(vertices + 1, tris)
    assert digest != accept.surface_digest(vertices, tris[:, ::-1])
    assert digest == accept.surface_digest(np.vstack((vertices + 4, vertices)),
                                           tris + len(vertices))


@pytest.mark.parametrize("result", [{"PASS_THROUGH"}, {"CANCELLED"}, {"FINISHED"}])
def test_pause_requires_observed_stop_and_records_operator_status(result):
    accept = _harness()
    run = object.__new__(accept.LiveAcceptance)
    run.window = SimpleNamespace(screen=SimpleNamespace(is_animation_playing=True),
                                 scene=SimpleNamespace(frame_current=15))
    run.report = {"controls": {}}
    run.stage = "live"
    run.override = nullcontext

    def cancel(*, restore_frame):
        assert restore_frame is False
        run.window.screen.is_animation_playing = False
        return result

    run.bpy = SimpleNamespace(ops=SimpleNamespace(screen=SimpleNamespace(animation_cancel=cancel)))
    run.pause()
    receipt = run.report["controls"]["pause_transitions"][-1]
    assert receipt["operator_result"] == sorted(result)
    assert receipt["playing_before"] is True and receipt["playing_after"] is False
    assert receipt["frame_before"] == receipt["frame_after"] == 15


def test_pause_rejects_success_status_when_playback_did_not_stop():
    accept = _harness()
    run = object.__new__(accept.LiveAcceptance)
    run.window = SimpleNamespace(screen=SimpleNamespace(is_animation_playing=True),
                                 scene=SimpleNamespace(frame_current=15))
    run.report = {"controls": {}}
    run.stage = "live"
    run.override = nullcontext
    run.bpy = SimpleNamespace(ops=SimpleNamespace(screen=SimpleNamespace(
        animation_cancel=lambda **kwargs: {"FINISHED"})))
    with pytest.raises(AssertionError, match="Playback did not stop"):
        run.pause()
