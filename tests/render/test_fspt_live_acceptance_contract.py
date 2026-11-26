"""Controls for the independent live-acceptance content oracle."""
import importlib.util
from pathlib import Path

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
