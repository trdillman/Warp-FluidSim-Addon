"""Real gatherers use persisted input epochs; Blender resources are CPU doubles."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_scene_bridge_collider_cache import (
    _animate_native_inputs,
    _field_source,
    _native_plan_inputs,
    _plans_by_name,
    _point_source,
    _region_inputs,
    _volume_source,
)

from fluidSim_warp_blender51.blender.scene_inputs import (
    SceneInputs,
    validate_frozen_scene,
)


@pytest.mark.parametrize(
    "factory",
    [
        _point_source,
        _volume_source,
        lambda mode: _volume_source(mode, role="SINK"),
        _field_source,
    ],
    ids=["points", "volume-source", "volume-sink", "field"],
)
def test_persisted_native_snapshot_survives_a_new_owner_at_later_frame(
    monkeypatch, tmp_path, factory
):
    original = _native_plan_inputs(monkeypatch)
    obj = factory("FIRST_SIM_FRAME")
    if obj.type == "VOLUME":
        path = tmp_path / "first.vdb"
        path.write_bytes(b"original-volume-payload")
        obj.data.grids.frame_filepath = str(path)
    scene = SimpleNamespace(
        objects=[obj], frame_current=1, render=SimpleNamespace(fps=24, fps_base=1)
    )
    expected = _plans_by_name(original, scene)[obj.name]
    state = original.export_frozen_state()
    _animate_native_inputs([obj], scene)
    validate_frozen_scene(state, scene)
    resumed = SceneInputs()
    resumed.restore_frozen_state(state)
    actual = _plans_by_name(resumed, scene)[obj.name]
    assert _region_inputs(actual.region) == _region_inputs(expected.region)
    recaptured = _plans_by_name(SceneInputs(), scene)[obj.name]
    assert _region_inputs(recaptured.region) != _region_inputs(expected.region)


def test_every_frame_point_input_is_recaptured_at_resume_frame(monkeypatch):
    original = _native_plan_inputs(monkeypatch)
    obj = _point_source("EVERY_SIM_FRAME")
    scene = SimpleNamespace(
        objects=[obj], frame_current=1, render=SimpleNamespace(fps=24, fps_base=1)
    )
    _plans_by_name(original, scene)
    state = original.export_frozen_state()
    assert not state.metadata["records"]
    _animate_native_inputs([obj], scene)
    resumed = SceneInputs()
    resumed.restore_frozen_state(state)
    actual = _plans_by_name(resumed, scene)[obj.name]
    expected = _plans_by_name(original, scene)[obj.name]
    np.testing.assert_array_equal(actual.region.positions, expected.region.positions)
