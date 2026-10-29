"""The SceneInputs owner supplies the shared capture recipe to both callers."""

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs


class RecordingInputs(SceneInputs):
    def __init__(self):
        super().__init__()
        self.calls = []

    def gather_sources(self, scene, depsgraph, **kwargs):
        self.calls.append(("sources", depsgraph, kwargs))
        return [SimpleNamespace(mesh=None)]

    def gather_source_plans(self, scene, depsgraph):
        self.calls.append(("plans", depsgraph))
        return ["plan"]

    def gather_colliders(self, scene, depsgraph, **kwargs):
        self.calls.append(("colliders", depsgraph, kwargs))
        return [SimpleNamespace(mesh=None)]

    def gather_force_fields(self, scene, depsgraph):
        self.calls.append(("forces", depsgraph))
        return ["force"]

    def gather_height_source(self, scene, depsgraph):
        self.calls.append(("height", depsgraph))
        return "height"

    def materialize_device_meshes(self, sources, colliders, device, *, staged=False):
        assert staged
        self.calls.append(("staging", device, sources, colliders))


def test_evaluation_owns_generation_staging_and_bake_first_capture_flag():
    inputs, depsgraph = RecordingInputs(), object()
    result = inputs.evaluate(
        object(), depsgraph, device="cuda:7", staged=True,
        capture_first_mesh=False)
    assert inputs.device == "cpu"
    assert inputs.stage_meshes is True
    assert result.session_generation == inputs._session_generation
    assert [call[1] for call in inputs.calls[:5]] == [depsgraph] * 5
    assert inputs.calls[0][2] == {
        "capture_first_mesh": False, "sync_generation": 1}
    assert inputs.calls[2][2] == {"sync_generation": 1}
    assert inputs.calls[5] == ("staging", "cpu", result.descriptors, result.colliders)
    assert result.source_plans == ("plan",)


def test_refused_sources_leave_source_lifecycle_and_cache_keys_untouched():
    inputs = RecordingInputs()
    inputs._source_mesh_cache_keys = {"admitted"}
    admitted = {"mesh": object()}
    inputs._evaluated_mesh_cache = {"admitted": admitted, "removed": {}}
    result = inputs.evaluate(
        object(), object(), device="cpu", include_sources=False)
    assert [call[0] for call in inputs.calls] == ["colliders", "forces", "height"]
    assert result.descriptors == result.source_plans == ()
    assert result.force_fields == ("force",)
    assert result.height_source == "height"
    assert inputs._evaluated_mesh_cache == {"admitted": admitted}


def test_failed_projection_ends_evaluation_without_clearing_admitted_lifecycle():
    inputs = RecordingInputs()
    generation = inputs._session_generation
    def fail(*_args):
        raise RuntimeError("height unavailable")
    inputs.gather_height_source = fail
    with pytest.raises(RuntimeError, match="height unavailable"):
        inputs.evaluate(object(), object(), device="cpu")
    assert inputs._evaluation_active is False
    assert inputs._session_generation == generation
