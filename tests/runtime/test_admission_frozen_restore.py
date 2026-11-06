"""Resume state must be restored before admission evaluates new source inputs."""
from types import SimpleNamespace

import pytest
from test_scene_bridge_settings import _install_scene_bridge_stubs


def test_invalid_frozen_state_stops_capture_before_evaluation(monkeypatch):
    _install_scene_bridge_stubs(monkeypatch)
    from fluidSim_warp_blender51.blender import simulation_admission as admission
    from fluidSim_warp_blender51.sim import source_regions

    events = []
    frozen = object()

    class Inputs:
        def restore_frozen_state(self, value):
            assert value is frozen
            events.append("restore")
            raise ValueError("invalid frozen input state")

        def evaluate(self, *_args, **_kwargs):
            pytest.fail("source evaluation ran before resume-state validation")

        def clear(self):
            events.append("clear")

    monkeypatch.setattr(admission, "SceneInputs", Inputs)
    monkeypatch.setattr(source_regions, "SourceCompiler", lambda: SimpleNamespace(
        release=lambda: events.append("compiler-release")))
    scene = SimpleNamespace(frame_current=1, fluidsim=SimpleNamespace(domain_object=None))
    with pytest.raises(ValueError, match="invalid frozen input state"):
        admission.capture_snapshot(
            scene, SimpleNamespace(device="cpu"), {},
            depsgraph=SimpleNamespace(updates=[]), frozen_input_state=frozen)
    assert events == ["restore", "clear", "compiler-release"]
