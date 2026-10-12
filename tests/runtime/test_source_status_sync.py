"""B78: source-status sync touches only simulation inputs, and only on change.

``SceneInputs.update_source_status`` runs after every live step. It visited
every object (every Blender object carries a FluidSim group), stamped a
persistent source id custom property onto cameras and lights, and rewrote
every status field on every frame even when nothing had changed.
"""

from __future__ import annotations

from types import SimpleNamespace

from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs


class _Group:
    """An object FluidSim group that records every attribute write."""

    def __init__(self, role, **values):
        object.__setattr__(self, "writes", [])
        defaults = dict(
            role=role, source_mode="PARTICLES", enabled=True,
            auto_inject_seed=False, source_status_severity="",
            source_status_code="", source_status_message="",
            source_status_emitted=0, source_status_capacity_limited=False)
        defaults.update(values)
        for name, value in defaults.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        self.writes.append(name)
        object.__setattr__(self, name, value)


class _Object:
    """A Blender object double: custom properties plus a FluidSim group."""

    def __init__(self, name, role, pointer, **values):
        self.name = self.name_full = name
        self.fluidsim = _Group(role, **values)
        self.custom = {}
        self._pointer = pointer
        self.library = None
        self.data = None

    def as_pointer(self):
        return self._pointer

    def get(self, key, default=None):
        return self.custom.get(key, default)

    def __setitem__(self, key, value):
        self.custom[key] = value


def _diagnostic(obj, code, message="bad input", severity="ERROR"):
    from fluidSim_warp_blender51.blender.scene_inputs import persistent_source_key

    return SimpleNamespace(stable_id=persistent_source_key(obj), code=code,
                           message=message, severity=severity)


def test_objects_outside_the_simulation_are_never_touched():
    camera = _Object("Camera", "NONE", 1, source_status_message="stale")
    light = _Object("Light", "NONE", 2)
    source = _Object("Source", "SOURCE", 3)
    scene = SimpleNamespace(objects=[camera, light, source])

    SceneInputs().update_source_status(scene, (), {"emitted": 5})

    assert camera.custom == {} and light.custom == {}
    assert camera.fluidsim.writes == [] and light.fluidsim.writes == []
    assert source.fluidsim.source_status_emitted == 5


def test_an_unchanged_status_is_not_rewritten_every_frame():
    source = _Object("Source", "SOURCE", 3)
    sink = _Object("Sink", "SINK", 4)
    seed = _Object("Seed", "NONE", 5, auto_inject_seed=True)
    scene = SimpleNamespace(objects=[source, sink, seed])
    inputs = SceneInputs()
    warning = (_diagnostic(sink, "SINK_EMPTY", "no voxels", "WARNING"),)

    inputs.update_source_status(scene, warning, {"emitted": 7})
    assert sink.fluidsim.source_status_code == "SINK_EMPTY"
    assert source.fluidsim.source_status_emitted == 7
    for obj in (source, sink, seed):
        obj.fluidsim.writes.clear()

    inputs.update_source_status(scene, warning, {"emitted": 7})

    assert source.fluidsim.writes == []
    assert sink.fluidsim.writes == []
    assert seed.fluidsim.writes == []

    # A real change is still written, and a cleared diagnostic is cleared.
    inputs.update_source_status(scene, (), {"emitted": 9})
    assert sink.fluidsim.source_status_code == ""
    assert sink.fluidsim.source_status_message == ""
    assert source.fluidsim.source_status_emitted == 9


def test_an_input_error_set_by_gathering_survives_the_status_sync():
    source = _Object("Source", "SOURCE", 3,
                     source_status_severity="ERROR",
                     source_status_code="SOURCE_INPUT_INVALID",
                     source_status_message="missing point input")
    scene = SimpleNamespace(objects=[source])

    SceneInputs().update_source_status(scene, (), None)

    assert source.fluidsim.source_status_code == "SOURCE_INPUT_INVALID"
    assert source.fluidsim.source_status_message == "missing point input"
