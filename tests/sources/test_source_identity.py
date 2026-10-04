from types import SimpleNamespace

from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs


class Object:
    name = "original"
    def as_pointer(self): return 1234


class Matrix:
    translation = (0.0,0.0,0.0)
    def to_quaternion(self): raise AssertionError


def test_scene_input_session_generation_changes_only_on_clear():
    inputs = SceneInputs()
    first = inputs._session_generation
    inputs.begin_sync(); inputs.begin_sync()
    assert inputs._session_generation == first
    inputs.clear()
    assert inputs._session_generation == first + 1


def test_object_velocity_history_is_rename_stable():
    inputs = SceneInputs()
    obj = Object(); obj.matrix_world = Matrix()
    scene = SimpleNamespace(frame_current=1,
        render=SimpleNamespace(fps=24,fps_base=1))
    assert inputs.object_velocity(obj,scene) == (0,0,0)
    obj.name = "renamed"; scene.frame_current = 2
    assert inputs.object_velocity(obj,scene) == (0,0,0)
    assert list(inputs._collider_prev) == [1234]
