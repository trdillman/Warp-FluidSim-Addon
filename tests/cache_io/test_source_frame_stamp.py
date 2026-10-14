"""The per-frame source stamp, as the Blender adapter produces it.

``source_frame_stamp`` is what a v3 bake writes for every frame and what a
resume compares against, so these pin what moves it (a transform, geometry)
and what must not (a rename, a panel setting, the frame number itself).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender.cache_identity import source_frame_stamp

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class _Object(dict):
    def __init__(self, key, *, role="COLLIDER"):
        super().__init__(fluidsim_persistent_id=key)
        properties = {
            "role": SimpleNamespace(type="ENUM", is_array=False),
            "friction": SimpleNamespace(type="FLOAT", is_array=False),
        }
        self.name = key
        self.name_full = key
        self.library = None
        self.type = "EMPTY"
        self.data = SimpleNamespace(points=())
        self.matrix_world = _IDENTITY
        self.fluidsim = SimpleNamespace(
            role=role,
            friction=0.5,
            bl_rna=SimpleNamespace(properties=properties),
        )


def _scene(*objects, frame=1):
    return SimpleNamespace(objects=tuple(objects), frame_current=frame)


def _translated(x):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_stamp_follows_a_moving_collider():
    paddle = _Object("paddle")
    scene = _scene(paddle, frame=1)
    at_rest = source_frame_stamp(scene, None, 1)

    paddle.matrix_world = _translated(0.25)
    scene.frame_current = 30

    assert source_frame_stamp(scene, None, 30) != at_rest


def test_stamp_is_repeatable_for_a_static_scene():
    scene = _scene(_Object("a"), _Object("b", role="SOURCE"), frame=1)
    first = source_frame_stamp(scene, None, 1)
    scene.frame_current = 30

    assert source_frame_stamp(scene, None, 30) == first


def test_stamp_ignores_panel_settings_and_names():
    paddle = _Object("paddle")
    scene = _scene(paddle)
    before = source_frame_stamp(scene, None, 1)

    paddle.fluidsim.friction = 0.9
    paddle.name_full = "paddle.001"

    assert source_frame_stamp(scene, None, 1) == before


def test_stamp_does_not_depend_on_object_order():
    a, b = _Object("a"), _Object("b")
    b.matrix_world = _translated(2.0)

    assert (
        source_frame_stamp(_scene(a, b), None, 1)
        == source_frame_stamp(_scene(b, a), None, 1)
    )


def test_untagged_objects_do_not_contribute():
    paddle = _Object("paddle")
    bystander = _Object("bystander", role="NONE")
    before = source_frame_stamp(_scene(paddle), None, 1)
    bystander.matrix_world = _translated(5.0)

    assert source_frame_stamp(_scene(paddle, bystander), None, 1) == before


def test_stamp_refuses_to_hash_another_frames_scene_under_this_frames_name():
    scene = _scene(_Object("paddle"), frame=7)

    with pytest.raises(ValueError, match="frame_set"):
        source_frame_stamp(scene, None, 8)
