"""Dynamic collider scene rules (4.0 lane R): bpy-free, one answer for every caller.

docs/specs/4.0/SPEC-rigid-coupling.md trap 6: a Dynamic collider that is also a
source or sink is refused at admission with a reason. Roles are exclusive per
object, so "also a source or sink" means an enabled source/sink that takes its
geometry from the Dynamic collider (``mesh_object``) or rides on it (parented
under it). Also refused: more than 64 bodies, and a Dynamic collider animated by
someone else (its motion is the solver's output, published as FluidSim-owned
keyframes).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.sim import rigid_scene as rs


def _object(name, role="NONE", *, dynamic=False, enabled=True, mesh_object=None,
            parent=None, action=None, obj_type="MESH"):
    fluidsim = SimpleNamespace(
        role=role, enabled=enabled, collider_dynamic=dynamic,
        mesh_object=mesh_object)
    animation = None if action is None else SimpleNamespace(action=action)
    return SimpleNamespace(name=name, type=obj_type, fluidsim=fluidsim,
                           parent=parent, animation_data=animation)


def _cork(**kwargs):
    return _object("Cork", "COLLIDER", dynamic=True, **kwargs)


def test_static_scene_has_no_reasons():
    wall = _object("Wall", "COLLIDER")
    source = _object("Tap", "SOURCE", mesh_object=wall)
    assert rs.scene_blocking_reasons([wall, source]) == ()


def test_lone_dynamic_collider_is_admitted():
    assert rs.scene_blocking_reasons([_cork()]) == ()
    assert rs.dynamic_collider_objects([_cork(), _object("Wall", "COLLIDER")])[0].name == "Cork"


@pytest.mark.parametrize("role", ["SOURCE", "SINK"])
def test_source_or_sink_using_dynamic_geometry_is_refused(role):
    cork = _cork()
    other = _object("Emitter", role, mesh_object=cork)
    reasons = rs.scene_blocking_reasons([cork, other])
    assert len(reasons) == 1
    assert "Cork" in reasons[0] and "Emitter" in reasons[0]
    assert role.lower() in reasons[0]


@pytest.mark.parametrize("role", ["SOURCE", "SINK"])
def test_source_or_sink_parented_to_dynamic_collider_is_refused(role):
    cork = _cork()
    middle = _object("Empty", obj_type="EMPTY", parent=cork)
    other = _object("Drain", role, parent=middle)
    reasons = rs.scene_blocking_reasons([cork, middle, other])
    assert len(reasons) == 1 and "Drain" in reasons[0]


def test_disabled_source_and_disabled_or_static_collider_are_ignored():
    cork = _cork()
    off = _object("Off", "SOURCE", mesh_object=cork, enabled=False)
    assert rs.scene_blocking_reasons([cork, off]) == ()
    disabled = _cork(enabled=False)
    source = _object("Tap", "SOURCE", mesh_object=disabled)
    assert rs.scene_blocking_reasons([disabled, source]) == ()


def test_foreign_animation_is_refused_but_owned_keyframes_are_not():
    foreign = _cork(action={"name": "user"})
    reasons = rs.scene_blocking_reasons([foreign])
    assert len(reasons) == 1 and "animat" in reasons[0]
    owned = _cork(action={rs.OWNED_ACTION_TAG: True})
    assert rs.scene_blocking_reasons([owned]) == ()


def _curve(keys=0, modifiers=0):
    return SimpleNamespace(keyframe_points=[object()] * keys, modifiers=[object()] * modifiers)


def _layered(*curves):
    bag = SimpleNamespace(fcurves=list(curves))
    return SimpleNamespace(layers=[SimpleNamespace(strips=[SimpleNamespace(channelbags=[bag])])],
                           get=lambda key, default=None: default)


def test_emptied_action_is_not_animation():
    # Deleting the last keyframe leaves the emptied action assigned; the
    # refusal told the artist to remove the keyframes, so Play must now run.
    assert rs.scene_blocking_reasons([_cork(action=_layered())]) == ()
    legacy = SimpleNamespace(fcurves=[], get=lambda key, default=None: default)
    assert rs.scene_blocking_reasons([_cork(action=legacy)]) == ()


@pytest.mark.parametrize("curve", [_curve(keys=1), _curve(modifiers=1)])
def test_keyed_or_modified_curve_is_animation(curve):
    reasons = rs.scene_blocking_reasons([_cork(action=_layered(_curve(), curve))])
    assert len(reasons) == 1 and "animat" in reasons[0]


def test_more_than_the_body_limit_is_refused():
    bodies = [_object(f"B{i}", "COLLIDER", dynamic=True) for i in range(rs.MAX_BODIES + 1)]
    reasons = rs.scene_blocking_reasons(bodies)
    assert len(reasons) == 1 and str(rs.MAX_BODIES) in reasons[0]
    assert rs.scene_blocking_reasons(bodies[:-1]) == ()


def test_refusal_message_lists_reasons():
    message = rs.refusal_message(("a", "b"))
    assert message.startswith("Dynamic colliders cannot run with this setup: a; b")


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_pose_matrix_at_rest_is_the_rest_matrix():
    rest = np.eye(4)
    rest[:3, :3] = np.diag((1.0, 2.0, 0.5))
    rest[:3, 3] = (0.3, -0.2, 0.7)
    record = {"pos": (0.4, 0.1, 0.9), "quat": (1.0, 0.0, 0.0, 0.0),
              "rest_com": (0.4, 0.1, 0.9)}
    assert np.allclose(rs.pose_matrix(record, rest), rest)


def test_pose_matrix_moves_and_turns_about_the_centre_of_mass():
    rest = np.eye(4)
    rest[:3, 3] = (1.0, 0.0, 0.0)
    com = np.array([1.0, 0.0, 0.5])
    half = np.pi / 4.0
    record = SimpleNamespace(pos=(2.0, 1.0, 0.5),
                             quat=(np.cos(half), 0.0, 0.0, np.sin(half)),
                             rest_com=tuple(com))
    world = rs.pose_matrix(record, rest)
    # The centre of mass lands on pos; a point offset from it turns 90 degrees.
    point = np.array([2.0, 0.0, 0.5, 1.0])     # com + (1, 0, 0) in rest
    local = np.linalg.solve(rest, point)
    moved = world @ local
    assert np.allclose(moved[:3], (2.0, 2.0, 0.5))
    assert np.allclose(world[:3, :3], _rotation_z(np.pi / 2.0))
