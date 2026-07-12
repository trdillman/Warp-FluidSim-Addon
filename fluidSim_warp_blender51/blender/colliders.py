"""Collider tagging and compatibility entry points."""

from __future__ import annotations


def tag_collider(obj, friction=0.0, restitution=0.0, transfer_velocity=True):
    obj.fluidsim.role = "COLLIDER"
    obj.fluidsim.collider_friction = friction
    obj.fluidsim.collider_restitution = restitution
    obj.fluidsim.transfer_velocity = transfer_velocity
    return obj


def collider_objects(scene):
    return [o for o in scene.objects
            if getattr(o, "fluidsim", None) is not None
            and o.fluidsim.role == "COLLIDER" and o.fluidsim.enabled]


def gather_colliders(scene, session, depsgraph=None):
    """Compatibility wrapper for the SceneInputs owner."""
    return getattr(session, "scene_inputs", session).gather_colliders(
        scene, depsgraph)
