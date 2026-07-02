"""Collider-object helpers.

Mesh-to-SDF conversion happens on the GPU (coreCreateSDFAxis kernels); the
scene bridge builds world-space ``wp.Mesh`` objects per collider. This module
provides tagging and animated-collider refresh utilities used by operators.
"""

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


def aggregate_material_response(scene):
    """Combine per-collider friction/restitution into solver-wide values.

    The solver applies a single restitution/friction pair during particle
    collision response; use the maximum across enabled colliders combined
    with the scene-level settings.
    """
    fs = scene.fluidsim
    friction = fs.friction
    restitution = fs.restitution
    for obj in collider_objects(scene):
        friction = max(friction, obj.fluidsim.collider_friction)
        restitution = max(restitution, obj.fluidsim.collider_restitution)
    return restitution, friction
