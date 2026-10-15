"""Dynamic collider scene rules (4.0 two-way rigid coupling), free of bpy and Warp.

Two answers every caller shares:

* which authored setups admission refuses (``scene_blocking_reasons``), read
  duck-typed from scene objects so the Blender adapter and tests ask the same
  question;
* how a solved body state becomes the object's world matrix (``pose_matrix``):
  ``T(pos) R(quat) T(-rest_com) M_rest``, where ``M_rest`` is the authored
  world matrix the body was built from and ``rest_com`` its centre of mass in
  that pose.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

MAX_BODIES = 64

# Custom property on the FluidSim-owned pose action: keyframes carrying it are
# solver output (cleared on Reset), anything else is the user's animation.
OWNED_ACTION_TAG = "fluidsim_rigid_poses"

KINEMATIC_REASON = (
    "FluidSim moves this Dynamic collider from the liquid and publishes its "
    "pose as keyframes, so its Blender rigid body is set to Animated "
    "(kinematic) to keep Bullet from moving it too")


def _settings(obj):
    return getattr(obj, "fluidsim", None)


def is_dynamic_collider(obj) -> bool:
    fs = _settings(obj)
    return bool(
        fs is not None
        and getattr(fs, "role", "NONE") == "COLLIDER"
        and getattr(fs, "enabled", True)
        and getattr(fs, "collider_dynamic", False)
        and getattr(obj, "type", "MESH") == "MESH")


def dynamic_collider_objects(scene_objects: Iterable[object]) -> list:
    return [obj for obj in scene_objects or () if is_dynamic_collider(obj)]


def has_foreign_animation(obj) -> bool:
    """True when the object is keyframed by an action FluidSim does not own."""
    animation = getattr(obj, "animation_data", None)
    action = getattr(animation, "action", None) if animation is not None else None
    if action is None:
        return False
    getter = getattr(action, "get", None)
    return not (callable(getter) and bool(getter(OWNED_ACTION_TAG, False)))


def _rides_on(obj, bodies) -> object | None:
    parent = getattr(obj, "parent", None)
    seen = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        if any(parent is body for body in bodies):
            return parent
        parent = getattr(parent, "parent", None)
    return None


def scene_blocking_reasons(scene_objects: Iterable[object]) -> tuple[str, ...]:
    """Every reason the scene's Dynamic colliders cannot be admitted, in order."""
    objects = list(scene_objects or ())
    bodies = dynamic_collider_objects(objects)
    if not bodies:
        return ()
    reasons = []
    if len(bodies) > MAX_BODIES:
        reasons.append(
            f"{len(bodies)} Dynamic colliders exceed the limit of {MAX_BODIES}")
    for body in bodies:
        if has_foreign_animation(body):
            reasons.append(
                f"Dynamic collider '{body.name}' is animated; its motion comes "
                "from the liquid, so remove its keyframes or turn Dynamic off")
    for obj in objects:
        fs = _settings(obj)
        role = getattr(fs, "role", "NONE") if fs is not None else "NONE"
        if role not in ("SOURCE", "SINK") or not getattr(fs, "enabled", True):
            continue
        geometry = getattr(fs, "mesh_object", None)
        body = next((b for b in bodies if geometry is b), None)
        how = "uses the geometry of"
        if body is None:
            body = _rides_on(obj, bodies)
            how = "is parented to"
        if body is not None:
            reasons.append(
                f"{role.lower()} '{obj.name}' {how} Dynamic collider "
                f"'{body.name}'; a Dynamic collider cannot also be a "
                f"{role.lower()}")
    return tuple(reasons)


def refusal_message(reasons: Iterable[str]) -> str:
    return (
        "Dynamic colliders cannot run with this setup: "
        + "; ".join(reasons)
        + ". Remove the conflict or turn Dynamic off on the collider.")


def _field(record, name):
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def quat_matrix(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_matrix(record, rest_matrix) -> np.ndarray:
    """The 4x4 world matrix of a body state (``RigidBodyRecord`` or dict)."""
    rest = np.asarray(rest_matrix, dtype=np.float64).reshape(4, 4)
    pos = np.asarray(_field(record, "pos"), dtype=np.float64)
    com = np.asarray(_field(record, "rest_com"), dtype=np.float64)
    motion = np.eye(4)
    rotation = quat_matrix(_field(record, "quat"))
    motion[:3, :3] = rotation
    motion[:3, 3] = pos - rotation @ com
    return motion @ rest
