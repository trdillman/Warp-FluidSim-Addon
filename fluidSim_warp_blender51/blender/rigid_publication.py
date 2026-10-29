"""Output Publication of Dynamic collider poses (4.0 two-way rigid coupling).

FluidSim owns a Dynamic collider's 6-DoF state; Blender sees it as keyframes
on the object, so the motion renders, exports (Alembic) and plays back with
no solver. docs/specs/4.0/SPEC-rigid-coupling.md, traps 1 and 5:

* Keyframes live in a FluidSim-owned action, tagged with
  ``rigid_scene.OWNED_ACTION_TAG``; Reset removes it and restores the pose.
* The pose the body was built from (the *authored* pose) is stored on the
  object the first time a pose is published. Scene inputs and cache identity
  read it through :func:`authored_world`, so published keyframes neither move
  the collider's rest geometry nor make a bake look stale.
* A Blender rigid body on the object is set to kinematic (Animated) with a
  reason, so Bullet does not fight the published motion.
"""

from __future__ import annotations

import numpy as np

from ..sim import rigid_scene

REST_PROP = "fluidsim_rigid_rest"
REASON_PROP = "fluidsim_rigid_reason"
KEY_GROUP = "FluidSim Dynamic"


def _stored_rest(obj):
    getter = getattr(obj, "get", None)
    value = getter(REST_PROP) if callable(getter) else None
    if value is None:
        return None
    values = [float(v) for v in value]
    if len(values) != 16:
        return None
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def Matrix(rows):
    from mathutils import Matrix as _Matrix

    return _Matrix(rows)


def authored_world(obj):
    """The object's authored world matrix: the stored rest pose once FluidSim
    has published a solved pose on it, otherwise ``matrix_world``."""
    rest = _stored_rest(obj)
    if rest is None:
        return obj.matrix_world
    return Matrix(rest.tolist())


def _store_rest(obj):
    rest = _stored_rest(obj)
    if rest is None:
        rest = np.asarray(obj.matrix_world, dtype=np.float64)
        obj[REST_PROP] = [float(v) for v in rest.reshape(-1)]
    return rest


def ensure_kinematic(obj) -> str:
    """Set a Blender rigid body on ``obj`` to kinematic; return the reason or ""."""
    body = getattr(obj, "rigid_body", None)
    if body is None:
        return ""
    if not body.kinematic:
        body.kinematic = True
    obj[REASON_PROP] = rigid_scene.KINEMATIC_REASON
    return rigid_scene.KINEMATIC_REASON


def _record_uid(record):
    if isinstance(record, dict):
        return str(record["object_uid"])
    return str(record.object_uid)


def _rotation_path(obj):
    mode = getattr(obj, "rotation_mode", "XYZ")
    if mode == "QUATERNION":
        return "rotation_quaternion"
    if mode == "AXIS_ANGLE":
        return "rotation_axis_angle"
    return "rotation_euler"


def _insert_keys(obj, frame):
    animation = obj.animation_data
    if (animation is not None and animation.action is not None
            and not animation.action.get(rigid_scene.OWNED_ACTION_TAG)):
        # Admission let it through, so it animates nothing (its keyframes were
        # deleted): unassign it so the solver keys go to an owned action.
        animation.action = None
    for path in ("location", _rotation_path(obj), "scale"):
        obj.keyframe_insert(data_path=path, frame=float(frame), group=KEY_GROUP)
    action = obj.animation_data.action
    if not action.get(rigid_scene.OWNED_ACTION_TAG):
        action[rigid_scene.OWNED_ACTION_TAG] = True
        action.name = f"FluidSim Dynamic {obj.name}"


def _bodies_by_uid(scene):
    from .scene_inputs import persistent_source_key

    return {persistent_source_key(obj): obj
            for obj in rigid_scene.dynamic_collider_objects(scene.objects)}


def publish(scene, records, frame) -> int:
    """Key each record's pose on its Dynamic collider at ``frame``."""
    records = tuple(records or ())
    if not records:
        return 0
    bodies = _bodies_by_uid(scene)
    published = 0
    for record in records:
        obj = bodies.get(_record_uid(record))
        if obj is None or rigid_scene.has_foreign_animation(obj):
            # Admission refuses foreign animation; never overwrite it.
            continue
        rest = _store_rest(obj)
        ensure_kinematic(obj)
        obj.matrix_world = Matrix(rigid_scene.pose_matrix(record, rest).tolist())
        _insert_keys(obj, frame)
        published += 1
    return published


def publish_live(scene, sess) -> int:
    """Publish the live solver's body states at the current frame."""
    states = getattr(getattr(sess, "solver", None), "rigid_body_states", None)
    if states is None:
        return 0
    records = states()
    if not isinstance(records, (list, tuple)):
        return 0
    return publish(scene, records, live_frame(scene, sess))


def live_frame(scene, sess) -> int:
    """The scene frame the live solver state belongs to.

    A run starts at ``sess.start_frame`` and each step advances one frame, so
    stepping without moving the timeline (bake loops, demo builders) still
    keys each pose on its own frame. Falls back to the timeline cursor.
    """
    start = getattr(sess, "start_frame", None)
    steps = getattr(getattr(sess, "solver", None), "frame", None)
    if isinstance(start, int) and isinstance(steps, int) and not isinstance(
            start, bool) and not isinstance(steps, bool):
        return start + steps
    return int(scene.frame_current)


def publish_cache_frame(scene, frame_data) -> int:
    """Key a cached frame's v3.1 rigid records (a v3 cache has none)."""
    records = frame_data.get("rigid_bodies", ()) if frame_data else ()
    if not records:
        return 0
    frame = frame_data.get("frame", scene.frame_current)
    frame = int(frame.item() if hasattr(frame, "item") else frame)
    return publish(scene, records, frame)


def cache_records(base_dir, frames, *, sequence=None) -> dict:
    """Rigid records from one verified committed sequence observation."""
    from ..io import cache_schema

    if sequence is None:
        sequence = cache_schema.observe_committed_sequence(base_dir)
    return {} if sequence is None else sequence.rigid_records(frames)


def carried_objects(scene, base_dir, frames, *, sequence=None) -> tuple:
    """Dynamic colliders whose poses the cache holds for ``frames``."""
    uids = {record.object_uid
            for records in cache_records(base_dir, frames, sequence=sequence).values()
            for record in records}
    if not uids:
        return ()
    return tuple(obj for uid, obj in _bodies_by_uid(scene).items() if uid in uids)


def publish_cache_frames(scene, base_dir, frames, *, sequence=None) -> int:
    """Key every cached pose in ``frames`` (the Alembic export pre-pass)."""
    published = 0
    for frame, records in sorted(cache_records(base_dir, frames, sequence=sequence).items()):
        published += publish(scene, records, frame)
    return published


def _owned_action(obj):
    animation = getattr(obj, "animation_data", None)
    action = getattr(animation, "action", None) if animation is not None else None
    if action is not None and action.get(rigid_scene.OWNED_ACTION_TAG):
        return action
    return None


RETIRED_ACTION_NAME = "FluidSim Dynamic (cleared)"


def _remove_retired_actions():
    import bpy

    for action in list(bpy.data.actions):
        if action.users == 0 and action.get(rigid_scene.OWNED_ACTION_TAG):
            bpy.data.actions.remove(action)
    return None


def _retire_action(action):
    """Unassigned owned action: free it later, never here.

    Reset runs inside frame-change handlers, whose depsgraph ``updates`` still
    reference the action; freeing it there crashed the next collider sync
    (use-after-free in ``update.id``). Renaming frees the name for the next
    run's action; a timer removes the orphan once the handler has returned,
    and a zero-user action is never saved.
    """
    action.name = RETIRED_ACTION_NAME
    try:
        import bpy

        if not bpy.app.timers.is_registered(_remove_retired_actions):
            bpy.app.timers.register(_remove_retired_actions, first_interval=0.0)
    except (ImportError, AttributeError, ValueError):
        pass


def clear_poses(scene) -> int:
    """Reset: drop the published keyframes and restore every authored pose."""
    cleared = 0
    for obj in list(scene.objects):
        rest = _stored_rest(obj)
        action = _owned_action(obj)
        if rest is None and action is None:
            continue
        if action is not None:
            obj.animation_data.action = None
            if action.users == 0:
                _retire_action(action)
        if rest is not None:
            obj.matrix_world = Matrix(rest.tolist())
            del obj[REST_PROP]
        cleared += 1
    return cleared


def begin_run(scene) -> None:
    """A new simulation run: clear published poses, make rigid bodies kinematic."""
    clear_poses(scene)
    for obj in rigid_scene.dynamic_collider_objects(scene.objects):
        ensure_kinematic(obj)
