"""Shared operator runtime guards and scene helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def _bridge():
    from ..blender import scene_bridge

    return scene_bridge


def _pending():
    from ..blender import pending_settings

    return pending_settings


def _solver_settings():
    from ..blender import solver_settings

    return solver_settings


def _configured_device():
    from ..preferences import configured_device

    return configured_device()


def _runtime_poll(cls) -> bool:
    """Block GPU actions until Warp and the selected CUDA GPU are ready."""
    from ..runtime.warp_runtime import runtime_status_for_device

    ready, message = runtime_status_for_device(_configured_device())
    if not ready:
        cls.poll_message_set(message)
    return ready


def _bake_poll(cls, context) -> bool:
    """Refuse while any Production Bake runs (owner decision D2).

    The Solver Session is a process singleton, so a bake in one scene owns it
    for every scene.
    """
    from ..blender import production_bake

    scene = getattr(context, "scene", None)
    if scene is not None and production_bake.bake_is_active(scene):
        cls.poll_message_set("Production Bake owns this scene")
        return False
    if production_bake.any_bake_active():
        cls.poll_message_set("A Production Bake is running in another scene")
        return False
    return True


def _mutation_poll(cls, context) -> bool:
    """Block solver mutations while a Production Bake owns the session."""
    if not _bake_poll(cls, context):
        return False
    return _runtime_poll(cls)


def _resolved_cache_path(scene) -> Path:
    """Resolve the exact authored cache target used by mutation dialogs."""
    raw = str(bpy.path.abspath(scene.fluidsim.cache_path) or "").strip()
    if not raw:
        raise ValueError("The cache path is empty")
    return Path(raw).expanduser().resolve()


def _owned_capacity_plan(scene):
    bridge = _bridge()
    getter = getattr(bridge, "capacity_plan_for_scene", None)
    if callable(getter):
        return getter(scene)
    sess = bridge.session()
    getter = getattr(sess, "capacity_plan_for_scene", None)
    return getter(scene) if callable(getter) else getattr(
        sess, "capacity_plan", None)


def _owned_capacity_signature(scene, plan):
    bridge = _bridge()
    getter = getattr(bridge, "capacity_plan_signature_for_scene", None)
    if callable(getter):
        return str(getter(scene))
    return str(getattr(plan, "scene_signature", ""))


class _FLUIDSIM_OT_session_base:
    """Grey out operators that cannot do anything without a live solver."""

    @classmethod
    def poll(cls, context):
        if not _mutation_poll(cls, context):
            return False
        if not _bridge().session().active:
            cls.poll_message_set("Initialize the solver first")
            return False
        return True
