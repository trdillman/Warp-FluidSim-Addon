"""Recognize .blend files saved before FluidSim 1.1.

FluidSim 2.0 opens files saved by 1.1 or later. Before 1.1 the domain settings
lived on the scene group. 1.1 moved them onto the domain object on load and
stamped ``schema_version = 1`` on the scene group. 2.0 no longer reads the
scene copy, so a file that never went through 1.1 opens with default domain
settings. Such a file is recognized from its stored scene group, and the user
is told to open and re-save it in 1.1 first.
"""

from __future__ import annotations

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

PRE_BRIDGE_MESSAGE = (
    "Saved before FluidSim 1.1: domain settings were not carried over. "
    "Open and re-save this file in FluidSim 1.1 first")


def _domain_only_fields() -> frozenset[str]:
    from ..properties import FLUIDSIM_PG_domain, FLUIDSIM_PG_scene

    return frozenset(FLUIDSIM_PG_domain.__annotations__) - frozenset(
        FLUIDSIM_PG_scene.__annotations__)


def _stored_scene_group(scene):
    """The scene group's stored values, including keys no class declares."""
    getter = getattr(scene, "bl_system_properties_get", None)
    if not callable(getter):
        return None
    try:
        stored = getter()
        return None if stored is None else stored.get("fluidsim")
    except (AttributeError, KeyError, TypeError):
        return None


def is_pre_bridge_scene(scene) -> bool:
    """True when ``scene`` holds domain settings that only 1.1 could move."""
    stored = _stored_scene_group(scene)
    if stored is None:
        return False
    try:
        if int(stored.get("schema_version", 0)) >= 1:
            return False
        keys = set(stored.keys())
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(keys & _domain_only_fields())


def draw_pre_bridge_warning(layout, scene) -> None:
    if not is_pre_bridge_scene(scene):
        return
    box = layout.box()
    box.alert = True
    box.label(text="Saved before FluidSim 1.1", icon="ERROR")
    box.label(text="Domain settings were not carried over.")
    box.label(text="Open and re-save this file in FluidSim 1.1 first.")


@persistent
def _on_load_post(_arg):
    for scene in getattr(getattr(bpy, "data", None), "scenes", ()):
        if is_pre_bridge_scene(scene):
            print(f"FluidSim: scene {scene.name!r}: {PRE_BRIDGE_MESSAGE}")


def register_handlers():
    unregister_handlers()
    load_post = getattr(getattr(getattr(bpy, "app", None), "handlers", None),
                        "load_post", None)
    if load_post is not None:
        load_post.append(_on_load_post)


def unregister_handlers():
    load_post = getattr(getattr(getattr(bpy, "app", None), "handlers", None),
                        "load_post", None)
    if load_post is None:
        return
    while _on_load_post in load_post:
        load_post.remove(_on_load_post)
