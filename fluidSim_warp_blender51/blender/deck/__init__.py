"""Liquid Deck: the in-viewport bar, command palette, radial wheel and gizmos.

The Deck is a second way to reach controls the panels already have. It stores
no settings of its own: every button calls an existing operator and every
gizmo or inline edit writes an existing property (see ``controls.py``).

Visibility is decided by two things the user owns:

- the add-on preference ``deck_mode``: ``OFF`` hides it everywhere, ``ON`` shows
  it wherever the scene's ``deck_shown`` flag is set, and ``AUTO`` (the default)
  also shows it, except that a file saved before FluidSim 4.0 opens with the
  flag cleared, so files from 3.x look as they did;
- the scene flag ``scene.fluidsim.deck_shown`` (the header toggle).

The load-post handler only reads the file into the Deck (it clears the flag of
pre-4.0 scenes under ``AUTO``); nothing in the Deck writes the file's state
back on load.

Everything the Deck adds to Blender goes through ``registry.REGISTRY``.
Submodules load on first use, so this package imports without ``gpu``.
"""

from __future__ import annotations

import importlib

try:
    import bpy as _bpy

    persistent = _bpy.app.handlers.persistent
except (ImportError, AttributeError):  # pragma: no cover - outside Blender
    def persistent(fn):
        return fn

ADDON_PACKAGE = __name__.rsplit(".", 2)[0]
DECK_MODES = ("AUTO", "ON", "OFF")
_SUBMODULES = frozenset({"_draw", "bar", "controls", "gizmos", "modal", "palette",
                         "registry", "tokens", "wheel"})


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def deck_mode(context) -> str:
    """The ``deck_mode`` preference; ``OFF`` when the add-on preferences are missing."""
    try:
        addon = context.preferences.addons.get(ADDON_PACKAGE)
        mode = str(addon.preferences.deck_mode)
    except (AttributeError, KeyError, TypeError):
        return "OFF"
    return mode if mode in DECK_MODES else "OFF"


def deck_visible(context, scene=None) -> bool:
    """Whether the Deck is drawn for ``scene`` (default: the context scene)."""
    if deck_mode(context) == "OFF":
        return False
    scene = scene if scene is not None else getattr(context, "scene", None)
    return bool(getattr(getattr(scene, "fluidsim", None), "deck_shown", False))


def sync_runtime(context=None) -> None:
    """Redraw 3D Views after the Deck's visibility or the scene changed."""
    import bpy

    from .bar import forget_scene_state

    forget_scene_state()
    context = context or bpy.context
    for window in getattr(getattr(context, "window_manager", None), "windows", ()):
        for area in getattr(getattr(window, "screen", None), "areas", ()):
            if area.type == "VIEW_3D":
                area.tag_redraw()


@persistent
def _on_deck_load_post(_filepath):
    """Read the loaded file into the Deck: under AUTO, pre-4.0 scenes open hidden."""
    import bpy

    from ..file_compat import scene_saved_by_deck_release

    context = bpy.context
    data = getattr(bpy, "data", None)
    if deck_mode(context) == "AUTO" and getattr(data, "filepath", ""):
        for scene in getattr(data, "scenes", ()):
            settings = getattr(scene, "fluidsim", None)
            if settings is None or scene_saved_by_deck_release(scene):
                continue
            if getattr(settings, "deck_shown", False):
                settings.deck_shown = False
    try:
        sync_runtime(context)
    except Exception as exc:  # noqa: BLE001 - a load handler must never raise
        print(f"FluidSim Liquid Deck: could not refresh the viewport: {exc!r}")


def register():
    import bpy

    from . import bar, gizmos, modal, palette, wheel
    from .registry import REGISTRY

    try:
        for cls in palette.CLASSES + wheel.CLASSES + modal.CLASSES + gizmos.CLASSES:
            REGISTRY.register_class(cls)
        REGISTRY.add_draw_handler("deck_bar", bpy.types.SpaceView3D, bar.draw_deck, (),
                                  "WINDOW", "POST_PIXEL")
        REGISTRY.add_keymap_item("3D View", "VIEW_3D", palette.FLUIDSIM_OT_command_palette.bl_idname,
                                 "K", "PRESS", ctrl=True)
        REGISTRY.add_keymap_item("3D View", "VIEW_3D", wheel.FLUIDSIM_OT_radial_wheel.bl_idname,
                                 "RIGHTMOUSE", "PRESS", alt=True)
        REGISTRY.add_app_handler("load_post", _on_deck_load_post)
        modal.start(REGISTRY)
    except Exception:
        REGISTRY.unregister_all()
        raise


def unregister():
    from . import bar, modal
    from .registry import REGISTRY

    modal.new_generation()
    REGISTRY.unregister_all()
    bar.forget_scene_state()
