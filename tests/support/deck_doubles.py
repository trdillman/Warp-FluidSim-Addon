"""A recording stand-in for the ``bpy`` surface the Liquid Deck registers into.

The Deck (``blender/deck/``) adds classes, a viewport draw handler, add-on keymap
items, app handlers and timers. This double records each so a test can count
what is left after ``unregister()``; nothing here draws or runs a solver.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

PACKAGE = "fluidSim_warp_blender51"


def _prop(**kwargs):
    return kwargs.get("default")


class _Base:
    _poll_message = ""

    @classmethod
    def poll_message_set(cls, message, *_args):
        cls._poll_message = message

    def report(self, *_args, **_kwargs):
        return None


class _SpaceView3D:
    handlers: list = []

    @classmethod
    def draw_handler_add(cls, callback, args, region, draw_type):
        handle = object()
        cls.handlers.append((handle, callback, region, draw_type))
        return handle

    @classmethod
    def draw_handler_remove(cls, handle, region):
        for entry in list(cls.handlers):
            if entry[0] is handle and entry[2] == region:
                cls.handlers.remove(entry)
                return
        raise ValueError("draw handler is not registered")


class _KeymapItems(list):
    def new(self, idname, key_type, value, **modifiers):
        item = SimpleNamespace(idname=idname, type=key_type, value=value, **modifiers)
        self.append(item)
        return item


class _Keymaps(list):
    def new(self, name, space_type="EMPTY", **_kwargs):
        for keymap in self:
            if keymap.name == name:
                return keymap
        keymap = SimpleNamespace(name=name, space_type=space_type, keymap_items=_KeymapItems())
        self.append(keymap)
        return keymap


class _Timers:
    def __init__(self):
        self.registered = []

    def register(self, fn, first_interval=0.0, persistent=False):
        self.registered.append(fn)

    def unregister(self, fn):
        if fn not in self.registered:
            raise ValueError("timer is not registered")
        self.registered.remove(fn)

    def is_registered(self, fn):
        return fn in self.registered


def install_deck_bpy(monkeypatch, *, deck_mode=None, refuse=()):
    """Install a recording ``bpy``; ``deck_mode`` sets the add-on preference.

    ``refuse`` names classes whose ``unregister_class`` raises, to drive the
    registry's tolerant teardown.
    """
    bpy = types.ModuleType("bpy")
    registered = []

    def register_class(cls):
        if cls in registered:
            raise ValueError(f"{cls.__name__} already registered")
        registered.append(cls)

    def unregister_class(cls):
        if cls.__name__ in refuse:
            raise RuntimeError(f"refused {cls.__name__}")
        registered.remove(cls)

    _SpaceView3D.handlers = []

    class Operator(_Base):
        pass

    class Gizmo(_Base):
        def new_custom_shape(self, kind, verts):
            return (kind, tuple(verts))

        def draw_custom_shape(self, *_args, **_kwargs):
            return None

    class GizmoGroup(_Base):
        pass

    class AddonPreferences(_Base):
        pass

    bpy.types = SimpleNamespace(
        Operator=Operator, Gizmo=Gizmo, GizmoGroup=GizmoGroup,
        AddonPreferences=AddonPreferences, Panel=_Base, PropertyGroup=object,
        SpaceView3D=_SpaceView3D,
    )
    bpy.props = SimpleNamespace(**{
        name: (lambda **kwargs: _prop(**kwargs)) for name in (
            "BoolProperty", "BoolVectorProperty", "EnumProperty", "FloatProperty", "FloatVectorProperty",
            "IntProperty", "StringProperty", "PointerProperty", "CollectionProperty")
    })
    bpy.utils = SimpleNamespace(register_class=register_class,
                                unregister_class=unregister_class)

    def persistent(fn):
        return fn

    handlers = SimpleNamespace(persistent=persistent, load_post=[], load_pre=[],
                               save_pre=[], depsgraph_update_post=[])
    bpy.app = SimpleNamespace(handlers=handlers, timers=_Timers(), background=True)
    keyconfigs = SimpleNamespace(addon=SimpleNamespace(keymaps=_Keymaps()))
    addons = {}
    if deck_mode is not None:
        addons[PACKAGE] = SimpleNamespace(preferences=SimpleNamespace(deck_mode=deck_mode))
    bpy.context = SimpleNamespace(
        window_manager=SimpleNamespace(keyconfigs=keyconfigs, windows=()),
        preferences=SimpleNamespace(addons=addons),
        scene=None,
    )
    bpy.data = SimpleNamespace(filepath="", scenes=())
    bpy.ops = SimpleNamespace(fluidsim=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    bpy.registered = registered
    return bpy


def reload_deck():
    """Re-import the Deck package against the currently installed ``bpy``."""
    prefix = f"{PACKAGE}.blender.deck"
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)
    return importlib.import_module(prefix)


def keymap_item_count(bpy):
    return sum(len(km.keymap_items) for km in bpy.context.window_manager.keyconfigs.addon.keymaps)


def gizmo_group_count(bpy):
    return sum(1 for cls in bpy.registered if issubclass(cls, bpy.types.GizmoGroup))


def owned_app_handlers(bpy):
    return [fn for name in ("load_post", "load_pre", "save_pre", "depsgraph_update_post")
            for fn in getattr(bpy.app.handlers, name)
            if getattr(fn, "__module__", "").startswith(PACKAGE)]
