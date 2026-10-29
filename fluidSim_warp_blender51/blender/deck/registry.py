"""The one owner of every Liquid Deck draw handler, keymap item, app handler and class.

Anything the Deck (or the Liquid Preview lane, which shares this registry) adds to
Blender is added through :data:`REGISTRY`, so ``unregister_all`` can take all of
it out again: on add-on disable, reload, or a failed registration. Teardown walks
every entry even when one removal raises, forgets the failed entry (a later
``register`` starts clean), prints each failure, and returns them.

Draw handlers are keyed: registering the same key again replaces the old handler
instead of stacking a second one, which is how handler leaks across reloads are
prevented.
"""

from __future__ import annotations


def _bpy():
    import bpy

    return bpy


class DeckRegistry:
    def __init__(self):
        self._classes = []
        self._draw_handlers = {}
        self._keymap_items = []
        self._app_handlers = []
        self._timers = {}

    # Classes ---------------------------------------------------------------

    def register_class(self, cls):
        _bpy().utils.register_class(cls)
        self._classes.append(cls)
        return cls

    # Draw handlers ---------------------------------------------------------

    def add_draw_handler(self, key, space_type, callback, args, region, draw_type):
        """Add ``callback`` to ``space_type`` under ``key``, replacing a previous one."""
        self.remove_draw_handler(key)
        handle = space_type.draw_handler_add(callback, tuple(args), region, draw_type)
        self._draw_handlers[key] = (space_type, handle, region)
        return handle

    def remove_draw_handler(self, key) -> bool:
        entry = self._draw_handlers.pop(key, None)
        if entry is None:
            return False
        space_type, handle, region = entry
        space_type.draw_handler_remove(handle, region)
        return True

    def has_draw_handler(self, key) -> bool:
        return key in self._draw_handlers

    def draw_handle(self, key):
        """The handle registered under ``key`` now, or ``None``; tells a modal whether it
        still owns its handler."""
        entry = self._draw_handlers.get(key)
        return entry[1] if entry is not None else None

    # Keymaps, app handlers, timers -----------------------------------------

    def add_keymap_item(self, keymap_name, space_type, idname, key_type, value, **modifiers):
        """Add an add-on keymap item; ``None`` when there is no add-on keyconfig
        (``--background``)."""
        keyconfigs = getattr(_bpy().context.window_manager, "keyconfigs", None)
        addon = getattr(keyconfigs, "addon", None)
        if addon is None:
            return None
        keymap = addon.keymaps.new(name=keymap_name, space_type=space_type)
        item = keymap.keymap_items.new(idname, key_type, value, **modifiers)
        self._keymap_items.append((keymap, item))
        return item

    def add_app_handler(self, list_name, fn):
        handlers = getattr(_bpy().app.handlers, list_name)
        if fn not in handlers:
            handlers.append(fn)
            self._app_handlers.append((list_name, fn))

    def add_timer(self, key, fn, first_interval=0.0, persistent=True):
        """Register a timer under ``key``; persistent by default, so it survives file loads."""
        timers = _bpy().app.timers
        self.remove_timer(key)
        timers.register(fn, first_interval=first_interval, persistent=persistent)
        self._timers[key] = fn

    def has_timer(self, key) -> bool:
        return key in self._timers

    def remove_timer(self, key) -> bool:
        fn = self._timers.pop(key, None)
        if fn is None:
            return False
        timers = _bpy().app.timers
        # A one-shot timer that returned None is already gone.
        if timers.is_registered(fn):
            timers.unregister(fn)
        return True

    # Teardown ----------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "classes": len(self._classes),
            "draw_handlers": len(self._draw_handlers),
            "keymap_items": len(self._keymap_items),
            "app_handlers": len(self._app_handlers),
            "timers": len(self._timers),
        }

    def unregister_all(self) -> list[str]:
        """Remove everything, newest first, and return the removals that failed."""
        bpy = _bpy()
        failures = []

        def attempt(label, action):
            try:
                action()
            except Exception as exc:  # noqa: BLE001 - keep walking, report at the end
                failures.append(f"{label}: {exc!r}")

        for key in list(self._timers):
            attempt(f"timer {key}", lambda key=key: self.remove_timer(key))
        self._timers.clear()

        for list_name, fn in reversed(self._app_handlers):
            def remove_handler(list_name=list_name, fn=fn):
                handlers = getattr(bpy.app.handlers, list_name)
                while fn in handlers:
                    handlers.remove(fn)
            attempt(f"app handler {list_name}:{getattr(fn, '__name__', fn)}", remove_handler)
        self._app_handlers.clear()

        for keymap, item in reversed(self._keymap_items):
            attempt(f"keymap item {getattr(item, 'idname', item)}",
                    lambda keymap=keymap, item=item: keymap.keymap_items.remove(item))
        self._keymap_items.clear()

        for key in list(self._draw_handlers):
            attempt(f"draw handler {key}", lambda key=key: self.remove_draw_handler(key))
        self._draw_handlers.clear()

        for cls in reversed(self._classes):
            attempt(f"class {cls.__name__}",
                    lambda cls=cls: bpy.utils.unregister_class(cls))
        self._classes.clear()

        for failure in failures:
            print(f"FluidSim Liquid Deck: could not unregister {failure}")
        return failures


REGISTRY = DeckRegistry()
