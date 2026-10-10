"""Recognize .blend files saved before FluidSim 1.1.

FluidSim 2.0 opens files saved by 1.1 or later. Before 1.1 the domain settings
lived on the scene group. 1.1 moved them onto the domain object on load and
stamped ``schema_version = 1`` on the scene group. 2.0 no longer reads the
scene copy, so a file that never went through 1.1 opens with default domain
settings. Such a file is recognized from its stored scene group, and the user
is told to open and re-save it in 1.1 first.

1.0 also stored Custom Warp Snippets on the scene group. 1.1 copied them to
disabled Custom Effects on load and marked the effect stack
``migration_complete``. 2.0 no longer reads snippets, so a stored snippet list
without that mark is flagged the same way, and admission refuses the scene
rather than run without the user's code.
"""

from __future__ import annotations

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

_deferred_imports = {}
_migrating_imports = False

PRE_BRIDGE_MESSAGE = (
    "Saved before FluidSim 1.1: domain settings were not carried over. "
    "Open and re-save this file in FluidSim 1.1 first")
UNMIGRATED_SNIPPETS_MESSAGE = (
    "Saved before FluidSim 1.1: its Custom Warp Snippets were not carried over "
    "to Custom Effects. Open and re-save this file in FluidSim 1.1 first")


def _domain_only_fields() -> frozenset[str]:
    from ..properties import FLUIDSIM_PG_domain, FLUIDSIM_PG_scene

    return frozenset(FLUIDSIM_PG_domain.__annotations__) - frozenset(
        FLUIDSIM_PG_scene.__annotations__)


def _stored_scene_group(scene, name="fluidsim"):
    """A scene group's stored values, including keys no class declares."""
    getter = getattr(scene, "bl_system_properties_get", None)
    if not callable(getter):
        return None
    try:
        stored = getter()
        return None if stored is None else stored.get(name)
    except (AttributeError, KeyError, TypeError):
        return None


def has_unmigrated_snippets(scene) -> bool:
    """True when ``scene`` holds 1.0 snippets that 1.1 never copied."""
    stored = _stored_scene_group(scene)
    try:
        if stored is None or not len(stored.get("snippets", ())):
            return False
        effects = _stored_scene_group(scene, "fluidsim_effects")
        return not (effects is not None and effects.get("migration_complete"))
    except (AttributeError, TypeError):
        return False


def is_pre_bridge_scene(scene) -> bool:
    """True when ``scene`` holds domain settings that only 1.1 could move."""
    if has_unmigrated_snippets(scene):
        return True
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
    if has_unmigrated_snippets(scene):
        box.label(text="Custom Warp Snippets were not carried over.")
    else:
        box.label(text="Domain settings were not carried over.")
    box.label(text="Open and re-save this file in FluidSim 1.1 first.")


def migrate_domain_resolution(objects=None, scenes=None, *, _evaluate_views=True) -> int:
    """Give a pre-3.0 domain object the Resolution Divisions its voxel size implied.

    Before 3.0 the grid was authored as an absolute ``voxel_size``. It is now
    derived from ``resolution_divisions``, so a domain group that stores a
    ``voxel_size`` and no ``resolution_divisions`` gets
    ``round(longest axis / voxel_size)``. The legacy value is read from the stored
    raw group, which keeps keys the class no longer stores. Idempotent: a group
    that already stores divisions is left alone. Returns the number migrated.
    """
    from contextlib import ExitStack

    from ..properties import _domain_longest_axis
    from ..sim.quality_contract import divisions_for_voxel_size
    from . import pending_settings

    data = getattr(bpy, "data", None)
    objects = getattr(data, "objects", ()) if objects is None else objects
    scenes = getattr(data, "scenes", ()) if scenes is None else scenes
    # A freshly loaded object's matrix_world is not evaluated yet (it reads as
    # identity), and the longest axis is measured from it.
    if _evaluate_views:
        for scene in scenes:
            for view_layer in getattr(scene, "view_layers", ()):
                view_layer.update()
    pending = []
    for obj in objects:
        if getattr(obj, "library", None) is not None:
            continue
        stored = _stored_scene_group(obj, "fluidsim_domain")
        try:
            if stored is None or "resolution_divisions" in stored or (
                    "voxel_size" not in stored):
                continue
            voxel = float(stored["voxel_size"])
        except (AttributeError, TypeError, ValueError):
            continue
        group = getattr(obj, "fluidsim_domain", None)
        if group is not None and voxel > 0.0:
            if getattr(obj, "users_scene", None) == ():
                # Library-load can return an unlinked object. Its matrix_world
                # is still identity, including transformed parents. Preserve
                # the raw voxel size until Blender evaluates its first scene.
                _deferred_imports[obj.as_pointer()] = obj
                continue
            pending.append((group, voxel))
    with ExitStack() as stack:
        for scene in scenes:
            stack.enter_context(pending_settings.suppress_callbacks(scene))
        for group, voxel in pending:
            group.resolution_divisions = divisions_for_voxel_size(
                voxel, _domain_longest_axis(group))
    return len(pending)


def migrate_removed_particle_color_mode(scenes=None) -> int:
    """Send a scene that stored the removed Time Offset particle colour mode to ``NONE``.

    ``TIME_OFFSET`` was the last ``particle_attribute_mode`` item. Blender keeps
    an enum as its integer index, so after the item was removed a stored index
    past the end of the list reads back as an empty string, which no colour
    branch recognizes. Such a scene is set back to ``NONE`` (Material).
    Idempotent. Returns the number of scenes changed.
    """
    from ..properties import PARTICLE_ATTR_ITEMS
    from . import pending_settings

    scenes = getattr(getattr(bpy, "data", None), "scenes", ()) if scenes is None else scenes
    stale = []
    for scene in scenes:
        stored = _stored_scene_group(scene)
        try:
            index = None if stored is None else stored.get("particle_attribute_mode")
            if index is None or int(index) < len(PARTICLE_ATTR_ITEMS):
                continue
        except (AttributeError, TypeError, ValueError):
            continue
        if getattr(scene, "fluidsim", None) is not None:
            stale.append(scene)
    for scene in stale:
        with pending_settings.suppress_callbacks(scene):
            scene.fluidsim.particle_attribute_mode = "NONE"
    return len(stale)


@persistent
def _on_load_post(_arg):
    migrate_domain_resolution()
    for scene in getattr(getattr(bpy, "data", None), "scenes", ()):
        if has_unmigrated_snippets(scene):
            print(f"FluidSim: scene {scene.name!r}: {UNMIGRATED_SNIPPETS_MESSAGE}")
        elif is_pre_bridge_scene(scene):
            print(f"FluidSim: scene {scene.name!r}: {PRE_BRIDGE_MESSAGE}")
    migrate_removed_particle_color_mode()


@persistent
def _on_blend_import_post(context):
    """Migrate only newly appended local IDs, never links or re-used data.

    Blender supplies dependent objects in import_items as well as the root
    Scene/Collection. Do not sweep bpy.data: a later append must not rewrite
    an existing domain or library data.
    """
    if "LINK" in getattr(context, "options", ()):
        return
    objects = []
    imported_scenes = []
    scenes = []
    for item in getattr(context, "import_items", ()):
        if getattr(item, "append_action", "UNSET") not in {"MAKE_LOCAL", "COPY_LOCAL"}:
            continue
        data = getattr(item, "id", None)
        if data is None or getattr(data, "library", None) is not None:
            continue
        if hasattr(data, "fluidsim_domain"):
            objects.append(data)
            for scene in getattr(data, "users_scene", ()):
                if scene not in scenes:
                    scenes.append(scene)
        elif hasattr(data, "view_layers"):
            imported_scenes.append(data)
            if data not in scenes:
                scenes.append(data)
    if objects or imported_scenes:
        migrate_domain_resolution(objects=objects, scenes=scenes)
        migrate_removed_particle_color_mode(scenes=imported_scenes)


@persistent
def _on_compat_load_pre(_arg):
    _deferred_imports.clear()


@persistent
def _on_imported_domain_update(scene, _depsgraph):
    """Resolve only queued, newly linked domains after dependency evaluation."""
    global _migrating_imports
    if _migrating_imports or not _deferred_imports:
        return
    _migrating_imports = True
    try:
        for key, obj in tuple(_deferred_imports.items()):
            try:
                if obj.library is not None:
                    _deferred_imports.pop(key, None)
                    continue
                if scene not in obj.users_scene:
                    continue
                migrate_domain_resolution(objects=(obj,), scenes=(scene,),
                                          _evaluate_views=False)
                _deferred_imports.pop(key, None)
            except ReferenceError:  # removed before ever being linked
                _deferred_imports.pop(key, None)
    finally:
        _migrating_imports = False


_COMPAT_HANDLERS = (
    ("load_post", _on_load_post),
    ("blend_import_post", _on_blend_import_post),
    ("load_pre", _on_compat_load_pre),
    ("depsgraph_update_post", _on_imported_domain_update),
)


def register_handlers():
    unregister_handlers()
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    for name, callback in _COMPAT_HANDLERS:
        callbacks = getattr(handlers, name, None)
        if callbacks is not None:
            callbacks.append(callback)
    try:
        migrate_domain_resolution()  # the file that is already open
    except (AttributeError, RuntimeError):  # bpy.data is restricted during enable
        pass


def unregister_handlers():
    _deferred_imports.clear()
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    for name, callback in _COMPAT_HANDLERS:
        callbacks = getattr(handlers, name, None)
        if callbacks is not None:
            while callback in callbacks:
                callbacks.remove(callback)
