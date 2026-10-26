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


# The domain-settings schema a domain object was saved under, stored raw in its
# ``fluidsim_domain`` group. 3 = the grid is authored as Resolution Divisions
# (default 64). Files saved before 3.1 carry no stamp, and an unstamped domain
# that stores neither ``voxel_size`` nor ``resolution_divisions`` may come from
# 2.x (default voxel 0.05) or from 3.0 (default 64 divisions): nothing in the
# file tells them apart, so it keeps the 3.0 default (decision B62). The stamp
# lets the next change to a default migrate untouched domains exactly.
# 4 = Multigrid is the default pressure preconditioner (4.0). A domain saved
# under an earlier schema that never stored a preconditioner ran Jacobi, so it
# keeps Jacobi on load and the Bake stage offers the switch once.
DOMAIN_SCHEMA_KEY = "domain_schema_version"
CURRENT_DOMAIN_SCHEMA = 4
MULTIGRID_DEFAULT_SCHEMA = 4
# Raw flag left by migrate_domain_preconditioner; cleared by either notice button.
KEPT_JACOBI_KEY = "kept_jacobi_preconditioner"


def _writable_domain_group(obj):
    """``obj``'s raw ``fluidsim_domain`` group, created when missing."""
    getter = getattr(obj, "bl_system_properties_get", None)
    if not callable(getter):
        return None
    try:
        stored = getter(do_create=True)
        if stored is None:
            return None
        if stored.get("fluidsim_domain") is None:
            stored["fluidsim_domain"] = {}
        return stored.get("fluidsim_domain")
    except (AttributeError, KeyError, TypeError):
        return None


def _local_domain_objects(objects=None, scenes=None) -> list:
    """Each scene's designated domain plus every object storing a domain group.

    Linked library data belongs to its own file and is left out.
    """
    from .scene_access import active_domain_object

    data = getattr(bpy, "data", None)
    objects = getattr(data, "objects", ()) if objects is None else objects
    scenes = getattr(data, "scenes", ()) if scenes is None else scenes
    candidates = {}
    for obj in [active_domain_object(scene) for scene in scenes] + [
            obj for obj in objects
            if _stored_scene_group(obj, "fluidsim_domain") is not None]:
        if obj is not None and getattr(obj, "library", None) is None:
            pointer = getattr(obj, "as_pointer", None)
            candidates[pointer() if callable(pointer) else id(obj)] = obj
    return list(candidates.values())


def stamp_domain_schema(objects=None, scenes=None) -> int:
    """Stamp every local domain object with :data:`CURRENT_DOMAIN_SCHEMA`.

    Covers each scene's designated domain, even one whose group was never
    written, and any object that stores a domain group. Linked library data
    belongs to its own file and is left alone. Returns the number stamped.
    """
    stamped = 0
    for obj in _local_domain_objects(objects, scenes):
        group = _writable_domain_group(obj)
        if group is None or group.get(DOMAIN_SCHEMA_KEY) == CURRENT_DOMAIN_SCHEMA:
            continue
        group[DOMAIN_SCHEMA_KEY] = CURRENT_DOMAIN_SCHEMA
        stamped += 1
    return stamped


# Saving stamps each local scene's group with the release that wrote it, so a
# load can tell a file saved by 4.0 or later (the Liquid Deck's AUTO mode shows
# the Deck) from one saved by 3.x (AUTO keeps the Deck off, as 3.x looked).
SCENE_SCHEMA_KEY = "scene_schema_version"
CURRENT_SCENE_SCHEMA = 4


def scene_saved_by_deck_release(scene) -> bool:
    """Whether ``scene`` was saved by FluidSim 4.0 or later."""
    group = _stored_scene_group(scene)
    try:
        return group is not None and int(group.get(SCENE_SCHEMA_KEY, 0)) >= CURRENT_SCENE_SCHEMA
    except (AttributeError, TypeError, ValueError):
        return False


def stamp_scene_schema(scenes=None) -> int:
    """Stamp every local scene with :data:`CURRENT_SCENE_SCHEMA`; returns the number stamped."""
    data = getattr(bpy, "data", None)
    scenes = getattr(data, "scenes", ()) if scenes is None else scenes
    stamped = 0
    for scene in scenes:
        if getattr(scene, "library", None) is not None:
            continue
        getter = getattr(scene, "bl_system_properties_get", None)
        if not callable(getter):
            continue
        try:
            stored = getter(do_create=True)
            if stored is None:
                continue
            if stored.get("fluidsim") is None:
                stored["fluidsim"] = {}
            group = stored.get("fluidsim")
            if group.get(SCENE_SCHEMA_KEY) == CURRENT_SCENE_SCHEMA:
                continue
            group[SCENE_SCHEMA_KEY] = CURRENT_SCENE_SCHEMA
        except (AttributeError, KeyError, TypeError):
            continue
        stamped += 1
    return stamped


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


def migrate_domain_preconditioner(objects=None, scenes=None) -> int:
    """Keep Jacobi on a domain saved before Multigrid became the default.

    A domain stamped below :data:`MULTIGRID_DEFAULT_SCHEMA` (or unstamped) that
    never stored ``pressure_preconditioner`` ran the old Jacobi default. It is
    given Jacobi explicitly, so its results do not change on open, and is
    flagged for the one-time Bake-stage notice. A stored choice is kept.
    Idempotent: the explicit value is stored. Returns the number changed.
    """
    from contextlib import ExitStack

    from . import pending_settings

    pending = []
    for obj in _local_domain_objects(objects, scenes):
        stored = _stored_scene_group(obj, "fluidsim_domain")
        try:
            if stored is not None and (
                    "pressure_preconditioner" in stored
                    or int(stored.get(DOMAIN_SCHEMA_KEY, 0) or 0)
                    >= MULTIGRID_DEFAULT_SCHEMA):
                continue
        except (AttributeError, TypeError, ValueError):
            continue
        group = getattr(obj, "fluidsim_domain", None)
        if group is not None:
            pending.append((obj, group))
    with ExitStack() as stack:
        seen = set()
        for obj, _group in pending:
            for scene in getattr(obj, "users_scene", ()) or ():
                if id(scene) not in seen:
                    seen.add(id(scene))
                    stack.enter_context(pending_settings.suppress_callbacks(scene))
        for obj, group in pending:
            group.pressure_preconditioner = "JACOBI"
            raw = _writable_domain_group(obj)
            if raw is not None:
                raw[KEPT_JACOBI_KEY] = 1
    return len(pending)


def shows_kept_jacobi_notice(settings) -> bool:
    """True while a migrated domain still runs the Jacobi it was given."""
    if settings is None:
        return False
    stored = _stored_scene_group(getattr(settings, "id_data", None), "fluidsim_domain")
    try:
        flagged = stored is not None and bool(stored.get(KEPT_JACOBI_KEY))
    except (AttributeError, TypeError):
        return False
    return flagged and getattr(settings, "pressure_preconditioner", "") == "JACOBI"


def clear_kept_jacobi_notice(settings) -> None:
    raw = _writable_domain_group(getattr(settings, "id_data", None))
    if raw is not None and raw.get(KEPT_JACOBI_KEY) is not None:
        del raw[KEPT_JACOBI_KEY]


def draw_kept_jacobi_notice(layout, settings) -> None:
    if not shows_kept_jacobi_notice(settings):
        return
    box = layout.box()
    box.label(text="4.0 default is Multigrid; this file keeps Jacobi", icon="INFO")
    box.label(text="Multigrid holds deep liquid at its true height.")
    row = box.row(align=True)
    switch = row.operator("fluidsim.preconditioner_notice", text="Use Multigrid")
    switch.action = "MULTIGRID"
    keep = row.operator("fluidsim.preconditioner_notice", text="Keep Jacobi")
    keep.action = "KEEP"


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
    migrate_domain_preconditioner()
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
        from . import sources

        migrate_domain_resolution(objects=objects, scenes=scenes)
        migrate_domain_preconditioner(objects=objects, scenes=imported_scenes)
        migrate_removed_particle_color_mode(scenes=imported_scenes)
        sources.migrate_source_schema(objects=objects, scenes=scenes)


@persistent
def _on_compat_load_pre(_arg):
    _deferred_imports.clear()


@persistent
def _on_save_pre(_arg):
    stamp_domain_schema()
    stamp_scene_schema()


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
    ("save_pre", _on_save_pre),
    ("depsgraph_update_post", _on_imported_domain_update),
)


def register_handlers():
    from ..unregister_guard import handler_registration

    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    with handler_registration(
        (getattr(handlers, name, None), callback)
        for name, callback in _COMPAT_HANDLERS
    ):
        try:
            # The file that is already open; sources migrate it on their own.
            migrate_domain_resolution()
            # Only a file read from disk: a domain made in this session is new.
            if getattr(getattr(bpy, "data", None), "filepath", ""):
                migrate_domain_preconditioner()
            migrate_removed_particle_color_mode()
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
