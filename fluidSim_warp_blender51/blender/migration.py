"""One-way migration of domain settings from Scene.fluidsim to obj.fluidsim_domain.

A .blend saved before the Phase 2 split keeps its domain settings on the scene
group. ``Scene.fluidsim`` is now bound to ``FLUIDSIM_PG_scene``, which declares
none of them, so they survive only as hidden IDProperties. ``_on_load_post``
rebinds the scene pointer to the pre-split union class
(``properties.legacy_scene_settings``) around the copy so it can read them, and
moves them onto the domain object.
"""

from __future__ import annotations

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

_MIGRATION_SKIP = frozenset({"domain_object", "schema_version"})
_TARGET_SCHEMA_VERSION = 1

# Faces of a closed box over the 8 corners emitted by ``_default_domain_geometry``.
_DEFAULT_DOMAIN_FACES = (
    (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
    (1, 5, 7, 3), (3, 7, 6, 2), (2, 6, 4, 0),
)


def _default_domain_geometry():
    """Corner verts + faces for a box spanning the canonical fallback domain.

    A freshly created domain must reproduce the bounds an unset ``domain_object``
    used to yield, or the auto-created object would collapse ``domain_bounds`` to
    a degenerate AABB and break solver initialization.
    """
    from .solver_settings import DEFAULT_DOMAIN_MAX, DEFAULT_DOMAIN_MIN

    lo, hi = DEFAULT_DOMAIN_MIN, DEFAULT_DOMAIN_MAX
    verts = [(x, y, z)
             for x in (lo[0], hi[0])
             for y in (lo[1], hi[1])
             for z in (lo[2], hi[2])]
    return verts, list(_DEFAULT_DOMAIN_FACES)


def migrated_field_names(settings_cls) -> tuple[str, ...]:
    return tuple(
        name for name in getattr(settings_cls, "__annotations__", {})
        if name not in _MIGRATION_SKIP
    )


def _domain_settings_cls():
    from ..properties import FLUIDSIM_PG_domain

    return FLUIDSIM_PG_domain


def _is_authored(group, name) -> bool:
    """True when ``group`` stores its own value for ``name``.

    Blender groups answer through ``is_property_set``: an unset field only
    reads as its RNA default, and copying that default would overwrite
    whatever the domain object already holds. Plain value objects have no RNA
    default to fall back to, so every attribute they carry counts as authored.
    """
    is_set = getattr(group, "is_property_set", None)
    return True if is_set is None else bool(is_set(name))


def copy_domain_settings(source, target, field_names) -> None:
    for name in field_names:
        try:
            setattr(target, name, getattr(source, name))
        except Exception as exc:  # noqa: BLE001 — read-only/computed props tolerated
            print(f"FluidSim migration: skipped field {name!r}: {exc!r}")


def migrate_domain_settings_schema(scene, domain_obj) -> bool:
    """Move domain settings onto ``domain_obj.fluidsim_domain``. Idempotent, safe.

    Returns True iff it acted. Never stamps/clears when ``domain_obj`` is None
    (that would drop data); the caller owns resolving/creating the object. On a
    real scene the caller must hold ``properties.legacy_scene_settings()``: the
    narrow scene group cannot see the legacy values, and copying from it would
    stamp the guard having moved nothing.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None or int(getattr(fs, "schema_version", 0)) != 0:
        return False
    if domain_obj is None:
        return False
    field_names = migrated_field_names(_domain_settings_cls())
    hidden = [name for name in field_names if not hasattr(fs, name)]
    if hidden:
        raise RuntimeError(
            "FluidSim migration: scene.fluidsim does not expose the legacy "
            f"domain settings ({', '.join(hidden[:3])}, ...); migrate inside "
            "properties.legacy_scene_settings()")
    authored = tuple(name for name in field_names if _is_authored(fs, name))
    target = domain_obj.fluidsim_domain
    with _side_effect_free_copy():
        copy_domain_settings(fs, target, authored)
        target.schema_version = _TARGET_SCHEMA_VERSION
        fs.schema_version = _TARGET_SCHEMA_VERSION   # guard stamped LAST; never re-zeroed
    return True


def _side_effect_free_copy():
    """Scope that neutralizes RNA update side effects fired by the field copy.

    ``copy_domain_settings`` writes with ``setattr``, which invokes each field's
    RNA update callback in real Blender. ``live``/``cache_playback_enabled``
    would otherwise spawn a Solver Session mid-load and bind the scene to a
    bogus domain. ``playback.suppress_mode_transitions`` makes migration a pure
    value copy. Imported lazily and guarded so the pure bpy-stub tests — which
    call this without a live playback module — remain a plain no-op.
    """
    try:
        from . import playback

        return playback.suppress_mode_transitions()
    except Exception:  # noqa: BLE001 — pure/stub import context: no live callbacks fire
        from contextlib import nullcontext

        return nullcontext()


def create_domain_object(scene):
    """Create, link and designate a fresh FluidSim domain object.

    The single source of domain-object creation. Both the migration load
    handler (for an unset scene) and the ``VIEW3D_MT_add`` operator route
    through here so every domain reproduces the canonical fallback geometry
    and becomes the scene's designated ``fluidsim.domain_object``. Unlike
    :func:`_resolve_or_create_domain`, this always creates a new object and
    re-designates it -- the Add-menu path uses that to introduce an
    additional domain and make it the active one. Returns None only when
    ``scene.fluidsim`` is absent.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return None
    mesh = bpy.data.meshes.new("FluidSim Domain")
    verts, faces = _default_domain_geometry()
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("FluidSim Domain", mesh)
    scene.collection.objects.link(obj)
    fs.domain_object = obj
    return obj


def _resolve_or_create_domain(scene):
    """Return the scene's domain object, creating/designating one if unset.

    The only bpy-touching helper, keeping the pure layer testable. Returns
    None only when ``scene.fluidsim`` is absent (the pure orchestrator then
    no-ops, preserving data). Today's load-time model is exactly one domain
    per scene, so an unset domain deterministically gets a freshly created
    object.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return None
    existing = getattr(fs, "domain_object", None)
    if existing is not None:
        return existing
    return create_domain_object(scene)


def _needs_migration(scene) -> bool:
    fs = getattr(scene, "fluidsim", None)
    return fs is not None and int(getattr(fs, "schema_version", 0)) == 0


def _legacy_scene_settings():
    """Scope exposing a pre-split file's domain settings on scene.fluidsim."""
    from .. import properties

    return properties.legacy_scene_settings()


@persistent
def _on_load_post(_arg):
    pending = [
        scene for scene in getattr(getattr(bpy, "data", None), "scenes", ())
        if _needs_migration(scene)
    ]
    if not pending:
        return
    # One rebind around the whole loop: the pointer type is per class, so a
    # per-scene rebind would only repeat the same re-registration.
    with _legacy_scene_settings():
        for scene in pending:
            migrate_domain_settings_schema(scene, _resolve_or_create_domain(scene))


def register_handlers():
    unregister_handlers()
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    load_post = getattr(handlers, "load_post", None)
    if load_post is not None:
        load_post.append(_on_load_post)


def unregister_handlers():
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    load_post = getattr(handlers, "load_post", None)
    if load_post is None:
        return
    while _on_load_post in load_post:
        load_post.remove(_on_load_post)
