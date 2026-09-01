"""One-way migration of domain settings from Scene.fluidsim to obj.fluidsim_domain."""

from __future__ import annotations

import bpy

_handlers = getattr(getattr(bpy, "app", None), "handlers", None)
persistent = getattr(_handlers, "persistent", lambda fn: fn)

_MIGRATION_SKIP = frozenset({"domain_object", "schema_version"})
_TARGET_SCHEMA_VERSION = 1


def migrated_field_names(settings_cls) -> tuple[str, ...]:
    return tuple(
        name for name in getattr(settings_cls, "__annotations__", {})
        if name not in _MIGRATION_SKIP
    )


def copy_domain_settings(source, target, field_names) -> None:
    for name in field_names:
        try:
            setattr(target, name, getattr(source, name))
        except Exception as exc:  # noqa: BLE001 — read-only/computed props tolerated
            print(f"FluidSim migration: skipped field {name!r}: {exc!r}")


def migrate_domain_settings_schema(scene, domain_obj) -> bool:
    """Move domain settings onto ``domain_obj.fluidsim_domain``. Idempotent, safe.

    Returns True iff it acted. Never stamps/clears when ``domain_obj`` is None
    (that would drop data); the caller owns resolving/creating the object.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None or int(getattr(fs, "schema_version", 0)) != 0:
        return False
    if domain_obj is None:
        return False
    target = domain_obj.fluidsim_domain
    field_names = migrated_field_names(type(fs))
    copy_domain_settings(fs, target, field_names)
    target.schema_version = _TARGET_SCHEMA_VERSION
    fs.schema_version = _TARGET_SCHEMA_VERSION   # guard stamped LAST; never re-zeroed
    return True


def _resolve_or_create_domain(scene):
    """Return the scene's domain object, creating/designating one if unset.

    The only bpy-touching helper, keeping the pure layer testable. Returns
    None only when ``scene.fluidsim`` is absent (the pure orchestrator then
    no-ops, preserving data). Today's model is exactly one domain per scene,
    so an unset domain deterministically gets a freshly created object.
    """
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return None
    existing = getattr(fs, "domain_object", None)
    if existing is not None:
        return existing
    mesh = bpy.data.meshes.new("FluidSim Domain")
    obj = bpy.data.objects.new("FluidSim Domain", mesh)
    scene.collection.objects.link(obj)
    fs.domain_object = obj
    return obj


@persistent
def _on_load_post(_arg):
    for scene in getattr(getattr(bpy, "data", None), "scenes", ()):
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
