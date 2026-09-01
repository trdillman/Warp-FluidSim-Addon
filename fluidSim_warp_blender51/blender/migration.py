"""One-way migration of domain settings from Scene.fluidsim to obj.fluidsim_domain."""

from __future__ import annotations

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
