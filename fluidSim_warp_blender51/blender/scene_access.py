"""Scene access: the Active Domain and a Scene's identity.

Domain settings live on the designated domain object's ``fluidsim_domain``
group; ``scene.fluidsim`` holds only scene-level state (see ``CONTEXT.md``,
"Active Domain"). Every reader goes through ``active_domain_object`` /
``active_domain_settings`` so none needs to know which object is the domain.

This is a leaf module: it imports neither ``bpy`` nor any other FluidSim
module, so any module may import it at the top. That is what lets it replace
the function-level ``from .scene_bridge import active_domain_settings`` imports
that existed only to avoid an import cycle with the Solver Session owner.
"""

from __future__ import annotations


def active_domain_object(scene):
    """Return the scene's designated domain object, or ``None``."""
    fs = getattr(scene, "fluidsim", None)
    if fs is None:
        return None
    return getattr(fs, "domain_object", None)


def active_domain_settings(scene):
    """Return the designated domain object's ``fluidsim_domain`` group.

    Domain settings exist only on the domain object: ``scene.fluidsim`` is a
    ``FLUIDSIM_PG_scene`` and declares none of them, so there is no fallback.
    ``None`` means the scene has no FluidSim settings or no designated domain
    object; callers treat that as "no domain", never as a cue to read
    ``scene.fluidsim``.
    """
    obj = active_domain_object(scene)
    if obj is None:
        return None
    return getattr(obj, "fluidsim_domain", None)


def scene_identity(scene):
    """Return a key that identifies ``scene`` across Python wrappers.

    Blender hands out a fresh Python wrapper per access, so ``id(scene)`` is not
    stable; the datablock pointer is. A dead datablock (``ReferenceError``), a
    zero pointer, or a test double without ``as_pointer`` falls back to
    ``id(scene)``.
    """
    pointer = getattr(scene, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
            if value:
                return value
        except (TypeError, ValueError, ReferenceError):
            pass
    return id(scene)


def scene_get(scene, key: str, default=""):
    """Read a scene custom property, tolerating objects without ``get``."""
    try:
        return scene.get(key, default)
    except AttributeError:
        return getattr(scene, key, default)


def scene_set(scene, key: str, value):
    """Write a scene custom property, tolerating objects without item assignment."""
    try:
        scene[key] = value
    except TypeError:
        setattr(scene, key, value)
