"""Scene doubles shaped like the split FluidSim settings groups.

Production reads domain settings only from the designated domain object's
``fluidsim_domain`` group (``scene_bridge.active_domain_settings``). A test that
builds one flat ``scene.fluidsim`` namespace therefore has to hand its
domain-owned attributes to a domain object double. ``split_settings`` does
that using the partition declared in ``properties.py``, so the doubles cannot
drift from the real classes.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

_PROPERTIES = (
    Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51" / "properties.py"
)


def _declared(class_name):
    tree = ast.parse(_PROPERTIES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return frozenset(
                statement.target.id for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
    raise LookupError(class_name)


SCENE_FIELDS = _declared("FLUIDSIM_PG_scene")
DOMAIN_FIELDS = _declared("FLUIDSIM_PG_domain")


def _property_default(call):
    """The default value of one ``XProperty(...)`` annotation, as Blender gives it."""
    import re

    keywords = {kw.arg: kw.value for kw in call.keywords}
    kind = call.func.id if isinstance(call.func, ast.Name) else ""
    if "default" in keywords:
        node = keywords["default"]
        try:
            return ast.literal_eval(node)
        except ValueError:
            pass
        text = ast.unparse(node)
        constant = re.fullmatch(r"_C\.(\w+)", text)
        if constant:
            from fluidSim_warp_blender51.sim import constants

            return getattr(constants, constant.group(1))
        member = re.fullmatch(r"\w+\.(\w+)\.name", text)
        if member:
            return member.group(1)
        if text == "PRODUCT_PRESET_NONE_NUMBER":
            # A dynamic enum's numeric default reads back as its item identifier.
            return "NONE"
        raise LookupError(f"cannot evaluate property default {text!r}")
    if kind == "FloatVectorProperty":
        size = keywords.get("size")
        return (0.0,) * (ast.literal_eval(size) if size is not None else 3)
    return {
        "BoolProperty": False, "IntProperty": 0, "FloatProperty": 0.0,
        "StringProperty": "", "EnumProperty": "", "PointerProperty": None,
    }[kind]


def _declared_defaults(class_name):
    tree = ast.parse(_PROPERTIES.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                statement.target.id: _property_default(statement.annotation)
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            }
    raise LookupError(class_name)


def full_settings(**overrides):
    """A flat settings double declaring every scene and domain field.

    Each field starts at its RNA default, as it would on the real property
    groups, and ``overrides`` replace chosen values. Production code reads a
    setting directly from the group that declares it, so a double that leaves a
    declared field out fails on the read instead of quietly supplying a default.
    Pass the result through :func:`split_settings` to hand the domain-owned
    fields to a domain object.
    """
    unknown = overrides.keys() - (SCENE_FIELDS | DOMAIN_FIELDS)
    if unknown:
        raise KeyError(f"not a declared FluidSim setting: {sorted(unknown)}")
    values = _declared_defaults("FLUIDSIM_PG_scene")
    values.update(_declared_defaults("FLUIDSIM_PG_domain"))
    values.update(overrides)
    return SimpleNamespace(**values)


def domain_group(**overrides):
    """A ``fluidsim_domain`` double: every domain field at its RNA default.

    Declares only the fields ``FLUIDSIM_PG_domain`` declares, so a read of a scene
    field through it raises, as it does on the real group.
    """
    unknown = overrides.keys() - DOMAIN_FIELDS
    if unknown:
        raise KeyError(f"not a declared domain setting: {sorted(unknown)}")
    values = _declared_defaults("FLUIDSIM_PG_domain")
    values.update(overrides)
    return SimpleNamespace(**values)


def scene_group(**overrides):
    """A ``scene.fluidsim`` double: every scene field at its RNA default."""
    unknown = overrides.keys() - SCENE_FIELDS
    if unknown:
        raise KeyError(f"not a declared scene setting: {sorted(unknown)}")
    values = _declared_defaults("FLUIDSIM_PG_scene")
    values.update(overrides)
    return SimpleNamespace(**values)


class IdentityMatrix:
    """``matrix_world`` of an object at the origin, unrotated and unscaled."""

    def __matmul__(self, vector):
        x, y, z = (float(component) for component in vector)
        return SimpleNamespace(x=x, y=y, z=z)

    def __getitem__(self, row):
        return tuple(1.0 if row == column else 0.0 for column in range(4))


def domain_object(**domain_values):
    """A designated domain object spanning the default domain box.

    Its world AABB equals ``DEFAULT_DOMAIN_MIN``/``DEFAULT_DOMAIN_MAX``, the
    bounds an unset ``domain_object`` resolves to, so designating it changes no
    solver bounds a test relied on.
    """
    from fluidSim_warp_blender51.blender.solver_settings import (
        DEFAULT_DOMAIN_MAX,
        DEFAULT_DOMAIN_MIN,
    )

    lo, hi = DEFAULT_DOMAIN_MIN, DEFAULT_DOMAIN_MAX
    corners = [(x, y, z)
               for x in (lo[0], hi[0])
               for y in (lo[1], hi[1])
               for z in (lo[2], hi[2])]
    return SimpleNamespace(
        name="FluidSim Domain",
        matrix_world=IdentityMatrix(),
        bound_box=corners,
        fluidsim_domain=SimpleNamespace(**domain_values),
    )


def split_settings(fs):
    """Move a flat settings double's domain-owned attributes onto its domain object.

    Returns ``fs`` so it can wrap a constructor in place. A namespace carrying no
    settings field (an object-role double) and a scene whose domain object
    already owns a ``fluidsim_domain`` group are returned untouched. Without a
    designated domain, one is created with :func:`domain_object`.
    """
    values = vars(fs)
    if not values.keys() & (SCENE_FIELDS | DOMAIN_FIELDS):
        return fs
    obj = values.get("domain_object")
    if obj is not None and hasattr(obj, "fluidsim_domain"):
        return fs
    moved = {name: values.pop(name) for name in list(values) if name in DOMAIN_FIELDS}
    if obj is None:
        fs.domain_object = domain_object(**moved)
    else:
        obj.fluidsim_domain = SimpleNamespace(**moved)
    return fs


def setting(fs, name):
    """Read ``name`` from whichever group of a split settings double owns it."""
    owner = fs.domain_object.fluidsim_domain if name in DOMAIN_FIELDS else fs
    return getattr(owner, name)
