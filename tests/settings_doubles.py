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
    Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51" / "properties.py"
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
