"""Public registration facade for FluidSim operators.

Operators are defined in the ``operator_domains`` modules and nowhere else is
one named: every ``FLUIDSIM_OT_*`` class a domain module defines is registered,
so adding an operator is writing its class. They are also reachable as
attributes of this module (``operators.FLUIDSIM_OT_<name>``).
"""

from __future__ import annotations

import bpy

from .operator_domains import (
    cache,
    diagnostics,
    export,
    presets,
    rendering,
    session,
    source_roles,
)
from .operator_domains import common as _operator_runtime

CLASSES = tuple(
    cls
    for domain in (session, source_roles, diagnostics, export, presets, cache, rendering)
    for cls in vars(domain).values()
    if isinstance(cls, type)
    and cls.__module__ == domain.__name__
    and cls.__name__.startswith("FLUIDSIM_OT_")
)
globals().update({cls.__name__: cls for cls in CLASSES})


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    # One class that will not unregister must not leave every later class registered.
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - report and keep walking
            print(f"FluidSim: could not unregister {cls.__name__}: {exc!r}")
