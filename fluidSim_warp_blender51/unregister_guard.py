"""One guarded teardown loop shared by every registration site.

A class that will not unregister must not leave every later class
registered, and a disable that partly failed must still say what it left
behind. Each site walks its classes in reverse through
:func:`unregister_classes`, which keeps going past a failure and reports all
failures together at the end.
"""

from __future__ import annotations


def unregister_classes(classes) -> list[str]:
    """Unregister ``classes`` in reverse order; return one line per failure."""
    import bpy  # per call: the add-on can outlive a reloaded bpy

    failed = []
    for cls in reversed(tuple(classes)):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # noqa: BLE001 - keep walking, report at the end
            failed.append(f"{getattr(cls, '__name__', cls)}: {exc!r}")
    if failed:
        print(f"FluidSim: could not unregister {len(failed)} class(es): "
              + "; ".join(failed))
    return failed
