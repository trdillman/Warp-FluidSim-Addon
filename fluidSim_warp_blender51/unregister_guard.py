"""One guarded teardown loop shared by every registration site.

A class that will not unregister must not leave every later class
registered, and a disable that partly failed must still say what it left
behind. Each site walks its classes in reverse through
:func:`unregister_classes`, which keeps going past a failure and reports all
failures together at the end.
"""

from __future__ import annotations

from contextlib import contextmanager


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


@contextmanager
def class_registration(classes):
    """Keep completed classes only if the entire local registration step succeeds.

    The journal is local to this call. A refusing (including already registered)
    class never enters it, so rollback cannot unregister pre-existing classes.
    """
    import bpy

    completed = []
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
            completed.append(cls)
        yield
    except Exception as exc:
        for failure in unregister_classes(completed):
            exc.add_note(f"FluidSim registration rollback: {failure}")
        raise


def register_classes(classes):
    """Atomically register one class-only step."""
    with class_registration(classes):
        pass


@contextmanager
def handler_registration(pairs):
    """Install callbacks atomically; a failure removes only this call's additions."""
    added = []
    replaced = []
    pairs = tuple(pairs)
    try:
        for callbacks, callback in pairs:
            if callbacks is not None and callback not in callbacks:
                callbacks.append(callback)
                added.append((callbacks, callback))
        yield
        # Reloads can leave an older function from this exact callback owner.
        # Retire it only after all additions and local followup work succeeded.
        for callbacks, callback in pairs:
            if callbacks is None:
                continue
            seen = False
            for index in range(len(callbacks) - 1, -1, -1):
                existing = callbacks[index]
                same_owner = (
                    getattr(existing, "__module__", None) == callback.__module__
                    and getattr(existing, "__name__", None) == callback.__name__
                )
                if existing is callback and not seen:
                    seen = True
                elif same_owner:
                    callbacks.pop(index)
                    replaced.append((callbacks, index, existing))
    except Exception as exc:
        for callbacks, index, callback in reversed(replaced):
            try:
                callbacks.insert(index, callback)
            except Exception as undo_exc:
                exc.add_note(f"FluidSim prior handler restore: {undo_exc!r}")
                print(f"FluidSim: could not restore prior handler: {undo_exc!r}")
        for callbacks, callback in reversed(added):
            try:
                callbacks.remove(callback)
            except Exception as undo_exc:
                exc.add_note(f"FluidSim handler rollback: {undo_exc!r}")
                print(f"FluidSim: could not roll back handler: {undo_exc!r}")
        raise
