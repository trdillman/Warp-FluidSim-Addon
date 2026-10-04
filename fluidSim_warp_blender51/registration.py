"""Ordered add-on registration: one table, walked forward and in reverse.

``STEPS`` lists, in the order they must be registered, everything the add-on
puts into Blender. ``register_all`` walks it forward and, if a step raises,
undoes the steps that succeeded; ``unregister_all`` walks the same table
backward. There is no second list to keep in step with the first: a step added
to the table is registered and torn down.

This module imports neither ``bpy`` nor any step module at import time, so the
root ``__init__`` can use it without touching Blender state, and the walk is
testable with plain stand-in modules.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One thing the add-on registers.

    ``module`` is relative to the package. ``register`` and ``unregister`` name
    attributes of that module. ``register=None`` marks a teardown-only step:
    state the add-on creates lazily and releases when it is disabled.
    """

    name: str
    module: str
    register: str | None
    unregister: str


STEPS: tuple[Step, ...] = (
    Step("properties", "properties", "register", "unregister"),
    Step("custom effects model", "blender.effects", "register", "unregister"),
    Step("preferences", "preferences", "register", "unregister"),
    Step("operators", "operators", "register", "unregister"),
    Step("viewport launcher", "panels", "register", "unregister"),
    Step("Scene Properties UI", "blender.ui_scene", "register", "unregister"),
    Step("Object Properties UI", "blender.ui_object", "register", "unregister"),
    Step("custom effects UI", "blender.effect_ui", "register", "unregister"),
    Step("Warp Code workspace", "blender.effect_workspace", "register", "unregister"),
    Step("Physics Properties UI", "blender.ui_physics", "register", "unregister"),
    Step("viewport workflow stages", "blender.ui_stages", "register", "unregister"),
    Step("viewport header popover", "blender.ui_header", "register", "unregister"),
    Step("Add-menu domain entry", "blender.menus", "register", "unregister"),
    Step("source migration handlers", "blender.sources",
         "register_handlers", "unregister_handlers"),
    Step("pre-1.1 file check", "blender.file_compat",
         "register_handlers", "unregister_handlers"),
    # Native CUDA modules load on first use; release them once the handlers
    # that could start a Solver Session are gone.
    Step("native CUDA cache", "sim.cuda_reduction", None, "reset_cuda_module_cache"),
    Step("playback handlers", "blender.playback",
         "register_handlers", "unregister_handlers"),
    Step("pending handlers", "blender.pending_settings",
         "register_handlers", "unregister_handlers"),
    Step("render handlers", "blender.render",
         "register_handlers", "unregister_handlers"),
    Step("viewport overlay", "blender.viewport_overlay",
         "register_overlay", "unregister_overlay"),
)


def register_all(package: str, steps: tuple[Step, ...] = STEPS) -> None:
    """Register every step in order; on a failure undo what succeeded, then raise.

    Blender's enable path does not unregister a half-registered add-on after
    ``register()`` raises, so the rollback happens here. Every module is
    imported before the first registration, so an import error leaves Blender
    untouched. A failing step is not undone (it did not complete); a failing
    rollback is reported and the remaining rollbacks still run.
    """
    registrable = [step for step in steps if step.register is not None]
    modules = [importlib.import_module(f"{package}.{step.module}") for step in registrable]
    completed = []
    try:
        for step, module in zip(registrable, modules):
            getattr(module, step.register)()
            completed.append((step, module))
    except Exception:
        for step, module in reversed(completed):
            try:
                getattr(module, step.unregister)()
            except Exception as undo_exc:  # noqa: BLE001
                print(f"FluidSim: rollback of {step.name!r} failed: {undo_exc!r}")
        raise


def unregister_all(package: str, steps: tuple[Step, ...] = STEPS) -> None:
    """Tear every step down in reverse order, tolerating any partial state.

    Safe after a rolled-back ``register_all`` and when called twice (Reload
    Scripts). A module is never imported just to tear it down: a partial
    registration may not have had the Blender API surface to import it. A
    step whose module was never loaded, or that lacks the teardown attribute,
    is skipped; a step that raises is reported and the walk continues.
    """
    for step in reversed(steps):
        module = sys.modules.get(f"{package}.{step.module}")
        undo = getattr(module, step.unregister, None) if module is not None else None
        if not callable(undo):
            continue
        try:
            undo()
        except Exception as exc:  # noqa: BLE001
            print(f"FluidSim: unregister step {step.name!r} failed: {exc!r}")
