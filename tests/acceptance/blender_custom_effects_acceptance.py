"""Blender 5.1 acceptance for built-in Custom Effects and package import."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon


def main() -> None:
    enable_repo_addon(REPO_ROOT, default_set=False, persistent=False)

    from fluidSim_warp_blender51.blender import effect_library, effect_ui, effects, scene_bridge
    from fluidSim_warp_blender51.sim.effects.builtins import (
        builtin_effect_definitions,
    )
    from fluidSim_warp_blender51.sim.effects.package import write_effect_package

    if not hasattr(bpy.types, "FLUIDSIM_PT_custom_effects"):
        raise AssertionError("Custom Effects panel is not registered")

    builtin_count = len(builtin_effect_definitions())
    library_items = effect_ui._effect_items(None, bpy.context)
    if builtin_count != 8 or len(library_items) != 8:
        raise AssertionError(
            f"expected eight visible built-ins, got {builtin_count}/{len(library_items)}"
        )
    if effect_ui._LAST_LIBRARY_FATAL:
        raise AssertionError(f"library enumeration failed: {effect_ui._LAST_LIBRARY_FATAL}")

    result = bpy.ops.fluidsim.effect_add(effect_id="fluidsim.vortex_force")
    if result != {"FINISHED"}:
        raise AssertionError(f"built-in Add Effect failed: {result}")

    stack = bpy.context.scene.fluidsim_effects
    vortex = stack.items[0]
    instances = effects.scene_effect_instances(bpy.context.scene)
    effects._publish_applied_state(bpy.context.scene, instances)
    updates = []

    class FakeSolver:
        def update_custom_effect_runtime_bindings(self, updated):
            updates.append(tuple(updated))
            return True

    fake_session = type("FakeSession", (), {
        "active": True,
        "solver": FakeSolver(),
        "owns_scene": lambda self, scene: scene is bpy.context.scene,
    })()
    original_session = scene_bridge.session
    original_test_draft = effects.test_draft
    try:
        scene_bridge.session = lambda: fake_session
        effects.test_draft = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("runtime-bindable edit attempted CUDA compile/apply")
        )
        suction = next(item for item in vortex.parameters if item.key == "suction")
        suction.float_value = 48.54
    finally:
        effects.test_draft = original_test_draft
        scene_bridge.session = original_session
    if not updates or abs(updates[-1][0].parameter_values["suction"] - 48.54) > 1.0e-4:
        raise AssertionError(
            "runtime-bindable parameter was not sent to the live solver; "
            f"stack={stack.status}, item={vortex.status}, blocking={stack.blocking_reason!r}"
        )
    if stack.status != "APPLIED" or vortex.status != "APPLIED":
        raise AssertionError("runtime-bindable parameter edit was incorrectly left dirty")
    if stack.blocking_reason:
        raise AssertionError(f"runtime-bindable update was blocked: {stack.blocking_reason}")

    original_user_library_path = effect_library.user_library_path
    try:
        with tempfile.TemporaryDirectory(
            prefix="fluidsim_effect_acceptance_", dir=REPO_ROOT
        ) as temp_dir:
            root = Path(temp_dir)
            library = root / "library"
            library.mkdir()
            effect_library.user_library_path = lambda: library
            definition = replace(
                builtin_effect_definitions()[0],
                effect_id="acceptance.imported-vortex",
                name="Imported Vortex Acceptance",
                built_in=False,
                package_origin="acceptance://custom-effects",
            )
            incoming = root / "imported-vortex.fseffect"
            write_effect_package(incoming, definition)
            result = bpy.ops.fluidsim.effect_import_package(
                filepath=str(incoming), conflict="cancel"
            )
            if result != {"FINISHED"}:
                raise AssertionError(f"Import Effect Package failed: {result}")
            installed = effect_library.installed_package_for_effect(
                definition.effect_id
            )
            item = bpy.context.scene.fluidsim_effects.items[
                bpy.context.scene.fluidsim_effects.active_index
            ]
            if not installed.path.is_file():
                raise AssertionError("imported package was not persisted")
            if item.effect_id != definition.effect_id or item.enabled:
                raise AssertionError("imported scene effect is not disabled and review-ready")
    finally:
        effect_library.user_library_path = original_user_library_path

    print("CUSTOM-EFFECTS-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
