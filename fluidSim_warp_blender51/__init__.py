"""FluidSim Warp Blender extension registration.

Release metadata is owned by ``blender_manifest.toml`` and runtime identity by
``version.py``. Import-time registration neither probes nor installs runtime
dependencies: Blender installs the bundled, hash-locked NVIDIA Warp wheel with
the extension, and ``runtime/warp_runtime.py`` validates it lazily.
"""


def register():
    from . import operators, panels, preferences, properties
    from .blender import (
        effect_ui,
        effect_workspace,
        effects,
        file_compat,
        menus,
        pending_settings,
        playback,
        render,
        sources,
        ui_object,
        ui_physics,
        ui_scene,
        ui_view3d,
        viewport_overlay,
    )

    # register()/unregister() must be idempotent in both directions: if a
    # later step raises, Blender's add-on enable path leaves everything
    # registered so far behind with no automatic rollback, and calling
    # unregister() to clean that up. Track exactly what succeeded and unwind
    # it in reverse order on any failure, instead of leaving Blender with a
    # half-registered add-on it believes failed to enable.
    steps = (
        ("properties", properties.register, properties.unregister),
        ("custom effects model", effects.register, effects.unregister),
        ("preferences", preferences.register, preferences.unregister),
        ("operators", operators.register, operators.unregister),
        ("viewport launcher", panels.register, panels.unregister),
        ("Scene Properties UI", ui_scene.register, ui_scene.unregister),
        ("Object Properties UI", ui_object.register, ui_object.unregister),
        ("custom effects UI", effect_ui.register, effect_ui.unregister),
        ("Warp Code workspace", effect_workspace.register, effect_workspace.unregister),
        ("Physics Properties UI", ui_physics.register, ui_physics.unregister),
        ("viewport Live Tuning shelf", ui_view3d.register, ui_view3d.unregister),
        ("Add-menu domain entry", menus.register, menus.unregister),
        ("source migration handlers", sources.register_handlers,
         sources.unregister_handlers),
        ("pre-1.1 file check", file_compat.register_handlers,
         file_compat.unregister_handlers),
        ("playback handlers", playback.register_handlers,
         playback.unregister_handlers),
        ("pending handlers", pending_settings.register_handlers,
         pending_settings.unregister_handlers),
        ("render handlers", render.register_handlers,
         render.unregister_handlers),
        ("viewport overlay", viewport_overlay.register_overlay,
         viewport_overlay.unregister_overlay),
    )
    completed = []
    try:
        for name, do_register, _undo in steps:
            do_register()
            completed.append((name, _undo))
    except Exception:
        for name, undo in reversed(completed):
            try:
                undo()
            except Exception as undo_exc:  # noqa: BLE001
                print(f"FluidSim: rollback of {name!r} failed: {undo_exc!r}")
        raise


def unregister():
    import sys

    # Reverse of register(); every step must tolerate being called on a
    # partially-registered state (e.g. after a register() rollback, or a
    # Reload Scripts that re-enters unregister() twice). Never import a module
    # merely to tear it down: a partial registration may not have had enough
    # Blender API surface to import later UI adapters in the first place.
    steps = (
        ("viewport overlay", "blender.viewport_overlay", "unregister_overlay"),
        ("render handlers", "blender.render", "unregister_handlers"),
        ("pending handlers", "blender.pending_settings", "unregister_handlers"),
        ("playback handlers", "blender.playback", "unregister_handlers"),
        ("native CUDA cache", "sim.cuda_reduction", "reset_cuda_module_cache"),
        ("legacy effect migration handlers", "blender.effects", "unregister_handlers"),
        ("pre-1.1 file check", "blender.file_compat", "unregister_handlers"),
        ("source migration handlers", "blender.sources", "unregister_handlers"),
        ("Add-menu domain entry", "blender.menus", "unregister"),
        ("viewport Live Tuning shelf", "blender.ui_view3d", "unregister"),
        ("Physics Properties UI", "blender.ui_physics", "unregister"),
        ("Warp Code workspace", "blender.effect_workspace", "unregister"),
        ("custom effects UI", "blender.effect_ui", "unregister"),
        ("Object Properties UI", "blender.ui_object", "unregister"),
        ("Scene Properties UI", "blender.ui_scene", "unregister"),
        ("viewport launcher", "panels", "unregister"),
        ("operators", "operators", "unregister"),
        ("preferences", "preferences", "unregister"),
        ("custom effects model", "blender.effects", "unregister"),
        ("properties", "properties", "unregister"),
    )
    for name, relative_name, function_name in steps:
        module = sys.modules.get(f"{__name__}.{relative_name}")
        step = getattr(module, function_name, None) if module is not None else None
        if not callable(step):
            continue
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            print(f"FluidSim: unregister step {name!r} failed: {exc!r}")
