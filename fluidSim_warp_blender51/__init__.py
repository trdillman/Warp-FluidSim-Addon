"""FluidSim Warp - GPU liquid simulation for Blender 5.1.

Native Blender port of the FluidSim OpenCL tool (fluidSim.cl 1.0.226) on
NVIDIA Warp/CUDA. Interactive viewport simulation with sparse tiled leaf-grid fields,
PIC/FLIP transfer, beta ST-FLIP, pressure projection (Jacobi/PCG), viscosity,
decoupled-resolution surface extraction, whitewater, profiling, validation, and custom
user-kernel extensibility.
"""

bl_info = {
    "name": "FluidSim Warp",
    "author": "FluidSim port (synthesized reference)",
    "version": (1, 0, 241),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > FluidSim",
    "description": "Interactive GPU liquid simulation with beta ST-FLIP",
    "category": "Physics",
}


from .warp_discovery import ensure_warp_discoverable

ensure_warp_discoverable()


def register():
    from . import operators, panels, preferences, properties
    from .blender import playback, render, scene_bridge, viewport_overlay

    # register()/unregister() must be idempotent in both directions: if a
    # later step raises, Blender's add-on enable path leaves everything
    # registered so far behind with no automatic rollback, and calling
    # unregister() to clean that up. Track exactly what succeeded and unwind
    # it in reverse order on any failure, instead of leaving Blender with a
    # half-registered add-on it believes failed to enable.
    steps = (
        ("properties", properties.register, properties.unregister),
        ("preferences", preferences.register, preferences.unregister),
        ("operators", operators.register, operators.unregister),
        ("panels", panels.register, panels.unregister),
        ("pending handlers", scene_bridge.register_handlers,
         scene_bridge.unregister_handlers),
        ("playback handlers", playback.register_handlers,
         playback.unregister_handlers),
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
    from . import operators, panels, preferences, properties
    from .blender import playback, render, scene_bridge, viewport_overlay

    # Reverse of register(); every step must tolerate being called on a
    # partially-registered state (e.g. after a register() rollback, or a
    # Reload Scripts that re-enters unregister() twice), so one step's
    # failure never blocks the rest of teardown.
    steps = (
        ("viewport overlay", viewport_overlay.unregister_overlay),
        ("render handlers", render.unregister_handlers),
        ("playback handlers", playback.unregister_handlers),
        ("pending handlers", scene_bridge.unregister_handlers),
        ("panels", panels.unregister),
        ("operators", operators.unregister),
        ("preferences", preferences.unregister),
        ("properties", properties.unregister),
    )
    for name, step in steps:
        try:
            step()
        except Exception as exc:  # noqa: BLE001
            print(f"FluidSim: unregister step {name!r} failed: {exc!r}")
