"""FluidSim Warp - GPU liquid simulation for Blender 5.1.

Native Blender port of the FluidSim OpenCL tool (fluidSim.cl 1.0.226) on
NVIDIA Warp/CUDA. Interactive viewport simulation with sparse tiled leaf-grid fields,
PIC/FLIP transfer, pressure projection (Jacobi/PCG), viscosity, surface SDF
extraction, whitewater, profiling, benchmarking, validation, and custom
user-kernel extensibility.
"""

bl_info = {
    "name": "FluidSim Warp",
    "author": "FluidSim port (synthesized reference)",
    "version": (1, 0, 227),
    "blender": (5, 1, 0),
    "location": "3D Viewport > Sidebar > FluidSim",
    "description": "Interactive GPU liquid simulation (NVIDIA Warp/CUDA)",
    "category": "Physics",
}


from .warp_discovery import ensure_warp_discoverable

ensure_warp_discoverable()


def register():
    from . import operators, panels, preferences, properties
    from .blender import render, scene_bridge, viewport

    properties.register()
    preferences.register()
    operators.register()
    panels.register()
    scene_bridge.register_handlers()
    render.register_handlers()
    viewport.register_overlay()


def unregister():
    from . import operators, panels, preferences, properties
    from .blender import render, scene_bridge, viewport

    viewport.unregister_overlay()
    render.unregister_handlers()
    scene_bridge.unregister_handlers()
    scene_bridge.release_session()
    panels.unregister()
    operators.unregister()
    preferences.unregister()
    properties.unregister()
