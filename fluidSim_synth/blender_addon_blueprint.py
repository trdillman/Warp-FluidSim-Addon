"""Target Blender 5.1 add-on blueprint."""

ADDON_ID = "fluidSim_warp_blender51"
ADDON_NAME = "FluidSim Warp"
BLENDER_TARGET = "5.1"

PACKAGE_LAYOUT = {
    "addon": ["preferences.py", "properties.py", "operators.py", "panels.py"],
    "sim": [
        "constants.py", "enums.py", "fields.py", "particles.py", "kernels_core.py", "kernels_fluid.py",
        "pressure.py", "viscosity.py", "whitewater.py", "solver.py", "validation.py",
    ],
    "blender": ["scene_bridge.py", "sources.py", "colliders.py", "materials.py", "viewport.py", "render.py"],
    "diagnostics": ["profiler.py", "benchmark.py", "report.py"],
    "io": ["cache.py", "export.py"],
    "docs": ["user_guide.md", "technical_report.md"],
    "tests": [
        "test_field_indexing.py", "test_particle_grid_transfer.py", "test_pressure_projection.py",
        "test_viscosity.py", "test_solver_validation.py",
    ],
}

PERFORMANCE_REQUIREMENTS = {
    "simulation_mode": "interactive viewport playback",
    "cache_mode": "optional only",
    "compute": "GPU-only Warp/CUDA for solver kernels",
    "goal": "orders-of-magnitude faster than comparable CPU liquid workflows where GPU work dominates",
}
