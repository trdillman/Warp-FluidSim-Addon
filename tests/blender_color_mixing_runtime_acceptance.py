"""Focused Blender/CUDA proof for the saved chromatic-vortex workflow."""

from pathlib import Path
import sys

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _test_device import script_device  # noqa: E402

from tools.repo_addon_loader import enable_repo_addon


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    bpy.ops.wm.open_mainfile(filepath=str(ROOT / "demos" / "chromatic_vortex.blend"))
    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    from fluidSim_warp_blender51.sim.solver import FluidSolver, SolverSettings
    from fluidSim_warp_blender51.blender.scene_bridge import active_domain_settings

    scene = bpy.context.scene
    fs = scene.fluidsim
    sources = [
        obj for obj in scene.objects
        if hasattr(obj, "fluidsim") and obj.fluidsim.role == "SOURCE"
    ]
    colors = [tuple(obj.fluidsim.source_color) for obj in sources]
    require(any(color[0] > 0.8 and color[2] < 0.2 for color in colors),
            "saved demo has no red liquid source")
    require(any(color[2] > 0.8 and color[0] < 0.2 for color in colors),
            "saved demo has no blue liquid source")
    require(any(hasattr(obj, "fluidsim") and obj.fluidsim.role == "FORCE"
                for obj in scene.objects),
            "saved demo has no force-driven mixer")
    require(active_domain_settings(scene).species_enabled,
            "saved demo mixing is disabled")
    require(fs.surface_attribute_mode == "COLOR",
            "saved demo surface does not display mixed color")

    settings = SolverSettings(
        domain_min=(0.0, 0.0, 0.0),
        domain_max=(1.0, 1.0, 1.0),
        voxel_size=0.25,
        leaf_capacity=32,
        max_particles=16,
        max_whitewater=1,
        surface_enabled=False,
        whitewater_enabled=False,
        species_enabled=True,
        species_mixing_rate=20.0,
        device=script_device(),
    )
    solver = FluidSolver(settings)
    try:
        solver.inject_particles(
            [(0.50, 0.50, 0.50)], color=(1.0, 0.0, 0.0, 1.0))
        solver.inject_particles(
            [(0.51, 0.50, 0.50)], color=(0.0, 0.0, 1.0, 1.0))
        solver.step_frame()
        mixed = solver.particles.attr("color").numpy()[:2]
        require(np.all(mixed[:, 0] > 0.15), "mixed result lacks red")
        require(np.all(mixed[:, 2] > 0.15), "mixed result lacks blue")

        solver.settings.species_enabled = False
        solver.reset()
        require(solver.color_mixing is None, "disable left transport alive")
        solver.settings.species_enabled = True
        solver.reset()
        require(solver.color_mixing is not None, "re-enable did not restore transport")
    finally:
        solver.release()

    print("COLOR-MIXING-RUNTIME-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
