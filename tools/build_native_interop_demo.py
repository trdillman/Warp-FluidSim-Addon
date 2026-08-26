"""Build the configured native CUDA/Vulkan particle interop demo scene."""

from tools.repo_addon_loader import enable_repo_addon
from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=True)
    assert bpy.ops.fluidsim.setup_scene() == {"FINISHED"}
    scene = bpy.context.scene
    settings = scene.fluidsim
    settings.voxel_size = 0.10
    settings.max_particles = 500_000
    settings.max_whitewater = 50_000
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.liquid_view_mode = "PARTICLES"
    settings.particle_display_fraction = 1.0
    settings.particle_display_scale = 1.6
    settings.particle_attribute_mode = "NONE"
    settings.native_gpu_interop = True
    scene.frame_start = 1
    scene.frame_end = 180
    scene["native_gpu_interop_demo"] = True
    scene["native_gpu_interop_backend"] = "Vulkan + CUDA/Warp"
    scene["native_gpu_interop_requirement"] = (
        "Use the interop-enabled Blender build, Initialize Solver, then enable Live Preview or play."
    )

    text = bpy.data.texts.get("NATIVE_GPU_INTEROP_README")
    if text is None:
        text = bpy.data.texts.new("NATIVE_GPU_INTEROP_README")
    text.clear()
    text.write(
        "Native GPU Interop Particle Demo\n"
        "================================\n\n"
        "This scene presents Warp/CUDA liquid particles through a Blender-owned Vulkan VBO.\n"
        "No bulk particle positions cross CPU memory during native viewport presentation.\n\n"
        "Run:\n"
        "1. Start the interop-enabled Blender build with the Vulkan backend.\n"
        "2. In Fluid Simulation, click Initialize Solver.\n"
        "3. Enable Live Preview or play the timeline.\n\n"
        "Native GPU Interop is enabled. Surface meshing, attributes, fractional display, and\n"
        "whitewater remain on their existing paths in this milestone. Unsupported settings\n"
        "fall back to the CPU datablock path and report the reason in the runtime session.\n"
    )

    output = REPO_ROOT / "demos" / "native_gpu_interop_particles.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print("NATIVE_INTEROP_DEMO", output)


if __name__ == "__main__":
    main()
