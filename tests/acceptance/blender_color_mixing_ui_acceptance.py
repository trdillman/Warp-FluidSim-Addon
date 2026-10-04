"""Focused Blender 5.1 acceptance for color-mixing authoring controls."""

from pathlib import Path
import sys

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.repo_addon_loader import enable_repo_addon


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    bpy.ops.wm.open_mainfile(filepath=str(ROOT / "demos" / "chromatic_vortex.blend"))
    enable_repo_addon(str(ROOT), default_set=True, persistent=True)
    scene = bpy.context.scene
    domain = scene.fluidsim.domain_object
    require(domain is not None, "saved demo has no designated domain")
    domain_settings = domain.fluidsim_domain

    require(hasattr(domain_settings, "species_enabled"),
            "domain mixing toggle is not registered")
    require(hasattr(domain_settings, "species_mixing_rate"),
            "domain mixing strength is not registered")
    require(hasattr(bpy.types, "FLUIDSIM_PT_sim_color_mixing"),
            "current Scene Properties color-mixing panel is not registered")
    require(hasattr(bpy.types, "FLUIDSIM_PT_physics_domain_sim_color_mixing"),
            "current Physics Properties domain mirror is not registered")
    require(hasattr(bpy.types, "FLUIDSIM_PT_physics_input_object_source"),
            "per-object source panel is not mirrored into Physics Properties")

    scene.fluidsim.particle_attribute_mode = "COLOR"
    scene.fluidsim.surface_attribute_mode = "COLOR"
    require(scene.fluidsim.particle_attribute_mode == "COLOR",
            "mixed particle color cannot be selected for viewport display")
    require(scene.fluidsim.surface_attribute_mode == "COLOR",
            "mixed surface color cannot be selected for viewport display")

    from fluidSim_warp_blender51.blender.output_publication import (
        _nearest_values,
        _values_to_colors,
    )
    authored = np.asarray([
        (1.0, 0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5, 1.0),
    ], dtype=np.float32)
    displayed = _values_to_colors(authored)
    require(displayed.shape == (2, 4),
            "RGBA particle values are not published as RGBA colors")
    require(np.allclose(displayed, authored),
            "viewport remaps authored mixed colors through a scalar ramp")
    surface_colors = _nearest_values(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
        authored,
        0.25,
    )
    require(surface_colors.shape == (2, 4),
            "surface color sampling collapses RGBA values to scalars")
    require(np.allclose(surface_colors, authored),
            "surface vertices do not inherit the nearest mixed color")

    bpy.ops.mesh.primitive_cube_add(size=1.0)
    emitter = bpy.context.object
    require(bpy.ops.fluidsim.tag_liquid_emitter() == {"FINISHED"},
            "liquid emitter role assignment failed")
    emitter.fluidsim.source_color = (1.0, 0.0, 0.0, 1.0)
    require(tuple(emitter.fluidsim.source_color) == (1.0, 0.0, 0.0, 1.0),
            "per-emitter source color is not authorable")

    print("COLOR-MIXING-UI-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
