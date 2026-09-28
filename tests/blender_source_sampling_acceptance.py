"""Blender RNA acceptance for source sampling defaults and migration."""

from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fluidSim_warp_blender51 as addon  # noqa: E402
from fluidSim_warp_blender51.blender.scene_inputs import (  # noqa: E402
    _effective_source_target,
)
from fluidSim_warp_blender51.blender.sources import (  # noqa: E402
    migrate_source_sampling_schema,
)


def cube(name):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def main():
    addon.register()
    scene = bpy.context.scene
    # Domain settings live only on the designated domain object.
    domain = scene.objects.get("Cube")
    assert domain is not None, "factory-startup Cube is missing"
    scene.fluidsim.domain_object = domain
    domain.fluidsim_domain.particles_per_voxel = 12

    new = cube("New Inherited Emitter")
    bpy.context.view_layer.objects.active = new
    new.select_set(True)
    assert bpy.ops.fluidsim.tag_liquid_emitter() == {"FINISHED"}
    assert new.fluidsim.source_ui_schema_version == 4
    assert new.fluidsim.source_sampling_mode == "INHERIT"
    assert new.fluidsim.source_sampling_multiplier == 1.0
    assert _effective_source_target(scene, new.fluidsim) == 12

    old = cube("Old Absolute Emitter")
    old.fluidsim.role = "SOURCE"
    old.fluidsim.source_ui_schema_version = 3
    old.fluidsim.emit_rate = 5.0
    old.fluidsim.source_sampling_mode = "INHERIT"  # default must not guess intent
    assert _effective_source_target(scene, old.fluidsim) == 5
    assert migrate_source_sampling_schema(scene) == 1
    assert old.fluidsim.source_ui_schema_version == 4
    assert old.fluidsim.source_sampling_mode == "LEGACY_ABSOLUTE"
    assert old.fluidsim.emit_rate == 5.0
    assert _effective_source_target(scene, old.fluidsim) == 5

    new.fluidsim.source_sampling_mode = "MULTIPLIER"
    new.fluidsim.source_sampling_multiplier = 2.0
    assert _effective_source_target(scene, new.fluidsim) == 24
    new.fluidsim.source_sampling_multiplier = 0.0
    assert _effective_source_target(scene, new.fluidsim) == 0

    addon.unregister()
    print("SOURCE-SAMPLING-ACCEPTANCE-PASS")


if __name__ == "__main__":
    main()
