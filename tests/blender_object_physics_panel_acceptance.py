"""Blender 5.1 acceptance for per-object FluidSim Physics Properties."""

from __future__ import annotations

import os
import sys

import bpy


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon


enable_repo_addon(REPO, default_set=False, persistent=False)
from fluidSim_warp_blender51 import panels

panel = panels.FLUIDSIM_PT_object_physics
assert hasattr(bpy.types, panel.__name__)
assert panel.bl_space_type == "PROPERTIES"
assert panel.bl_region_type == "WINDOW"
assert panel.bl_context == "physics"

scene = bpy.context.scene
roles = ("NONE", "SOURCE", "SINK", "COLLIDER", "FORCE", "FIELD")
checked = []
for index, role in enumerate(roles):
    bpy.ops.mesh.primitive_cube_add(location=(index * 2.0, 0.0, 0.0))
    obj = bpy.context.object
    obj.name = f"Physics Panel {role}"
    obj.fluidsim.role = role
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    assert panel.poll(bpy.context), role
    checked.append(role)

assert {
    "source_region_kind", "species_concentration", "sink_region_kind",
    "collider_friction", "force_type", "field_operation",
}.issubset(type(bpy.context.object.fluidsim).bl_rna.properties.keys())

print(
    "OBJECT-PHYSICS-PANEL-ACCEPTANCE-PASS "
    f"context={panel.bl_context} roles={','.join(checked)}",
    flush=True,
)
