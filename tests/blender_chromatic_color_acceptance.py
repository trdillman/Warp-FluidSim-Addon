"""Live viewport-color acceptance for the Chromatic Vortex demo."""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.repo_addon_loader import enable_repo_addon


enable_repo_addon(REPO, default_set=False, persistent=False)
scene = bpy.context.scene
fs = scene.fluidsim
assert os.path.basename(bpy.data.filepath) == "chromatic_vortex.blend"
assert fs.live and fs.species_enabled

# The initially opened frame must already be visibly authored; advancing the
# timeline would hide the exact handoff regression this test protects.
surface = bpy.data.objects.get("FluidSim Surface")
assert surface is not None and surface.type == "MESH"
assert len(surface.data.vertices) > 0

attribute = surface.data.attributes.get("fluidsim_color")
assert attribute is not None, "live species surface is missing fluidsim_color"
assert attribute.domain == "POINT"
assert len(attribute.data) == len(surface.data.vertices)

colors = np.empty(len(attribute.data) * 4, dtype=np.float32)
attribute.data.foreach_get("color", colors)
colors = colors.reshape((-1, 4))
positions = np.empty(len(surface.data.vertices) * 3, dtype=np.float32)
surface.data.vertices.foreach_get("co", positions)
positions = positions.reshape((-1, 3))

left = colors[positions[:, 0] < -0.08, :3]
right = colors[positions[:, 0] > 0.08, :3]
assert len(left) > 0 and len(right) > 0
left_mean = np.mean(left, axis=0)
right_mean = np.mean(right, axis=0)
assert left_mean[0] > left_mean[2], (left_mean, right_mean)
assert right_mean[2] > right_mean[0], (left_mean, right_mean)

material = surface.data.materials[0]
attribute_nodes = [
    node for node in material.node_tree.nodes
    if node.bl_idname == "ShaderNodeAttribute"]
assert any(node.attribute_name == "fluidsim_color" for node in attribute_nodes)
print(
    "CHROMATIC-COLOR-ACCEPTANCE-PASS "
    f"left={left_mean.tolist()} right={right_mean.tolist()}",
    flush=True,
)
