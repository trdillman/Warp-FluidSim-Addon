"""Real Blender gate for pre-allocation stop and one-click recovery."""

from __future__ import annotations

import pathlib
import sys

import bpy
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fluidSim_warp_blender51 as addon
from fluidSim_warp_blender51.blender import scene_bridge


addon.register()
scene = bpy.context.scene
fs = scene.fluidsim
fs.leaf_capacity = 4096
fs.max_particles = 250_000
fs.max_whitewater = 100_000
fs.voxel_size = 0.02
fs.surface_resolution_mode = "MULTIPLIER"
fs.surface_resolution_multiplier = 4.0
fs.surface_leaf_capacity_scale = 64.0
fs.surface_max_dense_dim = 2048

try:
    result = bpy.ops.fluidsim.initialize()
except RuntimeError:
    result = {"CANCELLED"}
error = scene_bridge.current_error(scene)
assert result == {"CANCELLED"}, result
assert not scene_bridge.session().active
assert "memory safeguard stopped initialization" in error.lower(), error

result = bpy.ops.fluidsim.apply_safe_settings()
assert result == {"FINISHED"}, result
assert scene_bridge.session().active
assert fs.surface_resolution_mode == "SIMULATION"
assert fs.surface_leaf_capacity_scale == 1.0
assert fs.surface_max_dense_dim <= 256
assert not scene_bridge.current_error(scene)
injected = scene_bridge.session().solver.inject_particles(np.asarray([
    (-0.1, -0.1, 0.5), (0.1, -0.1, 0.5),
    (-0.1, 0.1, 0.5), (0.1, 0.1, 0.5),
], dtype=np.float32))
assert injected == 4, injected
result = bpy.ops.fluidsim.step()
assert result == {"FINISHED"}, result
assert scene_bridge.session().active
assert scene_bridge.session().solver.grid.active_leaf_count > 0
assert scene_bridge.session().solver.surface_extractor._dense is not None

scene_bridge.release_session(scene)
addon.unregister()
print("MEMORY-SAFETY-GATE-PASS")
