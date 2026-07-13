"""Real Blender gate for pre-allocation stop and one-click recovery."""

from __future__ import annotations

import pathlib
import sys

import bpy
import numpy as np
import warp as wp


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
device = wp.get_device("cuda:0")
wp.synchronize_device(device)
baseline_free = int(device.free_memory)
baseline_pool = int(wp.get_mempool_used_mem_current(device))

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

released_free = []
released_pool = []
scene_bridge.release_session(scene)
released_free.append(int(device.free_memory))
released_pool.append(int(wp.get_mempool_used_mem_current(device)))
for _ in range(3):
    assert bpy.ops.fluidsim.initialize() == {"FINISHED"}
    assert scene_bridge.session().active
    scene_bridge.release_session(scene)
    released_free.append(int(device.free_memory))
    released_pool.append(int(wp.get_mempool_used_mem_current(device)))

assert all(value <= baseline_pool + 1024**2 for value in released_pool), (
    baseline_pool, released_pool)
assert all(value >= baseline_free - 512 * 1024**2 for value in released_free), (
    baseline_free, released_free)
assert max(released_free) - min(released_free) <= 1024**2, released_free

from fluidSim_warp_blender51.sim import memory_safety
real_assess = memory_safety.assess_configuration
memory_safety.assess_configuration = lambda settings, resources: {
    "safe": False,
    "estimated_bytes": 2 * 1024**3,
    "budget_bytes": 1024**3,
    "resources": dict(resources),
    "recommended": {},
}
try:
    result = bpy.ops.fluidsim.initialize(ignore_memory_safeguard=True)
    assert result == {"FINISHED"}, result
    assert scene_bridge.session().active
finally:
    memory_safety.assess_configuration = real_assess

scene_bridge.release_session(scene)
addon.unregister()
print("MEMORY-SAFETY-GATE-PASS")
