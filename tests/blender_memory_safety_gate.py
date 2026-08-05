"""Real Blender gate for pre-allocation stop and confirmed recovery."""

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


def main():
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

    authored_before = {
        "voxel_size": float(fs.voxel_size),
        "max_particles": int(fs.max_particles),
        "leaf_capacity": int(fs.leaf_capacity),
    }
    result = bpy.ops.fluidsim.apply_safe_settings()
    assert result == {"CANCELLED"}, result
    assert not scene_bridge.session().active
    assert authored_before == {
        "voxel_size": float(fs.voxel_size),
        "max_particles": int(fs.max_particles),
        "leaf_capacity": int(fs.leaf_capacity),
    }
    plan = scene_bridge.session().capacity_plan_for_scene(scene)
    assert plan is not None and plan.suggestions, "verified recommendation missing"
    recommendation = plan.suggestions[0]
    print("MEMORY-GATE-RECOMMENDATION", recommendation.key)
    approved = set(bpy.ops.fluidsim.apply_capacity_suggestion(
        "EXEC_DEFAULT",
        suggestion_key=recommendation.key,
        expected_scene_signature=(
            scene_bridge.session().capacity_plan_signature_for_scene(scene)),
    ))
    assert approved == {"FINISHED"}, approved
    assert scene_bridge.session().active
    assert not scene_bridge.current_error(scene)
    injected = scene_bridge.session().solver.inject_particles(np.asarray([
        (-0.1, -0.1, 0.5), (0.1, -0.1, 0.5),
        (-0.1, 0.1, 0.5), (0.1, 0.1, 0.5),
    ], dtype=np.float32))
    assert injected == 4, injected
    result = bpy.ops.fluidsim.step()
    print("MEMORY-GATE-STEP", result, scene_bridge.session().active,
          scene_bridge.session().solver.grid.active_leaf_count)
    assert result == {"FINISHED"}, result
    assert scene_bridge.session().active, "solver was released during step"
    assert scene_bridge.session().solver.grid.active_leaf_count > 0, "topology stayed empty"

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

    scene_bridge.release_session(scene)
    addon.unregister()


try:
    main()
except AssertionError as exc:
    detail = str(exc) or "assertion failed"
    print(f"MEMORY-SAFETY-GATE-FAIL: {detail}")
    sys.exit(1)

print("MEMORY-SAFETY-GATE-PASS")
