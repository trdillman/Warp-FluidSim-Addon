"""Real Blender/CUDA acceptance for object-level FluidSim FORCE regions.

Run:
    blender --background --factory-startup --python tests/blender_force_region_acceptance.py

The FORCE role is stored as a source-shape ABI descriptor (`mode=FIELD`,
`emit_rate=0`) but must act in the solver forces phase after P2G. It must not
create liquid or write unweighted MAC values during sourcing.
"""

from __future__ import annotations

import os
import sys

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon


def velocity_summary():
    from fluidSim_warp_blender51.blender.scene_bridge import session

    solver = session().solver
    count = solver.particles.refresh_host_count()
    velocity = solver.particles.velocity.numpy()[:count]
    return count, np.mean(velocity, axis=0) if count else np.zeros(3)


def step_once(scene):
    result = bpy.ops.fluidsim.step()
    if result != {"FINISHED"}:
        raise RuntimeError(f"manual step failed: {result}, {scene.fluidsim.status}")
    return velocity_summary()


try:
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.liquid_view_mode = "PARTICLES"
    if bpy.ops.fluidsim.setup_scene() != {"FINISHED"}:
        raise RuntimeError("quick setup failed")
    from fluidSim_warp_blender51.blender.scene_bridge import (
        active_domain_settings,
    )

    domain_settings = active_domain_settings(scene)
    assert domain_settings is not None, "quick setup did not designate a domain"
    domain_settings.voxel_size = 0.1
    domain_settings.max_particles = 120_000
    domain_settings.gravity = (0.0, 0.0, 0.0)

    source = next(
        (
            obj for obj in scene.objects
            if getattr(getattr(obj, "fluidsim", None), "role", "NONE")
            == "SOURCE"
        ),
        None,
    )
    if source is None:
        raise RuntimeError("quick setup did not tag a source")
    source.fluidsim.source_mode = "PARTICLES"
    source.fluidsim.emission_mode = "INITIAL_FILL"
    source.fluidsim.emit_rate = 8.0
    source.fluidsim.initial_velocity = (0.0, 0.0, 0.0)

    # Enclose the source in a directional analytic FORCE region.
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=source.location)
    force = bpy.context.object
    force.name = "FluidSim Force Region Acceptance"
    force.scale = (2.0, 2.0, 2.0)
    bpy.ops.object.select_all(action="DESELECT")
    force.select_set(True)
    bpy.context.view_layer.objects.active = force
    tag_result = bpy.ops.fluidsim.tag_object(role="FORCE")
    if tag_result != {"FINISHED"} or force.fluidsim.role != "FORCE":
        raise RuntimeError(
            f"normal-user Tag Force Region workflow failed: {tag_result}, "
            f"role={force.fluidsim.role}")
    force.fluidsim.shape = "BOX"
    force.fluidsim.force_type = "DIRECTIONAL"
    force.fluidsim.force_strength = 12.0
    force.fluidsim.force_direction = (1.0, 0.0, 0.0)
    force.fluidsim.enabled = False

    if bpy.ops.fluidsim.initialize() != {"FINISHED"}:
        raise RuntimeError(f"initialize failed: {fs.status}")
    baseline_count, baseline_mean = step_once(scene)
    if baseline_count <= 0:
        raise RuntimeError("baseline source emitted no particles")

    # Re-run from the same reset state with the region enabled. Initial-fill
    # sources are re-armed by reset and the source RNG is deterministic.
    force.fluidsim.enabled = True
    if bpy.ops.fluidsim.reset() != {"FINISHED"}:
        raise RuntimeError(f"reset failed: {fs.status}")
    forced_count, forced_mean = step_once(scene)

    if forced_count != baseline_count:
        raise RuntimeError(
            f"FORCE region changed particle count: baseline={baseline_count}, "
            f"forced={forced_count}; force regions must not source liquid")
    delta_x = float(forced_mean[0] - baseline_mean[0])
    if not delta_x > 0.05:
        raise RuntimeError(
            f"directional FORCE region did not accelerate particles: "
            f"baseline mean={baseline_mean}, forced mean={forced_mean}, "
            f"delta_x={delta_x}")
    if abs(float(forced_mean[1] - baseline_mean[1])) > 0.02:
        raise RuntimeError(
            f"directional FORCE leaked materially into Y: "
            f"baseline={baseline_mean}, forced={forced_mean}")

    print(
        "FORCE-REGION-ACCEPTANCE-PASS "
        f"count={forced_count:,} baseline_mean={baseline_mean.tolist()} "
        f"forced_mean={forced_mean.tolist()} delta_x={delta_x:.6f}"
    )
except Exception as exc:
    print(f"FORCE-REGION-ACCEPTANCE-FAIL: {exc}")
    raise
