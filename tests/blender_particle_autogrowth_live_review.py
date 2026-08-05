"""Exercise particle auto-growth and leave the GUI open for review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

import bpy

RESULT_PATH = Path(os.environ["FLUIDSIM_PROFILE_SAFE_REVIEW_RESULT"])


def write_result(payload):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf8")


def prepare_review():
    from fluidSim_warp_blender51.blender import scene_bridge

    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.auto_grow_particles = True
    before = {
        "initial_reserve": int(fs.max_particles),
        "leaf_capacity": int(fs.leaf_capacity),
        "voxel_size": float(fs.voxel_size),
    }
    operator_result = sorted(bpy.ops.fluidsim.initialize())
    sess = scene_bridge.session()
    plan = sess.capacity_plan
    runtime_capacity = (
        int(sess.solver.particles.capacity) if sess.active else 0)
    after = {
        "initial_reserve": int(fs.max_particles),
        "leaf_capacity": int(fs.leaf_capacity),
        "voxel_size": float(fs.voxel_size),
    }
    errors = []
    if operator_result != ["FINISHED"]:
        errors.append(f"initialize returned {operator_result}, expected FINISHED")
    if not sess.active:
        errors.append("solver is not active after guarded auto-growth")
    if before != after:
        errors.append("authored settings changed during runtime growth")
    if runtime_capacity <= before["initial_reserve"]:
        errors.append("runtime particle pool did not grow")
    if plan.particle_demand.value is not None and runtime_capacity < plan.particle_demand.value:
        errors.append("runtime capacity is below planned initial demand")

    # Stage one restart-only edit after validating growth so the visible review
    # exercises the new single neutral pending surface without saving the file.
    fs.voxel_size = float(before["voxel_size"]) * 1.05
    pending = scene_bridge.pending_presentation(scene)
    if pending["action"] != "REINIT_PENDING":
        errors.append("restart-only review edit did not produce pending state")

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            area.spaces.active.show_region_ui = True
            area.tag_redraw()

    payload = {
        "ok": not errors,
        "ready_for_user_review": not errors,
        "blend": str(Path(bpy.data.filepath).resolve()),
        "operator_result": operator_result,
        "solver_active": bool(sess.active),
        "before": before,
        "after": after,
        "required_particles": getattr(plan.particle_demand, "value", None),
        "runtime_capacity": runtime_capacity,
        "growth_failure_policy": fs.particle_growth_failure_policy,
        "status": fs.status,
        "pending": pending,
        "errors": errors,
        "instructions": [
            "Open N > FluidSim > Simulation and Particles.",
            "Confirm the authored Initial Particle Reserve is unchanged.",
            "Confirm Current GPU Capacity shows the larger runtime pool.",
            "Review the staged Voxel Size change in the single neutral pending summary.",
            "Close this owned Blender window when finished.",
        ],
    }
    write_result(payload)
    print("PARTICLE-AUTOGROWTH-LIVE-REVIEW-READY " + json.dumps(payload, default=str))


try:
    prepare_review()
except Exception as exc:
    payload = {
        "ok": False,
        "ready_for_user_review": False,
        "errors": [repr(exc)],
        "traceback": traceback.format_exc(),
    }
    write_result(payload)
    print("PARTICLE-AUTOGROWTH-LIVE-REVIEW-FAIL " + json.dumps(payload, default=str))
