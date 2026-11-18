"""CUDA acceptance: the sparse-leaf allocation grows by itself during stepping.

Opens a saved demo, starts it with a deliberately small Leaf Capacity so the
liquid outgrows it, steps through the real operator path, and requires that
(1) capacity grew automatically with no error or pause, and (2) the grown run
matches a control run that started with plenty of leaves.

Environment:
  FLUIDSIM_LEAF_GROWTH_BLEND    demo to open (default demos/tsunami_parthenon.blend)
  FLUIDSIM_LEAF_GROWTH_CAPACITY starting Leaf Capacity for the growing run (default 512)
  FLUIDSIM_LEAF_GROWTH_FRAMES   frames to step (default 40)
  FLUIDSIM_LEAF_GROWTH_QUALITY  optional scene Quality tier (DRAFT/PREVIEW/FINAL)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402


def check(label, condition, detail=""):
    if not condition:
        raise RuntimeError(f"{label}: {detail}")


def run(capacity, frames, quality):
    from fluidSim_warp_blender51.blender.scene_bridge import (
        active_domain_settings,
        session,
    )

    bpy.ops.wm.open_mainfile(filepath=str(BLEND))
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    scene = bpy.context.scene
    fs = scene.fluidsim
    domain = active_domain_settings(scene)
    domain.leaf_capacity = int(capacity)
    if quality:
        fs.quality = quality
    fs.live = False
    scene.frame_set(scene.frame_start)
    sess = session()
    if sess.active:
        sess.release(scene)
    check("initialize", bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status)
    start_capacity = int(sess.solver.grid.capacity)
    growths = []
    for frame in range(frames):
        result = bpy.ops.fluidsim.step(frames=1)
        check(f"step {frame + 1}", result == {"FINISHED"},
              f"{result} {fs.status} {sess.current_error(scene)}")
        growth = sess.last_leaf_growth
        if growth is not None and (not growths or growths[-1] != growth):
            growths.append(growth)
    solver = sess.solver
    count = int(solver.particles.refresh_host_count())
    position = solver.particles.position[:count].numpy().copy()
    report = {
        "start_capacity": start_capacity,
        "end_capacity": int(solver.grid.capacity),
        "total_leaf_cells": int(solver.grid.num_leaf_cells),
        "voxel_size": float(solver.settings.voxel_size),
        "growths": [[g.previous, g.current, g.required] for g in growths],
        "particles": count,
        "frame": int(solver.frame),
        "status": fs.status,
    }
    bpy.ops.fluidsim.release()
    return report, position


def main():
    global BLEND
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    BLEND = Path(os.environ.get(
        "FLUIDSIM_LEAF_GROWTH_BLEND", ROOT / "demos" / "tsunami_parthenon.blend"))
    capacity = int(os.environ.get("FLUIDSIM_LEAF_GROWTH_CAPACITY", "512"))
    frames = int(os.environ.get("FLUIDSIM_LEAF_GROWTH_FRAMES", "40"))
    quality = os.environ.get("FLUIDSIM_LEAF_GROWTH_QUALITY", "")

    grown, grown_positions = run(capacity, frames, quality)
    print("LEAF-GROWTH grown run:", json.dumps(grown, sort_keys=True), flush=True)
    check("capacity grew", grown["growths"] and
          grown["end_capacity"] > grown["start_capacity"], grown)
    check("every growth contained its demand",
          all(current >= required for _, current, required in grown["growths"]),
          grown)
    check("whole run completed", grown["frame"] >= frames, grown)

    control, control_positions = run(
        max(grown["end_capacity"], 8192), frames, quality)
    print("LEAF-GROWTH control run:", json.dumps(control, sort_keys=True), flush=True)
    check("control never grew", not control["growths"], control)
    check("same particle count", grown["particles"] == control["particles"],
          (grown["particles"], control["particles"]))
    drift = float(np.max(np.abs(grown_positions - control_positions)))
    mean_drift = float(np.mean(np.linalg.norm(
        grown_positions - control_positions, axis=1)))
    print(f"LEAF-GROWTH max drift {drift:.6f} m, mean {mean_drift:.6f} m", flush=True)
    check("grown run follows the control",
          mean_drift < control["voxel_size"], mean_drift)
    print("LEAF-GROWTH-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    main()
