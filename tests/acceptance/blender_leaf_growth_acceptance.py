"""CUDA acceptance: the sparse-leaf allocation grows by itself during stepping.

Opens a saved demo and steps it twice through the real operator path:

1. a control run with every leaf the domain can hold, recording how many leaves
   each frame uses;
2. a growing run whose Leaf Capacity is just enough for the first frames, so the
   liquid outgrows it mid-run.

It requires that the growing run raised no error and paused nothing, that the
allocation grew and always contained its demand, and that the result matches
the control. (A starting capacity too small for the *initial* liquid is refused
at Initialize by the capacity planner, a separate guard, so the starting
capacity is derived from the control run.)

Environment:
  FLUIDSIM_LEAF_GROWTH_BLEND     demo to open (default demos/tsunami_parthenon.blend)
  FLUIDSIM_LEAF_GROWTH_FRAMES    frames to step (default 40)
  FLUIDSIM_LEAF_GROWTH_QUALITY   optional scene Quality tier (DRAFT/PREVIEW/FINAL)
  FLUIDSIM_LEAF_GROWTH_CAPACITY  optional starting Leaf Capacity for the growing run
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

MAX_CAPACITY = 262_144  # clamped to the domain's own leaf count by the grid


def check(label, condition, detail=""):
    if not condition:
        raise RuntimeError(f"{label}: {detail}")


def run(blend, capacity, frames, quality):
    # open_mainfile and enable_repo_addon both reload the add-on modules, so
    # the session must be looked up only after they ran.
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender.scene_bridge import (
        active_domain_settings,
        session,
    )

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
    solver = sess.solver
    check("solver after initialize", solver is not None, fs.status)
    start_capacity = int(solver.grid.capacity)
    counts, growths = [], []
    for frame in range(frames):
        result = bpy.ops.fluidsim.step(frames=1)
        check(f"step {frame + 1}", result == {"FINISHED"},
              f"{result} {fs.status} {sess.current_error(scene)}")
        solver = sess.solver
        counts.append(int(solver.grid.refresh_topology_counts()))
        growth = sess.last_leaf_growth
        if growth is not None and (not growths or growths[-1] != growth):
            growths.append(growth)
    count = int(solver.particles.refresh_host_count())
    position = solver.particles.position[:count].numpy().copy()
    report = {
        "start_capacity": start_capacity,
        "end_capacity": int(solver.grid.capacity),
        "total_leaf_cells": int(solver.grid.num_leaf_cells),
        "voxel_size": float(solver.settings.voxel_size),
        "leaves_per_frame": counts,
        "growths": [[g.previous, g.current, g.required] for g in growths],
        "particles": count,
        "frame": int(solver.frame),
        "status": fs.status,
    }
    bpy.ops.fluidsim.release()
    return report, position


def main():
    enable_repo_addon(str(ROOT), default_set=True, persistent=False)
    blend = Path(os.environ.get(
        "FLUIDSIM_LEAF_GROWTH_BLEND", ROOT / "demos" / "tsunami_parthenon.blend"))
    frames = int(os.environ.get("FLUIDSIM_LEAF_GROWTH_FRAMES", "40"))
    quality = os.environ.get("FLUIDSIM_LEAF_GROWTH_QUALITY", "")

    control, control_positions = run(blend, MAX_CAPACITY, frames, quality)
    print("LEAF-GROWTH control:", json.dumps(control, sort_keys=True), flush=True)
    check("control never grew", not control["growths"], control)
    counts = control["leaves_per_frame"]

    forced = os.environ.get("FLUIDSIM_LEAF_GROWTH_CAPACITY")
    if forced:
        capacity = int(forced)
    else:
        growth_frames = [i for i in range(1, len(counts))
                         if counts[i] > max(counts[:i])]
        check("the liquid never needed more leaves", growth_frames, counts)
        capacity = max(counts[:growth_frames[0]])
    print(f"LEAF-GROWTH starting capacity {capacity}", flush=True)

    grown, grown_positions = run(blend, capacity, frames, quality)
    print("LEAF-GROWTH grown:", json.dumps(grown, sort_keys=True), flush=True)
    check("capacity grew", grown["growths"]
          and grown["end_capacity"] > grown["start_capacity"], grown)
    check("every growth contained its demand",
          all(current >= required for _, current, required in grown["growths"]),
          grown)
    check("whole run completed", grown["frame"] == control["frame"], grown)
    check("same leaves per frame", grown["leaves_per_frame"] == counts,
          (grown["leaves_per_frame"], counts))
    check("same particle count", grown["particles"] == control["particles"],
          (grown["particles"], control["particles"]))
    delta = np.linalg.norm(grown_positions - control_positions, axis=1)
    print(f"LEAF-GROWTH drift max {float(delta.max()):.6f} m, "
          f"mean {float(delta.mean()):.6f} m", flush=True)
    check("grown run follows the control",
          float(delta.mean()) < control["voxel_size"], float(delta.mean()))
    print("LEAF-GROWTH-ACCEPTANCE-PASS", flush=True)


if __name__ == "__main__":
    main()
