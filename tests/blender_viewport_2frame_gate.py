"""Production gate: prove live viewport simulation advances for two frames.

Run: blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py
"""

from __future__ import annotations

import os
import sys

import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.repo_addon_loader import enable_repo_addon

FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def main():
    enable_repo_addon(REPO_ROOT, default_set=True, persistent=True)
    scene = bpy.context.scene
    fs = scene.fluidsim
    fs.voxel_size = 0.10
    fs.max_particles = 120_000
    fs.max_whitewater = 20_000
    fs.whitewater_enabled = True
    fs.surface_enabled = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0

    check("setup_scene", bpy.ops.fluidsim.setup_scene() == {"FINISHED"})
    from fluidSim_warp_blender51.blender.scene_bridge import session

    sess = session()
    fs.live = True
    bpy.ops.fluidsim.release()
    check("saved live scene starts inactive", not sess.active)
    scene.frame_set(scene.frame_current + 1)
    check("timeline auto-initialized solver", sess.active)
    check("cuda solver", str(sess.solver.device).startswith("cuda"), str(sess.solver.device))
    after_first = sess.solver.frame
    surface_after_first = bpy.data.objects.get("FluidSim Surface")
    surface_data_pointer = (
        surface_after_first.data.as_pointer() if surface_after_first else 0)
    scene.frame_set(scene.frame_current + 1)
    after = sess.solver.frame
    check("viewport timeline advanced two frames", after >= after_first + 1,
          f"{after_first} -> {after}")

    surf = bpy.data.objects.get("FluidSim Surface")
    parts = bpy.data.objects.get("FluidSim Particles")
    check("surface viewport object", surf is not None and surf.type == "MESH")
    check("surface viewport mesh updated", surf is not None and len(surf.data.vertices) > 0,
          f"{len(surf.data.vertices) if surf else 0} verts")
    check("surface mesh datablock reused",
          surf is not None and surf.data.as_pointer() == surface_data_pointer)
    check("particle viewport object", parts is not None)
    count = live_point_count(parts)
    check("particle viewport data updated", count > 0, f"{count} points")
    check("scene stats updated", fs.stat_particles > 0 and fs.stat_voxels > 0,
          f"{fs.stat_particles} particles, {fs.stat_voxels} voxels")

    check_pointcloud_churn(scene)
    check_drifting_count_reuses_datablock()

    bpy.ops.fluidsim.release()


def check_drifting_count_reuses_datablock():
    """A count that drifts inside the capacity band must not realloc.

    This scene's solver count happens to be constant and its whitewater stays
    empty, so drive the writer directly against the real object. Before
    capacity padding, reuse required an exactly equal count, so every one of
    these writes swapped the datablock -- which on the live path happens from
    inside frame_change_post.
    """
    import numpy as np

    from fluidSim_warp_blender51.blender import viewport

    parts = bpy.data.objects.get(viewport.PARTICLES_NAME)
    if parts is None or parts.type != "POINTCLOUD":
        check("drifting count needs a particle cloud", False)
        return

    def write(count):
        viewport._swap_points(
            parts, np.zeros((count, 3), dtype=np.float32), 0.05)
        return parts.data.as_pointer(), live_point_count(parts)

    base = 60_000
    write(base)
    capacity = len(parts.data.points)
    samples = [write(n) for n in
               (base, base - 1_000, base - 2_000, base + 1_000, base)]
    counts = [n for _p, n in samples]
    pointers = {p for p, _n in samples}
    check("drift stayed inside the capacity band",
          max(counts) <= capacity, f"{max(counts)} of {capacity}")
    check("drift actually moved the point count",
          len(set(counts)) > 1, f"counts {counts}")
    check("drifting point count reuses the attached cloud",
          len(pointers) == 1, f"{len(pointers)} datablocks, counts {counts}")
    check("drifted padding is parked at radius 0", tail_radii_are_zero(parts))

    # Leaving the band reallocs -- and must not leak the cloud it replaced.
    datablocks = len(bpy.data.pointclouds)
    grown, grown_count = write(capacity + 1)
    check("growing past capacity reallocates",
          grown not in pointers, f"count {grown_count}")
    check("realloc does not leak the replaced cloud",
          len(bpy.data.pointclouds) <= datablocks,
          f"{datablocks} -> {len(bpy.data.pointclouds)}")


def check_pointcloud_churn(scene, frames=10):
    """Playback must not allocate a point-cloud datablock per frame.

    Every datablock swap is an ``obj.data = <new ID>`` from inside
    frame_change_post, which forces a depsgraph relations rebuild on an
    object mid-evaluation. Padded capacity means the count can drift without
    one; a realloc only happens when the count leaves the band.
    """
    from fluidSim_warp_blender51.blender.viewport import (
        PARTICLES_NAME, WHITEWATER_NAME)

    tracked = {}
    for name in (PARTICLES_NAME, WHITEWATER_NAME):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "POINTCLOUD":
            check(f"{name} is a point cloud", False, f"{obj and obj.type}")
            continue
        tracked[name] = {"obj": obj, "samples": []}

    datablocks_before = len(bpy.data.pointclouds)
    for _ in range(frames):
        scene.frame_set(scene.frame_current + 1)
        for state in tracked.values():
            obj = state["obj"]
            state["samples"].append(
                (obj.data.as_pointer(), live_point_count(obj)))

    check("no point-cloud datablock leak across playback",
          len(bpy.data.pointclouds) <= datablocks_before,
          f"{datablocks_before} -> {len(bpy.data.pointclouds)}")

    for name, state in tracked.items():
        samples = state["samples"]
        reallocs = sum(1 for a, b in zip(samples, samples[1:])
                       if a[0] != b[0])
        # The lock: the old code swapped the datablock on every count change
        # (and unconditionally for whitewater), so a frame where the count
        # moved and the pointer held is only reachable with padding.
        reused_across_a_count_change = any(
            a[0] == b[0] and a[1] != b[1]
            for a, b in zip(samples, samples[1:]))
        counts = [n for _p, n in samples]
        check(f"{name} datablock churn is not per-frame",
              reallocs <= frames // 2,
              f"{reallocs} reallocs over {frames} frames, counts "
              f"{min(counts)}..{max(counts)}")
        if len(set(counts)) > 1:
            check(f"{name} datablock survives a count change",
                  reused_across_a_count_change,
                  f"counts {min(counts)}..{max(counts)}")
        check(f"{name} padding is parked at radius 0",
              tail_radii_are_zero(state["obj"]))


def live_point_count(obj):
    from fluidSim_warp_blender51.blender import viewport

    reader = getattr(viewport, "live_point_count", None)
    if reader is not None:
        return reader(obj)
    # Unpadded build: the datablock length is the count. Keeps this gate a
    # meaningful negative control against a checkout without capacity padding.
    return len(obj.data.points if obj.type == "POINTCLOUD" else obj.data.vertices)


def tail_radii_are_zero(obj):
    """Padded slots must be invisible, or the viewport shows phantom points."""
    attr = obj.data.attributes.get("radius")
    if attr is None:
        return False
    radii = [0.0] * len(attr.data)
    attr.data.foreach_get("value", radii)
    return all(r == 0.0 for r in radii[live_point_count(obj):])


try:
    main()
except Exception as exc:  # noqa: BLE001 - Blender may return 0 for script exceptions
    print(f"VIEWPORT-2FRAME-GATE-FAIL unexpected exception: {exc!r}")
    sys.exit(1)

print()
if FAILURES:
    print(f"VIEWPORT-2FRAME-GATE-FAIL ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("VIEWPORT-2FRAME-GATE-PASS")
sys.exit(0)
