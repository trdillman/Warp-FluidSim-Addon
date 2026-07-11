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
    fs.whitewater_enabled = False
    fs.surface_enabled = True
    fs.liquid_view_mode = "SURFACE_PARTICLES"
    fs.particle_display_fraction = 1.0

    check("setup_scene", bpy.ops.fluidsim.setup_scene() == {"FINISHED"})
    check("initialize", bpy.ops.fluidsim.initialize() == {"FINISHED"}, fs.status)

    from fluidSim_warp_blender51.blender.scene_bridge import session

    sess = session()
    check("session active", sess.active)
    check("cuda solver", str(sess.solver.device).startswith("cuda"), str(sess.solver.device))

    check("live toggle on", bpy.ops.fluidsim.toggle_live() == {"FINISHED"})
    before = sess.solver.frame
    scene.frame_set(scene.frame_current + 1)
    surface_after_first = bpy.data.objects.get("FluidSim Surface")
    surface_data_pointer = (
        surface_after_first.data.as_pointer() if surface_after_first else 0)
    scene.frame_set(scene.frame_current + 1)
    after = sess.solver.frame
    check("viewport live advanced two frames", after >= before + 2, f"{before} -> {after}")

    surf = bpy.data.objects.get("FluidSim Surface")
    parts = bpy.data.objects.get("FluidSim Particles")
    check("surface viewport object", surf is not None and surf.type == "MESH")
    check("surface viewport mesh updated", surf is not None and len(surf.data.vertices) > 0,
          f"{len(surf.data.vertices) if surf else 0} verts")
    check("surface mesh datablock reused",
          surf is not None and surf.data.as_pointer() == surface_data_pointer)
    check("particle viewport object", parts is not None)
    if parts is not None and hasattr(parts.data, "attributes"):
        attr = parts.data.attributes.get("position")
        count = len(attr.data) if attr is not None else 0
    elif parts is not None and hasattr(parts.data, "vertices"):
        count = len(parts.data.vertices)
    else:
        count = 0
    check("particle viewport data updated", count > 0, f"{count} points")
    check("scene stats updated", fs.stat_particles > 0 and fs.stat_voxels > 0,
          f"{fs.stat_particles} particles, {fs.stat_voxels} voxels")

    bpy.ops.fluidsim.toggle_live()
    bpy.ops.fluidsim.release()


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
