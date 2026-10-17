"""Blender/CUDA acceptance: Dynamic bodies rest on a Static collider in a bake.

A Production Bake of a scene with a Static shelf and two Dynamic cubes dropped
onto it, one above the other:

* the lower cube falls, lands on the shelf and stays on it (never through);
* the upper cube lands on the lower one and stays on it (body-body contact);
* the shelf is never moved: it has no rigid record, no published keyframes,
  and its world matrix is unchanged after the bake.

Prints ``STATIC-COLLIDER-BAKE-ACCEPTANCE-PASS`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_static_collider_bake_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
START, END = 1, 48


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _box(name, location, scale, body_type):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    obj.fluidsim.role = "COLLIDER"
    obj.fluidsim.enabled = True
    obj.fluidsim.collider_body_type = body_type
    return obj


def _bounds(obj):
    corners = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    return corners.min(axis=0), corners.max(axis=0)


def main(cache_root):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import production_bake, scene_bridge
    from fluidSim_warp_blender51.blender.scene_inputs import persistent_source_key
    from fluidSim_warp_blender51.io import cache_schema

    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    domain = scene_bridge.active_domain_settings(scene)
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    fs.cache_path = str(cache_root)
    fs.live = False
    from mathutils import Vector

    corners = np.array([fs.domain_object.matrix_world @ Vector(c)
                        for c in fs.domain_object.bound_box])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    extent = hi - lo
    voxel = float(extent.min()) / 24.0
    domain.voxel_size = voxel
    mid = (lo + hi) / 2.0

    # A Static shelf across the middle of the domain, a third of the way up.
    shelf_top = float(lo[2] + 0.35 * extent[2])
    thickness = 2.0 * voxel
    shelf = _box("Shelf", (mid[0], mid[1], shelf_top - thickness / 2.0),
                 (0.6 * extent[0], 0.6 * extent[1], thickness), "STATIC")
    half = 0.08 * float(extent.min())
    low = _box("Low", (mid[0], mid[1], shelf_top + half + 0.05 * extent[2]),
               (2 * half, 2 * half, 2 * half), "DYNAMIC")
    high = _box("High", (mid[0], mid[1], shelf_top + 3 * half + 0.15 * extent[2]),
                (2 * half, 2 * half, 2 * half), "DYNAMIC")
    for obj in (low, high):
        obj.fluidsim.collider_density = 2000.0
    bpy.context.view_layer.update()
    shelf_rest = np.asarray(shelf.matrix_world, dtype=np.float64).copy()
    shelf_top = float(_bounds(shelf)[1][2])
    check("fixture: body types", (shelf.fluidsim.collider_body_type, low.fluidsim.collider_body_type)
          == ("STATIC", "DYNAMIC"))

    result = production_bake.run_synchronous(scene, start=START, end=END)
    check("bake completes", result.last_committed_frame == END, result)
    if scene_bridge.session().active:
        scene_bridge.session().release(scene)

    manifest = cache_schema.load_manifest(cache_schema.manifest_path(Path(fs.cache_path)))
    uids = {persistent_source_key(obj): obj.name for obj in (shelf, low, high)}
    track = {"Low": {}, "High": {}}
    published = set()
    for frame, entry in manifest.frames.items():
        for record in entry.rigid_bodies:
            name = uids.get(record.object_uid, record.object_uid)
            published.add(name)
            if name in track:
                track[name][frame] = record
    check("only the Dynamic cubes are solved", published == {"Low", "High"}, published)
    check("every frame carries both cubes",
          all(len(track[n]) >= END - START for n in track), {n: len(t) for n, t in track.items()})
    if FAILURES:
        return

    tol = 1.5 * voxel
    lows = [track["Low"][f].pos[2] - half for f in sorted(track["Low"])]
    highs = [track["High"][f].pos[2] - half for f in sorted(track["High"])]
    low_tops = [track["Low"][f].pos[2] + half for f in sorted(track["Low"])]
    final_low, final_high = track["Low"][max(track["Low"])], track["High"][max(track["High"])]
    print(f"voxel={voxel:.4f} shelf_top={shelf_top:.4f} low_bottom={lows[-1]:.4f} "
          f"high_bottom={highs[-1]:.4f} low_top={low_tops[-1]:.4f}", flush=True)
    check("the lower cube never passes through the shelf",
          min(lows) >= shelf_top - tol, (min(lows), shelf_top))
    check("the lower cube rests on the shelf",
          abs(lows[-1] - shelf_top) <= tol, (lows[-1], shelf_top))
    check("the upper cube never passes through the lower one",
          all(h >= t - tol for h, t in zip(highs, low_tops)))
    check("the upper cube rests on the lower one",
          abs(highs[-1] - low_tops[-1]) <= tol, (highs[-1], low_tops[-1]))
    for name, record in (("lower", final_low), ("upper", final_high)):
        check(f"the {name} cube has settled", abs(record.lin_vel[2]) <= 0.2, record.lin_vel)

    for frame in (START, END // 2, END):
        scene.frame_set(frame)
        error = float(np.abs(np.asarray(shelf.matrix_world) - shelf_rest).max())
        check(f"shelf unmoved at frame {frame}", error < 1e-6, f"{error:.2e}")
    check("shelf has no published keyframes",
          shelf.animation_data is None or shelf.animation_data.action is None)


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(prefix="fluidsim-static-collider-"))
    try:
        main(root / "cache")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        if bridge := sys.modules.get("fluidSim_warp_blender51.blender.scene_bridge"):
            session = bridge.session()
            if session.active:
                session.release(bpy.context.scene)
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print("STATIC-COLLIDER-BAKE-ACCEPTANCE-FAIL", FAILURES, flush=True)
        sys.exit(1)
    print("STATIC-COLLIDER-BAKE-ACCEPTANCE-PASS", flush=True)
    sys.exit(0)
