"""Blender/CUDA acceptance: a deforming collider deforms in a Production Bake (B30).

A collider deformed by an animated shape key keeps its transform, so the
evaluated-mesh cache relied on ``depsgraph.updates`` to notice the change. A
bake syncs after ``frame_set`` from ``evaluated_depsgraph_get()``, where that
list is always empty, so the bake simulated the first-frame shape throughout
while preview followed the deformation. Checks the collider extent the solver
receives each frame, in a bake and in live preview, against the shape key.

Prints ``DEFORMING-COLLIDER-BAKE-ACCEPTANCE-PASS`` and exits 0 when every check passes.

Run:
  blender --background --factory-startup --python-use-system-env --python tests/acceptance/blender_deforming_collider_bake_acceptance.py
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.repo_addon_loader import enable_repo_addon  # noqa: E402

FAILURES = []
START, END = 1, 5
NAME = "Deforming Collider"


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _deforming_collider(scene, domain_object):
    bpy.ops.mesh.primitive_cube_add(
        size=0.3, location=domain_object.matrix_world.translation.copy())
    collider = bpy.context.active_object
    collider.name = NAME
    collider.fluidsim.role = "COLLIDER"
    collider.shape_key_add(name="Basis")
    grow = collider.shape_key_add(name="Grow")
    for vertex in grow.data:
        vertex.co = vertex.co * 2.0
    grow.value = 0.0
    grow.keyframe_insert("value", frame=START)
    grow.value = 1.0
    grow.keyframe_insert("value", frame=END)
    scene.frame_set(START)
    return collider


def _expected_widths(scene, collider):
    widths = {}
    for frame in range(START, END + 1):
        scene.frame_set(frame)
        evaluated = collider.evaluated_get(bpy.context.evaluated_depsgraph_get())
        xs = [vertex.co.x for vertex in evaluated.to_mesh().vertices]
        evaluated.to_mesh_clear()
        widths[frame] = round(max(xs) - min(xs), 4)
    scene.frame_set(START)
    return widths


def main(cache_root):
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)
    from fluidSim_warp_blender51.blender import production_bake, scene_bridge, scene_inputs

    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim
    domain = scene_bridge.active_domain_settings(scene)
    domain.voxel_size = 0.12
    domain.whitewater_enabled = False
    domain.surface_enabled = False
    fs.cache_path = str(cache_root)
    fs.live = False
    collider = _deforming_collider(scene, fs.domain_object)
    expected = _expected_widths(scene, collider)
    check("fixture: the shape key deforms the collider",
          expected[START] < expected[END], expected)

    seen = {}
    original = scene_inputs.SceneInputs._evaluated_mesh

    def recording(self, obj, depsgraph, cache_key, **kwargs):
        entry, error = original(self, obj, depsgraph, cache_key, **kwargs)
        if obj.name == NAME and entry is not None:
            seen[scene.frame_current] = round(
                float(entry["aabb_max"][0] - entry["aabb_min"][0]), 4)
        return entry, error

    scene_inputs.SceneInputs._evaluated_mesh = recording
    try:
        result = production_bake.run_synchronous(scene, start=START, end=END)
        check("bake completes", result.last_committed_frame == END, result)
        check("the bake simulates the deformed collider every frame",
              seen == expected, (seen, expected))

        seen.clear()
        if scene_bridge.session().active:
            scene_bridge.session().release(scene)
        scene.frame_set(START)
        check("live preview initializes",
              bpy.ops.fluidsim.initialize() == {"FINISHED"})
        fs.live = True
        for frame in range(START + 1, END + 1):
            scene.frame_set(frame)
        check("preview follows the same deformation",
              all(seen.get(frame) == expected[frame]
                  for frame in range(START + 1, END + 1)), (seen, expected))
    finally:
        scene_inputs.SceneInputs._evaluated_mesh = original
        if scene_bridge.session().active:
            scene_bridge.session().release(scene)


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp(prefix="fluidsim-deforming-collider-"))
    try:
        main(root / "cache")
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if FAILURES:
        print("DEFORMING-COLLIDER-BAKE-ACCEPTANCE-FAIL", FAILURES, flush=True)
        sys.exit(1)
    print("DEFORMING-COLLIDER-BAKE-ACCEPTANCE-PASS", flush=True)
    sys.exit(0)
