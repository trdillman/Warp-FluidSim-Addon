"""Blender acceptance: the collider Body Type (Static / Animated / Dynamic).

Solver-free (``CUDA_VISIBLE_DEVICES=-1``; meshes are built on the CPU):

* ``collider_body_type`` is an enum over the stored ``collider_static`` and
  ``collider_dynamic`` flags, so a file saved before Body Type existed (only
  ``collider_dynamic``) reads back as Dynamic / Animated unchanged;
* a Static collider is read once per lifecycle: keyframing the object later
  neither moves its mesh nor gives it velocity, while an Animated collider
  follows the keyframes with a velocity;
* Reset (``SceneInputs.clear``) re-captures the Static pose.

Prints ``COLLIDER_BODY_TYPE_OK`` on success.

Run (needs Warp importable; use a private extensions dir):
  blender -b --factory-startup --python-use-system-env \\
      --python tests/acceptance/blender_collider_body_type_acceptance.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import bpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILURES = []


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _collider(name, location):
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.fluidsim.role = "COLLIDER"
    obj.fluidsim.enabled = True
    return obj


def _move(obj, frame, x):
    bpy.context.scene.frame_set(frame)
    obj.location.x = x
    obj.keyframe_insert("location", frame=frame)


def _gather(inputs, scene, frame):
    scene.frame_set(frame)
    descs = inputs.gather_colliders(
        scene, bpy.context.evaluated_depsgraph_get())
    return descs


def main():
    from tools.repo_addon_loader import enable_repo_addon

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)

    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    scene = bpy.context.scene
    scene.render.fps = 24
    bpy.ops.fluidsim.setup_scene()

    props = type(bpy.context.scene.objects[0].fluidsim).bl_rna.properties
    items = {i.identifier for i in props["collider_body_type"].enum_items}
    check("body type offers Static, Animated, Dynamic",
          items == {"STATIC", "ANIMATED", "DYNAMIC"}, repr(items))
    for name in ("collider_body_type", "collider_static"):
        check(f"{name} has a tooltip", len(props[name].description) > 20)

    probe = _collider("Probe", (0.0, 0.0, 0.0))
    fs = probe.fluidsim
    check("default is Animated (pre-Body-Type behaviour)",
          fs.collider_body_type == "ANIMATED"
          and not fs.collider_static and not fs.collider_dynamic)
    fs.collider_body_type = "STATIC"
    check("Static sets collider_static only",
          fs.collider_static and not fs.collider_dynamic)
    fs.collider_body_type = "DYNAMIC"
    check("Dynamic sets collider_dynamic and clears collider_static",
          fs.collider_dynamic and not fs.collider_static)
    fs.collider_body_type = "ANIMATED"
    check("Animated clears both",
          not fs.collider_dynamic and not fs.collider_static)
    fs.collider_dynamic = True  # what a file saved before Body Type holds
    check("a legacy Dynamic file reads back as Dynamic",
          fs.collider_body_type == "DYNAMIC")
    fs.collider_dynamic = False
    fs.collider_static = True
    fs.collider_dynamic = True
    check("Dynamic wins over a stale Static flag",
          fs.collider_body_type == "DYNAMIC")
    bpy.data.objects.remove(probe)

    still = _collider("StaticWall", (0.0, 0.0, 0.5))
    still.fluidsim.collider_body_type = "STATIC"
    moving = _collider("AnimatedWall", (0.0, 0.5, 0.5))
    moving.fluidsim.collider_body_type = "ANIMATED"
    for obj in (still, moving):
        _move(obj, 1, 0.0)
        _move(obj, 12, 1.1)  # 1.1 m in 11 frames

    inputs = SceneInputs()
    inputs.device = "cpu"
    first = {d.center: d for d in _gather(inputs, scene, 1)}
    check("both colliders are gathered", len(first) == 2, repr(sorted(first)))
    later = _gather(inputs, scene, 12)
    by_x = {round(d.center[1], 3): d for d in later}  # y separates the two
    static_desc, moving_desc = by_x[0.0], by_x[0.5]
    check("Static keeps its first-frame centre after the object moved",
          abs(static_desc.center[0]) < 1e-6, repr(static_desc.center))
    static_points = static_desc.mesh.points.numpy()
    check("Static mesh stays at its first-frame pose",
          abs(float(static_points[:, 0].mean())) < 1e-5,
          repr(float(static_points[:, 0].mean())))
    check("Static has no velocity",
          not any(static_desc.linear_velocity)
          and not any(static_desc.angular_velocity))
    check("Animated follows the keyframes",
          abs(moving_desc.center[0] - 1.1) < 1e-4, repr(moving_desc.center))
    check("Animated pushes with a velocity",
          abs(moving_desc.linear_velocity[0]) > 0.1,
          repr(moving_desc.linear_velocity))

    inputs.clear()
    again = {round(d.center[1], 3): d for d in _gather(inputs, scene, 12)}
    check("Reset re-captures the Static collider at the current pose",
          abs(again[0.0].center[0] - 1.1) < 1e-4, repr(again[0.0].center))

    print("COLLIDER_BODY_TYPE_OK" if not FAILURES else
          f"COLLIDER_BODY_TYPE_FAILED {FAILURES}", flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 - reported as a failure below
        traceback.print_exc()
        code = 1
    sys.exit(code)
