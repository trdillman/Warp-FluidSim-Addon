"""Blender acceptance: Dynamic collider settings, admission and pose publication (4.0 lane R).

Solver-free (``CUDA_VISIBLE_DEVICES=-1``). docs/specs/4.0/SPEC-rigid-coupling.md:

* the five collider settings exist on the object, each with a tooltip;
* ``gather_colliders`` describes a Dynamic collider (identity, density, drag,
  initial velocity, locks) from its *authored* pose, even when published
  keyframes have moved the object;
* the cache identity (source signature / frame stamp) hashes the authored pose,
  so publishing poses never makes a bake look stale;
* admission refuses a source/sink built on a Dynamic collider, with a reason;
* cache playback publishes each frame's v3.1 rigid record as ``matrix_world``
  keyframes in a FluidSim-owned action, and a Blender rigid body on the object
  is set to kinematic (Animated) with a reason;
* Reset clears the published keyframes and restores the authored pose;
* the Alembic and USD exports carry the collider's solved motion.

Prints ``RIGID_PUBLICATION_OK`` on success.

Run (needs Warp importable, not a CUDA device; use a private extensions dir):
  blender -b --factory-startup --python-use-system-env \\
      --python tests/acceptance/blender_rigid_publication_acceptance.py
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "tests" / "cache_io" / "fixtures"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

FAILURES = []
FRAMES = 4
REST_LOCATION = (0.4, -0.2, 0.6)
SETTINGS = ("collider_dynamic", "collider_density", "collider_drag",
            "collider_initial_velocity", "collider_lock_axes")


def check(name, condition, detail=""):
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        FAILURES.append(name)


def _records(uid, rest_com):
    """A body that rises and turns about Z, one record per frame."""
    records = {}
    for frame in range(1, FRAMES + 1):
        angle = 0.2 * (frame - 1)
        records[frame] = {
            "object_uid": uid,
            "pos": (rest_com[0] + 0.05 * (frame - 1), rest_com[1],
                    rest_com[2] + 0.1 * (frame - 1)),
            "quat": (math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)),
            "lin_vel": (0.0, 0.0, 0.1),
            "ang_vel": (0.0, 0.0, 0.2),
            "rest_com": tuple(rest_com),
        }
    return records


def _matrix(obj):
    return np.asarray(obj.matrix_world, dtype=np.float64)


def main(work):
    from tools.repo_addon_loader import enable_repo_addon

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    enable_repo_addon(str(REPO_ROOT), default_set=True, persistent=False)

    from tiny_cache import build_tiny_cache

    from fluidSim_warp_blender51.blender import (
        cache_identity, interchange, playback, rigid_publication, scene_bridge,
        simulation_admission)
    from fluidSim_warp_blender51.blender.scene_inputs import (
        SceneInputs, persistent_source_key)
    from fluidSim_warp_blender51.io import cache_schema
    from fluidSim_warp_blender51.io.cache_schema import RigidBodyRecord
    from fluidSim_warp_blender51.sim import rigid_scene

    scene = bpy.context.scene
    bpy.ops.fluidsim.setup_scene()
    fs = scene.fluidsim

    # Settings and tooltips.
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=REST_LOCATION)
    cork = bpy.context.active_object
    cork.name = "Cork"
    props = type(cork.fluidsim).bl_rna.properties
    for name in SETTINGS:
        check(f"setting {name} exists with a tooltip",
              name in props and len(props[name].description) > 20)
    check("lock axes are six booleans",
          "collider_lock_axes" in props and props["collider_lock_axes"].array_length == 6)
    cork.fluidsim.role = "COLLIDER"
    cork.fluidsim.enabled = True
    cork.fluidsim.collider_dynamic = True
    cork.fluidsim.collider_density = 300.0
    cork.fluidsim.collider_drag = 0.5
    cork.fluidsim.collider_initial_velocity = (0.0, 0.0, -1.0)
    cork.fluidsim.collider_lock_axes = (False, True, False, True, False, False)
    bpy.context.view_layer.objects.active = cork
    bpy.ops.rigidbody.object_add()
    check("cork has a Blender rigid body", cork.rigid_body is not None)
    if cork.rigid_body is not None:
        cork.rigid_body.kinematic = False
    rest = _matrix(cork)
    uid = persistent_source_key(cork)

    # Admission refusal: a source built on the Dynamic collider.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0.1, 0.1, 0.9))
    tap = bpy.context.active_object
    tap.name = "Tap"
    tap.fluidsim.role = "SOURCE"
    tap.fluidsim.enabled = True
    tap.fluidsim.mesh_object = cork
    refusal = simulation_admission.rigid_refusal(scene.objects)
    check("source on a Dynamic collider is refused",
          "Tap" in refusal and "Cork" in refusal, repr(refusal))
    tap.fluidsim.mesh_object = None
    check("no refusal once the conflict is removed",
          simulation_admission.rigid_refusal(scene.objects) == "")
    bpy.data.objects.remove(tap)

    signature_before = cache_identity.source_signature(
        scene, bpy.context.evaluated_depsgraph_get())

    # A v3.1 cache whose frames carry the cork's pose.
    base = build_tiny_cache(work / "cache", frame_count=FRAMES,
                            settings_signature=cache_identity.settings_signature(scene))
    fs.cache_path = str(base)
    manifest_file = cache_schema.manifest_path(base)
    manifest = cache_schema.load_manifest(manifest_file)
    records = _records(uid, REST_LOCATION)
    for frame, record in records.items():
        manifest.frames[frame] = replace(
            manifest.frames[frame],
            rigid_bodies=(RigidBodyRecord.from_dict(record),))
    cache_schema.write_manifest_atomic(manifest_file, manifest)

    playback.set_cache_playback_enabled(scene, True)
    for frame in range(1, FRAMES + 1):
        scene.frame_set(frame)
        playback.play_cache_frame(scene_bridge.session(), scene, frame)
        expected = rigid_scene.pose_matrix(records[frame], rest)
        error = float(np.abs(_matrix(cork) - expected).max())
        check(f"frame {frame} pose published", error < 1e-5, f"max error {error:.2e}")

    action = getattr(cork.animation_data, "action", None)
    check("poses are keyframes in a FluidSim-owned action",
          action is not None and bool(action.get(rigid_scene.OWNED_ACTION_TAG)))
    check("owned keyframes are not refused as foreign animation",
          simulation_admission.rigid_refusal(scene.objects) == "")
    check("rigid body is set to kinematic",
          cork.rigid_body is not None and cork.rigid_body.kinematic)
    check("kinematic reason recorded",
          rigid_scene.KINEMATIC_REASON in str(cork.get(rigid_publication.REASON_PROP, "")))
    # Keyframes play back with no cache and no handler.
    playback.set_cache_playback_enabled(scene, False)
    fs.cache_playback_enabled = False
    scene.frame_set(3)
    error = float(np.abs(_matrix(cork) - rigid_scene.pose_matrix(records[3], rest)).max())
    check("keyframes replay the pose without cache playback", error < 1e-5,
          f"max error {error:.2e}")

    # Inputs and identity read the authored pose, not the published one.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    check("source signature ignores published poses",
          cache_identity.source_signature(scene, depsgraph) == signature_before)
    inputs = SceneInputs()
    inputs.device = "cpu"
    descriptors = inputs.gather_colliders(scene, depsgraph)
    check("one collider gathered", len(descriptors) == 1, str(len(descriptors)))
    if descriptors:
        desc = descriptors[0]
        check("descriptor is Dynamic with the object identity",
              desc.dynamic and desc.object_uid == uid)
        check("descriptor carries density, drag, velocity and locks",
              desc.density == 300.0 and desc.drag == 0.5
              and tuple(desc.initial_velocity) == (0.0, 0.0, -1.0)
              and tuple(desc.lock_axes) == (False, True, False, True, False, False))
        points = desc.mesh.points.numpy()
        centre = points.mean(axis=0)
        check("descriptor mesh is in the authored pose",
              np.allclose(centre, REST_LOCATION, atol=1e-5), str(centre))
    inputs.clear()

    # Alembic carries the solved motion.
    playback.set_cache_playback_enabled(scene, True)
    fs.cache_playback_enabled = True
    abc = work / "rigid.abc"
    plan = interchange.plan_export(scene)
    check("export plan includes the Dynamic collider",
          any(obj is cork for obj in plan.carried))
    result = bpy.ops.fluidsim.export_alembic(filepath=str(abc))
    check("export finished", "FINISHED" in result, str(result))
    fresh = bpy.data.scenes.new("ImportCheck")
    bpy.context.window.scene = fresh
    before = set(fresh.objects)
    bpy.ops.wm.alembic_import(filepath=str(abc), as_background_job=False)
    imported = [o for o in fresh.objects if o not in before and o.name.startswith("Cork")]
    check("cork imported from the Alembic file", len(imported) == 1,
          str([o.name for o in fresh.objects if o not in before]))
    if imported:
        carried = imported[0]
        for frame in range(1, FRAMES + 1):
            fresh.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            world = np.asarray(carried.evaluated_get(depsgraph).matrix_world)
            expected = rigid_scene.pose_matrix(records[frame], rest)
            error = float(np.abs(world[:3, 3] - expected[:3, 3]).max())
            check(f"alembic frame {frame} carries the pose", error < 1e-4,
                  f"max error {error:.2e}")
    bpy.context.window.scene = scene

    # USD carries it too, as time-sampled transforms with no sampler modifier.
    usda = work / "rigid.usda"
    usd_plan = interchange.plan_usd_export(scene)
    check("USD export plan includes the Dynamic collider",
          any(obj is cork for obj in usd_plan.carried))
    check("USD export gives the collider no sampler",
          not any(obj is cork for obj in usd_plan.objects))
    result = bpy.ops.fluidsim.export_usd(filepath=str(usda))
    check("USD export finished", "FINISHED" in result, str(result))
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usda))
    prims = [p for p in stage.Traverse() if p.GetName() == "Cork"]
    check("cork prim in the USD file", len(prims) == 1,
          str([str(p.GetPath()) for p in stage.Traverse()]))
    if prims:
        xform = UsdGeom.Xformable(prims[0])
        samples = set(xform.GetTimeSamples())
        check("cork transform is sampled every frame",
              all(frame in samples for frame in range(1, FRAMES + 1)), str(sorted(samples)))
        for frame in range(1, FRAMES + 1):
            # Gf matrices are row-vector; transpose to Blender's matrix_world.
            gf = xform.ComputeLocalToWorldTransform(frame)
            world = np.array([[gf[i][j] for j in range(4)] for i in range(4)]).T
            expected = rigid_scene.pose_matrix(records[frame], rest)
            error = float(np.abs(world - expected).max())
            check(f"usd frame {frame} carries the pose", error < 1e-4,
                  f"max error {error:.2e}")

    # Reset clears the solver output and restores the authored pose.
    cleared = rigid_publication.clear_poses(scene)
    check("reset clears one published body", cleared == 1, str(cleared))
    check("owned action removed",
          cork.animation_data is None or cork.animation_data.action is None)
    check("authored pose restored", np.allclose(_matrix(cork), rest, atol=1e-6))
    check("rest marker removed", rigid_publication.REST_PROP not in cork.keys())


if __name__ == "__main__":
    work = Path(tempfile.mkdtemp(prefix="fsrigid_"))
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        main(work)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        FAILURES.append("exception")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if hidden is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = hidden
    if FAILURES:
        print(f"RIGID_PUBLICATION_FAILED {FAILURES}", flush=True)
        sys.exit(1)
    print("RIGID_PUBLICATION_OK", flush=True)
