"""Fresh-process Blender frozen-input/checkpoint acceptance, without a GPU solve.

Run --phase write then --phase read with the same private --state-root. Uses
real evaluated meshes, native point/volume/field plans and collider history;
CPU staging does not simulate liquid or certify numerical continuation. The
production-bake orchestration and malformed-state refusal have host tests;
CUDA continuation remains a separate required release gate.

Every invocation must use private BLENDER_USER_CONFIG/SCRIPTS/EXTENSIONS/DATAFILES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.repo_addon_loader import enable_repo_addon


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("write", "read"), required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def animate(obj):
    for frame, x in ((1, 0.0), (10, 2.0), (20, 4.0)):
        obj.location.x = x
        obj.keyframe_insert(data_path="location", frame=frame)


def mesh(scene, name, *, role="NONE"):
    data = bpy.data.meshes.new(name)
    data.from_pydata([(0, 0, 0), (0, 0.3, 0), (0, 0, 0.3)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.fluidsim.role = role
    obj.fluidsim.enabled = True
    return obj


def owner(scene, name, role, region):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    pg = obj.fluidsim
    pg.role = role
    pg.enabled = True
    pg.source_ui_schema_version = 5
    pg.mesh_evaluation_mode = "FIRST_SIM_FRAME"
    if role == "SOURCE":
        pg.source_region_kind = region
    elif role == "SINK":
        pg.sink_region_kind = region
    return obj


def fixture(scene, root):
    import openvdb

    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    domain = mesh(scene, "Frozen Domain")
    scene.fluidsim.domain_object = domain
    domain.fluidsim_domain.voxel_size = 0.1
    point_geometry = mesh(scene, "Point Geometry")
    animate(point_geometry)
    point = owner(scene, "Frozen Points", "SOURCE", "POINTS")
    point.fluidsim.point_input_object = point_geometry
    geometry = mesh(scene, "Mesh Geometry")
    animate(geometry)
    source = owner(scene, "Frozen Mesh", "SOURCE", "MESH_SHELL")
    source.fluidsim.mesh_object = geometry
    field = owner(scene, "Frozen Field", "FIELD", "ANALYTIC")
    animate(field)
    grid = openvdb.FloatGrid()
    grid.name = "density"
    grid.getAccessor().setValueOn((0, 0, 0), 1.0)
    volume_path = root / "frozen-volume.vdb"
    openvdb.write(str(volume_path), grids=[grid])
    data = bpy.data.volumes.new("Frozen Volume")
    data.filepath = str(volume_path)
    volume = bpy.data.objects.new("Volume Geometry", data)
    scene.collection.objects.link(volume)
    animate(volume)
    sink = owner(scene, "Frozen Volume Sink", "SINK", "VOLUME_GRID")
    sink.fluidsim.volume_input_object = volume
    for name, static in [("Static Collider", True), ("Moving Collider", False)]:
        collider = mesh(scene, name, role="COLLIDER")
        collider.fluidsim.collider_static = static
        collider.fluidsim.collider_dynamic = False
        collider.fluidsim.transfer_velocity = True
        animate(collider)
    scene.frame_set(1)


def evaluate(inputs, scene):
    return inputs.evaluate(
        scene, bpy.context.evaluated_depsgraph_get(), device="cpu", staged=True
    )


def fingerprint(state):
    return {
        "metadata": state.metadata,
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for key, value in state.arrays.items()
        },
    }


def main(args):
    import warp as wp

    enable_repo_addon(ROOT, default_set=False, persistent=True)
    from fluidSim_warp_blender51.blender.scene_inputs import (
        FrozenInputState,
        SceneInputs,
        validate_frozen_scene,
    )
    from fluidSim_warp_blender51.io import checkpoint

    root = args.state_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if args.phase == "write":
        fixture(scene, root)
        inputs = SceneInputs()
        evaluate(inputs, scene)
        expected = inputs.export_frozen_state()
        scene.frame_set(10)
        evaluate(inputs, scene)
        assert fingerprint(inputs.export_frozen_state()) == fingerprint(expected)
        payload = checkpoint.attach_frozen_inputs(
            checkpoint.CheckpointPayload(
                10, 0.5, "host-fixture-settings", "host-fixture-inputs"
            ),
            expected.metadata,
            expected.arrays,
        )
        digest = checkpoint.write_checkpoint_atomic(root / "checkpoint.npz", payload)
        (root / "expected.json").write_text(
            json.dumps(
                {"state": fingerprint(expected), "checkpoint_sha256": digest}, indent=2
            )
        )
        bpy.ops.wm.save_as_mainfile(filepath=str(root / "frozen.blend"))
        kinds = [r["kind"] for r in expected.metadata["records"]]
        assert kinds.count("MESH") == 2 and {"POINTS", "VOLUME", "ANALYTIC"} <= set(
            kinds
        )
        inputs.clear()
        result = {"records": len(kinds), "checkpoint_sha256": digest}
    else:
        bpy.ops.wm.open_mainfile(filepath=str(root / "frozen.blend"))
        scene = bpy.context.scene
        expected = json.loads((root / "expected.json").read_text())
        payload = checkpoint.read_checkpoint(
            root / "checkpoint.npz",
            expected_sha256=expected["checkpoint_sha256"],
            expected_frame=10,
            expected_settings_signature="host-fixture-settings",
            expected_source_signature="host-fixture-inputs",
        )
        state = validate_frozen_scene(
            FrozenInputState(*checkpoint.frozen_inputs(payload)), scene
        )
        resumed = SceneInputs()
        resumed.restore_frozen_state(state)
        evaluate(resumed, scene)
        assert fingerprint(resumed.export_frozen_state()) == expected["state"]
        recaptured = SceneInputs()
        evaluate(recaptured, scene)
        assert fingerprint(recaptured.export_frozen_state()) != expected["state"], (
            "moving-input counterexample missing"
        )
        scene.frame_set(11)
        continued = evaluate(resumed, scene)
        control = evaluate(recaptured, scene)
        # Moving Collider is the second collider, after the static sibling.
        assert len(continued.colliders) == len(control.colliders) == 2
        actual_velocity = np.asarray(continued.colliders[1].linear_velocity)
        np.testing.assert_array_equal(
            actual_velocity, control.colliders[1].linear_velocity
        )
        assert np.linalg.norm(actual_velocity) > 0
        assert fingerprint(resumed.export_frozen_state()) == expected["state"]
        result = {
            "fresh_process_roundtrip": True,
            "recapture_counterexample": True,
            "moving_collider_velocity": actual_velocity.tolist(),
            "records": len(state.metadata["records"]),
        }
        resumed.clear()
        recaptured.clear()
    report = {
        "phase": args.phase,
        "lane": "real_blender_host_staging_no_fluid_solve",
        "blender": bpy.app.version_string,
        "warp": wp.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "result": result,
    }
    (root / f"{args.phase}-report.json").write_text(json.dumps(report, indent=2))
    print("FROZEN-INPUT-RESUME-HOST-PASS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main(arguments())
