"""Actual CUDA Production Bake continuation; launch with run_frozen_input_acceptance.py.

Each phase runs in a fresh Blender process. Exact checkpoint/frozen restoration
is asserted; compiled-emitter numerical differences are evidence, not an
unapproved tolerance or release qualification. No CPU solver fallback exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tests" / "support"))
from _frozen_acceptance_contract import (
    array_differences,
    assert_exact_state,
    qualification_status,
    require_cuda,
)
from _test_device import script_device

START, PARTIAL, END, PREVIEW = 1, 3, 6, 5


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(scene, root):
    import bpy
    from blender_frozen_input_resume_acceptance import fixture as host_fixture

    host_fixture(scene, root)
    domain = scene.fluidsim.domain_object
    data = bpy.data.meshes.new("Production Domain Box")
    data.from_pydata(
        [(x, y, z) for x in (0, 3) for y in (0, 3) for z in (0, 3)],
        [],
        [
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ],
    )
    domain.data = data
    settings = domain.fluidsim_domain
    settings.resolution_divisions = 16
    settings.max_particles = 20000
    settings.leaf_capacity = 64
    settings.whitewater_enabled = False
    settings.surface_enabled = False
    settings.min_substeps = settings.max_substeps = 1
    settings.use_scene_gravity = False
    settings.gravity = (0, 0, -1)
    settings.pressure_iterations = 30
    scene.fluidsim.live = False
    scene.frame_start, scene.frame_end = START, END
    for obj in scene.objects:
        if obj.animation_data is None:
            continue
        obj.animation_data_clear()
        offset = (
            1.8
            if obj.name in {"Volume Geometry", "Static Collider", "Moving Collider"}
            else 0.6
        )
        for frame, x in ((START, offset), (END, offset + 0.4)):
            obj.location = (x, 0.8, 0.8)
            obj.keyframe_insert(data_path="location", frame=frame)
    field = bpy.data.objects["Frozen Field"]
    field.fluidsim.field_operation = "SCALAR"
    field.fluidsim.field_scalar_value = 0.25
    scene.frame_set(START)
    # Assign identities before saving: otherwise a fresh process would mint
    # different UUIDs from the baseline's first admission.
    from fluidSim_warp_blender51.blender.scene_inputs import persistent_source_key

    for obj in scene.objects:
        persistent_source_key(obj)


def load_checkpoint(cache_root, frame):
    from fluidSim_warp_blender51.io.cache_schema import load_manifest, manifest_path
    from fluidSim_warp_blender51.io.checkpoint import read_checkpoint

    manifest = load_manifest(manifest_path(cache_root))
    entry = manifest.frames[frame]
    payload = read_checkpoint(
        cache_root / entry.checkpoint_file,
        expected_sha256=entry.checkpoint_sha256,
        expected_size=entry.checkpoint_size,
        expected_frame=frame,
        expected_settings_signature=manifest.settings_signature,
        expected_source_signature=manifest.source_signature,
        expected_effect_signature=manifest.effect_signature,
    )
    return manifest, payload


def capture(session, manifest):
    from fluidSim_warp_blender51.io.checkpoint import capture_solver

    return capture_solver(
        session.solver,
        settings_signature=manifest.settings_signature,
        source_signature=manifest.source_signature,
        effect_signature=manifest.effect_signature,
        effect_inventory=manifest.effect_inventory,
    )


def assert_restored(expected, actual):
    from fluidSim_warp_blender51.io.checkpoint import (
        FROZEN_INPUT_ARRAY_PREFIX,
        FROZEN_INPUT_STATE_KEY,
    )

    assert expected.frame == actual.frame
    assert expected.sim_time == actual.sim_time
    assert_exact_state(
        {k: v for k, v in expected.scalars.items() if k != FROZEN_INPUT_STATE_KEY},
        dict(actual.scalars),
    )
    assert_exact_state(dict(expected.source_state), dict(actual.source_state))
    assert_exact_state(
        {
            k: v
            for k, v in expected.arrays.items()
            if not k.startswith(FROZEN_INPUT_ARRAY_PREFIX)
        },
        dict(actual.arrays),
    )


@contextmanager
def observe_inputs():
    """Observe real evaluated collider descriptors without modifying admission/step inputs."""
    from fluidSim_warp_blender51.blender.scene_inputs import SceneInputs

    original = SceneInputs.evaluate
    seen = {}

    def evaluate(self, scene, *args, **kwargs):
        value = original(self, scene, *args, **kwargs)
        if not kwargs.get("staged", False):
            seen["velocities"] = [
                list(map(float, c.linear_velocity)) for c in value.colliders
            ]
        return value

    SceneInputs.evaluate = evaluate
    try:
        yield seen
    finally:
        SceneInputs.evaluate = original


def bake(
    scene, root, phase, *, resume=False, rebake_from=None, cancel_at=None, fail_at=None
):
    import bpy
    from blender_frozen_input_resume_acceptance import fingerprint

    from fluidSim_warp_blender51.blender import cache_bake, production_bake
    from fluidSim_warp_blender51.blender.scene_inputs import (
        FrozenInputState,
        SceneInputs,
    )
    from fluidSim_warp_blender51.io.checkpoint import frozen_inputs

    cache_root = root / (
        "continued" if phase in {"partial", "resume", "rebake"} else phase
    )
    scene.fluidsim.cache_path = str(cache_root)
    expected = None
    if resume:
        _, expected = load_checkpoint(
            cache_root, PARTIAL if rebake_from is None else rebake_from - 1
        )
    traces = {}
    admission = {}
    with observe_inputs() as observed:

        class Backend(production_bake.ProductionBakeBackend):
            def initialize(self, *args, **kwargs):
                next_frame = super().initialize(*args, **kwargs)
                state = self.sess.scene_inputs.export_frozen_state()
                kinds = [record["kind"] for record in state.metadata["records"]]
                assert len(kinds) == 5 and kinds.count("MESH") == 2
                assert {"POINTS", "VOLUME", "ANALYTIC"} <= set(kinds)
                admission["frozen"] = fingerprint(state)
                admission["frame"] = scene.frame_current
                if expected is not None:
                    assert_restored(expected, capture(self.sess, self.manifest))
                    assert_exact_state(
                        fingerprint(FrozenInputState(*frozen_inputs(expected))),
                        admission["frozen"],
                    )
                    recaptured = SceneInputs()
                    try:
                        recaptured.evaluate(
                            scene,
                            bpy.context.evaluated_depsgraph_get(),
                            device="cpu",
                            staged=True,
                        )
                        assert (
                            fingerprint(recaptured.export_frozen_state())
                            != admission["frozen"]
                        )
                    finally:
                        recaptured.clear()
                    admission["exact_checkpoint_restore"] = True
                return next_frame

            def step_and_commit(self, scene, frame):
                if frame == fail_at:
                    raise RuntimeError("FROZEN_ACCEPTANCE_INJECTED_STEP_FAILURE")
                entry = super().step_and_commit(scene, frame)
                state = fingerprint(self.sess.scene_inputs.export_frozen_state())
                assert_exact_state(admission["frozen"], state)
                assert entry.particle_count > 0, "fixture must simulate nonempty liquid"
                traces[str(frame)] = {
                    "velocities": observed["velocities"],
                    "frozen": state,
                    "particle_count": entry.particle_count,
                }
                return entry

        run = cache_bake.BakeRun(
            scene,
            start=START,
            end=END,
            resume=resume,
            rebake_from=rebake_from,
            backend=Backend(),
        )
        try:
            while not run.finished:
                run.tick()
                if (
                    cancel_at is not None
                    and run.controller.last_committed_frame == cancel_at
                ):
                    run.request_cancel()
            status = run.controller.status.value
            expected_status = (
                "FAILED"
                if fail_at is not None
                else "CANCELLED"
                if cancel_at is not None
                else "COMPLETE"
            )
            assert run.controller.status.name == expected_status, run.controller.error
            if fail_at is not None:
                assert "FROZEN_ACCEPTANCE_INJECTED_STEP_FAILURE" in run.controller.error
        finally:
            run.close()
    return {"status": status, "admission": admission, "frames": traces}


def compare_continuation(root, phase, trace):
    from fluidSim_warp_blender51.io import cache

    baseline = json.loads((root / "baseline-report.json").read_text())["result"]
    numerical = {}
    for frame in range(PARTIAL + 1, END + 1):
        actual = trace["frames"][str(frame)]
        control = baseline["frames"][str(frame)]
        assert_exact_state(control["frozen"], actual["frozen"])
        assert_exact_state(control["velocities"], actual["velocities"])
        assert np.linalg.norm(actual["velocities"]) > 0, (
            "dynamic history control is vacuous"
        )
        numerical[str(frame)] = array_differences(
            cache.read_frame(str(root / "baseline"), frame),
            cache.read_frame(str(root / "continued"), frame),
        )
    write_json(root / f"{phase}-numerical-differences.json", numerical)
    return numerical


def preview_cases(scene, root):
    import bpy
    from blender_frozen_input_resume_acceptance import fingerprint

    from fluidSim_warp_blender51.blender import playback, scene_bridge
    from fluidSim_warp_blender51.io.checkpoint import capture_solver

    results = {}
    for terminal, original_live in product(
        ("finish", "cancel", "failure"), (False, True)
    ):
        with playback.suppress_mode_transitions():
            scene.fluidsim.live = False
        session = scene_bridge.session()
        if session.active:
            session.release(scene)
        with playback.suppress_frame_handlers():
            scene.frame_set(START)
            session.initialize(scene, depsgraph=bpy.context.evaluated_depsgraph_get())
            for frame in range(START, PREVIEW + 1):
                scene.frame_set(frame)
                session.sync_scene(scene, bpy.context.evaluated_depsgraph_get())
                scene_bridge.step_frame(
                    scene,
                    session,
                    timeline_frame=frame,
                    write_cache=False,
                    publish_outputs=False,
                )
        before = capture_solver(
            session.solver, settings_signature="preview", source_signature="preview"
        )
        frozen = fingerprint(session.scene_inputs.export_frozen_state())
        scene.fluidsim.live = original_live
        case = f"preview-{terminal}-live{int(original_live)}"
        trace = bake(
            scene,
            root,
            case,
            cancel_at=2 if terminal == "cancel" else None,
            fail_at=3 if terminal == "failure" else None,
        )
        assert scene.frame_current == PREVIEW
        assert scene.fluidsim.live is original_live
        assert session.active
        assert_restored(
            before,
            capture_solver(
                session.solver, settings_signature="preview", source_signature="preview"
            ),
        )
        assert_exact_state(
            frozen, fingerprint(session.scene_inputs.export_frozen_state())
        )
        # Restoration must prime dynamic history at the preview playhead too.
        with observe_inputs() as observed, playback.suppress_frame_handlers():
            scene.frame_set(PREVIEW + 1)
            session.sync_scene(scene, bpy.context.evaluated_depsgraph_get())
            control = json.loads((root / "baseline-report.json").read_text())["result"]
            assert_exact_state(
                control["frames"][str(PREVIEW + 1)]["velocities"],
                observed["velocities"],
            )
            assert np.linalg.norm(observed["velocities"]) > 0
            scene.frame_set(PREVIEW)
        results[case] = trace
    with playback.suppress_mode_transitions():
        scene.fluidsim.live = False
    session.release(scene)
    return results


def main(args):
    root = args.state_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": args.phase,
        "pid": os.getpid(),
        "qualified": False,
        "status": "failed",
        "qualification": qualification_status(),
        "script_sha256": digest(Path(__file__)),
        "source_sha256": {
            path: digest(ROOT / path)
            for path in (
                "fluidSim_warp_blender51/blender/production_bake.py",
                "fluidSim_warp_blender51/blender/cache_bake.py",
                "fluidSim_warp_blender51/blender/scene_inputs.py",
                "fluidSim_warp_blender51/blender/simulation_admission.py",
                "fluidSim_warp_blender51/io/checkpoint.py",
            )
        },
    }
    try:
        # Preflight before registering the add-on or creating/changing any scene.
        for suffix in ("CONFIG", "SCRIPTS", "EXTENSIONS", "DATAFILES"):
            profile = os.environ.get("BLENDER_USER_" + suffix, "")
            assert profile and Path(profile).is_absolute(), (
                "private absolute Blender profile required"
            )
        import bpy
        import warp as wp

        wp.init()
        available = [str(device) for device in wp.get_cuda_devices()]
        report["runtime"] = {
            "blender": bpy.app.version_string,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "requested_device": args.device,
            "cuda_devices": available,
        }
        if args.device not in available:
            raise RuntimeError(
                f"CUDA_REQUIRED: requested {args.device}; no matching CUDA device"
            )
        device = wp.get_device(args.device)
        require_cuda(device)
        wp.synchronize_device(device)
        report["runtime"] = {
            "blender": bpy.app.version_string,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": device.name,
        }
        if args.phase != "preflight":
            from tools.repo_addon_loader import enable_repo_addon

            enable_repo_addon(ROOT, default_set=False, persistent=True)
            from fluidSim_warp_blender51.preferences import (
                configured_device,
                set_configured_device,
            )

            set_configured_device(args.device)
            assert configured_device() == args.device
            if args.phase == "baseline":
                fixture(bpy.context.scene, root)
                bpy.ops.wm.save_as_mainfile(
                    filepath=str(root / "production-frozen.blend")
                )
            else:
                baseline = json.loads((root / "baseline-report.json").read_text())
                assert baseline["script_sha256"] == report["script_sha256"]
                assert baseline["source_sha256"] == report["source_sha256"]
                assert baseline["pid"] != os.getpid(), (
                    "continuation requires a fresh process"
                )
                assert baseline["runtime"] == report["runtime"], (
                    "runtime changed across continuation"
                )
                assert baseline["blend_sha256"] == digest(
                    root / "production-frozen.blend"
                )
                bpy.ops.wm.open_mainfile(filepath=str(root / "production-frozen.blend"))
            report["blend_sha256"] = digest(root / "production-frozen.blend")
            scene = bpy.context.scene
            if args.phase == "preview":
                report["result"] = preview_cases(scene, root)
            else:
                if args.phase == "rebake":
                    # Capture committed prefix bytes, then prove Re-bake preserves them.
                    manifest, _ = load_checkpoint(root / "continued", PARTIAL)
                    prefix = {
                        name: digest(root / "continued" / name)
                        for frame in range(START, PARTIAL + 1)
                        for name in (
                            manifest.frames[frame].frame_file,
                            manifest.frames[frame].checkpoint_file,
                        )
                    }
                report["result"] = bake(
                    scene,
                    root,
                    args.phase,
                    resume=args.phase in {"resume", "rebake"},
                    rebake_from=PARTIAL + 1 if args.phase == "rebake" else None,
                    cancel_at=PARTIAL if args.phase == "partial" else None,
                )
                if args.phase in {"resume", "rebake"}:
                    compare_continuation(root, args.phase, report["result"])
                if args.phase == "rebake":
                    assert prefix == {
                        name: digest(root / "continued" / name) for name in prefix
                    }
        report["status"] = "structural_checks_passed"
        return 0
    except Exception as exc:  # noqa: BLE001 - retain diagnostic evidence on any failed invariant
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return 1
    finally:
        write_json(root / f"{args.phase}-report.json", report)
        print("FROZEN-PRODUCTION-DIAGNOSTIC", json.dumps(report), flush=True)


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "baseline", "partial", "resume", "rebake", "preview"),
    )
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--device", default=script_device())
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(arguments(sys.argv[sys.argv.index("--") + 1 :])))
