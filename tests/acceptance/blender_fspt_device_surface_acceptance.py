"""Native final-device surface proof against the real Blender publication owner.

This is a small translated fluid fixture, not a playable demo or an installed
renderer/pool qualification. The GPU route still supplies only positions and
indices; no normal, material-attribute or modifier parity is inferred.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as base


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def evaluated_surface(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    graph.update()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=graph)
    try:
        mesh.calc_loop_triangles()
        vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.vertices.foreach_get("co", vertices.ravel())
        mesh.loop_triangles.foreach_get("vertices", indices.ravel())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        vertices = (vertices @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
        if np.linalg.det(matrix[:3, :3]) < 0:
            indices = indices[:, [0, 2, 1]]
        return vertices, indices
    finally:
        evaluated.to_mesh_clear()


def require_parity(actual, expected):
    positions, triangles = actual
    reference, reference_triangles = expected
    assert (
        positions.shape == reference.shape
        and triangles.shape == reference_triangles.shape
    )
    assert np.isfinite(positions).all() and len(triangles) > 0
    np.testing.assert_allclose(positions, reference, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(triangles, reference_triangles)


def make_scene(modules, origin):
    scene = bpy.context.scene
    scene.fluidsim.live = False
    for obj in tuple(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.frame_set(1)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=tuple(np.asarray(origin) + 0.25))
    domain = bpy.context.object
    domain.name = "FSPT translated device surface domain"
    domain.display_type = "WIRE"
    domain.hide_render = True
    scene.fluidsim.domain_object = domain
    ds, fs = domain.fluidsim_domain, scene.fluidsim
    ds.surface_enabled = True
    ds.live_surface_publish_cadence = 1
    ds.whitewater_enabled = False
    fs.show_surface = fs.view_surface = True
    fs.show_particles = fs.view_particles = fs.show_whitewater = False
    fs.surface_attribute_mode = "NONE"
    fs.liquid_preview_enabled = False
    fs.liquid_display_mode = "SURFACE"
    glass, node = base.material(
        "Device surface glass", "ShaderNodeBsdfGlass", (0.9, 0.98, 1.0)
    )
    node.inputs["IOR"].default_value = 1.333
    fs.surface_material = glass
    return scene, ds


def make_solver(modules, device, origin, *, waves, cadence):
    model, enums = modules("sim.solver"), modules("sim.enums")
    settings = model.SolverSettings(
        domain_min=origin,
        domain_max=tuple(np.asarray(origin) + 0.5),
        voxel_size=1 / 32,
        leaf_capacity=256,
        max_particles=60000,
        min_substeps=1,
        max_substeps=1,
        surface_enabled=True,
        whitewater_enabled=False,
        narrow_band_enabled=False,
        surface_resolution_mode=int(enums.SurfaceResolutionMode.MULTIPLIER),
        surface_resolution_multiplier=2.0,
        surface_update_cadence=int(cadence),
        surface_sheet_detail_enabled=waves,
        surface_waves_enabled=waves,
        surface_waves_strength=4.0,
        device=device,
    )
    solver = model.FluidSolver(settings)
    axis = (np.arange(16) + 0.5) / 64 + 0.17
    x, y = np.meshgrid(axis, axis, indexing="ij")
    positions = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.3)], axis=1)
    positions += np.asarray(origin)
    solver.inject_particles(positions.astype(np.float32))
    return solver


def packet_arrays(modules, wp, packet):
    """A separately owned nonblocking reader waits on the checked copy event."""
    reader = modules("fspt.async_backend")._nonblocking_stream(wp, packet.device)
    receipt = reader.retirement
    host_positions = wp.empty(
        packet.vertex_count, dtype=wp.vec3, device="cpu", pinned=True
    )
    host_indices = wp.empty(
        packet.triangle_count * 3, dtype=wp.int32, device="cpu", pinned=True
    )
    try:
        reader.wait_checked(packet.copy_complete)
        wp.copy(host_positions, packet.positions, stream=reader)
        wp.copy(host_indices, packet.indices, stream=reader)
        done = reader.record_checked(reader)
        wp.synchronize_event(done)
        result = (
            host_positions.numpy().copy(),
            host_indices.numpy().reshape(-1, 3).copy(),
        )
        assert packet.retire_source(), (
            "Completed copy did not release its producer lease"
        )
    finally:
        wp.synchronize_stream(reader)
        del reader
    assert receipt.result == 0, receipt
    return result


def run_case(modules, device, report, folder, *, label, waves, cadence):
    wp = modules("runtime.warp_runtime")._import_warp()
    output = modules("blender.output_publication")
    publication = modules("blender.surface_publication")
    bridge = modules("blender.scene_bridge")
    access = modules("blender.scene_access")
    inputs = modules("blender.scene_inputs")
    ds_api = modules("sim.device_surface")
    origin = (3.0, -2.0, 1.0)
    scene, settings = make_scene(modules, origin)
    solver = make_solver(modules, device, origin, waves=waves, cadence=cadence)
    session = bridge.Session()
    session.install_solver(
        solver,
        profiler=None,
        device=solver.device,
        scene_inputs=inputs.SceneInputs(),
        solved_timeline_frame=1,
    )
    session.bind_scene(access.scene_identity(scene), "fspt-device-proof")
    destination = None
    try:
        initial_clock = (solver.frame, solver.sim_time)
        solver.step_frame()
        session.solved_timeline_frame = 2
        scene.frame_set(2)
        output.update_outputs(scene, session)
        stamp = publication.current_surface_publication(scene)
        assert (
            stamp
            and stamp.solver_frame == solver.frame
            and stamp.state_revision == solver.liquid_state_revision
        )
        assert stamp.source_timeline_frame == 2 and session.owns_scene(scene)
        surface = output.output_object(scene, output.SURFACE_NAME, ensure=False)
        expected = evaluated_surface(surface)
        pipe = solver.surface
        source_stamp = pipe.device_surface_stamp()
        assert source_stamp.build_key == stamp.build_key
        if not waves:
            substep = int(cadence) == int(
                modules("sim.enums").SurfaceUpdateCadence.SUBSTEP
            )
            expected_clock = (
                initial_clock if substep else (solver.frame, solver.sim_time)
            )
            assert (
                source_stamp.solver_frame,
                source_stamp.solver_time,
            ) == expected_clock
        else:
            assert pipe.sheet.count > 0 and pipe.waves.frame == solver.frame
            assert (
                pipe.last_sdf_splat["sheetDetail"]["fineParticles"] == pipe.sheet.count
            )
            heights = pipe.waves.h.numpy()
            assert np.max(np.abs(heights)) > 0, (
                "No real wave displacement was exercised"
            )
            undisplaced, unshifted_triangles = pipe.render_extractor.extract(
                pipe.render_sdf
            )
            np.testing.assert_array_equal(unshifted_triangles, expected[1])
            displacement = np.linalg.norm(expected[0] - undisplaced, axis=1)
            assert np.max(displacement) > 1e-7, (
                "Wave state did not displace the final mesh"
            )

        original_numpy, readbacks = wp.array.numpy, []

        def observed_numpy(array, *args, **kwargs):
            result = original_numpy(array, *args, **kwargs)
            if array.device.is_cuda:
                readbacks.append(
                    {"elements": int(array.size), "bytes": int(result.nbytes)}
                )
            return result

        destination = ds_api.DeviceSurfaceDestination(
            device,
            vertex_capacity=len(expected[0]) + 23,
            triangle_capacity=len(expected[1]) + 17,
        )
        captured_host = solver.host_frame()

        def state_clock():
            return (
                solver.frame,
                solver.sim_time,
                solver.liquid_state_revision,
                pipe.waves.frame if waves else None,
                pipe.waves.state_revision if waves else None,
            )

        before_copy = state_clock()
        with patch.object(wp.array, "numpy", observed_numpy):
            packet = captured_host.copy_final_device_surface(
                destination, expected_stamp=source_stamp
            )
        assert state_clock() == before_copy, (
            "Read-only capture advanced solver or wave state"
        )
        assert readbacks and all(row["elements"] <= 3 for row in readbacks), readbacks
        actual = packet_arrays(modules, wp, packet)
        require_parity(actual, expected)
        assert actual[0].mean(0)[0] > 3, "World translation was lost"
        for bad in (
            (actual[0] - np.asarray(origin), actual[1]),
            (actual[0], actual[1][:, [0, 2, 1]]),
        ):
            try:
                require_parity(bad, expected)
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    "The geometry comparator accepted an incorrect transform or winding"
                )
        snapshot_file = folder / f"{label}-device-snapshot.npz"
        np.savez(snapshot_file, positions=actual[0], triangles=actual[1])
        state = {
            "label": label,
            "source_clock": [source_stamp.solver_frame, source_stamp.solver_time],
            "completed_clock": [stamp.solver_frame, stamp.simulation_time],
            "publication": asdict(stamp),
            "vertices": packet.vertex_count,
            "triangles": packet.triangle_count,
            "positions_sha256": digest(actual[0]),
            "indices_sha256": digest(actual[1]),
            "original": str(snapshot_file),
            "original_sha256": base.sha(snapshot_file),
            "materialization_readbacks": readbacks,
            "read_only_capture_clock": before_copy,
            "sheet_children": pipe.sheet.count if waves else 0,
            "wave_max_abs_height": float(np.max(np.abs(heights))) if waves else 0,
            "final_mesh_max_displacement": float(np.max(displacement)) if waves else 0,
        }
        # Capacity is refused before any optional extraction of the current field.
        undersized = ds_api.DeviceSurfaceDestination(
            device, vertex_capacity=1, triangle_capacity=1
        )
        with patch.object(
            pipe,
            "_final_device_arrays",
            side_effect=AssertionError("capacity refusal extracted"),
        ):
            try:
                pipe.copy_final_device_surface(undersized, expected_stamp=source_stamp)
            except ds_api.DeviceSurfaceUnavailable as error:
                assert "capacity" in str(error)
            else:
                raise AssertionError("Undersized destination was accepted")
        # The first solver frame always publishes, irrespective of cadence.
        # Exercise a genuine skipped frame after that startup exception.
        solver.step_frame()
        session.solved_timeline_frame = 3
        scene.frame_set(3)
        output.update_outputs(scene, session, force_surface=True)
        cadence_stamp = publication.current_surface_publication(scene)
        assert (
            cadence_stamp.solver_frame == 2 and cadence_stamp.source_timeline_frame == 3
        )
        assert not solver.host_frame().should_publish_surface(3)
        refused = []
        with patch.object(
            pipe,
            "_final_device_arrays",
            side_effect=AssertionError("ineligible capture extracted"),
        ):
            for label, call, reason in (
                (
                    "stale_host",
                    lambda: captured_host.copy_final_device_surface(
                        destination, expected_stamp=source_stamp
                    ),
                    "stale host frame",
                ),
                (
                    "stale_stamp",
                    lambda: pipe.copy_final_device_surface(
                        destination, expected_stamp=source_stamp
                    ),
                    "stale device surface stamp",
                ),
                (
                    "unsupported_normals",
                    lambda: pipe.copy_final_device_surface(
                        destination,
                        expected_stamp=pipe.device_surface_stamp(),
                        required_channels={"normals"},
                    ),
                    "required surface channels",
                ),
            ):
                try:
                    call()
                except ds_api.DeviceSurfaceUnavailable as error:
                    assert reason in str(error), error
                    refused.append(label)
                else:
                    raise AssertionError(f"Ineligible capture was accepted: {label}")
        state["refused_before_extraction"] = refused
        with modules("blender.pending_settings").suppress_callbacks(scene):
            settings.live_surface_publish_cadence = 3
        output.update_outputs(scene, session)
        assert publication.current_surface_publication(scene) == cadence_stamp
        output.update_outputs(scene, session, force_surface=True)
        refreshed = publication.current_surface_publication(scene)
        assert (
            refreshed.generation > cadence_stamp.generation
            and solver.frame == cadence_stamp.solver_frame
        )
        state["cadence_skip_generation"] = cadence_stamp.generation
        state["forced_generation"] = refreshed.generation
        solver.step_frame()
        require_parity(packet_arrays(modules, wp, packet), actual)
        state["immutable_after_next_solver_frame"] = solver.frame
        report.setdefault("cases", []).append(state)
    finally:
        try:
            if destination is not None:
                destination.drain()
        finally:
            session._clear_runtime()


def main():
    args = base.arguments("artifacts/fspt-p0/device-surface.json")
    path = Path(args.json_out).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = path.with_name(path.stem + "-originals-" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "verifier_sha256": base.sha(Path(__file__)),
        "scope": (
            "source final sparse positions/indices, real Blender evaluated publication"
        ),
        "not_qualified": [
            "GPU publication channel",
            "renderer pool",
            "packet tracing",
            "normals/attributes/modifiers",
            "installed route",
            "performance",
        ],
    }
    addon = None
    try:
        addon = base.load_addon(args, report)
        report["runtime"] = base.runtime_info(addon)
        assert report["runtime"]["warp"] == "1.19.0"

        def modules(name):
            return importlib.import_module(addon.__name__ + "." + name)

        cadence = modules("sim.enums").SurfaceUpdateCadence
        for label, waves, mode in (
            ("sheet-waves", True, cadence.FRAME),
            ("substep-clock", False, cadence.SUBSTEP),
            ("frame-clock", False, cadence.FRAME),
        ):
            base.checkpoint(path, report, label)
            run_case(
                modules,
                base.script_device(),
                report,
                folder,
                label=label,
                waves=waves,
                cadence=mode,
            )
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        try:
            if addon is not None:
                addon.unregister()
        except BaseException:
            report.update(status="FAILED", cleanup_error=traceback.format_exc())
            raise
        finally:
            base.write_report(path, report)
    print("FSPT-DEVICE-SURFACE-PASS", path, flush=True)


if __name__ == "__main__":
    main()
