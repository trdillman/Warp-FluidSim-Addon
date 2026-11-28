"""Incremental aggregate input uploads and exact fresh-BVH CUDA oracle parity."""
import gc
import hashlib
import importlib
import sys
import traceback
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


def oracle_kernel(wp):
    # Independent primary-ray implementation for this fixture's identity-view,
    # orthographic camera and purely emissive surfaces. No production traversal,
    # asset table, material offsets, or production integrator is called.
    @wp.kernel
    def trace_flat(mesh: wp.uint64, ids: wp.array(dtype=int), colors: wp.array(dtype=wp.vec3),
                   sky: wp.vec3, matrix: wp.mat44, width: int, height: int, seed: int,
                   output: wp.array(dtype=wp.vec4)):
        pixel = wp.tid()
        # Independent expression of the documented seed/pixel/sample-zero key.
        # Do not call the production sampler or integrator in this oracle.
        rng = wp.rand_init(wp.int32(wp.rand_init(seed, pixel)), 0)
        x = 2.0 * (float(pixel % width) + wp.randf(rng)) / float(width) - 1.0
        y = 2.0 * (float(pixel // width) + wp.randf(rng)) / float(height) - 1.0
        start = matrix * wp.vec4(x, y, -1.0, 1.0)
        end = matrix * wp.vec4(x, y, 1.0, 1.0)
        origin = wp.vec3(start[0], start[1], start[2]) / start[3]
        delta = wp.vec3(end[0], end[1], end[2]) / end[3] - origin
        color = sky
        if ids.shape[0] > 0:
            h = wp.mesh_query_ray(mesh, origin, wp.normalize(delta), wp.length(delta))
            if h.result:
                color = colors[ids[h.face]]
        output[pixel] = wp.vec4(color[0], color[1], color[2], 1.0)

    return trace_flat


def flat_pixels(wp, kernel, device, scene, camera, width, height, seed):
    points = wp.array(scene.vertices, dtype=wp.vec3, device=device)
    indices = wp.array(scene.triangles.ravel(), dtype=int, device=device)
    mesh = wp.Mesh(points=points, indices=indices) if len(scene.triangles) else None
    assignments = wp.array(scene.material_indices, dtype=int, device=device)
    emissions = wp.array([m.emission for m in scene.materials], dtype=wp.vec3, device=device)
    rgba = wp.empty(width * height, dtype=wp.vec4, device=device)
    wp.launch(kernel, dim=width * height, device=device, inputs=[
        mesh.id if mesh else wp.uint64(0), assignments, emissions, wp.vec3(*scene.world),
        wp.mat44(*camera.inverse_view_projection.ravel().tolist()), width, height, seed, rgba])
    return rgba.numpy().reshape(height, width, 4).copy()


def tie_scenes(contracts):
    quad = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
    fan = np.vstack([quad, [0, 0, 0]])
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8]]
    colors = tuple(contracts.Material(kind="emission", emission=c) for c in [(1, 0, 0), (0, 1, 0)])
    spans = (contracts.AssetSpan((1,), 4, 2), contracts.AssetSpan((2,), 5, 4))
    base = contracts.SceneSnapshot(np.vstack((quad, fan)), faces, [0, 0, 1, 1, 1, 1],
                                   colors, (0, 0, 0), 1, asset_spans=spans)
    eye = contracts.CameraSnapshot(np.eye(4))
    angle = np.deg2rad(32)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    yield "coplanar", base, eye
    yield "rotated_partial", replace(base, vertices=np.vstack((quad, fan @ rot.T + [0.38, 0, 0]))), eye
    far = contracts.CameraSnapshot(np.diag([1, 1, 1e5, 1]))
    yield "disjoint_equal_float_depth", replace(base, vertices=np.vstack((quad, fan + [0, 0, 0.001]))), far
    one = replace(base, asset_spans=(contracts.AssetSpan((1,), 9, 6),))
    yield "single_asset_disjoint_components", replace(one, vertices=np.vstack((quad, fan + [3, 0, 0]))), eye
    yield "single_asset_moved_components", one, eye


def verify_ties(backend, contracts, kernel):
    results = []
    for label, scene, camera in tie_scenes(contracts):
        reference = flat_pixels(backend.wp, kernel, backend.device, scene, camera, 32, 24, 2)
        actual = backend.render(scene, camera, 32, 24, max_bounces=1, seed=2).pixels
        np.testing.assert_array_equal(actual, reference, err_msg=label)
        results.append(dict(label=label, pixels_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    return results


def fixture(bpy, base, device_surface, module, device, session):
    scene, _ds = device_surface.make_scene(module, (0, 0, 0))
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    materials = [base.material(f"M3 emission {n}", "ShaderNodeEmission", c)[0]
                 for n, c in enumerate(colors)]
    scene.fluidsim.surface_material = materials[0]

    def quad(name, x0, y0, x1, y1, z, collection, mats):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)],
                         [], [(0, 1, 2), (0, 2, 3)])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        for material in mats:
            mesh.materials.append(material)
        mesh.polygons[1].material_index = len(mats) - 1
        return obj

    quad("Behind fluid", 0, 0, 0.5, 0.5, 0.8, scene.collection, materials[1:3])
    source = bpy.data.collections.new("M3 instance geometry")
    quad("In front of fluid", 0, 0.05, 0.035, 0.45, 0.01, source, materials[3:])
    for x in (0.20, 0.36):
        obj = bpy.data.objects.new("M3 collection instance", None)
        scene.collection.objects.link(obj)
        obj.instance_type = "COLLECTION"
        obj.instance_collection = source
        obj.location.x = x
    cadence = module("sim.enums").SurfaceUpdateCadence.FRAME
    solver = device_surface.make_solver(module, device, (0, 0, 0), waves=False, cadence=cadence)
    session.install_solver(solver, profiler=None, device=solver.device,
                           scene_inputs=module("blender.scene_inputs").SceneInputs(),
                           solved_timeline_frame=1)
    session.bind_scene(module("blender.scene_access").scene_identity(scene), "fspt-m3-proof")
    return scene, solver


def advance(bpy, module, scene, solver, session, timeline):
    # Only the fixture drives the owner. Rendering must preserve these clocks.
    solver.step_frame()
    session.solved_timeline_frame = timeline
    scene.frame_set(timeline)
    module("blender.output_publication").update_outputs(scene, session)
    bpy.context.view_layer.update()
    snapshot = module("blender.fspt_scene").extract_scene(bpy.context.evaluated_depsgraph_get())
    stamp, = snapshot.publications
    assert stamp.solver_frame == solver.frame and stamp.source_timeline_frame == timeline
    return snapshot, (0, stamp.object_token)


def verify_pending_uploads(module, original, camera, width, height, device, report):
    candidate = replace(original, materials=tuple(replace(m, emission=(0.3, 0.1, 0.7))
                                                  for m in original.materials))
    for name in ("CudaBackend", "AsyncCudaBackend"):
        backend = getattr(module("fspt.async_backend"), name)(str(device))
        wp = backend.wp
        owners, drains = [], []

        class FaultWarp:
            def __getattr__(self, attr):
                return getattr(wp, attr)

            def array(self, *args, **kwargs):
                if owners:
                    raise RuntimeError("injected second shading upload")
                first = wp.array(*args, **kwargs)
                owners.append(weakref.ref(first))
                return first

            def _refuse_drain(self, _):
                drains.append(owners[0]() is not None)
                raise RuntimeError("injected unavailable retirement")

            synchronize_device = synchronize_stream = _refuse_drain

        try:
            backend.render(original, camera, width, height, max_bounces=1)
            backend.wp = FaultWarp()
            try:
                backend.render(candidate, camera, width, height, max_bounces=1)
            except RuntimeError as exc:
                assert str(exc) == ("injected second shading upload" if name == "CudaBackend"
                                    else "injected unavailable retirement")
            else:
                raise AssertionError("missing partial-shading fault")
            assert backend.scene_digest == original.digest and drains == [True]
            error = backend._uncertain[backend][0] if name == "CudaBackend" else backend._job.failure
            error.__traceback__ = None
            gc.collect()
            assert owners[0]() is not None
        finally:
            backend.wp = wp
            backend.close()
        gc.collect()
        assert owners[0]() is None
        report.setdefault("partial_upload_retirement", {})[name] = "held through failed drain, released after close"


def verify_reused_reader_retirement(module, scene, camera, width, height, device, report):
    for name in ("CudaBackend", "AsyncCudaBackend"):
        for edit in ("unchanged", "camera", "shading"):
            backend = getattr(module("fspt.async_backend"), name)(str(device))
            wp = backend.wp
            reader_refs = []

            class LateFailure:
                def __getattr__(self, attr):
                    return getattr(wp, attr)

                def launch(self, *args, **kwargs):
                    result = wp.launch(*args, **kwargs)
                    reader_refs.extend(weakref.ref(owner) for owner in (
                        backend.mesh, backend.arrays["points"], backend.arrays["indices"],
                        backend.arrays["materials"], backend.arrays["kinds"], backend.arrays["colors"],
                        backend.arrays["emissions"], backend.arrays["iors"], backend.accumulation, backend.output))
                    if name == "AsyncCudaBackend":
                        reader_refs.append(weakref.ref(backend._pinned))
                    return result

                def refuse(self, *args, **kwargs):
                    raise RuntimeError("injected reader retirement failure")

                synchronize_device = synchronize_event = synchronize_stream = refuse

            try:
                backend.render(scene, camera, width, height, max_bounces=1)
                assert not backend._pending_resources
                old_mesh = weakref.ref(backend.mesh)
                candidate, view = scene, camera
                if edit == "camera":
                    matrix = camera.inverse_view_projection.copy()
                    matrix[0, 3] += 0.01
                    view = replace(camera, inverse_view_projection=matrix)
                if edit == "shading":
                    candidate = replace(scene, materials=tuple(replace(m, emission=(0.6, 0.2, 0.1))
                                                               for m in scene.materials))
                backend.wp = LateFailure()
                try:
                    backend.render(candidate, view, width, height, max_bounces=1)
                except RuntimeError as exc:
                    assert str(exc) == "injected reader retirement failure"
                else:
                    raise AssertionError("retirement fault did not fire")
                error = backend._uncertain[backend][0] if name == "CudaBackend" else backend._job.failure
                error.__traceback__ = None
                gc.collect()
                assert reader_refs and reader_refs[0]() is old_mesh()
                assert all(ref() is not None for ref in reader_refs), (name, edit, "lost reader owner")
            finally:
                backend.wp = wp
                backend.close()
            gc.collect()
            assert all(ref() is None for ref in reader_refs), (name, edit, "retired owner leaked")
            report.setdefault("reused_reader_retirement", []).append([name, edit])


def verify_geometry_faults(module, kernel, device, moved, rewired, camera, width, height, report):
    positions = moved.vertices.copy()
    positions[:, 0] += 0.02
    changed_points = replace(moved, vertices=positions)
    for name in ("CudaBackend", "AsyncCudaBackend"):
        for kind, changed in (("positions", changed_points), ("indices", rewired)):
            backend = getattr(module("fspt.async_backend"), name)(str(device))
            wp = backend.wp
            failed_build = []
            drains = []

            class RefuseRetirement:
                def __getattr__(self, attr):
                    return getattr(wp, attr)

                def refuse(self, _):
                    drains.append(True)
                    raise RuntimeError("injected unavailable geometry retirement")

                synchronize_device = synchronize_stream = refuse

            try:
                baseline = backend.render(moved, camera, width, height, max_bounces=1, seed=37)
                expected = flat_pixels(wp, kernel, device, moved, camera, width, height, 37)
                np.testing.assert_array_equal(baseline.pixels, expected)
                old = weakref.ref(backend.mesh)
                build = backend._build_mesh

                def fail_after_build(points, indices):
                    built = build(points, indices)
                    failed_build.append(weakref.ref(built))
                    raise RuntimeError("injected failure after aggregate write and build")

                backend.wp = RefuseRetirement()
                with patch.object(backend, "_build_mesh", fail_after_build):
                    try:
                        backend.render(changed, camera, width, height, max_bounces=1, seed=37)
                    except RuntimeError as exc:
                        assert str(exc) == ("injected failure after aggregate write and build"
                                            if name == "CudaBackend" else "injected unavailable geometry retirement")
                    else:
                        raise AssertionError("geometry failure control did not fire")
                assert drains and backend.mesh is None and backend.scene_digest is None
                error = backend._uncertain[backend][0] if name == "CudaBackend" else backend._job.failure
                error.__traceback__ = None
                gc.collect()
                assert old() is not None and failed_build[0]() is not None
                if name == "AsyncCudaBackend":
                    assert not backend.submit(moved, camera, width, height)
                else:
                    try:
                        backend.render(moved, camera, width, height)
                    except RuntimeError as exc:
                        assert str(exc) == "injected unavailable geometry retirement"
                    else:
                        raise AssertionError("uncertain geometry was dispatched")
                backend.wp = wp
                # Prove the intended write occurred before checking recovery.
                # Synchronize privately on async, as in the production abort.
                if name == "AsyncCudaBackend":
                    backend._drain()
                else:
                    wp.synchronize_device(backend.device)
                observed = old().points.numpy() if kind == "positions" else old().indices.numpy()
                target = changed.vertices if kind == "positions" else changed.triangles.ravel()
                np.testing.assert_array_equal(observed, target)
                if name == "AsyncCudaBackend":
                    try:
                        backend.collect(wait=True)
                    except RuntimeError as exc:
                        assert "discarded a failed submission" in str(exc)
                    else:
                        raise AssertionError("failed submission produced a frame")
                recovered = backend.render(moved, camera, width, height, max_bounces=1, seed=37)
                np.testing.assert_array_equal(recovered.pixels, expected)
                assert recovered.samples == 1 and backend.scene_digest == moved.digest
                gc.collect()
                assert failed_build[0]() is None and old() is None
                report.setdefault("geometry_write_failure_recovery", []).append([name, kind])
            finally:
                backend.wp = wp
                backend.close()


def check_pixel_cases(module, backend, scene, solver, session, bpy, report, path, base):
    device, wp = backend.device, backend.wp
    kernel = oracle_kernel(wp)
    width, height, seed = 80, 64, 91
    camera = module("fspt.contracts").CameraSnapshot(
        [[0.28, 0, 0, 0.25], [0, 0.28, 0, 0.25], [0, 0, 1, 0], [0, 0, 0, 1]])
    report["cases"] = []

    def compare(label, snapshot):
        base.checkpoint(path, report, label)
        before = (solver.frame, solver.sim_time, solver.liquid_state_revision)
        reference = flat_pixels(wp, kernel, device, snapshot, camera, width, height, seed)
        frame = backend.render(snapshot, camera, width, height, samples=1, max_bounces=1, seed=seed)
        assert frame.device == str(device) and frame.samples == 1
        assert frame.publications == snapshot.publications
        np.testing.assert_array_equal(frame.pixels, reference)
        assert before == (solver.frame, solver.sim_time, solver.liquid_state_revision)
        report["cases"].append(dict(label=label, pixels_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                                    assets=len(snapshot.assets), triangles=len(snapshot.triangles),
                                    counts=dict(frame.statistics), frame=snapshot.frame))
        return frame.pixels

    original, fluid = advance(bpy, module, scene, solver, session, 2)
    initial_pixels = compare("published_frame_2", original)
    verify_pending_uploads(module, original, camera, width, height, device, report)
    verify_reused_reader_retirement(module, original, camera, width, height, device, report)
    assert len(original.assets) == 4 and sum(a.key[0] == 1 for a in original.assets) == 2
    for rgb in [(5, 0, 0), (0, 5, 0), (0, 0, 5), (5, 5, 0)]:
        assert np.count_nonzero(np.all(initial_pixels[..., :3] == rgb, axis=-1)) > 10, rgb
    corrupt = type(backend)(str(device))
    try:
        prepare = corrupt._set_scene

        def wrong_materials(snapshot):
            prepare(snapshot)
            corrupt.arrays["materials"].zero_()

        with patch.object(corrupt, "_set_scene", wrong_materials):
            wrong = corrupt.render(original, camera, width, height, samples=1, max_bounces=1, seed=seed)
        mismatched = int(np.any(wrong.pixels != initial_pixels, axis=2).sum())
        assert mismatched > 10, "incorrect face materials escaped the flat oracle"
        report["wrong_material_control_mismatched_pixels"] = mismatched
    finally:
        corrupt.close()

    newer, next_key = advance(bpy, module, scene, solver, session, 3)
    assert next_key == fluid and newer.geometry_digest != original.geometry_digest
    compare("published_frame_3", newer)
    asset = next(a for a in newer.assets if a.key == fluid)
    start, count, _ = asset.position
    points = newer.vertices.copy()
    points[start:start + count, 0] += 0.10
    # Adversarial derived edits carry no production publication stamp.
    moved = replace(newer, vertices=points, publications=(), frame=4)
    old_mesh, prior = backend.mesh, backend.statistics.copy()
    position_store, index_store = backend.arrays["position_storage"], backend.arrays["index_storage"]
    moved_pixels = compare("position_span_only", moved)
    assert backend.mesh is not old_mesh and backend.statistics["mesh_builds"] == prior["mesh_builds"] + 1
    assert backend.arrays["position_storage"] is position_store and backend.arrays["index_storage"] is index_store
    assert backend.statistics["position_upload_bytes"] - prior["position_upload_bytes"] == 12 * count
    assert backend.statistics["position_upload_ranges"] - prior["position_upload_ranges"] == 1
    assert backend.statistics["index_upload_bytes"] == prior["index_upload_bytes"]
    assert backend.statistics["geometry_array_allocations"] == prior["geometry_array_allocations"]
    begin, length, _ = asset.index
    begin, faces_count = begin // 3, length // 3
    faces = moved.triangles.copy()
    faces[begin:begin + faces_count] = faces[begin]
    topology = replace(moved, triangles=faces, frame=5)
    old_mesh, prior = backend.mesh, backend.statistics.copy()
    changed_pixels = compare("equal_count_rewire", topology)
    assert backend.mesh is not old_mesh and backend.statistics["mesh_builds"] == prior["mesh_builds"] + 1
    assert backend.statistics["position_upload_bytes"] == prior["position_upload_bytes"]
    assert backend.statistics["index_upload_bytes"] - prior["index_upload_bytes"] == 4 * length
    assert backend.statistics["index_upload_ranges"] - prior["index_upload_ranges"] == 1
    assert backend.statistics["geometry_array_allocations"] == prior["geometry_array_allocations"]
    assert np.any(moved_pixels != changed_pixels)
    verify_geometry_faults(module, kernel, device, moved, topology, camera, width, height, report)
    empty = replace(topology, vertices=[], triangles=[], material_indices=[], asset_spans=(), frame=6)
    compare("empty_world", empty)
    assert backend.mesh is None
    compare("restore_all_assets", original)
    report["traversal_tie_cases"] = verify_ties(backend, module("fspt.contracts"), kernel)


def main():
    import bpy

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blender_fspt_acceptance as base
    import blender_fspt_device_surface_acceptance as device_surface

    args = base.arguments("artifacts/fspt-m3/persistent-assets.json")
    path = Path(args.json_out).resolve()
    addon = backend = session = None
    report = dict(status="RUNNING", verifier_sha256=base.sha(Path(__file__)))
    try:
        base.checkpoint(path, report, "load")
        addon = base.load_addon(args, report)
        report["runtime"] = base.runtime_info(addon)
        report["fixture_helper_sha256"] = {Path(m.__file__).name: base.sha(Path(m.__file__))
                                           for m in (base, device_surface)}
        def module(name):
            return importlib.import_module(addon.__name__ + "." + name)
        device = base.script_device()
        session = module("blender.scene_bridge").session()
        scene, solver = fixture(bpy, base, device_surface, module, device, session)
        backend = module("fspt.async_backend").AsyncCudaBackend(device)
        check_pixel_cases(module, backend, scene, solver, session, bpy, report, path, base)
        backend.close()
        backend = None
        session._clear_runtime()
        session = None
        addon.unregister()
        addon = None
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
            if session is not None:
                session._clear_runtime()
            if addon is not None:
                addon.unregister()
        finally:
            base.write_report(path, report)
    print("FSPT-PERSISTENT-ASSETS-PASS", path, flush=True)


if __name__ == "__main__":
    main()
