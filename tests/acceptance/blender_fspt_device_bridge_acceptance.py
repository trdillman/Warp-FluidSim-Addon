"""Standalone Blender/CUDA acceptance for the inactive FSPT surface producer."""

from __future__ import annotations

import importlib
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_device_surface_acceptance as surface_fixture
from blender_fspt_surface_pool_acceptance import StreamGate

base = surface_fixture.base
BYTE_LIMIT = 64 * 1024 * 1024
ORIGIN = (3.0, -2.0, 1.0)


@contextmanager
def live_case(modules, wp, alias, *, waves=False):
    scene, _settings = surface_fixture.make_scene(modules, ORIGIN)
    cadence = modules("sim.enums").SurfaceUpdateCadence.FRAME
    solver = surface_fixture.make_solver(modules, alias, ORIGIN, waves=waves, cadence=cadence)
    assert type(solver.device) is str and solver.device == alias
    device = wp.get_device(alias)
    assert device.is_cuda and not device.is_capturing
    producer_stream = wp.get_stream(device)
    publication = modules("blender.surface_publication")
    owner_api = modules("blender.fspt_device_bridge")
    session = modules("blender.scene_bridge").Session()
    owner = pool = None
    try:
        session.install_solver(solver, profiler=None, device=alias,
                               scene_inputs=modules("blender.scene_inputs").SceneInputs(),
                               solved_timeline_frame=1)
        session.bind_scene(modules("blender.scene_access").scene_identity(scene), "fspt-producer-proof")
        pool = modules("fspt.surface_pool").SurfacePool(device, byte_limit=BYTE_LIMIT)
        budget = modules("fspt.device_scene").GeometryArrayBudget(pool, byte_limit=BYTE_LIMIT)
        owner = owner_api.DeviceSurfaceBridge(budget)
        extracted = []
        extractor_type = type(solver.surface.render_extractor)
        extract_arrays = extractor_type._sparse_arrays

        def count_extract(extractor, sdf, *args, **kwargs):
            current = solver.surface
            if extractor is current.render_extractor and sdf is current.render_sdf:
                extracted.append({"solver_frame": int(solver.frame),
                                  "active_leaves": int(extractor.grid.active_leaf_count)})
            return extract_arrays(extractor, sdf, *args, **kwargs)

        with patch.object(extractor_type, "_sparse_arrays", count_extract):
            yield SimpleNamespace(
                modules=modules, wp=wp, device=device, stream=producer_stream,
                scene=scene, solver=solver, session=session, publication=publication,
                output=modules("blender.output_publication"), source_type=owner_api.LiveSurfaceSource,
                scene_token=publication.id_token(scene), owner=owner, budget=budget, pool=pool,
                extracted=extracted,
            )
    finally:
        if owner is not None:
            owner.close()
        elif pool is not None:
            pool.close()
        session._clear_runtime()


def step(case, timeline):
    case.scene.frame_set(timeline)
    case.session.step(case.scene, write_cache=False, timeline_frame=timeline, publish_outputs=False)
    ready = case.session.current_solver_completion(case.scene)
    assert ready is not None and ready.ready_at is not None
    session_scene = case.modules("blender.scene_access").scene_identity(case.scene)
    assert ready.source_timeline_frame == timeline and ready.scene_token == session_scene
    assert ready.owner_token == case.publication.surface_owner_token(case.scene)
    assert (ready.solver_frame, ready.state_revision, ready.simulation_time) == (
        case.solver.frame, case.solver.liquid_state_revision, case.solver.sim_time)
    return case.source_type(
        case.scene_token, ready.owner_token, ready.epoch, ready.source_timeline_frame,
        ready.state_revision, ready.solver_frame, ready.simulation_time, ready.ready_at,
    )


def prepare(case, source, *, host_required):
    return case.owner.prepare(case.solver.host_frame(), source, producer_stream=case.stream,
                              host_required=host_required)


def publish_host(case, source):
    result = case.output.update_outputs(case.scene, case.session, force_surface=True)
    assert result.synchronized, result.warning
    stamp = case.publication.current_surface_publication(case.scene)
    assert stamp is not None and stamp.state_ready_at == source.state_ready_at
    assert (stamp.epoch, stamp.source_timeline_frame, stamp.solver_frame, stamp.state_revision) == (
        source.epoch, source.source_timeline_frame, source.solver_frame, source.state_revision)
    mesh = case.output.output_object(case.scene, case.output.SURFACE_NAME, ensure=False)
    assert mesh is not None
    return surface_fixture.evaluated_surface(mesh)


def ready_counts(case, key, source, expected):
    offer = case.owner.poll(case.scene_token)
    assert offer.status == "READY" and offer.key == key and offer.source == source
    assert (offer.vertex_count, offer.triangle_count) == (len(expected[0]), len(expected[1]))
    assert case.budget.surface_ready(key) == (len(expected[0]), len(expected[1]))


@contextmanager
def producer_record(case, on_ready):
    """Intercept only the owned record immediately after real GPU extraction."""
    host = case.solver.host_frame()
    copy = host.copy_final_device_surface

    def controlled(destination, **kwargs):
        normal_record = destination._driver.record

        def record(warp, device, stream, event):
            if event is destination._ready:
                return on_ready(normal_record, warp, device, stream, event)
            return normal_record(warp, device, stream, event)

        with patch.object(destination._driver, "record", record):
            return copy(destination, **kwargs)

    with patch.object(host, "copy_final_device_surface", controlled):
        yield


@contextmanager
def delayed_prepare(case, source):
    gate = StreamGate(case.modules, case.wp, case.device, case.stream)

    def delay(record, *args):
        record(*args)
        gate.arm()

    try:
        with producer_record(case, delay):
            offer = prepare(case, source, host_required=False)
        assert offer.status == "PREPARED" and gate.armed, offer
        yield offer.key
    finally:
        gate.open()
        if gate.armed:
            case.wp.synchronize_stream(case.stream)
            gate.verify()


def packet_arrays(case, lease, reader):
    wp = case.wp
    packet = lease.packet
    assert packet.device == case.device and packet.vertex_count and packet.triangle_count
    positions = wp.empty(packet.vertex_count, dtype=wp.vec3, device="cpu", pinned=True)
    indices = wp.empty(packet.triangle_count * 3, dtype=wp.int32, device="cpu", pinned=True)
    lease.retain((positions, indices))
    reader.wait_checked(packet.copy_complete)
    wp.copy(positions, packet.positions, stream=reader)
    wp.copy(indices, packet.indices, stream=reader)
    final_read = reader.record_checked(reader)
    wp.synchronize_event(final_read)
    return positions.numpy().copy(), indices.numpy().reshape(-1, 3).copy()


def retain_geometry(folder, label, source, host, *, source_route, evaluated_route):
    surface_fixture.require_parity(source, host)
    artifact = folder / f"{label}.npz"
    np.savez(artifact, positions=source[0], triangles=source[1], host_positions=host[0],
             host_triangles=host[1])
    return {"vertices": len(host[0]), "triangles": len(host[1]),
            "source_route": source_route,
            "source_geometry_sha256": [surface_fixture.digest(array) for array in source],
            "evaluated_route": evaluated_route,
            "evaluated_geometry_sha256": [surface_fixture.digest(array) for array in host],
            "original": str(artifact), "original_sha256": base.sha(artifact)}


def no_demand(case, source):
    old_extractions = len(case.extracted)
    host = case.solver.host_frame()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("No-demand producer issued CUDA work or a host readback")

    with (patch.object(case.pool, "poll", wraps=case.pool.poll) as poll,
          patch.object(case.budget, "submit_surface", wraps=case.budget.submit_surface) as submit,
          patch.object(case.wp, "launch", unexpected), patch.object(case.wp, "copy", unexpected),
          patch.object(case.wp, "synchronize_stream", unexpected),
          patch.object(case.wp.array, "numpy", unexpected)):
        result = case.owner.prepare(host, source, producer_stream=case.stream, host_required=True)
        assert result.status == "NO_DEMAND" and result.key is None
        assert case.owner.poll(case.scene_token).status == "NO_DEMAND"
        poll.assert_not_called()
        submit.assert_not_called()
    assert len(case.extracted) == old_extractions


def pressure_control(case, source, *, host_required):
    before = (case.pool.allocations, len(case.extracted), case.budget.denials)
    outcome = prepare(case, source, host_required=host_required)
    assert outcome.status == "DEFERRED" and outcome.key is None, outcome
    assert (case.pool.allocations, len(case.extracted), case.budget.denials) == (
        before[0], before[1], before[2] + 1)
    assert not case.pool.poisoned


def exercise_demand(modules, wp, alias, report, path, folder):
    with live_case(modules, wp, alias) as case:
        reader = modules("fspt.async_backend")._nonblocking_stream(wp, case.device)
        retirement = reader.retirement
        leases, charge = [], None
        try:
            first = step(case, 2)
            assert case.solver.surface.mesh_cache is None and not case.extracted
            no_demand(case, first)
            assert case.pool.allocations == 0 and case.budget.reserved_bytes == 0
            withdraw = case.owner.subscribe(case.scene_token)
            offered = prepare(case, first, host_required=True)
            assert offered.status == "PREPARED" and offered.source == first
            first_key = offered.key
            assert first_key.build_stamp == case.solver.surface.device_surface_stamp()
            assert len(case.extracted) == 1 and case.pool.allocations == 1
            active = case.extracted[-1]["active_leaves"]
            assert active > 0 and case.pool.reserved_bytes == 148_716 * active
            assert case.solver.surface.mesh_cache is not None
            wp.synchronize_stream(case.stream)
            pending = case.owner.poll(case.scene_token)
            assert pending.status == "PENDING" and pending.key is None
            assert case.budget.surface_ready(first_key) is not None
            first_host = publish_host(case, first)
            assert len(case.extracted) == 1, "Cold host publication extracted twice"
            assert case.owner.commit(first_key)
            ready_counts(case, first_key, first, first_host)
            same = prepare(case, first, host_required=True)
            assert same.key == first_key and len(case.extracted) == 1 and case.pool.allocations == 1

            lease = case.pool.acquire(first_key, reader)
            assert lease is not None
            leases.append(lease)
            first_packet = packet_arrays(case, lease, reader)
            report["cold_shared"] = dict(retain_geometry(
                folder, "cold-shared", first_packet, first_host, source_route="checked_cuda_packet",
                evaluated_route="native_publication_from_shared_packet_cache"),
                                         source=asdict(first), active_leaves=active,
                                         cold_extractions=len(case.extracted),
                                         fence_ready_before_commit_rejected=True,
                                         same_build_reused=True, matching_solver_alias=case.solver.device,
                                         no_demand_before_subscribe=True)
            base.checkpoint(path, report, "byte-pressure")

            following = step(case, 3)
            # Charge remaining shared bytes while retaining one old aggregate
            # generation. The pool alone can admit another packet.
            charge = case.budget.reserve(BYTE_LIMIT - case.budget.reserved_bytes,
                                         generation=(first_key, "retained aggregate control"))
            assert charge.size > 0 and case.budget.generation_count == 1
            assert case.pool.occupied == 1 and case.budget.reserved_bytes == BYTE_LIMIT
            pressure_control(case, following, host_required=False)
            assert case.pool.acquire(first_key, reader) is None
            assert lease.packet.vertex_count == len(first_host[0])
            charge.release()
            charge = None

            base.checkpoint(path, report, "delayed-copy")
            before = len(case.extracted)
            with delayed_prepare(case, following) as second_key:
                assert case.owner.commit(second_key)
                waiting = case.owner.poll(case.scene_token)
                assert waiting.status == "PENDING" and waiting.key is None
                assert case.budget.surface_ready(second_key) is None
                assert case.pool.acquire(second_key, reader) is None
                assert case.solver.surface.mesh_cache is None
            assert len(case.extracted) == before + 1
            second_host = publish_host(case, following)
            assert len(case.extracted) == before + 2  # The earlier copy was GPU-only.
            ready_counts(case, second_key, following, second_host)
            lease2 = case.pool.acquire(second_key, reader)
            assert lease2 is not None
            leases.append(lease2)
            second_packet = packet_arrays(case, lease2, reader)
            assert surface_fixture.digest(first_packet[0]) != surface_fixture.digest(second_packet[0])
            report["gpu_only"] = dict(retain_geometry(
                folder, "gpu-only", second_packet, second_host, source_route="checked_cuda_packet",
                evaluated_route="native_publication_from_separate_extraction"),
                                      source=asdict(following), copy_pending_after_commit_rejected=True,
                                      byte_denial_without_extraction=True, host_cache_remained_cold=True)

            base.checkpoint(path, report, "retained-readers")
            third = step(case, 4)
            assert case.pool.occupied == case.budget.generation_count == 2
            pressure_control(case, third, host_required=False)
            # Old immutable data remains intact after solver advance and after
            # selection moves. Read rights belong to the open lease.
            surface_fixture.require_parity(packet_arrays(case, lease, reader), first_host)
            gate = StreamGate(modules, wp, case.device, reader)
            try:
                gate.arm()
                for owned in leases:
                    owned.finish()
                case.pool.poll()
                assert all(not owned.retired for owned in leases)
                assert case.pool.occupied == case.budget.generation_count == 2
                pressure_control(case, third, host_required=False)
            finally:
                gate.open()
                wp.synchronize_stream(reader)
                gate.verify()
            case.pool.poll()
            assert all(owned.retired for owned in leases)
            assert case.pool.occupied == case.budget.generation_count == 0

            base.checkpoint(path, report, "cancel-pending")
            with delayed_prepare(case, third) as canceled:
                assert case.pool.occupied == 1
                assert case.owner.commit(canceled)
                withdraw()
                assert case.pool.acquire(canceled, reader) is None
                assert not case.owner.commit(canceled)
                no_demand(case, third)
            # No-demand checks intentionally didn't poll. Explicit retirement
            # drains the now-unselected packet.
            case.owner.close()
            assert case.pool.occupied == case.budget.generation_count == 0
            report["lifetime"] = {"third_generation_denied": True,
                                  "last_read_fence_pending_denied": True,
                                  "pool_reserved_peak_bytes": case.pool.peak_bytes,
                                  "shared_reserved_peak_bytes": case.budget.peak_bytes,
                                  "shared_peak_generations": case.budget.peak_generations,
                                  "independent_denials": case.budget.denials,
                                  "canceled_pending_copy": True, "remaining_bytes": case.budget.reserved_bytes}
            assert case.budget.denials == 3 and case.budget.peak_generations == 2
            assert case.budget.peak_bytes == BYTE_LIMIT and case.budget.reserved_bytes == 0
        finally:
            if charge is not None:
                charge.release()
            for owned in leases:
                owned.finish()
            wp.synchronize_stream(reader)
            case.owner.close()
            del reader
        # A checked stream wrapper is not destroyed while a lease retains it.
        del gate
        assert retirement.result == 0, retirement
        report["lifetime"]["reader_destroy_result"] = retirement.result


def exercise_warmed_waves(modules, wp, alias, report, folder):
    with live_case(modules, wp, alias, waves=True) as case:
        source = step(case, 2)
        pipeline = case.solver.surface
        assert len(case.extracted) == 1 and pipeline.waves.frame == case.solver.frame
        assert pipeline.sheet.count > 0 and pipeline.waves.count > 0
        assert np.max(np.abs(pipeline.waves.h.numpy())) > 0
        assert pipeline.mesh_cache is not None and pipeline.mesh_key == pipeline.built_key
        wave_receipt = (pipeline.waves.frame, pipeline.waves.state_revision)
        reference = tuple(array.copy() for array in pipeline.mesh_cache)
        case.owner.subscribe(case.scene_token)
        with (patch.object(case.pool, "poll", wraps=case.pool.poll) as poll,
              patch.object(case.budget, "submit_surface", wraps=case.budget.submit_surface) as submit):
            for mandatory in (True, False):
                result = prepare(case, source, host_required=mandatory)
                assert result.status == "UNSUPPORTED" and "already materialized" in result.reason
                assert result.key is None
            poll.assert_not_called()
            submit.assert_not_called()
        assert len(case.extracted) == 1 and case.pool.allocations == 0 and not case.pool.poisoned
        host = publish_host(case, source)
        assert len(case.extracted) == 1 and wave_receipt == (pipeline.waves.frame, pipeline.waves.state_revision)
        report["waves_cache"] = dict(retain_geometry(
            folder, "native-waves", reference, host, source_route="preexisting_native_wave_cache",
            evaluated_route="native_publication_reusing_wave_cache"),
                                     preexisting_host_extractions=1, denied_before_poll=True,
                                     waves_frame=wave_receipt[0], wave_revision=wave_receipt[1],
                                     sheet_particles=int(pipeline.sheet.count),
                                     evaluated_fallback_preserved=True)


def exercise_copy_fault(modules, wp, alias, report, folder):
    with live_case(modules, wp, alias) as case:
        source = step(case, 2)
        pipeline = case.solver.surface
        # Direct extractor oracle leaves the pipeline and HostFrame caches cold.
        reference = pipeline.render_extractor.extract(pipeline.render_sdf)
        assert pipeline.mesh_cache is None and len(case.extracted) == 1
        case.owner.subscribe(case.scene_token)
        attempts = []

        def fault(_record, _warp, _device, _stream, _event):
            attempts.append(len(case.extracted))
            raise RuntimeError("injected producer-ready fence failure")

        with producer_record(case, fault):
            failed = prepare(case, source, host_required=True)
        assert attempts == [2] and len(case.extracted) == 2, attempts
        assert failed.status == "FAILED" and "producer-ready fence failure" in failed.reason
        assert failed.key is None and case.pool.poisoned and case.pool.occupied == 1
        assert pipeline.mesh_cache is None and case.pool.allocations == 1
        assert case.owner.poll(case.scene_token).status == "FAILED"
        host = publish_host(case, source)
        assert len(case.extracted) == 3 and case.pool.occupied == 1
        assert prepare(case, source, host_required=True).status == "FAILED"
        assert len(case.extracted) == 3 and case.pool.allocations == 1
        case.owner.close()
        assert case.pool.occupied == 0 and case.budget.reserved_bytes == 0
        report["copy_fault"] = dict(retain_geometry(
            folder, "native-after-optional-fault", reference, host, source_route="pre_fault_native_sparse_extraction",
            evaluated_route="native_publication_after_optional_failure"),
                                    error=failed.reason, injected_after_real_extraction=True,
                                    ordinary_host_republished=True, quarantine_drained=True)


def main():
    args = base.arguments("artifacts/fspt-p0/device-bridge.json")
    path = Path(args.json_out).resolve()
    folder = path.with_name(path.stem + "-originals")
    if path.exists():
        raise FileExistsError(f"Use a fresh report path: {path}")
    folder.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "inactive source producer with real CUDA and native mesh output",
              "verifier_sha256": base.sha(Path(__file__)),
              "fixtures": {str(file.name): base.sha(file) for file in (
                  Path(surface_fixture.__file__), Path(base.__file__),
                  Path(sys.modules[StreamGate.__module__].__file__))},
              "byte_limit": BYTE_LIMIT}
    addon = None
    try:
        base.checkpoint(path, report, "load")
        assert bpy.app.background, "Use the owned background CUDA acceptance profile"
        addon = base.load_addon(args, report)
        report["runtime"] = base.runtime_info(addon)
        assert report["runtime"]["warp"] == "1.19.0", report["runtime"]

        def modules(name):
            return importlib.import_module(f"{addon.__name__}.{name}")

        package = Path(addon.__file__).resolve().parent
        report["loaded_sources"] = {}
        for name in ("blender.fspt_device_bridge", "fspt.device_scene", "fspt.surface_pool",
                     "fspt.async_backend", "runtime.cuda_driver", "sim.device_surface", "sim.surface",
                     "sim.frame_output", "blender.output_publication", "blender.scene_bridge"):
            file = Path(modules(name).__file__).resolve()
            assert file.is_relative_to(package), file
            report["loaded_sources"][name] = {"path": str(file), "sha256": base.sha(file)}
        wp = modules("runtime.warp_runtime")._import_warp()
        alias = base.script_device()
        assert type(alias) is str and wp.get_device(alias).is_cuda
        base.checkpoint(path, report, "cold-demand")
        exercise_demand(modules, wp, alias, report, path, folder)
        base.checkpoint(path, report, "warm-waves")
        exercise_warmed_waves(modules, wp, alias, report, folder)
        base.checkpoint(path, report, "optional-fault")
        exercise_copy_fault(modules, wp, alias, report, folder)
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
    print("FSPT-DEVICE-BRIDGE-PASS", path, flush=True)


if __name__ == "__main__":
    main()
