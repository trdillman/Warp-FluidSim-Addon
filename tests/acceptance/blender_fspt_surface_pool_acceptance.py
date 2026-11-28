"""Native two-slot geometry lifetime proof. Run through the private Blender gate."""

from __future__ import annotations

import ctypes
import importlib
import sys
import threading
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_device_surface_acceptance as fixture

base = fixture.base


class StreamGate:
    """Test-only CUDA host function. Never call CUDA inside the callback.

    An independently releasable fence makes pending/completed observations
    deterministic across devices. The timeout turns a blocked verifier into failure.
    """

    def __init__(self, modules, wp, device, stream):
        self.release = threading.Event()
        self.timed_out = threading.Event()
        self.armed = False
        convention = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
        prototype = convention(None, ctypes.c_void_p)

        release, timed_out = self.release, self.timed_out

        @prototype
        def wait(_):
            if not release.wait(15):
                timed_out.set()

        self.callback = wait
        driver = modules("runtime.cuda_driver")
        self.launch = driver.cuda_driver().library.cuLaunchHostFunc
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.wp, self.device, self.stream = wp, device, stream
        self.check = driver.check_cuda

    def arm(self):
        assert not self.armed
        with self.wp.ScopedDevice(self.device):
            self.check(
                self.launch(self.stream.cuda_stream, self.callback, None),
                "cuLaunchHostFunc",
            )
        self.armed = True

    def open(self):
        self.release.set()

    def verify(self):
        assert self.armed and not self.timed_out.is_set(), "test stream gate timed out"


def run(modules, report, folder):
    wp = modules("runtime.warp_runtime")._import_warp()
    api = modules("fspt.surface_pool")
    device = wp.get_device(base.script_device())
    producer = wp.get_stream(device)
    reader = modules("fspt.async_backend")._nonblocking_stream(wp, device)
    receipt = reader.retirement
    pool = api.SurfacePool(device, byte_limit=32 * 1024 * 1024)
    bridge, output = (
        modules("blender.scene_bridge"),
        modules("blender.output_publication"),
    )
    publication = modules("blender.surface_publication")
    origin = (3.0, -2.0, 1.0)
    session = bridge.Session()
    gates, leases = [], []
    rows, expectations = [], []
    try:
        scene, _ = fixture.make_scene(modules, origin)
        cadence = modules("sim.enums").SurfaceUpdateCadence.FRAME
        solver = fixture.make_solver(
            modules, device, origin, waves=True, cadence=cadence
        )
        session.install_solver(
            solver,
            profiler=None,
            device=device,
            scene_inputs=modules("blender.scene_inputs").SceneInputs(),
            solved_timeline_frame=1,
        )
        session.bind_scene(
            modules("blender.scene_access").scene_identity(scene), "fspt-pool-proof"
        )
        for generation in (1, 2):
            solver.step_frame()
            frame = generation + 1
            session.solved_timeline_frame = frame
            scene.frame_set(frame)
            output.update_outputs(scene, session, force_surface=True)
            host_stamp = publication.current_surface_publication(scene)
            assert host_stamp.source_timeline_frame == frame
            surface = output.output_object(scene, output.SURFACE_NAME, ensure=False)
            expected = fixture.evaluated_surface(surface)
            expectations.append(expected)
            stamp = solver.surface.device_surface_stamp()
            assert stamp.build_key == host_stamp.build_key
            # This verifier owns its GPU generation. The host generation remains
            # independently recorded and never certifies completion of this copy.
            key = api.SurfaceKey(
                host_stamp.scene_token,
                host_stamp.owner_token,
                host_stamp.epoch,
                "LIVE",
                "GPU_FINAL",
                generation,
                stamp,
                frame,
                host_stamp.state_revision,
                host_stamp.solver_frame,
                host_stamp.simulation_time,
            )
            assert pool.select(key)
            clock = (solver.frame, solver.sim_time, solver.liquid_state_revision)
            captured = []
            gate = StreamGate(modules, wp, device, producer)
            gates.append(gate)

            def capture(destination, expected_stamp, gate=gate):
                checked = destination._driver.record

                def delay_before_copy(warp, dev, stream, event):
                    checked(warp, dev, stream, event)
                    if event is destination._ready:
                        gate.arm()

                with patch.object(destination._driver, "record", delay_before_copy):
                    packet = solver.host_frame().copy_final_device_surface(
                        destination, expected_stamp=expected_stamp
                    )
                captured.append(destination)
                return packet

            try:
                assert pool.submit(
                    key,
                    len(expected[0]),
                    len(expected[1]),
                    capture,
                    producer_stream=producer,
                )
                pool.poll()
                assert pool.acquire(key, reader) is None
                dest = captured.pop()
                assert dest._sources is not None
                # Enqueue producer reuse of the *actual* fresh extracted arrays
                # behind the owned copy, while its completion is still pending.
                dest._sources.positions.zero_()
                dest._sources.indices.zero_()
            finally:
                gate.open()
            wp.synchronize_stream(producer)
            gate.verify()
            pool.poll()
            assert dest._sources is None
            del dest
            assert clock == (
                solver.frame,
                solver.sim_time,
                solver.liquid_state_revision,
            )
            lease = pool.acquire(key, reader)
            assert lease is not None
            leases.append(lease)
            rows.append(
                {
                    "epoch": key.epoch,
                    "device_generation": generation,
                    "evaluated_generation": host_stamp.generation,
                    "clock": clock,
                    "source_timeline_frame": frame,
                    "vertices": len(expected[0]),
                    "triangles": len(expected[1]),
                    "pending_copy_rejected": True,
                    "source_reuse_queued_while_pending": True,
                }
            )

        assert fixture.digest(expectations[0][0]) != fixture.digest(expectations[1][0])
        assert leases[0].packet.positions.ptr != leases[1].packet.positions.ptr
        assert leases[0].packet.indices.ptr != leases[1].packet.indices.ptr
        future = replace(key, epoch=key.epoch + 1, generation=1)
        assert pool.select(future)

        def forbidden(*_):
            raise AssertionError("pool exhaustion invoked materializer")

        readbacks = []
        for lease, expected in zip(leases, expectations):
            # Both actual reads occur after solver advance and producer overwrite.
            p = wp.empty(len(expected[0]), dtype=wp.vec3, device="cpu", pinned=True)
            t = wp.empty(expected[1].size, dtype=wp.int32, device="cpu", pinned=True)
            lease.retain((p, t))
            wp.copy(p, lease.packet.positions, stream=reader)
            wp.copy(t, lease.packet.indices, stream=reader)
            readbacks.append((p, t))
        gate = StreamGate(modules, wp, device, reader)
        gates.append(gate)
        try:
            gate.arm()
            for lease in leases:
                lease.finish()
            pool.poll()
            assert pool.occupied == 2 and pool.acquire(key, reader) is None
            assert not pool.submit(future, 1, 1, forbidden, producer_stream=producer)
        finally:
            gate.open()
        wp.synchronize_stream(reader)
        gate.verify()
        pool.poll()
        assert pool.occupied == 0 and not pool.select(key)
        for i, ((p, t), expected) in enumerate(zip(readbacks, expectations)):
            actual = (p.numpy().copy(), t.numpy().reshape(-1, 3).copy())
            fixture.require_parity(actual, expected)
            path = folder / f"snapshot-{i + 1}.npz"
            np.savez(path, positions=actual[0], triangles=actual[1])
            rows[i].update(
                positions_sha256=fixture.digest(actual[0]),
                indices_sha256=fixture.digest(actual[1]),
                original=str(path),
                original_sha256=base.sha(path),
            )
        assert pool.allocations == 2 and pool.denials == 1
        report.update(
            snapshots=rows,
            allocations=pool.allocations,
            peak_bytes=pool.peak_bytes,
            bounded_denials=pool.denials,
            retired_slots=2,
            byte_limit=32 * 1024 * 1024,
            last_read_pending_observed=True,
        )
    finally:
        for gate in gates:
            gate.open()
        for lease in leases:
            lease.finish()
        pool.close()
        session._clear_runtime()
    # The host-callback closures retain the reader wrapper until their scopes end.
    gates.clear()
    del gate, reader
    assert receipt.result == 0, receipt
    report["reader_destroy_result"] = receipt.result


def main():
    args = base.arguments("artifacts/fspt-p0/surface-pool.json")
    path = Path(args.json_out).resolve()
    now = datetime.now(timezone.utc)
    folder = path.with_name(
        path.stem + "-originals-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    )
    folder.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "started_at": now.isoformat(),
        "scope": "two-slot source CUDA consumer; live geometry positions/indices only",
        "verifier_sha256": base.sha(Path(__file__)),
        "fixture_sha256": base.sha(Path(fixture.__file__)),
        "base_helper_sha256": base.sha(Path(base.__file__)),
    }
    addon = None
    try:
        base.checkpoint(path, report, "loading")
        addon = base.load_addon(args, report)
        report["runtime"] = base.runtime_info(addon)
        assert report["runtime"]["warp"] == "1.19.0"

        def modules(name):
            return importlib.import_module(addon.__name__ + "." + name)

        package = Path(addon.__file__).resolve().parent
        report["loaded_sources"] = {}
        for name in (
            "fspt.surface_pool",
            "sim.device_surface",
            "sim.surface",
            "sim.frame_output",
            "runtime.cuda_driver",
            "fspt.async_backend",
        ):
            loaded = Path(modules(name).__file__).resolve()
            assert loaded.is_relative_to(package)
            report["loaded_sources"][name] = {
                "path": str(loaded),
                "sha256": base.sha(loaded),
            }
        base.checkpoint(path, report, "two_solver_publications")
        run(modules, report, folder)
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
    print("FSPT-SURFACE-POOL-PASS", path, flush=True)


if __name__ == "__main__":
    main()
