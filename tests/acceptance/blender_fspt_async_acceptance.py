"""Native Blender/CUDA backend acceptance using the P1 evaluated optical fixture."""
import hashlib
import importlib
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_fspt_acceptance as p1


def poll_frame(backend):
    deadline = time.monotonic() + 60
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        frame = backend.collect()
        if frame is not None:
            return frame, polls
        time.sleep(0.001)
    raise TimeoutError("FSPT copy event did not retire within 60 seconds")


def main():
    args = p1.arguments("artifacts/fspt-p2/async-backend.json")
    path = Path(args.json_out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "scope": "native direct backend, evaluated P1 optical fixture",
              "verifier_sha256": p1.sha(Path(__file__))}
    addon = reference = first = second = None
    try:
        p1.checkpoint(path, report, "register")
        addon = p1.load_addon(args, report)
        report["runtime"] = p1.runtime_info(addon)
        module = importlib.import_module(addon.__name__ + ".fspt.async_backend")
        adapter = importlib.import_module(addon.__name__ + ".blender.fspt_scene")
        report["backend_file"] = module.__file__
        report["backend_sha256"] = p1.sha(Path(module.__file__))
        scene, _node, _sky = p1.fixture()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        snapshot = adapter.extract_scene(depsgraph)
        width, height = p1.WIDTH, p1.HEIGHT
        camera = adapter.camera_snapshot(depsgraph, width, height)
        half = adapter.camera_snapshot(depsgraph, width // 2, height // 2)
        options = {"max_bounces": 6, "seed": 31}
        device = p1.script_device()

        p1.checkpoint(path, report, "synchronous_reference")
        reference = module.CudaBackend(device)
        baseline = reference.render(snapshot, camera, width, height, samples=3, **options)
        refined = reference.render(snapshot, camera, width, height, samples=2, **options)
        smaller = reference.render(snapshot, half, width // 2, height // 2, samples=2, **options)
        reference.close()
        reference = None
        assert np.std(baseline.pixels[..., :3]) > 0.03
        first, second = module.AsyncCudaBackend(device), module.AsyncCudaBackend(device)
        assert first.arbiter is second.arbiter
        wp = first.wp
        copy_array, read_array = wp.copy, wp.array.numpy
        transfers = []
        fault_next_copy = False

        def observed_copy(dest, src, *a, **kw):
            nonlocal fault_next_copy
            if src.device.is_cuda and dest.device.is_cpu:
                stream = kw["stream"]
                assert dest.pinned and not stream.is_blocking
                transfers.append({"pixels": kw["count"], "stream": int(stream.cuda_stream),
                                  "pinned": bool(dest.pinned), "injected": fault_next_copy})
                if fault_next_copy:
                    fault_next_copy = False
                    raise RuntimeError("injected pinned transfer refusal")
            return copy_array(dest, src, *a, **kw)

        def host_numpy(array, *a, **kw):
            assert array.device.is_cpu, "device array performed implicit bulk readback"
            return read_array(array, *a, **kw)

        def no_global_wait(*a, **kw):
            raise AssertionError("async acceptance reached a device-wide wait")

        with ExitStack() as guards:
            guards.enter_context(patch.object(wp, "copy", observed_copy))
            guards.enter_context(patch.object(wp.array, "numpy", host_numpy))
            guards.enter_context(patch.object(wp, "synchronize_device", no_global_wait))
            p1.checkpoint(path, report, "event_poll_and_exact_pixels")
            assert first.submit(snapshot, camera, width, height, samples=3, **options)
            assert first.pending and not first._render_stream.is_blocking
            assert not second.submit(snapshot, camera, width, height, samples=3, **options)
            frame, polls = poll_frame(first)
            assert frame.key == baseline.key and frame.samples == 3
            assert frame.scene_frame == snapshot.frame and frame.publications == snapshot.publications
            np.testing.assert_array_equal(frame.pixels, baseline.pixels)
            report["nonblocking_poll_calls"] = polls
            report["frame"] = {"key": frame.key, "scene_frame": frame.scene_frame,
                               "samples": frame.samples, "device": frame.device,
                               "backend": frame.backend, "transport": frame.transport,
                               "geometry_digest": frame.geometry_digest}
            original = path.with_name(path.stem + "-rgba.npy")
            np.save(original, frame.pixels)
            report["original_rgba"] = {"path": str(original), "sha256": p1.sha(original),
                                       "shape": list(frame.pixels.shape)}
            old_hash = hashlib.sha256(frame.pixels.tobytes()).hexdigest()

            p1.checkpoint(path, report, "second_view_and_accumulation")
            assert second.submit(snapshot, camera, width, height, samples=3, **options)
            assert not first.submit(snapshot, camera, width, height, samples=2, **options)
            np.testing.assert_array_equal(second.collect(wait=True).pixels, baseline.pixels)
            assert first.submit(snapshot, camera, width, height, samples=2, **options)
            continued = first.collect(wait=True)
            assert continued.samples == 5
            np.testing.assert_array_equal(continued.pixels, refined.pixels)
            assert first.submit(snapshot, half, width // 2, height // 2, samples=2, **options)
            reduced = first.collect(wait=True)
            assert reduced.samples == 2
            np.testing.assert_array_equal(reduced.pixels, smaller.pixels)
            assert hashlib.sha256(frame.pixels.tobytes()).hexdigest() == old_hash
            counts = dict(reduced.statistics)
            assert counts["image_allocations"] == counts["mesh_builds"] == counts["geometry_uploads"] == 1
            report["counts_after_resize"] = counts

            p1.checkpoint(path, report, "partial_copy_failure_then_retry")
            fault_next_copy = True
            try:
                first.submit(snapshot, camera, width, height, samples=3, **options)
            except RuntimeError as exc:
                assert "injected pinned transfer refusal" in str(exc), exc
                report["reached_copy_fault"] = str(exc)
            else:
                raise AssertionError("Injected partial submission was accepted")
            assert not first.pending and first.history.samples == 0
            assert first.submit(snapshot, camera, width, height, samples=3, **options)
            retry = first.collect(wait=True)
            assert retry.samples == 3
            np.testing.assert_array_equal(retry.pixels, baseline.pixels)
            assert len(transfers) == 6 and sum(t["injected"] for t in transfers) == 1
            report["pinned_transfers"] = transfers

            # Keep only the retirement receipts. Keeping a borrowed Warp Stream
            # here would itself retain its native handle after backend close.
            retirements = [first._render_stream.retirement, second._render_stream.retirement]
            first.close()
            second.close()
            assert [r.result for r in retirements] == [0, 0], retirements
            report["cuStreamDestroy_results"] = [r.result for r in retirements]
        scene.render.engine = "BLENDER_WORKBENCH"
        addon.unregister()
        addon = None
        report.update(status="VERIFIED", stage="complete")
    except BaseException:
        report.update(status="FAILED", error=traceback.format_exc())
        raise
    finally:
        try:
            for backend in (first, second, reference):
                if backend is not None:
                    backend.close()
            if addon is not None:
                addon.unregister()
        finally:
            p1.write_report(path, report)
    print("FSPT-ASYNC-BACKEND-PASS", path, flush=True)


if __name__ == "__main__":
    main()
