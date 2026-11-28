"""Bind native FSPT ownership reports to current sources and retained originals.

The verifier performs numerical checks in Blender. This controller independently
requires its substantive cases, exact loaded code/runtime, and original files.
No status string, success marker, or zero process exit substitutes for those.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

BROKER = "blender_fspt_viewport_backend_acceptance.py"
VIEWPORT = "blender_fspt_device_viewport_acceptance.py"
SOURCES = {
    BROKER: {"fspt.viewport_backend", "fspt.device_backend", "fspt.device_scene",
             "fspt.device_scene_kernels", "fspt.backend", "fspt.async_backend",
             "fspt.surface_pool", "sim.device_surface", "runtime.cuda_driver",
             "blender.fspt_device_scene", "blender.fspt_bridge", "blender.surface_publication",
             "sim.surface", "sim.frame_output"},
    VIEWPORT: {"blender.fspt_engine", "blender.fspt_device_runtime", "blender.fspt_device_bridge",
               "blender.fspt_device_scene", "blender.fspt_bridge", "blender.output_publication",
               "blender.surface_publication", "blender.scene_bridge", "fspt.viewport",
               "fspt.viewport_backend", "fspt.final_handoff", "fspt.device_scene",
               "fspt.device_backend", "fspt.surface_pool", "fspt.async_backend"},
}
FINAL_ORDER = ("render_enter", "retirement_enter", "retirement_receipt", "final_backend_allocate",
               "final_backend_allocated", "final_backend_close_enter", "final_backend_closed", "render_exit")


def require(condition, reason):
    if not condition:
        raise RuntimeError("FSPT evidence: " + reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def python_tree(package):
    result = hashlib.sha256()
    for source in sorted(package.rglob("*.py")):
        result.update(source.relative_to(package).as_posix().encode() + b"\0")
        result.update(bytes.fromhex(sha(source)))
    return result.hexdigest()


def original(parent, record, files, path_key="original", hash_key="original_sha256"):
    value, expected = record.get(path_key), record.get(hash_key)
    require(isinstance(value, str) and digest(expected), "missing original path or SHA-256")
    path = Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(parent.resolve()),
            "original must belong to this report directory")
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0 and sha(path) == expected,
            "original changed, missing, or is a symbolic link: " + value)
    canonical = str(path.resolve())
    require(canonical not in files, "each required observation needs a distinct original")
    files[canonical] = expected


def _identity(root, name, report, context):
    package = root / "fluidSim_warp_blender51"
    require(report.get("stage") == "complete" and bool(report.get("scope")), "incomplete scope/stage")
    require(report.get("verifier_sha256") == sha(root / "tests/acceptance" / name),
            "verifier source identity differs")
    require(Path(report.get("addon", "")).resolve() == (package / "__init__.py").resolve(),
            "loaded add-on is not the qualified source")
    require(report.get("python_tree_sha256") == python_tree(package), "loaded Python tree differs")
    loaded = report.get("loaded_sources")
    require(isinstance(loaded, dict) and SOURCES[name] <= loaded.keys(), "required loaded sources missing")
    for module, item in loaded.items():
        require(isinstance(module, str) and re.fullmatch(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*", module),
                "invalid source module")
        expected = package.joinpath(*module.split(".")).with_suffix(".py").resolve()
        require(isinstance(item, dict) and Path(item.get("path", "")).resolve() == expected,
                "loaded module path differs: " + module)
        require(expected.is_file() and item.get("sha256") == sha(expected), "loaded source hash differs: " + module)
    runtime = report.get("runtime", {})
    require(isinstance(runtime, dict) and runtime.get("warp") == "1.19.0"
            and re.fullmatch(r"cuda:\d+", str(runtime.get("device")))
            and runtime.get("gpu") and runtime.get("warp_path"), "missing certified CUDA runtime")
    observed = (context or {}).get("observed_environment")
    if observed is not None:
        require(runtime["warp"] == observed["warp_version"]
                and runtime["gpu"] == observed["gpu_model"]
                and Path(runtime["warp_path"]).resolve() == Path(observed["warp_loaded_path"]).resolve()
                and report.get("blender_build") == observed["blender_build"]
                and report.get("python") == observed["python_version"],
                "report runtime differs from admitted environment")


def _broker(report, parent, files):
    broker, empty, live = (report.get(key, {}) for key in ("broker", "empty_generation", "live_adapter"))
    require(all(isinstance(item, dict) and item.get("status") == "VERIFIED" for item in (broker, empty, live)),
            "broker, zero-byte generation and live adapter cases required")
    require(integer(broker.get("worker_thread"), 1) and integer(broker.get("publication_thread"), 1)
            and broker["worker_thread"] != broker["publication_thread"], "worker collection unproven")
    require(broker.get("queued_stale_rejected") is True and broker.get("poisoned_recovery_checked") is True,
            "stale/recovery controls missing")
    require(integer(broker.get("peak_generations"), 1) and broker["peak_generations"] <= 2
            and integer(broker.get("peak_explicit_array_bytes"), 1)
            and integer(broker.get("two_facade_shared_bytes"), 1), "shared geometry accounting missing")
    receipts = broker.get("render_stream_retirement_results")
    require(isinstance(receipts, list) and receipts and all(type(value) is int and value == 0 for value in receipts),
            "render stream retirement not checked")
    require(all(type(empty.get(key)) is int and empty[key] == value for key, value in
                (("pending_bytes", 0), ("pending_generations", 1), ("final_generations", 0))),
            "zero-byte generation lifetime unproven")
    require(live.get("guard_negative_controls") == 3 and live.get("fluid_bulk_accesses") == []
            and type(live.get("render_stream_retirement_result")) is int
            and live["render_stream_retirement_result"] == 0, "live adapter controls incomplete")
    original(parent, broker, files, "originals", "sha256")
    original(parent, live, files, "originals", "sha256")


def _viewport(report, parent, files):
    require(bool(report.get("frontend_fixture")), "controlled frontend scope missing")
    frames = report.get("frames")
    require(isinstance(frames, list) and len(frames) == 2 and all(isinstance(row, dict) for row in frames),
            "two advancing GPU frames required")
    for index, row in enumerate(frames):
        publication = row.get("publication", {})
        require(isinstance(publication, dict) and publication.get("channel") == "GPU_FINAL"
                and integer(publication.get("generation"), 1), "GPU publication identity missing")
        require(all(integer(row.get(key), 1) for key in ("timeline_frame", "solver_frame", "state_revision"))
                and row.get("samples") == 4 and row.get("backend") and row.get("transport")
                and row.get("device_geometry_copies") == index + 1
                and row.get("publication_clock_order_verified") is True, "frame provenance incomplete")
        require(all(isinstance(row.get(key), list) and len(row[key]) == 2 and all(map(digest, row[key]))
                    for key in ("source_geometry_sha256", "evaluated_geometry_sha256"))
                and digest(row.get("pixels_sha256")), "geometry/image hashes missing")
        original(parent, row, files)
    require(all(frames[1][key] > frames[0][key] for key in ("timeline_frame", "solver_frame", "state_revision"))
            and frames[1]["publication"]["generation"] > frames[0]["publication"]["generation"]
            and frames[1]["pixels_sha256"] != frames[0]["pixels_sha256"]
            and frames[1]["source_geometry_sha256"] != frames[0]["source_geometry_sha256"]
            and frames[1].get("stale_gpu_publication_rejected_before_poll") is True,
            "GPU states did not advance with distinct geometry and images")
    final = report.get("final", {})
    require(isinstance(final, dict) and "RUNNING_MODAL" in final.get("operator_result", [])
            and final.get("event_order_verified") is True
            and final.get("render_info", {}).get("status") == "COMPLETE", "actual GUI Final incomplete")
    events = final.get("events", [])
    require(isinstance(events, list) and all(isinstance(row, dict) for row in events), "invalid Final events")
    ordered = [row for row in events if row.get("event") in FINAL_ORDER]
    require(tuple(row.get("event") for row in ordered) == FINAL_ORDER, "Final ownership event order differs")
    enter, retire, receipt, allocate, allocated, close, closed, leave = ordered
    require(integer(enter.get("thread"), 1) and enter.get("on_main") is False
            and all(row.get("on_main") is False and row.get("thread") == enter["thread"]
                    for row in (allocate, allocated, close, closed, leave))
            and retire.get("on_main") is True and receipt.get("on_main") is True
            and integer(retire.get("thread"), 1) and retire["thread"] == receipt.get("thread") != enter["thread"],
            "Final worker/main thread boundary unproven")
    require(final.get("viewport_before", {}).get("trace_pending") is True
            and retire.get("trace_pending") is False and retire.get("view_owned") is True
            and retire.get("aggregate_open") is True and integer(retire.get("reserved_bytes"), 1)
            and integer(retire.get("generation_count"), 1), "pending viewport drain and owned aggregate unproven")
    require(receipt.get("request_matches") is True and receipt.get("owners_retired") is True
            and all(type(receipt.get(key)) is int and receipt[key] == 0 for key in ("reserved_bytes", "generation_count"))
            and receipt.get("resources_remaining") is False and receipt.get("view_owned") is False,
            "Final retirement receipt is not identity-bound zero ownership")
    require(all(row.get("hold_active") is True and row.get("coordinator_blocked") is True
                for row in (retire, receipt, allocate, allocated, close, closed))
            and leave.get("hold_active") is False and leave.get("coordinator_blocked") is False
            and all(row.get("resources_remaining") is False for row in (allocate, allocated)),
            "Final device hold or allocation exclusion unproven")
    require(allocated.get("backend_type") == "AsyncCudaBackend" and closed.get("backend_closed") is True
            and type(closed.get("render_stream_retirement")) is int and closed["render_stream_retirement"] == 0
            and final.get("handoff_statistics_delta") == {"requests": 1, "serviced": 1, "timeouts": 0, "failures": 0}
            and sum(row.get("event") == "render_complete" for row in events) == 1
            and not any(row.get("event") == "render_cancel" for row in events), "Final completion/checked close unproven")
    values = final.get("pixel_range", [])
    require(final.get("image_size") == [128, 96] and digest(final.get("pixels_sha256"))
            and isinstance(values, list) and len(values) == 2
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            and values[1] > max(0, values[0]), "Final original has no finite nonconstant image evidence")
    original(parent, final, files)
    cleanup = report.get("cleanup", {})
    require(isinstance(cleanup, dict) and cleanup.get("viewport_closed") is True
            and cleanup.get("resources_remaining") is False and cleanup.get("hold_remaining") is False
            and cleanup.get("addon_unregistered") is True, "checked cleanup missing")


def validate_report(root, script, path, report, qualification_context=None):
    name = Path(script).name
    if name not in SOURCES:
        return {}
    files = {}
    try:
        _identity(Path(root), name, report, qualification_context)
        (_broker if name == BROKER else _viewport)(report, Path(path).parent, files)
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as exc:
        raise RuntimeError(f"FSPT evidence: malformed or missing report field: {exc}") from exc
    return files
