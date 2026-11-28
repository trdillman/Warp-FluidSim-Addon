"""Independent controller rejects plausible but incomplete native receipts."""
import copy
import json
from pathlib import Path

import pytest

from tools import run_production_gate as gate
from tools.gate import fspt_evidence as evidence


@pytest.fixture
def case(tmp_path):
    package = tmp_path / "fluidSim_warp_blender51"
    package.mkdir()
    (package / "__init__.py").write_text("# source fixture\n")
    loaded = {}
    for module in set.union(*evidence.SOURCES.values()):
        source = package.joinpath(*module.split(".")).with_suffix(".py")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# " + module + "\n")
        loaded[module] = {"path": str(source), "sha256": evidence.sha(source)}
    for script in evidence.SOURCES:
        source = tmp_path / "tests/acceptance" / script
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# verifier fixture " + script)
    folder = tmp_path / "artifacts/run"
    folder.mkdir(parents=True)
    originals = []
    for name in ("frame-a.npz", "frame-b.npz", "final.exr"):
        source = folder / name
        source.write_bytes(name.encode())
        originals.append({"original": str(source), "original_sha256": evidence.sha(source)})
    common = {"status": "VERIFIED", "stage": "complete", "scope": "controlled fixture",
              "addon": str(package / "__init__.py"), "python_tree_sha256": evidence.python_tree(package),
              "loaded_sources": loaded, "runtime": {"warp": "1.19.0", "device": "cuda:0",
              "gpu": "test GPU", "warp_path": str(tmp_path / "warp/__init__.py")},
              "python": "test Python", "blender_build": "test build"}
    broker = dict(common, verifier_sha256=evidence.sha(tmp_path / "tests/acceptance" / evidence.BROKER))
    broker.update(broker={"status": "VERIFIED", "worker_thread": 2, "publication_thread": 1,
                         "queued_stale_rejected": True, "poisoned_recovery_checked": True,
                         "peak_generations": 2, "peak_explicit_array_bytes": 100,
                         "two_facade_shared_bytes": 80, "render_stream_retirement_results": [0, 0],
                         "originals": originals[0]["original"], "sha256": originals[0]["original_sha256"]},
                  empty_generation={"status": "VERIFIED", "pending_bytes": 0, "pending_generations": 1,
                                    "final_generations": 0},
                  live_adapter={"status": "VERIFIED", "guard_negative_controls": 3, "fluid_bulk_accesses": [],
                                "render_stream_retirement_result": 0, "originals": originals[1]["original"],
                                "sha256": originals[1]["original_sha256"]})
    events = []
    for name in evidence.FINAL_ORDER:
        main = name in ("retirement_enter", "retirement_receipt")
        row = {"event": name, "on_main": main, "thread": 1 if main else 2,
               "hold_active": name not in ("render_enter", "render_exit"),
               "coordinator_blocked": name not in ("render_enter", "render_exit"),
               "resources_remaining": False}
        events.append(row)
    events[1].update(view_owned=True, aggregate_open=True, trace_pending=False, reserved_bytes=100, generation_count=1)
    events[2].update(request_matches=True, owners_retired=True, reserved_bytes=0, generation_count=0, view_owned=False)
    events[4]["backend_type"] = "AsyncCudaBackend"
    events[6].update(backend_closed=True, render_stream_retirement=0)
    events.append({"event": "render_complete"})
    frames = []
    for i in range(2):
        frames.append(dict(originals[i], timeline_frame=i + 2, solver_frame=i + 1, state_revision=i + 1,
                           publication={"channel": "GPU_FINAL", "generation": i + 1}, samples=4,
                           backend="Warp-CUDA", transport="pinned-host", device_geometry_copies=i + 1,
                           publication_clock_order_verified=True, stale_gpu_publication_rejected_before_poll=bool(i),
                           source_geometry_sha256=[str(i) * 64, "a" * 64],
                           evaluated_geometry_sha256=[str(i) * 64, "a" * 64], pixels_sha256=str(i) * 64))
    viewport = dict(common, verifier_sha256=evidence.sha(tmp_path / "tests/acceptance" / evidence.VIEWPORT),
                    frontend_fixture="controlled ViewportState", frames=frames,
                    final=dict(originals[2], operator_result=["RUNNING_MODAL"], event_order_verified=True,
                               render_info={"status": "COMPLETE"}, events=events, viewport_before={"trace_pending": True},
                               handoff_statistics_delta={"requests": 1, "serviced": 1, "timeouts": 0, "failures": 0},
                               pixels_sha256="a" * 64, image_size=[128, 96], pixel_range=[0.0, 1.0]),
                    cleanup={"viewport_closed": True, "resources_remaining": False,
                             "hold_remaining": False, "addon_unregistered": True})
    return tmp_path, folder / "acceptance.json", broker, viewport


@pytest.mark.parametrize("index,script,count", [(2, evidence.BROKER, 2), (3, evidence.VIEWPORT, 3)])
def test_complete_report_binds_all_originals(case, index, script, count):
    root, report_path, *_ = case
    result = evidence.validate_report(root, script, report_path, case[index])
    assert len(result) == count
    assert all(evidence.sha(path) == expected for path, expected in result.items())


@pytest.mark.parametrize("field,value", [
    ("loaded_sources", {}), ("verifier_sha256", "0" * 64), ("python_tree_sha256", "0" * 64),
    ("addon", "/foreign/__init__.py"), ("runtime", {"warp": "1.18.0"}),
    ("stage", "two-facade-transition"), ("broker", {}), ("empty_generation", {}), ("live_adapter", {}),
])
def test_incomplete_or_transplanted_broker_report_rejected(case, field, value):
    root, path, broker, _ = case
    broker[field] = value
    with pytest.raises(RuntimeError, match="FSPT evidence"):
        evidence.validate_report(root, evidence.BROKER, path, broker)


@pytest.mark.parametrize("event,key,value", [
    (1, "trace_pending", True), (1, "reserved_bytes", 0), (2, "reserved_bytes", False),
    (2, "generation_count", 1), (2, "request_matches", False), (2, "on_main", False),
    (3, "resources_remaining", True), (3, "thread", 1), (4, "backend_type", "fake"),
    (5, "hold_active", False), (6, "render_stream_retirement", None), (7, "hold_active", True),
])
def test_final_ownership_evidence_must_be_complete(case, event, key, value):
    root, path, _, viewport = case
    viewport["final"]["events"][event][key] = value
    with pytest.raises(RuntimeError, match="FSPT evidence"):
        evidence.validate_report(root, evidence.VIEWPORT, path, viewport)


@pytest.mark.parametrize("fault", ["identical_frames", "missing_frame", "reordered", "cancel", "cleanup", "outside_original"])
def test_gui_proof_requires_progression_final_and_cleanup(case, fault):
    root, path, _, viewport = case
    if fault == "identical_frames":
        viewport["frames"][1] = copy.deepcopy(viewport["frames"][0])
    elif fault == "missing_frame":
        viewport["frames"].pop()
    elif fault == "reordered":
        viewport["final"]["events"][1:3] = viewport["final"]["events"][1:3][::-1]
    elif fault == "cancel":
        viewport["final"]["events"].append({"event": "render_cancel"})
    elif fault == "cleanup":
        viewport["cleanup"]["hold_remaining"] = True
    else:
        viewport["final"]["original"] = str(root / "foreign.exr")
    with pytest.raises(RuntimeError, match="FSPT evidence"):
        evidence.validate_report(root, evidence.VIEWPORT, path, viewport)


def test_admitted_runtime_is_compared(case):
    root, path, broker, _ = case
    observed = {"warp_version": "1.19.0", "gpu_model": "test GPU", "warp_loaded_path": broker["runtime"]["warp_path"],
                "blender_build": "test build", "python_version": "test Python"}
    evidence.validate_report(root, evidence.BROKER, path, broker, {"observed_environment": observed})
    observed["gpu_model"] = "other GPU"
    with pytest.raises(RuntimeError, match="admitted environment"):
        evidence.validate_report(root, evidence.BROKER, path, broker, {"observed_environment": observed})


def test_manifest_rechecks_originals_after_success(case, monkeypatch):
    root, path, broker, _ = case
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "_RUN_DIRECTORY", path.parent)
    monkeypatch.setattr(gate, "_QUALIFICATION", None)
    monkeypatch.setattr(gate, "_FSPT_REPORTS", {})
    path.write_text(json.dumps(broker))
    gate._record_fspt_report("broker", evidence.BROKER, path)
    assert len(gate._fspt_report_manifest()[evidence.BROKER]["evidence_files"]) == 2
    Path(broker["broker"]["originals"]).write_bytes(b"changed after native exit")
    with pytest.raises(RuntimeError, match="original evidence changed"):
        gate._fspt_report_manifest()


def test_reused_original_cannot_stand_for_two_observations(case):
    root, path, broker, _ = case
    broker["live_adapter"]["originals"] = broker["broker"]["originals"]
    broker["live_adapter"]["sha256"] = broker["broker"]["sha256"]
    with pytest.raises(RuntimeError, match="distinct original"):
        evidence.validate_report(root, evidence.BROKER, path, broker)


def test_empty_original_rejected_even_with_matching_hash(case):
    root, path, broker, _ = case
    original = Path(broker["broker"]["originals"])
    original.write_bytes(b"")
    broker["broker"]["sha256"] = evidence.sha(original)
    with pytest.raises(RuntimeError, match="original changed"):
        evidence.validate_report(root, evidence.BROKER, path, broker)
