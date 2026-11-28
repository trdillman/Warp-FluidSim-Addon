"""Host controls for the native verifier's strict worker/ownership evidence."""
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


@pytest.fixture
def verifier(monkeypatch):
    # The pure evidence checker does not use Blender or import CUDA.
    path = Path(__file__).resolve().parents[1] / "acceptance" / "blender_fspt_device_viewport_acceptance.py"
    for name in ("bpy", "mathutils", "blender_fspt_acceptance",
                 "blender_fspt_device_bridge_acceptance", "blender_fspt_device_surface_acceptance"):
        stub = ModuleType(name)
        if name == "mathutils":
            stub.Vector = object
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return runpy.run_path(str(path))


@pytest.fixture
def evidence(verifier):
    events = [{"event": name, "thread": 12, "on_main": False, "hold_active": True,
               "coordinator_blocked": True, "resources_remaining": False}
              for name in verifier["FINAL_ORDER"]]
    events[1].update(thread=9, on_main=True, view_owned=True, aggregate_open=True,
                     reserved_bytes=128, generation_count=1, resources_remaining=True, trace_pending=False)
    events[2].update(thread=9, on_main=True, view_owned=False, request_matches=True,
                     owners_retired=True, reserved_bytes=0, generation_count=0)
    events[4].update(backend_type="AsyncCudaBackend")
    events[6].update(backend_closed=True, render_stream_retirement=0)
    events[7].update(hold_active=False, coordinator_blocked=False)
    events.append({"event": "render_complete"})
    return ({"events": events, "viewport_before": {"trace_pending": True}},
            {"requests": 1, "serviced": 1, "timeouts": 0, "failures": 0})


def test_worker_after_identity_bound_zero_owner_receipt(verifier, evidence):
    verifier["verify_handoff"](*evidence)


@pytest.mark.parametrize(("index", "field", "value"), [
    (0, "on_main", True),
    (1, "aggregate_open", False),
    (1, "trace_pending", True),
    (2, "request_matches", False),
    (2, "generation_count", 1),
    (3, "resources_remaining", True),
    (5, "hold_active", False),
    (6, "render_stream_retirement", 1),
    (7, "hold_active", True),
    (8, "event", "render_cancel"),
])
def test_rejects_incomplete_or_overlapping_worker_evidence(verifier, evidence, index, field, value):
    report, _ = evidence
    report["events"][index][field] = value
    with pytest.raises(AssertionError):
        verifier["verify_handoff"](*evidence)


def test_rejects_constructor_before_retirement(verifier, evidence):
    report, _ = evidence
    events = report["events"]
    events[2], events[3] = events[3], events[2]
    with pytest.raises(AssertionError):
        verifier["verify_handoff"](*evidence)
