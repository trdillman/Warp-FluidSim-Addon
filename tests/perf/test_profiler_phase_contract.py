"""Contract tests for the canonical solver profiler phase vocabulary."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.diagnostics.profiler import Profiler
from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS
from tools.profiling.profile_gpu_solver import _phase_labels


def test_phase_slots_are_complete_and_unique():
    assert set(TIMER_PHASE_SLOTS.values()) == set(range(16))
    assert len(TIMER_PHASE_SLOTS) == len(set(TIMER_PHASE_SLOTS))


def test_unknown_profiler_phase_fails_instead_of_using_misc_slot():
    profiler = Profiler(device="cpu", sync=False)
    profiler._wp = SimpleNamespace()

    with pytest.raises(KeyError, match="unknown profiler phase"):
        with profiler.track("typo"):
            pass


def test_unknown_named_track_and_nonpositive_summary_window_fail_closed():
    profiler = Profiler(device="cpu", sync=False)

    with pytest.raises(KeyError, match="unknown profiler track"):
        with profiler.track_track("typo"):
            pass
    with pytest.raises(ValueError, match="last_n must be positive"):
        profiler.summary(0)


def test_profile_tool_derives_labels_from_canonical_slots():
    assert _phase_labels() == {
        f"compute_{slot}": phase
        for phase, slot in TIMER_PHASE_SLOTS.items()
    }


def test_blender_profiler_labels_derive_from_canonical_slots():
    root = Path(__file__).resolve().parents[2] / "fluidSim_warp_blender51"
    for relative_path in ("blender/viewport_overlay.py", "blender/ui_scene.py"):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "TIMER_PHASE_SLOTS.items()" in source
        assert '"compute_0"' not in source


def _counting_warp():
    """Minimal ``warp`` stand-in that records ``synchronize_device`` calls."""
    calls = []
    return calls, SimpleNamespace(synchronize_device=calls.append)


def test_nested_phase_spans_do_not_double_count_frame_aggregates():
    """An inner span must not inflate the "compute"/"solver" frame totals.

    ``SurfaceTensionSolver`` times its solver-grid SDF rebuild under the
    "surface" slot while already inside the enclosing "forces" span.  Per-slot
    tracks stay supersets (compute_3 includes compute_11), but the aggregate
    tracks are frame totals and must count the wall interval exactly once.
    """
    profiler = Profiler(device="cpu", sync=False)
    profiler._wp = SimpleNamespace()

    with profiler.track("forces"):
        with profiler.track("surface"):
            pass

    timers = profiler.frame.timers
    forces = timers["compute_3"]
    surface = timers["compute_11"]

    assert surface > 0.0
    assert forces >= surface
    assert timers["compute"] == pytest.approx(forces)
    assert timers["solver"] == pytest.approx(forces)
    # The nested interval is counted once, not once per nesting level.
    assert timers["compute"] < forces + surface or surface == 0.0
    assert profiler._track_depth == 0


def test_sibling_phase_spans_still_sum_into_frame_aggregates():
    profiler = Profiler(device="cpu", sync=False)
    profiler._wp = SimpleNamespace()

    with profiler.track("forces"):
        pass
    with profiler.track("viscosity"):
        pass

    timers = profiler.frame.timers
    expected = timers["compute_3"] + timers["compute_4"]
    assert timers["compute"] == pytest.approx(expected)
    assert timers["solver"] == pytest.approx(expected)


def test_nesting_depth_unwinds_on_exception_and_resets_each_frame():
    profiler = Profiler(device="cpu", sync=False)
    profiler._wp = SimpleNamespace()

    with pytest.raises(RuntimeError):
        with profiler.track("forces"):
            with profiler.track("surface"):
                raise RuntimeError("boom")
    assert profiler._track_depth == 0

    profiler._track_depth = 2  # simulate an abandoned span
    profiler.begin_frame()
    assert profiler._track_depth == 0
    with profiler.track("forces"):
        pass
    assert profiler.frame.timers["compute"] == pytest.approx(
        profiler.frame.timers["compute_3"]
    )


def test_nested_spans_keep_per_span_cuda_synchronization():
    """Inner spans still sync, so an inner slot measures real GPU work.

    Two syncs per span is the documented cost of nesting: the inner entry sync
    is charged to the enclosing span and the inner exit sync to the inner span.
    """
    calls, fake_wp = _counting_warp()
    profiler = Profiler(device="cuda:0", sync=True)
    profiler._wp = fake_wp

    with profiler.track("forces"):
        with profiler.track("surface"):
            pass

    assert calls == ["cuda:0"] * 4


def test_sync_disabled_profiler_never_synchronizes():
    calls, fake_wp = _counting_warp()
    profiler = Profiler(device="cuda:0", sync=False)
    profiler._wp = fake_wp

    with profiler.track("forces"):
        with profiler.track("surface"):
            pass

    assert calls == []
