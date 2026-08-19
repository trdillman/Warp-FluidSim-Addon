"""Contract tests for the canonical solver profiler phase vocabulary."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.diagnostics.profiler import Profiler
from fluidSim_warp_blender51.sim.constants import TIMER_PHASE_SLOTS
from tools.profile_gpu_solver import _phase_labels


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
    root = Path(__file__).resolve().parents[1] / "fluidSim_warp_blender51"
    for relative_path in ("blender/viewport_overlay.py", "panels.py"):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "TIMER_PHASE_SLOTS.items()" in source
        assert '"compute_0"' not in source
