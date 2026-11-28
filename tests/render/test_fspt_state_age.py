"""Completed-state telemetry must not borrow the later publication clock."""
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.blender import fspt_bridge as bridge


def stamp(*, ready=10.0, source="LIVE", published=10.06):
    return SimpleNamespace(source=source, channel='EVALUATED_MESH', state_ready_at=ready,
                           captured_at=10.02, published_at=published)


def test_solver_age_includes_optional_work_before_publication(monkeypatch):
    captured = (stamp(),)
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    observed = bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)
    assert observed["solver_state_age_seconds"] == pytest.approx(0.1)
    assert observed["publication_age_seconds"] == pytest.approx(0.04)
    assert "Solver Session completion" in observed["solver_state_age_clock"]


@pytest.mark.parametrize("captured", [(), (stamp(ready=None),), (stamp(source="CACHE"),),
                                     (stamp(), stamp(ready=None))])
def test_missing_or_cache_clock_stays_unknown(monkeypatch, captured):
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    observed = bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)
    assert observed["solver_state_age_seconds"] is None
    assert observed["solver_state_age_clock"] is None


@pytest.mark.parametrize("ready", [float("nan"), float("inf"), 10.03, 10.2])
def test_invalid_completion_order_is_not_presented_as_a_measurement(monkeypatch, ready):
    captured = (stamp(ready=ready),)
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    with pytest.raises(bridge.SurfaceNotReadyError, match="completion"):
        bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)


def test_multiple_sources_report_the_oldest_completed_state(monkeypatch):
    captured = (stamp(ready=9.5), stamp(ready=9.8))
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    observed = bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)
    assert observed["solver_state_age_seconds"] == pytest.approx(0.6)


def test_paused_state_age_keeps_growing_without_relabeling_it_fresh(monkeypatch):
    captured = (stamp(),)
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    observed = bridge.validate_viewport_publications(object(), captured, clock=lambda: 610.0)
    assert observed["status"] == "CURRENT"
    assert observed["solver_state_age_seconds"] == pytest.approx(600.0)


def test_retained_image_uses_its_own_completion_not_the_latest_solve(monkeypatch):
    captured, current = (stamp(ready=9.5),), (stamp(ready=10.0),)
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: current)
    monkeypatch.setattr(bridge, "forward_live_publication", lambda *a: True)
    observed = bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)
    assert observed["status"] == "RETAINED"
    assert observed["solver_state_age_seconds"] == pytest.approx(0.6)
    assert observed["publication_age_seconds"] == pytest.approx(0.04)


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_unknown_source_does_not_hide_a_malformed_known_clock(monkeypatch, reverse):
    captured = (stamp(ready=None), stamp(ready=float("nan")))
    if reverse:
        captured = captured[::-1]
    monkeypatch.setattr(bridge, "capture_publications", lambda *a, **k: captured)
    with pytest.raises(bridge.SurfaceNotReadyError, match="completion"):
        bridge.validate_viewport_publications(object(), captured, clock=lambda: 10.1)
