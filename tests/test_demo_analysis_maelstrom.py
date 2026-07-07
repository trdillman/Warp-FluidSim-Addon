"""Regression tests for the maelstrom demo analysis contract."""

from copy import deepcopy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_demos", ROOT / "tools" / "analyze_demos.py"
)
analyze_demos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_demos)


GOOD_META = {
    "scene": "maelstrom_basin",
    "playable_live_setup": True,
    "seed_object": "Maelstrom Vortex-Ring Seed",
    "spin_force_object": "Maelstrom Spin Column",
    "spin_strength": 7.0,
    "max_particles": 2_000_000,
    "injected": 1_085_840,
    "fps": 20.89,
    "fps_target": 10.0,
    "p95_ms": 54.8,
    "max_whitewater": 84_778,
}


def _run(meta):
    analyze_demos.CHECKS.clear()
    analyze_demos.validate_maelstrom_basin(meta)
    checks = list(analyze_demos.CHECKS)
    analyze_demos.CHECKS.clear()
    return checks


def test_maelstrom_contract_accepts_playable_measured_basin():
    checks = _run(deepcopy(GOOD_META))

    assert len(checks) == 5
    assert all(item["passed"] for item in checks)
    names = {item["check"] for item in checks}
    assert "high-particle basin metadata" in names
    assert "live playback metadata meets FPS target" in names
    assert "whitewater churn metadata present" in names
    assert not any("measured" in item["check"] for item in checks)


def test_maelstrom_contract_rejects_lost_spin_and_churn():
    meta = deepcopy(GOOD_META)
    meta["spin_force_object"] = "Plain Cylinder"
    meta["spin_strength"] = 0.0
    meta["max_whitewater"] = 0

    checks = _run(meta)
    failed = {item["check"] for item in checks if not item["passed"]}

    assert "spin force configured" in failed
    assert "whitewater churn metadata present" in failed
