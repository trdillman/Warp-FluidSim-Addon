from __future__ import annotations

from pathlib import Path

from fluidsim_testpkg.diagnostics import validation
from fluidsim_testpkg.sim.validation import ValidationResult

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostics_orchestrator_appends_benchmark_contract(monkeypatch):
    simulation_result = ValidationResult("simulation contract", True)
    progress = []

    def run_simulation(*, device, progress):
        assert device == "cuda:test"
        progress(simulation_result)
        return [simulation_result]

    monkeypatch.setattr(
        validation.simulation_validation, "run_all", run_simulation
    )
    monkeypatch.setattr(
        validation,
        "test_benchmark_schema",
        lambda device: (device == "cuda:test", "schema checked"),
    )

    results = validation.run_all(device="cuda:test", progress=progress.append)

    assert [result.name for result in results] == [
        "simulation contract",
        validation.BENCHMARK_SCHEMA_TEST,
    ]
    assert progress == results
    assert results[-1].passed is True
    assert results[-1].detail == "schema checked"


def test_simulation_validation_does_not_import_diagnostics():
    source = (ROOT / "sim" / "validation.py").read_text(encoding="utf-8")

    assert "diagnostics" not in source
    assert validation.VALIDATION_TESTS[-1] == validation.BENCHMARK_SCHEMA_TEST
    assert len(validation.VALIDATION_TESTS) == 20
