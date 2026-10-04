"""Fail-closed benchmark comparison and baseline operator regressions."""

from __future__ import annotations

import importlib
import json
import math
import sys
import types
from types import SimpleNamespace

import pytest

from fluidSim_warp_blender51.diagnostics.benchmark import compare_benchmark


def _benchmark_record(**changes):
    record = {
        "totalTime": 0.2,
        "activeVoxelsAverage": 10,
        "activeVoxelsMinimum": 8,
        "activeVoxelsMaximum": 12,
        "timeStepAverage": 0.1,
        "timeStepMinimum": 0.08,
        "timeStepMaximum": 0.12,
        "voxelsPerSecondAverage": 100.0,
        "voxelsPerSecondMinimum": 80.0,
        "voxelsPerSecondMaximum": 120.0,
        "result": "completed",
    }
    record.update(changes)
    return record


@pytest.mark.parametrize(
    "value",
    [None, "0.1", [], {}, True, float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.parametrize("side", ["current", "baseline"])
def test_compare_benchmark_reports_malformed_performance_numbers(side, value):
    current = _benchmark_record()
    baseline = _benchmark_record()
    target = current if side == "current" else baseline
    target["timeStepAverage"] = value

    report = compare_benchmark(current, baseline)

    assert report["result"] == "fail"
    detail = report["performance"]["timeStepAverage"]
    assert detail["within_tolerance"] is False
    assert detail["error"] == f"{side} must be a finite JSON number"
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            assert math.isnan(detail[side])
        else:
            assert detail[side] == value


@pytest.mark.parametrize("value", [None, "10", [], {}, True, float("nan")])
def test_compare_benchmark_reports_malformed_exact_numbers(value):
    current = _benchmark_record(activeVoxelsAverage=value)

    report = compare_benchmark(current, _benchmark_record())

    assert report["result"] == "fail"
    detail = report["exact"]["activeVoxelsAverage"]
    assert detail["match"] is False
    assert detail["error"] == "current must be a finite JSON number"


@pytest.mark.parametrize(
    "field",
    [
        "totalTime",
        "timeStepMinimum",
        "timeStepMaximum",
        "voxelsPerSecondMinimum",
        "voxelsPerSecondMaximum",
    ],
)
@pytest.mark.parametrize("side", ["current", "baseline"])
def test_compare_benchmark_validates_every_numeric_contract_field(field, side):
    current = _benchmark_record()
    baseline = _benchmark_record()
    target = current if side == "current" else baseline
    target[field] = None

    report = compare_benchmark(current, baseline)

    assert report["result"] == "fail"
    assert report["error"] == f"{field}: {side} must be a finite JSON number"
    assert report["contract"][field]["valid"] is False


@pytest.mark.parametrize("payload", [None, "benchmark", [], 1, True])
def test_compare_benchmark_reports_non_object_payload(payload):
    report = compare_benchmark(_benchmark_record(), payload)

    assert report == {
        "contract": {},
        "exact": {},
        "performance": {},
        "result": "fail",
        "error": "baseline must be a JSON object",
    }


def _install_bpy_stub(monkeypatch):
    fake_bpy = types.ModuleType("bpy")

    class _Operator:
        @classmethod
        def poll_message_set(cls, _message):
            return None

        def report(self, *_args):
            return None

    def _property(**kwargs):
        return kwargs.get("default")

    fake_bpy.types = SimpleNamespace(Operator=_Operator)
    fake_bpy.props = SimpleNamespace(
        BoolProperty=_property,
        EnumProperty=_property,
        IntProperty=_property,
        StringProperty=_property,
    )
    fake_bpy.path = SimpleNamespace(abspath=lambda value: value)
    fake_bpy.app = SimpleNamespace(version=(5, 1, 0))
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "bpy.props", fake_bpy.props)


@pytest.fixture
def operators_module(monkeypatch):
    _install_bpy_stub(monkeypatch)
    module_name = "fluidSim_warp_blender51.operators"
    operator_domain_prefix = "fluidSim_warp_blender51.operator_domains"
    previous = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == module_name or name.startswith(operator_domain_prefix)
    }
    for name in previous:
        sys.modules.pop(name, None)
    try:
        yield importlib.import_module(module_name)
    finally:
        for name in tuple(sys.modules):
            if name == module_name or name.startswith(operator_domain_prefix):
                sys.modules.pop(name, None)
        sys.modules.update(previous)


@pytest.mark.parametrize(
    "baseline_text",
    [
        "{not-json",
        "null",
        '"benchmark"',
        json.dumps(_benchmark_record(timeStepAverage=None)),
        json.dumps(_benchmark_record(timeStepAverage=True)),
    ],
)
def test_benchmark_operator_cancels_with_actionable_malformed_baseline(
    monkeypatch, tmp_path, baseline_text, operators_module
):
    operators = operators_module
    from fluidSim_warp_blender51.diagnostics import benchmark as benchmark_module

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(baseline_text, encoding="utf-8")
    fs = SimpleNamespace(
        benchmark_frames=1,
        benchmark_baseline=str(baseline_path),
        status="",
    )
    session = SimpleNamespace(benchmark_result=None, benchmark_compare=None)
    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: session),
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_benchmark",
        lambda **_kwargs: SimpleNamespace(as_dict=lambda: _benchmark_record()),
    )

    messages = []
    operator = operators.FLUIDSIM_OT_run_benchmark()
    operator.save_baseline = False
    operator.report = lambda level, message: messages.append((level, message))

    result = operator.execute(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert result == {"CANCELLED"}
    assert fs.status.startswith("Benchmark baseline invalid:")
    assert messages[-1][0] == {"ERROR"}
    assert messages[-1][1] == fs.status
    assert session.benchmark_compare["result"] == "fail"
    assert session.benchmark_compare["error"] == fs.status


def test_benchmark_operator_preserves_valid_baseline_comparison(
    monkeypatch, tmp_path, operators_module
):
    operators = operators_module
    from fluidSim_warp_blender51.diagnostics import benchmark as benchmark_module

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_benchmark_record()), encoding="utf-8")
    fs = SimpleNamespace(
        benchmark_frames=1,
        benchmark_baseline=str(baseline_path),
        status="",
    )
    session = SimpleNamespace(benchmark_result=None, benchmark_compare=None)
    monkeypatch.setattr(
        operators._operator_runtime,
        "_bridge",
        lambda: SimpleNamespace(session=lambda: session),
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_benchmark",
        lambda **_kwargs: SimpleNamespace(as_dict=lambda: _benchmark_record()),
    )

    operator = operators.FLUIDSIM_OT_run_benchmark()
    operator.save_baseline = False
    operator.report = lambda *_args: None

    result = operator.execute(SimpleNamespace(scene=SimpleNamespace(fluidsim=fs)))

    assert result == {"FINISHED"}
    assert session.benchmark_compare["result"] == "pass"
    assert fs.status == "Benchmark completed"
