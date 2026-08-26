"""Focused regressions for bpy/Warp-independent add-on authorities."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluidSim_warp_blender51.diagnostics.benchmark import compare_benchmark
from fluidSim_warp_blender51.io.export import export_points_ply
from fluidSim_warp_blender51.sim.quality_contract import (
    AuthoredQualitySettings,
    SourceSamplingMode,
    resolve_quality,
    resolve_source_sampling,
)
from fluidSim_warp_blender51.sim.registry import (
    KERNEL_REGISTRY,
    SOURCE_ALIASES,
    register,
)


def _benchmark_record(**changes):
    record = {
        "activeVoxelsAverage": 0,
        "activeVoxelsMinimum": 0,
        "activeVoxelsMaximum": 0,
        "timeStepAverage": 0.0,
        "voxelsPerSecondAverage": 0.0,
    }
    record.update(changes)
    return record


def test_benchmark_zero_baseline_does_not_hide_nonzero_regression():
    report = compare_benchmark(
        _benchmark_record(timeStepAverage=1.0), _benchmark_record())

    assert report["result"] == "performance_regression"
    assert report["performance"]["timeStepAverage"]["relative_delta"] == float("inf")


def test_benchmark_missing_contract_field_fails_comparison():
    current = _benchmark_record()
    del current["activeVoxelsMaximum"]

    assert compare_benchmark(current, _benchmark_record())["result"] == "fail"


def test_point_export_rejects_malformed_or_misaligned_arrays_before_write(tmp_path):
    target = tmp_path / "points.ply"
    with pytest.raises(ValueError, match="positions must have shape"):
        export_points_ply(np.zeros((3,), np.float32), str(target))
    assert not target.exists()

    with pytest.raises(ValueError, match="velocities must match"):
        export_points_ply(
            np.zeros((2, 3), np.float32), str(target),
            velocities=np.zeros((1, 3), np.float32))
    assert not target.exists()


@pytest.fixture
def kernel_registry_isolation():
    before_registry = dict(KERNEL_REGISTRY)
    before_aliases = dict(SOURCE_ALIASES)
    try:
        yield
    finally:
        KERNEL_REGISTRY.clear()
        KERNEL_REGISTRY.update(before_registry)
        SOURCE_ALIASES.clear()
        SOURCE_ALIASES.update(before_aliases)
    assert KERNEL_REGISTRY == before_registry
    assert SOURCE_ALIASES == before_aliases


def test_duplicate_kernel_identity_is_rejected_without_overwrite(
        kernel_registry_isolation):
    name = "fluidSimRegressionDuplicateIdentity"

    @register(name)
    def first():
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register(name)
        def second():
            pass
    assert KERNEL_REGISTRY[name].fn is first


def test_multi_name_registration_failure_is_atomic(kernel_registry_isolation):
    existing = "fluidSimRegressionAtomicExisting"
    candidate = "fluidSimRegressionAtomicCandidate"
    candidate_alias = "paradigmRegressionAtomicCandidate"

    @register(existing)
    def first():
        pass

    with pytest.raises(ValueError, match="already registered"):
        @register(candidate, existing)
        def second():
            pass

    assert candidate not in KERNEL_REGISTRY
    assert candidate_alias not in SOURCE_ALIASES
    assert KERNEL_REGISTRY[existing].fn is first


def test_duplicate_name_within_registration_is_atomic(kernel_registry_isolation):
    name = "fluidSimRegressionAtomicRepeated"
    alias = "paradigmRegressionAtomicRepeated"

    with pytest.raises(ValueError, match="more than once"):
        @register(name, name)
        def repeated():
            pass

    assert name not in KERNEL_REGISTRY
    assert alias not in SOURCE_ALIASES


@pytest.mark.parametrize(
    ("mode", "field", "message", "expected"),
    [
        (SourceSamplingMode.MULTIPLIER, "multiplier",
         "invalid sampling multiplier", 8.0),
        (SourceSamplingMode.LEGACY_ABSOLUTE, "legacy_absolute",
         "invalid legacy source sampling", 0.0),
    ],
)
def test_source_sampling_recovers_from_non_numeric_authored_values(
        mode, field, message, expected):
    quality = resolve_quality(
        AuthoredQualitySettings(voxel_size=0.1), ((0, 0, 0), (1, 1, 1)))
    authored = SimpleNamespace(
        mode=mode.value, multiplier=1.0, legacy_absolute=8.0)
    setattr(authored, field, "not-a-number")

    result = resolve_source_sampling(quality, authored)

    assert result.value == expected
    assert any(message in diagnostic for diagnostic in result.diagnostics)
