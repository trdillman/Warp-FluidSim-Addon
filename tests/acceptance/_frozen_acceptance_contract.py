"""Host-testable assertions; no new numerical acceptance tolerance is invented."""

from __future__ import annotations

import numpy as np


def require_cuda(device):
    if not getattr(device, "is_cuda", False):
        raise RuntimeError("CUDA_REQUIRED: production continuation cannot run on CPU")


def qualification_status():
    return "pending_compiled_emitter_numerical_policy"


def assert_exact_state(expected, actual):
    """Checkpoint import and frozen snapshots have an exact round-trip contract."""
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_exact_state(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            assert_exact_state(left, right)
    elif isinstance(expected, np.ndarray):
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected, actual)
    else:
        assert expected == actual


def array_differences(expected, actual):
    """Retain raw-order differences, including shape/count drift; never sort away errors."""
    assert expected.keys() == actual.keys(), "array inventory changed"
    report = {}
    for key, left in expected.items():
        left, right = np.asarray(left), np.asarray(actual[key])
        numeric = left.dtype.kind in "biufc" and right.dtype.kind in "biufc"
        if numeric:
            assert np.isfinite(left).all() and np.isfinite(right).all(), key
        same_shape = left.shape == right.shape
        item = {
            "shape_matches": same_shape,
            "expected_shape": list(left.shape),
            "actual_shape": list(right.shape),
            "expected_dtype": str(left.dtype),
            "actual_dtype": str(right.dtype),
            "exact": bool(np.array_equal(left, right)),
        }
        if numeric and same_shape:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            item["max_abs"] = float(delta.max(initial=0))
            item["rms"] = float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0
        report[key] = item
    return report
