"""Grader controls only; native CUDA broker execution is a separate gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[1] / 'acceptance' / 'blender_fspt_viewport_backend_acceptance.py'
    spec = importlib.util.spec_from_file_location('fspt_broker_verifier_controls', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example():
    pixels = np.ones((3, 4, 4), np.float32)
    pixels[0, :, :3] = (0.1, 0.2, 0.3)
    frame = SimpleNamespace(device='cuda:0', key=('captured', 7), samples=1, pixels=pixels)
    return frame, pixels.copy()


def test_native_grader_accepts_exact_source_bound_nontrivial_image(verifier):
    frame, expected = example()
    verifier.check_frame(frame, expected, 'cuda:0', ('captured', 7))


@pytest.mark.parametrize('fault', ['stale', 'device', 'samples', 'constant', 'nonfinite', 'shape', 'missing'])
def test_native_grader_rejects_reached_bad_result(verifier, fault):
    frame, expected = example()
    if fault == 'stale':
        frame.key = ('captured', 6)
    elif fault == 'device':
        frame.device = 'cpu'
    elif fault == 'samples':
        frame.samples = 2
    elif fault == 'constant':
        frame.pixels[:] = frame.pixels[0, 0]
    elif fault == 'nonfinite':
        frame.pixels[0, 0, 0] = float('nan')
    elif fault == 'shape':
        frame.pixels = frame.pixels[:1]
    else:
        frame = None
    with pytest.raises(AssertionError):
        verifier.check_frame(frame, expected, 'cuda:0', ('captured', 7))


def test_native_grader_rejects_a_trivial_matching_oracle(verifier):
    frame, _ = example()
    frame.pixels[:] = (0.1, 0.2, 0.3, 1.0)
    with pytest.raises(AssertionError, match='oracle'):
        verifier.check_frame(frame, frame.pixels.copy(), 'cuda:0', ('captured', 7))


def test_native_deadline_does_not_turn_missing_completion_into_success(verifier):
    with pytest.raises(TimeoutError, match='no completion'):
        verifier.until(lambda: None, lambda value: value is not None, 'no completion', seconds=0)
