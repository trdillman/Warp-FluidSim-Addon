"""Every GPU self-validation must return its native CUDA kernel leases.

A leaked lease makes ``reset_cuda_module_cache`` refuse to run, so disabling
the add-on after Diagnostics > Run Validations logged
``unregister step 'native CUDA cache' failed ... 1 active lease(s)``.
"""

import pytest

from fluidSim_warp_blender51.sim import cuda_reduction
from fluidSim_warp_blender51.sim import validation


def _active_leases():
    cache = cuda_reduction._MODULE_CACHE
    with cache.lock:
        return sum(entry.references for entry in cache.entries.values())


@pytest.mark.parametrize(
    "check", validation._TESTS, ids=lambda fn: fn.__name__
)
def test_validation_returns_every_native_lease(cuda_device, check):
    before = _active_leases()
    check(cuda_device)
    assert _active_leases() == before, check.validation_name


def test_module_cache_resets_after_all_validations(cuda_device):
    results = validation.run_all(device=cuda_device)

    assert all(result.passed for result in results), [
        result.name for result in results if not result.passed
    ]
    cuda_reduction.reset_cuda_module_cache()
