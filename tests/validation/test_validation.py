"""Headless CUDA validation suite runner (no Blender required).

Runs all synthesized validation targets on the real Warp/CUDA device and
exits non-zero on any failure. Under pytest the same suite runs as one
``slow`` test on the shared ``cuda_device`` fixture.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from _subset_runner import script_device


@pytest.mark.slow
def test_validation_suite_passes(cuda_device):
    from fluidSim_warp_blender51.diagnostics import validation

    results = validation.run_all(device=cuda_device)
    assert results, "validation suite ran no targets"
    failed = [f"{r.name}: {r.detail}" for r in results if not r.passed]
    assert not failed, "validation failures:\n" + "\n".join(failed)


def main():
    import warp as wp

    wp.init()
    device = script_device()
    if device.startswith("cuda") and not wp.is_cuda_available():
        print("FATAL: CUDA not available")
        return 2

    from fluidSim_warp_blender51.diagnostics import validation

    print(f"device: {wp.get_device(device).name}")
    print(f"running {len(validation.VALIDATION_TESTS)} validation tests\n")

    failures = 0

    def progress(result):
        nonlocal failures
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name} ({result.elapsed:.2f}s)")
        print(f"       {result.detail}")
        if not result.passed:
            failures += 1

    results = validation.run_all(device=device, progress=progress)

    print(f"\n{len(results) - failures}/{len(results)} tests passed")
    if failures == 0:
        print("VALIDATION-PASS")
        return 0
    print("VALIDATION-FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
