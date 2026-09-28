"""Shared runner for focused validation-subset test modules.

Each subset module is collected by pytest (one case per validation target on
the shared ``cuda_device`` fixture) and remains runnable as a gate script.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _test_device import script_device  # noqa: E402,F401 - re-exported for scripts


def validation_id(index):
    """Stable pytest id for a ``sim.validation`` target index."""
    return f"v{index + 1:02d}"


def check_validation(index, device):
    """Run one ``sim.validation`` target and assert it passed."""
    from fluidSim_warp_blender51.sim import validation

    fn = validation._TESTS[index]
    passed, detail = fn(device)
    assert passed, f"{fn.validation_name}: {detail}"


def run_subset(indices, label):
    import warp as wp

    wp.init()
    device = script_device()
    if device.startswith("cuda") and not wp.is_cuda_available():
        print("FATAL: CUDA not available")
        return 2

    from fluidSim_warp_blender51.sim import validation

    tests = [validation._TESTS[i] for i in indices]
    failures = 0
    for fn in tests:
        t0 = time.perf_counter()
        try:
            passed, detail = fn(device)
        except Exception as exc:  # noqa: BLE001
            passed, detail = False, f"exception: {exc!r}"
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {fn.validation_name} ({time.perf_counter() - t0:.2f}s)")
        print(f"       {detail}")
        if not passed:
            failures += 1

    if failures == 0:
        print(f"{label}-PASS")
        return 0
    print(f"{label}-FAIL")
    return 1
