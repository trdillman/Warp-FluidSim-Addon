"""Shared runner for focused validation-subset test scripts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_subset(indices, label):
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        print("FATAL: CUDA not available")
        return 2

    from fluidSim_warp_blender51.sim import validation

    tests = [validation._TESTS[i] for i in indices]
    failures = 0
    for fn in tests:
        import time

        t0 = time.perf_counter()
        try:
            passed, detail = fn("cuda:0")
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
