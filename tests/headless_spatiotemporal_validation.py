"""Headless ST-FLIP validation for Blender + Warp.

Run from an unpacked add-on tree:
    blender --background --python fluidSim_warp_blender51/tests/headless_spatiotemporal_validation.py -- --device cuda:0

Run from an installed add-on by keeping the package importable and executing
this file with Blender's Python.  The test uses real Warp kernels; it is not a
mock or pure-Python substitute.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _ensure_package_on_path() -> None:
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    package_dir = root / "fluidSim_warp_blender51"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main(argv: list[str] | None = None) -> int:
    _ensure_package_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args, _ = parser.parse_known_args(argv)

    from fluidSim_warp_blender51.sim import validation

    matches = [fn for fn in validation._TESTS
               if fn.validation_name.startswith("spatiotemporal")]
    if not matches:
        raise RuntimeError("No spatiotemporal validation tests registered")
    failures = 0
    for fn in matches:
        passed, detail = fn(args.device)
        failures += 0 if passed else 1
        print(f"{fn.validation_name}: {'PASS' if passed else 'FAIL'} - {detail}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
