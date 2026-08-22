"""Blender ``--python`` shim for the installed FluidSim headless bake API."""

from __future__ import annotations

import importlib
import os
import sys

import bpy


def _installed_module() -> str:
    override = os.environ.get("FLUIDSIM_EXTENSION_MODULE", "").strip()
    if override:
        return override
    matches = [
        item.module
        for item in bpy.context.preferences.addons
        if item.module == "fluidsim_warp" or item.module.endswith(".fluidsim_warp")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one enabled FluidSim extension, found {matches}")
    return matches[0]


def main() -> int:
    module = importlib.import_module(f"{_installed_module()}.blender.command_line")
    return int(module.main())


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 15
    sys.stdout.flush()
    sys.stderr.flush()
    # Blender may catch SystemExit from --python and return zero. os._exit is
    # deliberate here so farm schedulers receive the documented process code.
    os._exit(exit_code)
