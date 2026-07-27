"""Verify every kernel in the synthesized manifest is registered in the port."""

import inspect
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

wp = pytest.importorskip("warp")

wp.init()

# Import every kernel module so registrations run.
from fluidSim_warp_blender51.sim import (  # noqa: F401
    fields, particles, kernels_core, kernels_fluid, pressure, viscosity,
    surface, whitewater, sourcing, solver,
)
from fluidSim_warp_blender51.sim.registry import KERNEL_REGISTRY, lookup

from fluidSim_synth.kernel_manifest import KERNELS

manifest_names = {k.canonical_name for k in KERNELS}
registered = set(KERNEL_REGISTRY.keys())

missing = sorted(manifest_names - registered)
extra = sorted(registered - manifest_names)

print(f"manifest kernels: {len(manifest_names)}")
print(f"registered names: {len(registered)}")
if missing:
    print(f"MISSING ({len(missing)}):")
    for name in missing:
        print("  ", name)
if extra:
    print(f"extra (port-internal, ok): {len(extra)}")

# Alias lookups must resolve.
assert lookup("paradigmAdvectParticles") is lookup("fluidSimAdvectParticles")

def _is_real_warp_entry(entry):
    if isinstance(entry.fn, wp.Kernel):
        return True
    if not inspect.isfunction(entry.fn) or entry.fn.__name__ == "<lambda>":
        return False
    try:
        source = inspect.getsource(entry.fn)
    except (OSError, TypeError):
        return False
    # kernels_core.py:94-104 intentionally wraps wp.utils device primitives
    # instead of exposing raw wp.Kernel objects for reduction aliases.
    return "wp." in source


invalid_entries = sorted(
    name for name, entry in KERNEL_REGISTRY.items()
    if not _is_real_warp_entry(entry)
)
assert not missing, f"{len(missing)} manifest kernels unregistered"
assert not invalid_entries, (
    "registry entries must be compiled Warp kernels or inspected Warp-device "
    f"orchestrators: {invalid_entries}"
)
print("KERNEL-COVERAGE-PASS")
