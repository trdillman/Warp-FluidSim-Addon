"""Verify every kernel in the synthesized manifest is registered in the port."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import warp as wp

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

assert not missing, f"{len(missing)} manifest kernels unregistered"
print("KERNEL-COVERAGE-PASS")
