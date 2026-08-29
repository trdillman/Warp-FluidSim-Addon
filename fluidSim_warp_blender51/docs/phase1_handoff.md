# Phase 1 Handoff

## P0 baseline

Phase 1 starts from FluidSim Warp **1.0.245** with the following proven release foundation:

- Blender **5.2.0 LTS**, Linux x64.
- External NVIDIA Warp **1.16.0+cu12**; no Blender or Warp runtime packages are included in FluidSim ZIPs.
- Extension ID `fluidsim_warp`; GPL-3.0-or-later.
- No runtime network, `pip`, subprocess dependency probe, or automatic dependency installation.
- Clean-profile source/archive validation, build, install, enable, disable, re-enable, fresh-process reload, Warp CPU-kernel execution, support-report generation, and uninstall gates.
- GPU actions remain CUDA-only and are rejected before allocation when no compatible device is available.
- Cache-v1 remains readable and non-resumable. P0 does not alter solver numerics or cache payloads.

## Required Phase 1 scope

Implement Tasks 4–6 from `FluidSim_High_ROI_Implementation_Plan.md`:

1. Cache-v2 manifest and complete solver checkpoint contract.
2. Atomic production bake state machine, cancellation, integrity checks, and resume.
3. Background/headless bake CLI with documented exit codes and machine-readable progress.

## Integration boundaries

- Keep `sim/` free of `bpy` imports.
- Add cache schema/checkpoint ownership under `io/`; do not turn `blender/scene_bridge.py` into a cache implementation.
- Preserve cache-v1 playback while writing new production bakes in cache v2.
- A resumable frame is committed only after frame output, checkpoint, checksums, and the atomic manifest entry are complete.
- Continue using `version.py`, `runtime/warp_runtime.py`, and the existing release scripts as the canonical P0 contracts.

## Validation constraints

The sandbox used for P0 exposed no NVIDIA CUDA driver. A separately provisioned Warp binary imported and executed a real CPU kernel inside Blender, while the CUDA preflight correctly reported no device. Before a public release, run `../tools/release/gpu_regression_blender.py` on a self-hosted NVIDIA runner and attach its JSON report.

Windows packaging is not part of this baseline. Add it only with a checksum-locked Warp 1.16.0+cu12 Windows wheel and a clean Blender 5.2 Windows install/smoke run.
