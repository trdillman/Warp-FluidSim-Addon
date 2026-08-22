# FluidSim Phase 2 handoff

## Release state

- Branch: `phase2/usable-first-run-workflow`
- Release: `1.0.247`
- Host target validated here: Blender 5.2.0 LTS, Linux x64
- External runtime validated here: NVIDIA Warp 1.16.0+cu12
- Available device: Warp CPU only; no usable NVIDIA CUDA driver/device

## Delivered contracts

### Phase 1 foundation reconstructed on the available 1.0.245 baseline

The handoff referenced commit `8be9535` and 1.0.246 artifacts, but none were
present in the supplied workspace or local Git objects. The cache-v2,
checkpoint, production-bake, cache-manager, and headless contracts were rebuilt
before Phase 2 work. The release report must preserve this provenance rather
than claim the unavailable commit was inspected.

### Phase 2 guided workflow

- Strict `PresetV1` allow-list schema and six built-in presets.
- Atomic extension-managed user preset persistence and migration seam.
- Selected-object setup modes: Quick Liquid, Quick Container, Quick Viscous,
  Add Collider, Add Inflow, and Add Outflow.
- Stable owned `FluidSim/Inputs/Outputs/Helpers` collection tree.
- Gravity-aware, outward voxel-aligned domain calculation.
- Idempotent surface/liquid-particle/whitewater output shells and materials.
- Capacity-preflight status without automatic solver playback.
- Four reproducibly generated and save/reopen-validated example scenes.

## Invariants to preserve

- The manifest is the only authority for a complete cache-v2 frame.
- Orphans are never playable or resumable.
- Checksum, size, schema, and compatibility failures fail closed.
- Resume begins only from the last contiguous verified checkpoint.
- Cancellation is observed only between frame transactions.
- Cache v1 remains playback-compatible and non-resumable.
- Presets may write only explicitly allow-listed settings.
- Guided setup never deletes unowned objects or silently starts simulation.
- Blender and Warp runtimes remain external to release artifacts.

## Remaining mandatory gate

Run the installed package with the supplied Warp 1.16.0+cu12 package on a
compatible NVIDIA GPU. Required evidence includes all registered GPU
regressions, uninterrupted-versus-resumed deterministic comparison, CUDA
corruption/resume rejection, memory admission/growth behavior, and a successful
headless two-frame production bake. Do not infer CUDA proof from CPU smoke.

## Next documented phase

Phase 3 is artist-ready rendering: versioned output-attribute schemas, motion
vectors, cache/checkpoint integration for persistent channels, Blender attribute
publication, and bundled Geometry Nodes/material assets. Preserve Phase 1/2
persistent schemas and treat new channels as opt-in.
