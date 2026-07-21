# Repair safe-settings Blender RNA type boundary

## State

completed

## What to build

Make memory-safety recommendations compatible with the Blender property types used by the live add-on, especially the integer `particles_per_voxel` property, and verify the recovery button through Blender 5.1 rather than only Python stubs.

## Acceptance criteria

- [x] Fractional recommendation math is normalized to the destination property type before assignment.
- [x] `blender --background --factory-startup --python tests/blender_memory_safety_gate.py` passes through safe-settings application and retry initialization.
- [x] A regression test covers the Blender RNA assignment boundary and prevents float assignment to `particles_per_voxel`.
- [x] `python tools/run_production_gate.py --include-gui` reaches and passes the memory-safety lane.

## Evidence

- `fluidSim_warp_blender51/operators.py` normalizes values for destination RNA `INT` properties at the direct assignment seam.
- `blender --background --factory-startup --python tests/blender_memory_safety_gate.py` emitted `MEMORY-SAFETY-GATE-PASS` on Blender 5.1 / `cuda:0`.
- `python tools/run_production_gate.py --include-gui` started at `2026-07-21T20:00:16.660750+00:00`, exited `0`, and emitted `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (35/35)`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`.

## Blocked by

None — completed; no prerequisite card was required.
