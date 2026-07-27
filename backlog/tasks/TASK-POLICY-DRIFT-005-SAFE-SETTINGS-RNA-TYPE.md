# Repair safe-settings Blender RNA type boundary

## State

completed

## What to build

Make memory-safety recommendations compatible with the Blender property types used by the live add-on, especially the integer `particles_per_voxel` property, and verify the recovery button through Blender 5.1 rather than only Python stubs.

## Acceptance criteria

- [x] Fractional recommendation math is normalized to the destination property type before assignment.
- [x] `blender --background --factory-startup --python tests/blender_memory_safety_gate.py` passes through safe-settings application and retry initialization.
- [x] A regression test covers the Blender RNA assignment boundary and prevents float assignment to `particles_per_voxel`.
- [x] The 2026-07-22 `python tools/run_production_gate.py --include-gui` run reaches and passes the memory-safety lane.

## Evidence

- `POLICY_DRIFT_REPORT.md:103-108` records that the Blender RNA safe-settings boundary is resolved, with integer `particles_per_voxel` recovery and solver advancement verified.
- `POLICY_DRIFT_REPORT.md:30-39` records the 2026-07-22 Blender 5.1 / `cuda:0` production-lane verification, including `MEMORY-SAFETY-GATE-PASS`, `FULL-PRODUCTION-GATE-PASS`, `GUI-ACCEPTANCE-PASS (35/35)`, and `SAVED-DEMO-GUI-PLAYBACK-PASS`.

## Blocked by

None — completed; no prerequisite card was required.
