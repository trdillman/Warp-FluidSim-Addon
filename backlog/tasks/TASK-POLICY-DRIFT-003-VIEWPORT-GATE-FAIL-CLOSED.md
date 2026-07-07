# TASK-POLICY-DRIFT-003: Make viewport gate fail closed

## State

completed

## What to build

Make the Blender viewport two-frame validation lane and aggregate production gate fail closed when the viewport script throws an unexpected Python exception, even if Blender returns process exit code 0.

## Acceptance criteria

- [x] `POLICY_DRIFT_REPORT.md` is changed to `in-progress` before project files are edited.
- [x] `tests/blender_viewport_2frame_gate.py` catches unexpected exceptions, prints a clear `VIEWPORT-2FRAME-GATE-FAIL` marker, and exits nonzero or otherwise makes the failure machine-detectable.
- [x] `tools/run_production_gate.py` rejects traceback or gate-fail markers from Blender subprocess output even when the Blender process exits 0.
- [x] A focused regression proves a simulated or real viewport-gate exception cannot be reported as `PRODUCTION-GATE-PASS`.
- [x] Direct `blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py` and `python tools/run_production_gate.py` pass after the runtime/import issue is fixed.
- [x] `POLICY_DRIFT_REPORT.md` is updated to `completed` only after verified repair.

## Blocked by

None - can start immediately
