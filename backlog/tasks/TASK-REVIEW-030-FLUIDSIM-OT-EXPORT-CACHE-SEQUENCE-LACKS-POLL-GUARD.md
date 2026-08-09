# TASK-REVIEW-030: FLUIDSIM_OT_export_cache_sequence lacks poll() guard, button always enabled without active solver

## State

ready-for-agent

## Severity

medium

## What to build

Unlike FLUIDSIM_OT_export_frame (line 1448) which inherits _FLUIDSIM_OT_session_base (providing a poll that requires an active solver), FLUIDSIM_OT_export_cache_sequence (line 1483) inherits only bpy.types.Operator and defines no poll method. The 'Export Cache Sequence' button in the Output panel (panels.py line 1008) is therefore always enabled regardless of solver state. When clicked with no active solver, bridge.session().export_cache_sequence(scene, start, end) at line 1508 fails and is caught by the generic except at line 1509, producing an error report rather than the correct UX of a greyed-out button.

**Location:** `fluidSim_warp_blender51/operators.py:1483`
**Dimension:** UI/UX & Operators

## Impact

Users can click an export button that cannot succeed, receiving a generic error instead of a visually disabled control. This is misleading about what actions are available.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:1483` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Either inherit _FLUIDSIM_OT_session_base or add a poll classmethod that returns False when bridge.session().active is False.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-002` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
