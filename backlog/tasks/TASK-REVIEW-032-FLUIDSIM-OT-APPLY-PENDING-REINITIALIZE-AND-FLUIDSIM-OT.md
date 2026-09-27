# TASK-REVIEW-032: FLUIDSIM_OT_apply_pending_reinitialize and FLUIDSIM_OT_apply_pending_reset declare REGISTER without UNDO despite modifying scene settings

## State

wont-fix

## Severity

medium

## What to build

FLUIDSIM_OT_apply_pending_reinitialize (line 395, bl_options = {"REGISTER"}) calls _bridge().apply_pending_reinitialize(context.scene) at line 410, which modifies scene-level RNA properties. FLUIDSIM_OT_apply_pending_reset (line 426, bl_options = {"REGISTER"}) similarly calls _bridge().apply_pending_reset(context.scene) at line 449. Neither includes "UNDO" in bl_options. The RNA property changes made by these bridge methods are not captured in an undo step, so they cannot be reverted through Blender's undo system. Compare with FLUIDSIM_OT_revert_pending (line 465) which correctly declares {"REGISTER", "UNDO"} for its RNA modifications.

**Location:** `fluidSim_warp_blender51/operators.py:395`
**Dimension:** UI/UX & Operators

## Impact

Restart/reset-tier pending settings applied through these operators are non-undoable. Users must manually revert each changed setting if they want to roll back.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:395` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add "UNDO" to bl_options for both operators.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-004` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Closed as won't-fix for 1.1.0. `FLUIDSIM_OT_apply_pending_reinitialize` and `FLUIDSIM_OT_apply_pending_reset` now live in `fluidSim_warp_blender51/operator_domains/session.py` and still declare `bl_options = {"REGISTER"}`.

These operators reinitialize or reset the GPU solver. Blender undo cannot restore GPU solver state, so adding `UNDO` would roll the RNA settings back while the solver keeps the applied ones, leaving the two out of sync. `FLUIDSIM_OT_revert_pending` (same file, `{"REGISTER", "UNDO"}`) is the supported way back.

## Blocked by

None - can start immediately
