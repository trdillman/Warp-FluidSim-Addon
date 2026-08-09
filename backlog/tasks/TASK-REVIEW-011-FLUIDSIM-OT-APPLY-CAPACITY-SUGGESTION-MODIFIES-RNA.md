# TASK-REVIEW-011: FLUIDSIM_OT_apply_capacity_suggestion modifies RNA properties and domain transforms but lacks UNDO in bl_options

## State

ready-for-agent

## Severity

high

## What to build

The operator declares bl_options = {"REGISTER"} (line 191) without "UNDO". Its execute method modifies multiple scene RNA properties via setattr(fs, name, value) (line 305) and modifies the domain object's location and dimensions via _apply_domain_bounds (lines 180-181: domain.location = ..., domain.dimensions = ...). These are standard Blender undoable data types. Without UNDO, Blender does not push an undo step for this operator, so pressing Ctrl+Z after approving capacity changes undoes whatever step preceded the approval, leaving stale fs.* settings and domain transforms that no longer match the running solver. The operator has explicit rollback logic (lines 316-319) for failures, confirming these are recognized as significant mutations.

**Location:** `fluidSim_warp_blender51/operators.py:191`
**Dimension:** UI/UX & Operators

## Impact

Users who approve capacity changes cannot revert them through Blender's undo system. They must manually undo each property change and domain resize, risking a desync between scene settings and the live solver state.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:191` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add "UNDO" to bl_options: bl_options = {"REGISTER", "UNDO"}.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-001` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
