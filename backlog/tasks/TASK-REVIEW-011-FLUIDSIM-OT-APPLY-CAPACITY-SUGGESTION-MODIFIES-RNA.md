# TASK-REVIEW-011: FLUIDSIM_OT_apply_capacity_suggestion modifies RNA properties and domain transforms but lacks UNDO in bl_options

## State

completed

## Severity

high

## What to build

The operator declares bl_options = {"REGISTER"} (line 191) without "UNDO". Its execute method modifies multiple scene RNA properties via setattr(fs, name, value) (line 305) and modifies the domain object's location and dimensions via _apply_domain_bounds (lines 180-181: domain.location = ..., domain.dimensions = ...). These are standard Blender undoable data types. Without UNDO, Blender does not push an undo step for this operator, so pressing Ctrl+Z after approving capacity changes undoes whatever step preceded the approval, leaving stale fs.* settings and domain transforms that no longer match the running solver. The operator has explicit rollback logic (lines 316-319) for failures, confirming these are recognized as significant mutations.

**Location:** `fluidSim_warp_blender51/operators.py:191`
**Dimension:** UI/UX & Operators

## Impact

Users who approve capacity changes cannot revert them through Blender's undo system. They must manually undo each property change and domain resize, risking a desync between scene settings and the live solver state.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/operators.py:191` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Add "UNDO" to bl_options: bl_options = {"REGISTER", "UNDO"}.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-001` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed for 2.0 (2026-09-27). `FLUIDSIM_OT_apply_capacity_suggestion` (now in `fluidSim_warp_blender51/operator_domains/session.py`) declares `bl_options = {"REGISTER", "UNDO"}`.

Why this differs from the TASK-REVIEW-032 won't-fix precedent: this operator writes RNA domain settings and the domain object's transform. Blender's memfile undo already reverts those writes even without an undo step, but merged into whatever step came before the approval. With its own step, Ctrl+Z reverts exactly this approval. The reinit signature in `blender/pending_settings.py` (voxel_size, capacities, domain bounds) then flags that the live solver needs reinitialising, so the solver cannot silently desync. Flagged for owner review.

Test: `tests/test_operator_panel_behavior.py::test_undo_metadata_is_limited_to_mutating_operators` now lists this operator in the undoable set.

Also fixed while here: the operator split `b77f4b9` had re-encoded `session.py`, so the capacity dialog and every status line in it showed mojibake (`â†’` for `→`, `â€”` for `—`). The text is corrected, and `tests/test_source_encoding.py` guards all add-on source against it.

## Blocked by

None - can start immediately
