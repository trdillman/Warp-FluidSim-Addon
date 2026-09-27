# TASK-REVIEW-050: FLUIDSIM_OT_step and FLUIDSIM_OT_release declare no bl_options, leaving fs.status changes outside the undo system

## State

wont-fix

## Severity

low

## What to build

FLUIDSIM_OT_step (line 486) and FLUIDSIM_OT_release (line 117) declare no bl_options at all. Both modify context.scene.fluidsim.status (step at line 513 via sess.error, release at line 125). While the primary effect of these operators is on external GPU solver state (which cannot be undone), the RNA property changes to fs.status are also not captured in any undo step. This means status text changes persist through undo operations, potentially showing stale status after an undo.

**Location:** `fluidSim_warp_blender51/operators.py:486`
**Dimension:** UI/UX & Operators

## Impact

Minor: status text may become stale relative to actual solver state after undo operations. Not a functional defect since GPU state cannot be undone regardless.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:486` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Consider adding bl_options = {"REGISTER"} to both operators so Blender tracks them in the operator stack, even if UNDO is intentionally omitted for solver lifecycle operations.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-006` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Closed as won't-fix for 1.1.0. `FLUIDSIM_OT_step` and `FLUIDSIM_OT_release` now live in `fluidSim_warp_blender51/operator_domains/session.py` and still declare no `bl_options`.

Both act on the external GPU solver session, which undo cannot reverse. The only RNA they write directly is transient status text (`release` sets `fluidsim.status`; `step` records failures on the session). An undo step here would only add noise.

`tests/test_operator_panel_behavior.py::test_undo_metadata_is_limited_to_mutating_operators` pins `release` as not undoable.

## Blocked by

None - can start immediately
