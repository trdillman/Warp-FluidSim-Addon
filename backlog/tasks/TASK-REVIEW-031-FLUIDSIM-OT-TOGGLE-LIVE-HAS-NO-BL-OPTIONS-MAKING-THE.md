# TASK-REVIEW-031: FLUIDSIM_OT_toggle_live has no bl_options, making the live-mode toggle non-undoable

## State

completed

## Severity

medium

## What to build

FLUIDSIM_OT_toggle_live (line 518) declares no bl_options at all. Its execute method sets fs.live = True/False and fs.status (lines 537-541). Because no undo step is pushed, pressing Ctrl+Z after toggling live on will undo whatever step preceded the toggle while fs.live remains True. Additionally, the operator lacks a poll() guard but is designed to self-initialize the solver (lines 530-536). The Simulation panel greys out the toggle via row.enabled = sess.active (line 216), hiding functionality the operator actually supports when invoked directly.

**Location:** `fluidSim_warp_blender51/operators.py:518`
**Dimension:** UI/UX & Operators

## Impact

The live simulation toggle is non-undoable, so users cannot revert it through standard Blender undo. The greyed-out UI also misrepresents the operator's capabilities since it can initialize the solver on its own.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:518` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add bl_options = {"REGISTER", "UNDO"}. Consider whether the panel enable logic at line 216 should be relaxed to reflect the operator's self-initialization capability.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-003` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Closed as won't-fix for 2.0 (2026-09-27), following the REVIEW-032 and REVIEW-050 precedent. `FLUIDSIM_OT_toggle_live` now lives in `fluidSim_warp_blender51/operator_domains/session.py` and still declares no `bl_options`. Turning Live on can initialize the GPU solver, which Blender undo cannot reverse, so an undo step would flip `fluidsim.live` back while the solver stays initialized. The toggle itself is the supported way back. The operator now has a `poll()` (it refuses while a Production Bake owns the scene), which covers the second half of the finding.
