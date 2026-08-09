# TASK-REVIEW-049: Snippet line-edit operators silently return CANCELLED without user feedback when no active snippet exists

## State

ready-for-agent

## Severity

low

## What to build

FLUIDSIM_OT_snippet_edit_line.invoke (line 1384) returns {"CANCELLED"} when _active_snippet(context) returns None. FLUIDSIM_OT_snippet_edit_line.execute (line 1393), FLUIDSIM_OT_snippet_line_add.execute (line 1420), and FLUIDSIM_OT_snippet_line_delete.execute (line 1438) also return {"CANCELLED"} without calling self.report(). No error or warning message is shown to the user. While the Snippet Editor panel (panels.py lines 1093-1094) does show 'Add or load a snippet first' text, the operators themselves fail silently if invoked through search menu or API.

**Location:** `fluidSim_warp_blender51/operators.py:1384`
**Dimension:** UI/UX & Operators

## Impact

Users invoking these operators outside the panel context get no feedback about why nothing happened.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:1384` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add self.report({'WARNING'}, 'No active snippet') before each bare return {"CANCELLED"}.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `UIUX-005` in `reviews/swarm-2026-08-08/findings/ui.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the UI/UX & Operators lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
