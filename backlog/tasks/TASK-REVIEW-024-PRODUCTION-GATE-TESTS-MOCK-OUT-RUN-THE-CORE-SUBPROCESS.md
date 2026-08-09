# TASK-REVIEW-024: Production gate tests mock out _run (the core subprocess executor) and all guard functions -- only verifies command assembly, never actual execution

## State

ready-for-agent

## Severity

medium

## What to build

test_memory_safety_recovery.py lines 136-146 read operators.py and panels.py as plain text strings and assert substrings are present: 'assert "bl_idname = \"fluidsim.apply_safe_settings\"" in operators', 'assert "is_memory_capacity_error" in panels'. Line 146 reads run_production_gate.py as text and asserts a filename reference exists. The test itself acknowledges the limitation at line 142-143: 'Limitation: this source-text check proves a live-looking reference, not that the production gate actually executes Blender successfully.' The same pattern appears in tests/test_surface_resolution_feature_contracts.py lines 41 and 130.

**Location:** `tests/test_memory_safety_recovery.py:136`
**Dimension:** Test Coverage

## Impact

These tests pass even if the operator is never registered, the function is dead code, or the reference is in a comment. They provide false confidence that the memory safety recovery UI is wired up correctly.

## Acceptance criteria

- [ ] Root cause at `tests/test_memory_safety_recovery.py:136` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Replace with import-based checks: import the module and verify the class/function exists as an attribute, verify bpy.types registration via mock. For gate references, verify the string appears in an executable context (not a comment) or better, run the gate with a mock Blender.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-006` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
