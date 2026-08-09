# TASK-REVIEW-027: No negative-path tests for solver initialization failure modes (invalid device, capacity overflow during live step, kernel launch failure)

## State

ready-for-agent

## Severity

medium

## What to build

The solver has explicit error handling at solver.py:600-630 (invalid device string -> RuntimeError with actionable message, non-CUDA device -> RuntimeError). The pressure solver has launch failure handling at cuda_reduction.py:133 (result != 0 -> RuntimeError). The sourcing module has invariant violation detection at sourcing.py (admitted/emitted count mismatch -> RuntimeError). None of these error paths are tested. While memory_safety_recovery.py tests the classification of error strings (is_memory_capacity_error) and capacity_planner tests capacity BLOCKED verdicts, there are no tests that trigger actual solver initialization failures, live capacity overflow during step_frame(), or kernel launch failures.

**Location:** `tests/test_memory_safety_recovery.py:105`
**Dimension:** Test Coverage

## Impact

The fail-closed error paths in the solver -- which are the last line of defense against silent data corruption -- are untested. A regression in the device validation or capacity guard could go undetected.

## Acceptance criteria

- [ ] Root cause at `tests/test_memory_safety_recovery.py:105` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add tests for: FluidSolver.__init__ with invalid device string (expect RuntimeError), FluidSolver.__init__ with non-CUDA device (expect RuntimeError), capacity overflow during step_frame() (expect RuntimeError or safe shutdown), and the sourcing invariant violation path.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-009` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
