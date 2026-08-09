# TASK-REVIEW-026: No conftest.py anywhere -- cuda_device fixture is defined locally in a single file, no shared fixtures or test isolation configuration

## State

ready-for-agent

## Severity

medium

## What to build

No conftest.py exists anywhere in the tests/ directory tree (confirmed: find returned 'no conftest found'). The cuda_device fixture is defined only in tests/test_pressure_projection_regressions.py line 112 as a module-scoped fixture. Every other CUDA test file hardcodes 'cuda:0' and calls wp.init() independently with no shared fixture. There is no shared setUp/teardown for Warp device initialization, no test isolation markers, and no pytest configuration file (no pytest.ini, no pyproject.toml [tool.pytest.ini_options]).

**Location:** `tests/:1`
**Dimension:** Test Coverage

## Impact

Device initialization is duplicated and inconsistent across files (some use wp.init() at module scope, some inside functions, some via pytest.importorskip). Test isolation is unmanaged. Adding a second CUDA device or testing multi-GPU behavior requires touching every file.

## Acceptance criteria

- [ ] Root cause at `tests/:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Create a tests/conftest.py with a session-scoped cuda_device fixture, standardize Warp initialization, and add pytest markers for CUDA-required vs CPU-only tests.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-008` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
