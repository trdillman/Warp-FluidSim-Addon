# TASK-REVIEW-008: Assertions run at module import time, not inside test functions -- will fail during pytest collection not as test results

## State

ready-for-agent

## Severity

high

## What to build

test_kernel_coverage.py has ZERO 'def test_*' functions (confirmed: grep returns no matches). The assertions at lines 41 ('assert lookup("paradigmAdvectParticles") is lookup("fluidSimAdvectParticles")'), 61 ('assert not missing'), and 62 ('assert not invalid_entries') execute at module-level during import. If pytest imports this file during collection, an assertion failure surfaces as a collection error (not a test failure), which many CI systems report differently. Additionally, wp.init() at line 14 runs at module scope, so importing this file without Warp installed crashes the entire pytest collection.

**Location:** `tests/test_kernel_coverage.py:41`
**Dimension:** Test Coverage

## Impact

Kernel coverage failures may be misreported or missed entirely depending on CI pytest configuration. The wp.init() at line 14 makes the file a collection-time landmine on any machine without Warp/CUDA.

## Acceptance criteria

- [ ] Root cause at `tests/test_kernel_coverage.py:41` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Wrap all assertions inside a 'def test_kernel_registry_coverage():' function. Move wp.init() and the imports inside the test function body (or use pytest.importorskip at module level, consistent with test_pressure_projection_regressions.py line 39).
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-002` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
