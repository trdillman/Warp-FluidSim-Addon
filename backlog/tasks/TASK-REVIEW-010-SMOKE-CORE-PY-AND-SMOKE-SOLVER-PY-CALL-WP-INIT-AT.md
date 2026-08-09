# TASK-REVIEW-010: smoke_core.py and smoke_solver.py call wp.init() at module scope -- will crash pytest collection without CUDA

## State

ready-for-agent

## Severity

high

## What to build

smoke_core.py lines 3-5: 'import warp as wp / wp.init() / dev = "cuda:0"'. smoke_solver.py lines 9-11: 'import warp as wp / wp.init()'. These execute at import time. When pytest discovers and imports these files during collection (which it will, since they are .py files in tests/), the import will raise if Warp is not installed or CUDA is not available, crashing the entire collection with a hard error rather than a skip. Neither file contains any 'def test_*' function.

**Location:** `tests/smoke_core.py:3`
**Dimension:** Test Coverage

## Impact

Running 'pytest tests/' on any CI machine or developer workstation without CUDA will fail at collection time due to these files, preventing any tests from running at all.

## Acceptance criteria

- [ ] Root cause at `tests/smoke_core.py:3` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Either move these files outside tests/ (e.g., to a scripts/ or smoke/ directory), or guard the module-level code inside 'if __name__ == "__main__":' blocks and add a pytest skip marker.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-004` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
