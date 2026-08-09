# TASK-REVIEW-009: 41 of 104 test files have no pytest-discoverable test functions -- entire acceptance/gate test suite is invisible to pytest

## State

ready-for-agent

## Severity

high

## What to build

Systematic analysis shows 41 of 104 test files in tests/ contain zero 'def test_*' functions. These are acceptance/gate/runner scripts designed for 'blender --python' invocation, not pytest. Critical files affected include: blender_memory_safety_gate.py (19 asserts, 0 test functions), blender_capacity_planner_acceptance.py (44 asserts, 0 test functions), blender_whitewater_truth_acceptance.py (18 asserts, 0 test functions), blender_mixed_source_batch_acceptance.py (14 asserts, 0 test functions), blender_gui_acceptance.py (647 lines, 0 test functions), blender_source_sampling_acceptance.py (13 asserts, 0 test functions). Five files (blender_animated_source_acceptance.py, blender_point_source_acceptance.py, blender_source_lifecycle_acceptance.py, blender_source_region_acceptance.py, blender_volume_source_acceptance.py) are 5-line stubs that just call main() from _phase2_source_acceptance_common at module scope.

**Location:** `tests/:1`
**Dimension:** Test Coverage

## Impact

A 'pytest tests/' run skips the entire acceptance/gate suite silently. Coverage reports will show these files as untested. Developers may believe they have 104 test files when pytest only discovers ~63. The memory safety gate, capacity planner acceptance, and whitewater truth acceptance tests are effectively invisible to standard Python test tooling.

## Acceptance criteria

- [ ] Root cause at `tests/:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: For acceptance tests that must run under Blender: either (a) add a pytest-collectible wrapper that skips when bpy is unavailable and delegates to the acceptance main(), or (b) document a separate test runner and exclude these from the pytest discovery path via pyproject.toml/markers. For the 5-line stub files: their module-scope main() calls execute on import, which is a side effect during pytest collection.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-003` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (High Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
