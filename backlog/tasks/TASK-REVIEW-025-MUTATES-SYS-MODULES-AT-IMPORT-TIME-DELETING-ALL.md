# TASK-REVIEW-025: Mutates sys.modules at import time, deleting all fluidSim_warp_blender51 modules

## State

ready-for-agent

## Severity

medium

## What to build

Lines 14-16: 'for _name in list(sys.modules): / if _name == "fluidSim_warp_blender51" or _name.startswith("fluidSim_warp_blender51."): / del sys.modules[_name]'. This runs at module import time. Any test file that imports from _phase2_source_acceptance_common (the 5 stub acceptance files: blender_animated_source_acceptance.py, blender_point_source_acceptance.py, blender_source_lifecycle_acceptance.py, blender_source_region_acceptance.py, blender_volume_source_acceptance.py) triggers this purge. If pytest collects these in the same process as other tests, the module purge can cause other tests to see stale or missing module references.

**Location:** `tests/_phase2_source_acceptance_common.py:14`
**Dimension:** Test Coverage

## Impact

Test-order-dependent failures in any pytest session that collects both the phase2 stubs and other tests importing fluidSim_warp_blender51 modules. The purge-and-reimport pattern assumes single-process isolation that pytest does not guarantee.

## Acceptance criteria

- [ ] Root cause at `tests/_phase2_source_acceptance_common.py:14` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Move the sys.modules purge inside the main() function so it only runs when the acceptance script is explicitly invoked, not during import.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TEST-007` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
