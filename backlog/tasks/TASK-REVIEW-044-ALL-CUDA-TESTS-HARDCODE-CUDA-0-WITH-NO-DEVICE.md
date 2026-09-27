# TASK-REVIEW-044: All CUDA tests hardcode 'cuda:0' with no device parameterization or fallback

## State

ready-for-agent

## Severity

low

## What to build

At least 20 test files hardcode the string 'cuda:0' (smoke_core.py:5, smoke_solver.py:11, _subset_runner.py:26, _phase2_source_acceptance_common.py:53/87/146/564, blender_capacity_planner_acceptance.py:48/70/72, blender_memory_safety_gate.py:33, blender_mixed_source_batch_acceptance.py:43/65/66, blender_topology_transaction_acceptance.py:20-21, blender_whitewater_truth_acceptance.py:165-168, headless_spatiotemporal_validation.py:29, headless_stflip_benchmark.py:217, headless_surface_resolution_pipeline.py:142, test_compiled_source_operations.py:38/56/80, test_device_preference_routing.py:62/66). Only headless_stflip_benchmark.py, headless_spatiotemporal_validation.py, and headless_surface_resolution_pipeline.py accept --device as a CLI argument (but default to cuda:0).

**Location:** `tests/:1`
**Dimension:** Test Coverage

## Impact

Tests cannot run on multi-GPU systems targeting a specific device, and cannot adapt to environments where cuda:0 is unavailable but cuda:1 is. This limits CI flexibility and hardware test coverage.

## Acceptance criteria

- [ ] Root cause at `tests/:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Use a shared fixture or environment variable (e.g., FLUIDSIM_TEST_DEVICE) that defaults to cuda:0 but can be overridden. Parameterize the device in tests that use it.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Progress

Partially addressed in `16e5090` on `claude/2.0-phase1-testinfra`: pytest
tests now take the shared `cuda_device` fixture, which honours
`FLUIDSIM_TEST_DEVICE` (default `cuda:0`), including
`test_compiled_source_operations.py` and `_subset_runner.py`; guarded by
`tests/test_cuda_device_fixture.py`.

Still open: the Blender/headless scripts listed above (`smoke_*`,
`_phase2_source_acceptance_common.py`, `blender_*_acceptance.py`,
`blender_memory_safety_gate.py`, `headless_*`) still hardcode or default to
`cuda:0`; they need Blender to verify. `test_device_preference_routing.py`
uses `cuda:0` as routing data, which is intended.

## Evidence

- Source finding: `TEST-010` in `reviews/swarm-2026-08-08/findings/tests.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Test Coverage lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
