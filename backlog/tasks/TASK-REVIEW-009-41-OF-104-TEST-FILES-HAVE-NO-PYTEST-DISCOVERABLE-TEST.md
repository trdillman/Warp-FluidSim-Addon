# TASK-REVIEW-009: 41 of 104 test files have no pytest-discoverable test functions -- entire acceptance/gate test suite is invisible to pytest

## State

completed

## Severity

high

## What to build

Systematic analysis shows 41 of 104 test files in tests/ contain zero 'def test_*' functions. These are acceptance/gate/runner scripts designed for 'blender --python' invocation, not pytest. Critical files affected include: blender_memory_safety_gate.py (19 asserts, 0 test functions), blender_capacity_planner_acceptance.py (44 asserts, 0 test functions), blender_whitewater_truth_acceptance.py (18 asserts, 0 test functions), blender_mixed_source_batch_acceptance.py (14 asserts, 0 test functions), blender_gui_acceptance.py (647 lines, 0 test functions), blender_source_sampling_acceptance.py (13 asserts, 0 test functions). Five files (blender_animated_source_acceptance.py, blender_point_source_acceptance.py, blender_source_lifecycle_acceptance.py, blender_source_region_acceptance.py, blender_volume_source_acceptance.py) are 5-line stubs that just call main() from _phase2_source_acceptance_common at module scope.

**Location:** `tests/:1`
**Dimension:** Test Coverage

## Impact

A 'pytest tests/' run skips the entire acceptance/gate suite silently. Coverage reports will show these files as untested. Developers may believe they have 104 test files when pytest only discovers ~63. The memory safety gate, capacity planner acceptance, and whitewater truth acceptance tests are effectively invisible to standard Python test tooling.

## Resolution

Fixed on `claude/2.0-phase1-platform` (2.0 Phase 2) with remediation (b): the
Blender scripts keep their `blender --python` form, and their runner is
`tools/run_production_gate.py`, not pytest. Pytest never collects
`blender_*.py`, so the module-scope `main()` stubs have no collection side
effect.

The real gap was that nothing ran most of them:

- Thirteen `tests/blender_*.py` acceptance scripts were outside the gate, and
  five had been broken since 1.0.255 (they read the removed `bl_info`, or wrote
  domain settings to `scene.fluidsim` after the 1.1 split). All are fixed and
  now run in `BACKGROUND_ACCEPTANCE`/`SOURCE_REPORT_ACCEPTANCE`. GUI-only
  probes are listed in `MANUAL_BLENDER_PROBES` with a reason.
- None of the six `tools/release/validate_*_blender.py` release validators was
  run by anything; five of the six had broken against the split. All six are
  fixed and pass on Blender 5.2, and they now run as the gate's
  `RELEASE_BLENDER_VALIDATORS` lane.
- Root cause of the silent breakage: Blender accepts any Python attribute on a
  PropertyGroup, so `scene.fluidsim.voxel_size = x` "worked" and never reached
  the solver. `FLUIDSIM_PG_scene`/`FLUIDSIM_PG_domain` now raise
  `AttributeError` naming the right group.
- Found along the way: the Live Tuning shelf had dropped its display-fraction
  knob since the split, and examples 01-04 shipped with pre-1.1 settings (2.0
  flagged its own examples). Both fixed; the examples were regenerated.

Regression gates:

- `tests/test_production_gate_coverage.py` fails if any `tests/blender_*.py`
  or `tools/release/validate_*_blender.py` is neither gated nor excused, or if
  a gate marker or argument does not exist in its script.
- `tests/test_file_compat.py::test_writing_a_setting_to_the_wrong_group_raises`
  covers the guard.
- `validate_examples_blender.py` refuses an example that 2.0 flags as pre-1.1.

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
