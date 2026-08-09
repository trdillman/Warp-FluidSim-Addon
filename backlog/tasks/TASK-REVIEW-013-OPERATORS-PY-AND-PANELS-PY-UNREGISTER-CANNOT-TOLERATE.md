# TASK-REVIEW-013: operators.py and panels.py unregister() cannot tolerate partial registration, leaking classes on rollback

## State

ready-for-agent

## Severity

medium

## What to build

operators.py unregister() (lines 1571-1573) and panels.py unregister() (lines 1310-1311) call bpy.utils.unregister_class(cls) in a bare loop with no error handling. In contrast, properties.py unregister() (lines 1312-1318) correctly wraps each unregister_class in try/except (RuntimeError, ValueError). The __init__.py register() rollback (lines 48-59) only calls the undo function for steps that fully completed (completed.append happens AFTER do_register() succeeds). If operators.register() or panels.register() fails partway through their CLASSES tuple, the module is never added to the completed list, so its unregister() is not called during rollback. Even if it were called, the bare unregister_class loop would crash on the first never-registered class, leaving all remaining registered classes leaked. operators.py has 42 classes and panels.py has 26 classes, so a mid-loop failure leaks a significant number of Blender RNA types that persist until Blender restart.

**Location:** `fluidSim_warp_blender51/operators.py:1573`
**Dimension:** Architecture

## Impact

A failed add-on enable leaves orphaned Blender RNA classes (operators, panels) registered with no owner. Subsequent enable attempts hit 'already registered' errors. The user must restart Blender to recover.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/operators.py:1573` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Wrap each bpy.utils.unregister_class(cls) call in try/except (RuntimeError, ValueError) in operators.py and panels.py, matching the pattern already in properties.py. Additionally, consider having each module's register() use an internal try/except that unwinds its own partial registrations before re-raising.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-002` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
