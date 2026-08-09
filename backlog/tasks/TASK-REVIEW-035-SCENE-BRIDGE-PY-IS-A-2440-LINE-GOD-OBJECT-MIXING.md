# TASK-REVIEW-035: scene_bridge.py is a 2440-line god-object mixing session lifecycle, scene translation, handler management, and schema migration

## State

ready-for-agent

## Severity

low

## What to build

scene_bridge.py at 2440 lines contains: the Session class (lines 99-1159, ~1060 lines), ScenePlanningSnapshot, SceneUpdate enum, SceneInteractionState, ~40 module-level functions, 5 @persistent handler callbacks, schema migration logic (migrate_source_sampling_schema at line 2376), cache validation (_check_cache_scene_key at line 1171), error management (set_error/clear_error/current_error), and handler registration/removal. This single module owns solver lifecycle, Blender scene-to-solver data translation, frame-change/depsgraph/undo/load handler registration, cache key validation, source schema migration, and the module-level singleton session instance. The Session class alone is ~1060 lines.

**Location:** `fluidSim_warp_blender51/blender/scene_bridge.py:99`
**Dimension:** Architecture

## Impact

High cognitive load for maintenance. Changes to handler logic risk unintended side effects on session management or scene translation. Difficult to test components in isolation. Ownership boundaries are unclear -- a bug in schema migration could affect handler registration.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender/scene_bridge.py:99` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Split into focused modules: session.py (Session class + singleton), handlers.py (callbacks + register/unregister), scene_sync.py (scene-to-solver translation), and schema_migration.py (migrate_source_sampling_schema). Keep scene_bridge.py as a thin re-export facade for backward compatibility with existing callers.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-005` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
