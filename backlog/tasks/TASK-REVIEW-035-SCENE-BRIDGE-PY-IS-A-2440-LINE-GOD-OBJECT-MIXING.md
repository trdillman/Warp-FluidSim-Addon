# TASK-REVIEW-035: scene_bridge.py is a 2440-line god-object mixing session lifecycle, scene translation, handler management, and schema migration

## State

completed

## Severity

low

## What to build

scene_bridge.py at 2440 lines contains: the Session class (lines 99-1159, ~1060 lines), ScenePlanningSnapshot, SceneUpdate enum, SceneInteractionState, ~40 module-level functions, 5 @persistent handler callbacks, schema migration logic (migrate_source_sampling_schema at line 2376), cache validation (_check_cache_scene_key at line 1171), error management (set_error/clear_error/current_error), and handler registration/removal. This single module owns solver lifecycle, Blender scene-to-solver data translation, frame-change/depsgraph/undo/load handler registration, cache key validation, source schema migration, and the module-level singleton session instance. The Session class alone is ~1060 lines.

**Location:** `fluidSim_warp_blender51/blender/scene_bridge.py:99`
**Dimension:** Architecture

## Impact

High cognitive load for maintenance. Changes to handler logic risk unintended side effects on session management or scene translation. Difficult to test components in isolation. Ownership boundaries are unclear -- a bug in schema migration could affect handler registration.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/blender/scene_bridge.py:99` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Split into focused modules: session.py (Session class + singleton), handlers.py (callbacks + register/unregister), scene_sync.py (scene-to-solver translation), and schema_migration.py (migrate_source_sampling_schema). Keep scene_bridge.py as a thin re-export facade for backward compatibility with existing callers.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-005` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed before 1.1.0. `fluidSim_warp_blender51/blender/scene_bridge.py` is down from 2440 to 567 lines. Its responsibilities were split out: handlers and cache playback (including `_check_cache_scene_key`) to `blender/playback.py`, source schema migration (`migrate_source_sampling_schema`) to `blender/sources.py`, pending settings to `blender/pending_settings.py`, admission to `blender/simulation_admission.py`, output publication to `blender/output_publication.py`, and bakes to `blender/cache_bake.py` / `blender/production_bake.py`.

Split landed in the architecture series `13a482c`, `7e19cb7`, `e363a6f`, `4b6c2b7`, `afbdf0e`, merged by `fff298e`.

No dedicated regression test (no size or structure guard); behaviour is covered by the existing per-module suites.

## Blocked by

None - can start immediately
