# TASK-REVIEW-046: Missing VELOCITY subtype on max_speed and height_velocity

## State

completed

## Severity

low

## What to build

The FloatProperty 'max_speed' (line 806-809) is described as 'particle speed limit in meters per second' but lacks subtype="VELOCITY". The FloatVectorProperty 'height_velocity' (line 1127-1130) is described as 'initial velocity for height-fill particles' but also lacks the subtype. In contrast, the analogous 'initial_velocity' (line 601-604) correctly uses subtype="VELOCITY", and 'field_vector_value' (line 512-515) also uses subtype="VELOCITY".

**Location:** `fluidSim_warp_blender51/properties.py:806`
**Dimension:** Type System

## Impact

Blender's UI will not display velocity unit suffixes for these properties. Minor UX inconsistency; no functional impact on solver behavior.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/properties.py:806` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Add subtype="VELOCITY" to max_speed and subtype="VELOCITY" to height_velocity.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-004` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed in `427a42f` on `claude/2.0-phase1-platform` (2.0 Phase 2, settings-drift pass).

`height_velocity` uses `subtype="VELOCITY"`. A scalar `FloatProperty` has no VELOCITY subtype, so `max_speed` uses `unit="VELOCITY"`, which gives the same display. Blender 5.2 registration was verified by the background acceptance run.

Regression guard: `tests/test_settings_default_agreement.py`.
