# TASK-REVIEW-028: dissipation_rate uses subtype FACTOR without max=1.0 bound

## State

completed

## Severity

medium

## What to build

The FloatProperty 'dissipation_rate' (line 851-854) declares subtype="FACTOR" but omits max=1.0. In Blender, the FACTOR subtype implies a normalized 0..1 range (displayed as a percentage slider). Every other property in this file that represents a 0..1 normalized value has an explicit max=1.0: flip_ratio (line 788), relax_strength (line 799), limit_percent (line 815), surface_filter_strength (line 1034), surface_dilate_strength (line 1042), surface_erode_strength (line 1050), surface_smooth_strength (line 1062). Without the max bound, values above 1.0 are accepted by RNA and could cause numerical instability in the exponential attribute decay solver (the description says 'Set exponential attribute decay per second').

**Location:** `fluidSim_warp_blender51/properties.py:851`
**Dimension:** Type System

## Impact

Users can set dissipation rates above 1.0 via the UI or API. Depending on solver implementation, this may cause exponential decay overshoot or instability. At minimum, the FACTOR slider displays inconsistently with its implied 0..1 semantics.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/properties.py:851` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Add max=1.0 to the FloatProperty definition, or if rates above 1.0 are intentionally allowed, remove subtype="FACTOR" to avoid the semantic mismatch.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-001` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed in `427a42f` on `claude/2.0-phase1-platform` (2.0 Phase 2, settings-drift pass).

`dissipation_rate` drops `subtype="FACTOR"` (the remediation's second option): the decay kernel computes `exp(-rate*dt)`, which is stable for any rate >= 0, so rates above 1.0 are intentional. The guard also requires every FACTOR property to be bounded at or below 1.0.

Regression guard: `tests/test_settings_default_agreement.py`.
