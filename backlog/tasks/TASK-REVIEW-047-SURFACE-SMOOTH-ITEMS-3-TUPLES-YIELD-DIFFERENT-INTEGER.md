# TASK-REVIEW-047: SURFACE_SMOOTH_ITEMS 3-tuples yield different integer backing values than SdfFilterType enum

## State

completed

## Severity

low

## What to build

SURFACE_SMOOTH_ITEMS (line 87-90) defines two items as 3-tuples: ("BOX", ...) and ("MEAN_CURVATURE_FLOW", ...). Blender assigns sequential index integers to 3-tuple items, so BOX gets backing value 0 and MEAN_CURVATURE_FLOW gets value 1. However, the same identifiers in the SdfFilterType IntEnum (enums.py line 64-69) have values BOX=2 and MEAN_CURVATURE_FLOW=3. The surface_filter_type property (line 1025-1028) uses _items(SdfFilterType) which generates 4-tuples with the correct enum values. If any downstream code maps the integer value from surface_smooth_type to SdfFilterType (e.g., via .value comparison or C-level RNA access), it would resolve to the wrong filter type: smooth BOX(0) would map to SdfFilterType.DIALATE and smooth MEAN_CURVATURE_FLOW(1) to SdfFilterType.ERODE.

**Location:** `fluidSim_warp_blender51/properties.py:87`
**Dimension:** Type System

## Impact

No active bug confirmed because Python-level Blender property access returns string identifiers (which are correct). However, the integer value mismatch is a latent trap for any code that reads the integer enum value through RNA internals or maps between smooth_type and filter_type.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/properties.py:87` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Define SURFACE_SMOOTH_ITEMS as 4-tuples with explicit SdfFilterType values: ("BOX", "Box", "...", SdfFilterType.BOX.value) and ("MEAN_CURVATURE_FLOW", "Mean Curvature", "...", SdfFilterType.MEAN_CURVATURE_FLOW.value).
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-005` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed in `427a42f` on `claude/2.0-phase1-platform` (2.0 Phase 2, settings-drift pass).

Equivalent approach, not the literal remediation. `SURFACE_SMOOTH_ITEMS` is now 4-tuples that pin the existing saved numbers 0/1 explicitly. Renumbering to the `SdfFilterType` values 2/3 would flip the smoothing type in every saved file, because Blender stores the item number. The solver converts by identifier (`enum_type[identifier]` in `blender/solver_settings.py`), so no code maps the number. The guard pins the 0/1 numbers and requires the identifiers to be `SdfFilterType` members.

Regression guard: `tests/test_settings_default_agreement.py`.
