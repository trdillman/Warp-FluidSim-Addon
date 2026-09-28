# TASK-REVIEW-045: Missing DISTANCE subtype on voxel_size, surface_voxel_size, height_base, and feather

## State

completed

## Severity

low

## What to build

Four FloatProperties represent world-space distances but lack subtype="DISTANCE": voxel_size (line 725, described as 'world units'), surface_voxel_size (line 992, described as 'world units'), height_base (line 1108, described as 'world-space base fill level'), and feather (line 613, consumed as a world-space dimension in AnalyticRegionPlan at scene_inputs.py line 496). In contrast, other distance properties in the same file correctly use subtype="DISTANCE": omnidirectional_force_radius (line 831), attraction_max_distance (line 866), mesh_surface_thickness (line 553), and point_rasterize_radius (line 452).

**Location:** `fluidSim_warp_blender51/properties.py:725`
**Dimension:** Type System

## Impact

Blender's UI will not display unit suffixes or apply scene-unit scaling for these four properties, creating an inconsistent user experience where some distance fields show meters and others do not.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/properties.py:725` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Add subtype="DISTANCE" to voxel_size, surface_voxel_size, height_base, and feather.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-003` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed in `427a42f` on `claude/2.0-phase1-platform` (2.0 Phase 2, settings-drift pass).

`voxel_size`, `surface_voxel_size`, `height_base` and `feather` use `subtype="DISTANCE"`.

Regression guard: `tests/test_settings_default_agreement.py`.
