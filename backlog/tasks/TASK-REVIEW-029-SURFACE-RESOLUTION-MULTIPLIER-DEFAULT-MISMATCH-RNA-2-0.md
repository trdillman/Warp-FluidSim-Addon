# TASK-REVIEW-029: surface_resolution_multiplier default mismatch: RNA 2.0 vs dataclass 1.0

## State

completed

## Severity

medium

## What to build

The RNA FloatProperty 'surface_resolution_multiplier' (properties.py line 989) has default=2.0. The corresponding field in AuthoredQualitySettings (quality_contract.py line 34) has default=1.0. In scene_inputs.py _effective_source_target() (line 20-25), an AuthoredQualitySettings is constructed passing only voxel_size and liquid_sampling, so surface_resolution_multiplier silently defaults to 1.0 instead of the RNA default of 2.0. While this particular code path only uses liquid_sampling downstream, any future caller of resolve_quality() via AuthoredQualitySettings without explicitly passing surface_resolution_multiplier will get the wrong default (1.0 instead of 2.0), producing a surface voxel size 2x coarser than expected.

**Location:** `fluidSim_warp_blender51/properties.py:989`
**Dimension:** Type System

## Impact

Latent inconsistency: AuthoredQualitySettings constructed without explicit surface parameters resolves surface resolution multiplier as 1.0 (no change) rather than 2.0 (2x finer). If this path is later used for surface resolution decisions, surface detail would silently halve.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/properties.py:989` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Align the defaults: either set AuthoredQualitySettings.surface_resolution_multiplier default to 2.0 to match RNA, or always pass the RNA value when constructing AuthoredQualitySettings.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `TYPE-002` in `reviews/swarm-2026-08-08/findings/types.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Type System lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately

## Resolution

Fixed in `427a42f` on `claude/2.0-phase1-platform` (2.0 Phase 2, settings-drift pass).

Every literal fallback for `surface_resolution_multiplier` now equals the declared 2.0 (`AuthoredQualitySettings`, `resolve_quality`, `frame_output`, `surface`, `capacity_planner`). The same pass aligned `surface_leaf_capacity_scale` (8.0) and `surface_smooth_iterations` (1). The guard also pins the RNA/`SolverSettings` default agreement across 110 fields; three known divergences (`viscosity_constant`, `ww_bubble_drag`, `ww_spray_drag`) are named in the test and carried to the 2.1 handoff.

Regression guard: `tests/test_settings_default_agreement.py`.
