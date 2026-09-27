# TASK-REVIEW-012: Manifest version desynchronized from bl_info after uncommitted version bump

## State

completed

## Severity

medium

## What to build

blender_manifest.toml line 6 declares version = "1.0.241" while __init__.py line 13 declares "version": (1, 0, 242). The uncommitted git diff bumps __init__.py from (1, 0, 241) to (1, 0, 242) but the manifest was not updated. The manifest's own comment on line 3 states: 'Mirrors the legacy bl_info dict in __init__.py; keep the two in sync.' The Blender extensions platform uses the manifest version (not bl_info) for update tracking and display, so users and the platform will see a stale version.

**Location:** `fluidSim_warp_blender51/blender_manifest.toml:6`
**Dimension:** Architecture

## Impact

Blender extensions platform reports an incorrect (lower) version. Update detection and changelog display are wrong. The 'keep in sync' contract documented in the manifest comment is violated.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/blender_manifest.toml:6` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Bump blender_manifest.toml version to "1.0.242" to match bl_info. Consider a single source-of-truth pattern (e.g., reading version from one location at build time) to prevent future drift.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-001` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed before 1.1.0. `bl_info` was removed from `fluidSim_warp_blender51/__init__.py`; `fluidSim_warp_blender51/version.py` (`VERSION`, `EXTENSION_ID`, `BLENDER_MIN_VERSION`) is the single release identity and `blender_manifest.toml` must match it (both 1.1.0).

Tests: `fluidSim_warp_blender51/tests/test_release_identity.py::test_manifest_matches_runtime_version_and_identity`, `::test_manifest_is_the_only_blender_metadata_surface`, and `tests/test_extension_manifest.py::test_source_manifest_matches_runtime_release_identity`.

`version.py` introduced in commit `bb6d3fc`.

## Blocked by

None - can start immediately
