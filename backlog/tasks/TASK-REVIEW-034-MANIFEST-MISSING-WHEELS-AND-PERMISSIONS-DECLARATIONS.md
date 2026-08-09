# TASK-REVIEW-034: Manifest missing wheels and permissions declarations for Warp dependency and filesystem access

## State

ready-for-agent

## Severity

low

## What to build

blender_manifest.toml (18 lines total) declares no [wheels] section and no permissions. The add-on depends on NVIDIA Warp (a native CUDA Python package), which is discovered at runtime via warp_discovery.py using subprocess calls, sys.path manipulation, and filesystem traversal. For Blender 5.1 extensions distributed via the extensions platform, the manifest is the mechanism for declaring external dependencies and required permissions. Without a wheels declaration, the extensions platform cannot bundle or validate the Warp dependency. Without a permissions declaration, filesystem and subprocess access may be restricted on locked-down Blender configurations.

**Location:** `fluidSim_warp_blender51/blender_manifest.toml:18`
**Dimension:** Architecture

## Impact

Users installing from the extensions platform get no dependency resolution for Warp. The add-on silently fails at runtime if Warp is not pre-installed in a discoverable location. Subprocess-based discovery in warp_discovery.py may be blocked by Blender's sandbox on some configurations.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender_manifest.toml:18` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Add a [wheels] section bundling or referencing the Warp wheel for target platforms, or document the external dependency in the manifest. Add appropriate permissions declarations if the extensions platform supports them for filesystem/subprocess access.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `ARCH-004` in `reviews/swarm-2026-08-08/findings/arch.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Architecture lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
