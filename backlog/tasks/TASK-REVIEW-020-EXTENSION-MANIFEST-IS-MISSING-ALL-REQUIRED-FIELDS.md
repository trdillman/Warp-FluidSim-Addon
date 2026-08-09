# TASK-REVIEW-020: Extension manifest is missing all required fields including permissions declarations

## State

ready-for-agent

## Severity

medium

## What to build

blender_manifest.toml contains only 'schema_version = "1.0.0"' (line 1). It is missing id, version, name, tagline, maintainer, type, license, and permissions. Blender 5.1's extension system requires declared permissions for filesystem access, network access, and subprocess execution. The add-on performs filesystem writes (cache.py write_frame, export.py), subprocess execution (memory_safety.py nvidia-smi, warp_discovery.py python discovery), and ctypes FFI (cuda_reduction.py, memory_safety.py), none of which are declared. The empty manifest means Blender's trust boundary cannot inform users what the extension accesses.

**Location:** `fluidSim_warp_blender51/blender_manifest.toml:1`
**Dimension:** Trust & Safety

## Impact

Users installing this extension receive no permission warnings. The extension system cannot enforce sandboxing or alert users to the filesystem, subprocess, and native code access the add-on performs. This breaks the trust model Blender 5.1 extensions are designed to provide.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender_manifest.toml:1` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Populate blender_manifest.toml with all required fields. Declare permissions for filesystem access (e.g., 'files' for cache/export), and document subprocess/native access. At minimum: id, version, name, tagline, maintainer, type='add-on', license=['SPDX:GPL-3.0-or-later'], and permissions = { files = 'Cache and export file access' }.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-003` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
