# TASK-REVIEW-021: cache_path RNA property used without sanitization against directory traversal

## State

completed

## Severity

medium

## What to build

write_frame() resolves the cache directory via 'base = bpy.path.abspath(fs.cache_path)' (line 340) and immediately calls os.makedirs(base, exist_ok=True) and writes files like fluidsim_00000.npz to that location. The cache_path StringProperty (properties.py:1250) is stored in the .blend file and can be set to any string including '../../' traversal sequences or absolute system paths. bpy.path.abspath resolves '//' prefixes relative to the blend file but does not sanitize '..' components. A crafted .blend file could direct cache writes to arbitrary filesystem locations.

**Location:** `fluidSim_warp_blender51/io/cache.py:340`
**Dimension:** Trust & Safety

## Impact

Opening a malicious .blend file and running the simulation could write cache files (fluidsim_NNNNN.npz, sequence_metadata.json) to unintended directories. While the filenames are fixed and content is simulation data, this could overwrite identically-named files or fill disk in sensitive locations.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/io/cache.py:340` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: After resolving with bpy.path.abspath, normalize the path with os.path.normpath and validate it is within an expected root (e.g., the blend file directory or user home). Reject paths containing '..' after normalization, or confine writes to a sandboxed subdirectory.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SEC-004` in `reviews/swarm-2026-08-08/findings/security.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Trust & Safety lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Resolved in 1.1.0. The cache path is an artist-chosen folder by design (scratch disks,
network shares), so confining it to the .blend directory would break real workflows;
FluidSim only writes its own fixed names (`fluidsim_cache.json`, `frames/frame_*.npz`,
`checkpoints/checkpoint_*.npz`). The real harm behind this finding was deletion:
Restart removed any `frames`, `checkpoints` or `logs` folder under the chosen path even
when the folder was not a FluidSim cache.

- `blender/cache_manager.py`: new `require_restartable_cache_directory` refuses a
  restart that would replace a non-empty `frames`/`checkpoints`/`logs` folder (or a
  stray `fluidsim_cache.json`) in a folder that is not a recognized FluidSim cache.
- Both restart paths call it before touching anything: `_RestartCacheProtection.stage`
  (`blender/cache_bake.py`) and `ProductionBakeBackend._clear_for_restart`
  (`blender/production_bake.py`). Free, move and rename were already guarded by
  `_require_managed_cache_directory`.
- Regression tests: `tests/test_artist_bake_safety.py::test_restart_never_stages_or_clears_user_folders_outside_a_cache`
  and two unit tests in `fluidSim_warp_blender51/tests/test_cache_manager.py`.

## Blocked by

None - can start immediately
