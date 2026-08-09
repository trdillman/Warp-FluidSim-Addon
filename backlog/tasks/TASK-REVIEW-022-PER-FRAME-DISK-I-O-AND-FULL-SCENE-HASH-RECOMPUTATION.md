# TASK-REVIEW-022: Per-frame disk I/O and full scene hash recomputation during cache playback

## State

ready-for-agent

## Severity

medium

## What to build

_check_cache_scene_key (called from play_cache_frame at line 876, invoked every frame from handle_frame_change at line 1079) reads the sequence metadata JSON from disk on every frame (line 1184: cache.read_sequence_metadata(base)) and fully recomputes the scene content hash (line 1190: cache.scene_key(scene)). scene_key builds a settings dataclass via build_settings, hashes every field, iterates all scene objects, and hashes RNA properties plus matrix_world transforms for each fluidsim-tagged object. This redundant O(n_settings + n_objects) computation plus file I/O runs at playback framerate (24-60 fps).

**Location:** `fluidSim_warp_blender51/blender/scene_bridge.py:1184`
**Dimension:** Serialization & IO

## Impact

Stuttering or frame drops during cache playback on complex scenes due to repeated disk reads and expensive hash recomputation per frame, degrading the user experience for the primary cache playback workflow.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender/scene_bridge.py:1184` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Cache the computed scene_key for the duration of a cache playback session (compute once when cache_playback_enabled is toggled on, invalidate on scene edit or playback toggle). Cache the metadata scene_key after first read rather than re-reading the JSON file every frame.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SERIO-001` in `reviews/swarm-2026-08-08/findings/io.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Serialization & IO lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
