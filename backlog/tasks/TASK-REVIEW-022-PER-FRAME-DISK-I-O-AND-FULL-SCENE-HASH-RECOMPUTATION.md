# TASK-REVIEW-022: Per-frame disk I/O and full scene hash recomputation during cache playback

## State

completed

## Severity

medium

## What to build

_check_cache_scene_key (called from play_cache_frame at line 876, invoked every frame from handle_frame_change at line 1079) reads the sequence metadata JSON from disk on every frame (line 1184: cache.read_sequence_metadata(base)) and fully recomputes the scene content hash (line 1190: cache.scene_key(scene)). scene_key builds a settings dataclass via build_settings, hashes every field, iterates all scene objects, and hashes RNA properties plus matrix_world transforms for each fluidsim-tagged object. This redundant O(n_settings + n_objects) computation plus file I/O runs at playback framerate (24-60 fps).

**Location:** `fluidSim_warp_blender51/blender/scene_bridge.py:1184`
**Dimension:** Serialization & IO

## Impact

Stuttering or frame drops during cache playback on complex scenes due to repeated disk reads and expensive hash recomputation per frame, degrading the user experience for the primary cache playback workflow.

## Acceptance criteria

- [x] Root cause at `fluidSim_warp_blender51/blender/scene_bridge.py:1184` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [x] Fix: Cache the computed scene_key for the duration of a cache playback session (compute once when cache_playback_enabled is toggled on, invalidate on scene edit or playback toggle). Cache the metadata scene_key after first read rather than re-reading the JSON file every frame.
- [x] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [x] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SERIO-001` in `reviews/swarm-2026-08-08/findings/io.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Medium Findings section).
- Identified by the Serialization & IO lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Fixed for 2.0 (2026-09-27). Working this card exposed a P1 defect that was worse than the reported cost.

The P1 bug: a production bake with an animated source could not play itself back. The shipped paddle_stir demo baked cleanly, played frame 1, then refused frames 2-5 with "Cache frame 2 was baked from different scene settings". The per-frame check compared the full `scene_key`, which also hashes every source's `matrix_world` and evaluated geometry, so a keyframed collider changed the key on every frame. Resume had the same fault: its signatures were taken at whatever frame the playhead was on. The unit tests all stubbed the identity, which is why this was never caught.

Fix:
- `playback._check_cache_scene_key(scene, frame_data)` now compares only the frame's `settings_signature` with `cache_identity.settings_signature(scene)`. That signature does not depend on the frame. The check no longer re-reads the sequence metadata: `read_frame` has already verified the frame's stamps against the manifest. Frames without a settings stamp are not checked.
- `ProductionBakeBackend._signatures_at(scene, frame)` computes the bundle at the bake's reference frame (the bake start), with frame handlers suppressed, and restores the playhead afterwards. `initialize` uses it to stamp the manifest, and `resume_decision` uses it to check against `manifest.frame_start`.

Cost: measured on Blender 5.2 across the four demos, the per-frame check is now about 0.5 ms (settings only) with no disk I/O. The full bundle it replaces cost up to 2.2 ms (chromatic_vortex), plus a JSON read. The card's session memo was not added: a cached signature would miss exactly the settings edits the check exists to catch, and 0.5 ms is about 3% of a 60 fps frame.

Tests:
- `tests/test_scene_bridge_cache_playback.py::test_cache_playback_compares_only_the_frame_settings_signature`: no metadata read and no source signature per frame, a scene-key difference alone is accepted, and a settings difference is refused.
- The existing playback tests were moved to the settings stamp.
- New real-Blender gate `tests/blender_animated_cache_playback_acceptance.py`, registered in `BACKGROUND_ACCEPTANCE`. It uses a keyframed collider and runs a partial bake, a resume with the playhead elsewhere, and a completed bake. Every frame must play back, and a settings edit must still be refused. Against the unfixed code this gate fails with SCENE_KEY_MISMATCH.

Owner note: playback now detects settings edits but not source edits (moved or re-modelled colliders/emitters). Resume still checks sources, at the start frame. A per-frame source stamp is proposed for 2.1.

## Blocked by

None - can start immediately
