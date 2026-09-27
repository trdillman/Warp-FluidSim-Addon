# TASK-REVIEW-043: Non-atomic gap between frame write and metadata update in live cache path

## State

obsolete

## Severity

low

## What to build

In Session.step, the live cache write path writes the frame NPZ (line 831: cache.write_frame) and then writes the sequence metadata JSON (lines 842-848: cache.write_sequence_metadata) as two separate atomic operations. If a crash occurs between them, the metadata's frame_end and settings_records are stale: the frame file exists on disk but the metadata does not reference it. The metadata's settings_segments, settings_baseline, and settings_changes would all be out of date.

**Location:** `fluidSim_warp_blender51/blender/scene_bridge.py:831`
**Dimension:** Serialization & IO

## Impact

After a crash, metadata reports a narrower or inconsistent range compared to what exists on disk. Practical impact is low because play_cache_frame reads frames by index directly and available_frames scans filenames, but metadata consumers see inconsistent state.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/blender/scene_bridge.py:831` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Accept the minor inconsistency for supplemental caching (current behavior is documented as best-effort), or write a preliminary metadata marker before the frame and a confirming one after, so post-crash recovery can detect the incomplete write.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SERIO-004` in `reviews/swarm-2026-08-08/findings/io.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Serialization & IO lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Resolution

Obsolete in 1.1.0. The live cache write in `Session.step` (`cache.write_frame` followed by `cache.write_sequence_metadata`) was removed in commit `bb6d3fc`. No add-on code calls either function any more.

Cache frames are written only by Production Bake (`fluidSim_warp_blender51/blender/production_bake.py`, also used by `cache_bake.bake`). It writes each frame with `cache.write_frame_v2_atomic` and a checkpoint with `write_checkpoint_atomic`, then commits the manifest last (`with_committed_frame`). A crash leaves uncommitted files as orphans, which the manifest-authoritative cache inspection ignores and reports (`fluidSim_warp_blender51/tests/test_cache_manager.py::test_inspection_is_manifest_authoritative_and_reports_orphans`, `::test_interrupted_first_transaction_is_recognized_as_managed_cache`).

## Blocked by

None - can start immediately
