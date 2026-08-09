# TASK-REVIEW-042: read_sequence_metadata silently swallows corrupt metadata files

## State

ready-for-agent

## Severity

low

## What to build

read_sequence_metadata catches (OSError, json.JSONDecodeError) at line 319 and returns None, treating a corrupt metadata file identically to a missing one. In _check_cache_scene_key (scene_bridge.py line 1184), None is coerced to an empty dict via the `or {}` fallback, so metadata_key becomes an empty string. This silently weakens the sequence-level scene_key validation rather than alerting the user. If the per-frame scene_key is also absent (old cache format or empty scene_key), all validation is bypassed without any warning.

**Location:** `fluidSim_warp_blender51/io/cache.py:319`
**Dimension:** Serialization & IO

## Impact

A corrupt metadata file silently disables cache integrity validation instead of surfacing the problem, potentially allowing playback of a stale or mismatched cache without warning.

## Acceptance criteria

- [ ] Root cause at `fluidSim_warp_blender51/io/cache.py:319` is fixed per the recommended remediation (or an equivalent approach with the same safety guarantee).
- [ ] Fix: Log a warning when a metadata file exists but cannot be parsed (distinguishing corruption from a simple missing file). Optionally raise rather than return None for corrupt-but-present files.
- [ ] A regression test/gate proves the defect cannot recur (unit test for logic bugs; gate/acceptance script for runtime/UI/registration issues).
- [ ] No existing test suite regresses as a result of the fix.

## Evidence

- Source finding: `SERIO-003` in `reviews/swarm-2026-08-08/findings/io.json`, cross-referenced in `reviews/swarm-2026-08-08/CONSOLIDATED_CODEBASE_REVIEW.md` (Low Findings section).
- Identified by the Serialization & IO lane of the 2026-08-08 swarm codebase review (8-lane read-only static review, ~56K LOC).
- Citation manually re-verified against source during /double-check on 2026-08-08 (see session record); this finding survived verification.

## Blocked by

None - can start immediately
