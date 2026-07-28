# Phase 4 — Master Integration Evidence (2026-07-28)

## Scope

Phase 4 was integrated in the clean worktree:

- Worktree: `C:\Users\Tyler\Downloads\liquid_warp_blender51_port\liquid_warp_blender51_phase4_commit`
- Branch: `phase4/live-interaction-merge-20260728`
- Base: master `0df90d4`

This is a parent-authored integration record, not a replacement for the independent Gate-A/Gate-B reviews.

## Preserved master behavior

- frame-handler reentrancy protection and dropped-frame catch-up;
- handler teardown and cache scene-key mismatch rejection;
- centralized configured-device ownership;
- operator/domain draw guards;
- device-side display stride and padded point-cloud capacity;
- current CFL, pressure, sourcing, packaging, and production-gate fixes.

## Preserved Phase-4 behavior

- atomic live-setting reconciliation and interaction tiers;
- per-scene pending signatures, rewind application, and explicit operators;
- fixed header, panel, and overlay feedback;
- ST-FLIP live reference-mass scaling;
- cache settings signatures with baseline/delta metadata;
- Phase-4 unit, Blender/CUDA, and GUI acceptance lanes.

## Verification

```text
python -m pytest -q
410 passed

blender --background --factory-startup --python tests/blender_phase4_live_interaction_acceptance.py
PHASE4-LIVE-INTERACTION-PASS

blender --background --factory-startup --python tests/blender_handler_persistence_gate.py
HANDLER-PERSISTENCE-GATE-PASS

blender --background --factory-startup --python tests/blender_viewport_2frame_gate.py
VIEWPORT-2FRAME-GATE-PASS

blender --background --factory-startup --python tests/blender_cache_export_pipeline.py
CACHE-EXPORT-PIPELINE-PASS

python tools/run_production_gate.py --include-gui
GUI-ACCEPTANCE-PASS (38/38)
GUI-PLAYBACK-PROBE-PASS
SAVED-DEMO-GUI-PLAYBACK-PASS
FULL-PRODUCTION-GATE-PASS
```

## Outstanding merge requirement

The main `master` worktree contains separate uncommitted playback/session-handler changes (`ensure_session_before_frames` / `_on_playback_pre` and tests). They are not part of committed master and therefore are not included in this branch. Preserve and commit that work before merging this Phase-4 branch into master, then reconcile it with the integrated `scene_bridge.py` and rerun the handler and full production gates.
