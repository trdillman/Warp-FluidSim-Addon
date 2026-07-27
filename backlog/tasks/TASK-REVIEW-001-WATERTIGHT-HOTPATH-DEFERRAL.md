# TASK-REVIEW-001: Defer watertight validation off the mesh gather hot path

## State

completed

## What to build

Stop running `_watertight_triangles` on every evaluated-mesh rebuild. The edge-manifold analysis is only ever consumed by SOLID mesh sinks, but it was computed unconditionally for every collider and mesh source inside `frame_change_post`, as a scalar Python loop over all triangles. Compute it lazily on first SOLID-sink read, memoize it into the cache entry, and vectorize the check itself.

## Acceptance criteria

- [x] `fluidSim_warp_blender51/blender/scene_inputs.py` no longer calls `_watertight_triangles` while building an evaluated-mesh cache entry; the entry stores a deferred sentinel plus the triangle array.
- [x] A new `_entry_is_watertight(entry)` helper evaluates once and memoizes, and is the only read path for the SOLID mesh sink branch.
- [x] `_watertight_triangles` is vectorized (`np.sort` + `np.unique(..., return_counts=True)`) with unchanged semantics, including degenerate inputs.
- [x] Red-first regression proves the deferral: `tests/test_scene_bridge_collider_cache.py::test_watertight_check_is_deferred_to_solid_sink_consumers` asserts zero calls for a plain COLLIDER and non-zero for a SOLID sink. Demonstrated failing before the fix.
- [x] Semantics guard pins existing behavior for empty / single-triangle / closed-cube / coincident-triangle inputs (`test_watertight_triangles_semantics`).
- [x] SOLID mesh sink rejection still works without fallback: `blender --background --factory-startup --python tests/blender_mesh_source_acceptance.py` prints `MESH-SOURCE-ACCEPTANCE-PASS` and reports "Solid mesh sink requires a watertight mesh".
- [x] Live measurement confirms the hot path is clear.

## Evidence

- Before: 8 calls over 4 playback frames on a 122,880-triangle animated COLLIDER at 394-459 ms per call; 4-frame playback 2171.7 ms (`artifacts/live_review/probe3.json`).
- After: **0 calls**, 4-frame playback **458.2 ms** (`artifacts/live_review/probe3_after.json`).
- Regression lanes: host pytest `261 passed`; `MESH-SOURCE-ACCEPTANCE-PASS`; `BACKGROUND-ACCEPTANCE-PASS`; `VIEWPORT-2FRAME-GATE-PASS`; `CACHE-EXPORT-PIPELINE-PASS`.
- Source finding: `reviews/ADDON_REVIEW.md` (B5, CONFIRMED-WORSE) and `reviews/CORRECTION_PLAN.md` §0.

## Notes

The first draft of the semantics guard asserted that two coincident triangles are *not* watertight. That was an unrequested behavior change, not the existing contract: every edge has count 2, so the existing criterion returns `True`. The test was changed to pin current behavior instead.

`entry["watertight_tris"]` retains a reference to the triangle array (not a new copy; the cache signature already holds those bytes). Revisit when the signature moves from raw bytes to a hash.

## Blocked by

None - completed
